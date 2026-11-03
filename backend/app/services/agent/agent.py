"""LangGraph ReAct agent executor (Module D) with reliability (Challenge 7).

Builds a manual ReAct loop with two phases per iteration:
  1. agent node — LLM decides the next tool or produces a final answer
  2. tools node — executes the selected tool and feeds result back

Every tool call is recorded as an AgentStep (with latency + tokens) so the
dashboard can show full agent execution visibility. Failures are fed back to
the model as tool messages so it can recover.

Thought-signature handling: Gemini 3+ models require a thought_signature
to be preserved and returned verbatim in subsequent turns. We implement a
native Google GenerativeAI SDK loop alongside the LangChain model chain so
that Gemini can be used reliably for tool calling. The native path is only
used when Gemini is the current provider; other providers use LangChain.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from ...config import settings
from ...models import AgentRun, AgentStep
from ..llm.base import LLMResponse, ProviderError
from ..llm.service import get_llm_service
from ..llm.templates import build_agent_system_prompt
from .reliability import with_timeout_and_retry
from .tools import TOOL_DESCRIPTIONS, AgentContext, build_tools

logger = logging.getLogger(__name__)

# Errors that mean "switch providers" rather than "die"
_FALLBACK_TRIGGERS = (
    "thought_signature",
    "rate limit",
    "rate_limit",
    "429",
    "overloaded",
    "capacity",
    "too many requests",
    "503",
    "unavailable",
    "resource_exhausted",
    "resourceexhausted",
    "quota",
    "timeout",
    "timed out",
)


def _is_fallback_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(t in msg for t in _FALLBACK_TRIGGERS)


class AgentState(TypedDict):
    messages: list[BaseMessage]
    query: str
    iteration: int
    steps: list[dict[str, Any]]
    final_answer: str
    total_tokens: int
    error: str


def _extract_text_content(content: Any) -> str:
    """Normalise LLM message content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", str(block)))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content) if content is not None else ""


# ------------------------------------------------------------------ Native Gemini loop
# We use the google-generativeai SDK directly for tool calling with Gemini,
# because it preserves thought_signature transparently.

def _make_native_gemini_runner(tools: list[Any], settings_ref: Any):
    """Build a callable that runs the Gemini ReAct loop natively, bypassing LangChain.

    Returns (final_answer, steps, total_tokens) or raises on unrecoverable error.
    """
    try:
        import google.generativeai as genai
        from google.generativeai.types import FunctionDeclaration, Tool as GeminiTool
    except ImportError:
        return None

    genai.configure(api_key=settings_ref.GEMINI_API_KEY)

    # Build Gemini-native tool declarations from our LangChain tools.
    gemini_tools = []

    def _clean_prop(schema: dict) -> dict:
        """Clean a single property schema for Gemini proto compatibility."""
        ALLOWED = {"type", "description", "items", "enum", "format", "nullable", "anyOf", "oneOf", "allOf"}
        cleaned = {}
        for k, v in schema.items():
            if k not in ALLOWED:
                continue
            if k in ("anyOf", "oneOf", "allOf") and isinstance(v, list):
                # For optional fields: anyOf: [{type: X}, {type: null}] -> type: X
                non_null = [i for i in v if i.get("type") != "null" and i.get("type") is not None]
                if non_null:
                    cleaned.update(_clean_prop(non_null[0]))
                else:
                    cleaned["nullable"] = True
                continue
            if isinstance(v, dict):
                cleaned[k] = _clean_prop(v)
            elif isinstance(v, list):
                cleaned[k] = [_clean_prop(i) if isinstance(i, dict) else i for i in v]
            else:
                cleaned[k] = v
        # Ensure every property has a type
        if "type" not in cleaned:
            cleaned["type"] = "string"
        return cleaned

    for lc_tool in tools:
        schema = lc_tool.args_schema.model_json_schema()
        props = {}
        for prop_name, prop_schema in schema.get("properties", {}).items():
            props[prop_name] = _clean_prop(prop_schema)
        # Only include required fields that exist in properties
        required = [r for r in schema.get("required", []) if r in props]
        clean_params = {
            "type": "object",
            "properties": props,
        }
        if required:
            clean_params["required"] = required
        try:
            fd = FunctionDeclaration(
                name=lc_tool.name,
                description=lc_tool.description,
                parameters=clean_params,
            )
            gemini_tools.append(fd)
        except Exception as exc:
            logger.warning("Could not create FunctionDeclaration for %s: %s", lc_tool.name, exc)

    gemini_tool_obj = GeminiTool(function_declarations=gemini_tools) if gemini_tools else None
    tool_map = {t.name: t for t in tools}

    model_name = settings_ref.GEMINI_CHAT_MODEL
    if model_name.startswith("models/"):
        model_name = model_name[7:]  # strip "models/" prefix for sdk

    def runner(query: str, system_prompt: str) -> tuple[str, list[dict], int]:
        model = genai.GenerativeModel(
            model_name=model_name,
            tools=[gemini_tool_obj] if gemini_tool_obj else [],
            system_instruction=system_prompt,
        )
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(query)

        steps: list[dict] = []
        step_index = 0
        total_tokens = 0
        max_iter = settings_ref.AGENT_MAX_ITERATIONS
        # Hard wall-clock budget — stop and answer before the HTTP timeout fires
        wall_budget = 90  # seconds total for the entire agent run
        wall_start = time.time()

        for iteration in range(max_iter):
            # Hard wall-clock check — if running too long, force a quick answer
            elapsed = time.time() - wall_start
            if elapsed > wall_budget:
                logger.warning("Agent hit wall-clock budget (%.0fs); forcing final answer", elapsed)
                try:
                    final_model = genai.GenerativeModel(model_name=model_name, system_instruction=system_prompt)
                    summary_prompt = f"Based on the tool results gathered so far (steps: {[s['tool_name'] for s in steps]}), provide a concise final answer to: {query}"
                    r2 = final_model.generate_content(summary_prompt)
                    text = r2.text if hasattr(r2, "text") else ""
                    return text or "I gathered some information but ran out of time to synthesize a full answer.", steps, total_tokens
                except Exception:
                    return "The agent ran out of time. Please try a simpler query.", steps, total_tokens

            # Count tokens from usage metadata
            try:
                um = response.usage_metadata
                total_tokens += (um.total_token_count or 0)
            except Exception:
                pass

            # Check if any function calls in response
            fn_calls = []
            for part in response.parts:
                if hasattr(part, "function_call") and part.function_call.name:
                    fn_calls.append(part.function_call)

            if not fn_calls:
                # No tool calls -> extract text answer
                text = ""
                for part in response.parts:
                    if hasattr(part, "text") and part.text:
                        text += part.text
                return text, steps, total_tokens

            # Execute each tool call
            from google.generativeai.types import content_types
            fn_responses = []
            for fc in fn_calls:
                step_index += 1
                tool_name = fc.name
                tool_args = dict(fc.args)
                lc_tool = tool_map.get(tool_name)
                if lc_tool is None:
                    result_data = {"error": f"Unknown tool: {tool_name}"}
                    steps.append({
                        "step_index": step_index,
                        "tool_name": tool_name,
                        "tool_input": tool_args,
                        "tool_output": "",
                        "success": False,
                        "error_message": f"Unknown tool: {tool_name}",
                        "latency_ms": 0,
                        "tokens": 0,
                    })
                else:
                    result = with_timeout_and_retry(
                        lambda tn=tool_name, ta=tool_args, lt=lc_tool: lt._run(**ta)
                    )
                    if result.success:
                        try:
                            result_data = json.loads(result.output)
                        except Exception:
                            result_data = {"output": str(result.output)}
                    else:
                        result_data = {"error": result.error}
                    steps.append({
                        "step_index": step_index,
                        "tool_name": tool_name,
                        "tool_input": tool_args,
                        "tool_output": result.output if result.success else "",
                        "success": result.success,
                        "error_message": result.error if not result.success else "",
                        "latency_ms": result.latency_ms,
                        "tokens": 0,
                    })
                fn_responses.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=tool_name,
                            response={"result": result_data},
                        )
                    )
                )

            # Send tool results back — the SDK handles thought_signature internally
            response = chat_session.send_message(fn_responses)

        # Hit iteration cap
        text = ""
        for part in response.parts:
            if hasattr(part, "text") and part.text:
                text += part.text
        if not text:
            text = "The agent reached its step limit and could not complete the task."
        return text, steps, total_tokens

    return runner


class AgentExecutor:
    """Wraps the agent execution loop with DB persistence."""

    def __init__(self, ctx: AgentContext):
        self.ctx = ctx
        self.llm = get_llm_service()
        self.tools = build_tools(ctx)
        self._tool_map = {t.name: t for t in self.tools}
        # List of (name, bound_model_or_runner) pairs — tried in order
        self._models: list[tuple[str, Any]] = []
        self._current_model_idx: int = 0
        self._use_native_gemini: bool = False
        self._build_model_chain()

    def _build_model_chain(self):
        """Build the model chain. If Gemini is primary, try native SDK first."""
        # Try native Gemini SDK runner first (avoids thought_signature issue)
        native_runner = None
        if settings.GEMINI_API_KEY:
            native_runner = _make_native_gemini_runner(self.tools, settings)
            if native_runner is not None:
                self._models.append(("gemini-native", native_runner))
                logger.info("Using native Google GenerativeAI SDK for Gemini (thought_signature safe)")

        # LangChain fallback providers
        for provider in self.llm._providers:
            if provider.name == "gemini" and native_runner is not None:
                continue  # already added native; skip LC Gemini to avoid duplicates
            chat = provider.bound
            try:
                bound = chat.bind_tools(self.tools)
                self._models.append((provider.name, bound))
                logger.info("Tools bound to LC provider: %s", provider.name)
            except Exception as exc:
                logger.info("bind_tools failed for %s (%s); skipping", provider.name, exc)

        if not self._models:
            raise ProviderError("No LLM provider available for agent execution.")

        logger.info("Agent model chain: %s", [name for name, _ in self._models])

    def _invoke_model(self, messages: list[BaseMessage]) -> "AIMessage | None":
        """Invoke the current model. For native Gemini, this is handled separately.

        Returns None if native Gemini runner should be used (handled in _loop).
        """
        name, model = self._models[self._current_model_idx]
        if name == "gemini-native":
            return None  # Handled by native runner path in _loop
        return model.invoke(messages)

    def _try_next_provider(self, exc: Exception) -> bool:
        """Try to fall back to the next provider. Returns True if switched."""
        if _is_fallback_error(exc) and self._current_model_idx + 1 < len(self._models):
            logger.warning(
                "Provider %s failed (%s); switching to next.",
                self._models[self._current_model_idx][0], exc,
            )
            self._current_model_idx += 1
            return True
        return False

    # ------------------------------------------------------------- run
    def run(self, query: str, conversation_id: str | None = None) -> AgentRun:
        db = self.ctx.db
        run = AgentRun(
            owner_id=self.ctx.owner_id,
            conversation_id=conversation_id,
            query=query,
            status="running",
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        self.ctx.run_id = run.id

        start = time.perf_counter()
        try:
            final_answer, steps, total_tokens = self._loop(query)
        except Exception as exc:
            logger.exception("Agent execution failed")
            run.status = "failed"
            run.error_message = str(exc)[:2000]
            run.completed_at = datetime.now(timezone.utc)
            run.latency_ms = int((time.perf_counter() - start) * 1000)
            db.commit()
            return run

        run.status = "completed"
        run.final_answer = (final_answer or "")[:10000]
        run.total_tokens = total_tokens
        run.latency_ms = int((time.perf_counter() - start) * 1000)
        run.completed_at = datetime.now(timezone.utc)

        for s in steps:
            db.add(AgentStep(
                run_id=run.id,
                step_index=s["step_index"],
                tool_name=s["tool_name"],
                tool_input=json.dumps(s.get("tool_input", {}), default=str),
                tool_output=str(s.get("tool_output", ""))[:5000],
                success=s.get("success", True),
                error_message=s.get("error_message", ""),
                latency_ms=s.get("latency_ms", 0),
                tokens=s.get("tokens", 0),
            ))
        db.commit()
        return run

    # ------------------------------------------------------------- loop
    def _loop(self, query: str) -> tuple[str, list[dict], int]:
        """Main execution loop. Returns (final_answer, steps, total_tokens)."""
        while self._current_model_idx < len(self._models):
            name, model = self._models[self._current_model_idx]

            if name == "gemini-native":
                # Use the native Gemini SDK runner — handles thought_signature
                try:
                    tool_desc = "\n".join(f"- {n}: {d}" for n, d in TOOL_DESCRIPTIONS.items())
                    system_msgs = build_agent_system_prompt(tool_desc)
                    system_prompt = system_msgs[0]["content"] if system_msgs else ""
                    final_answer, steps, total_tokens = model(query, system_prompt)
                    return final_answer, steps, total_tokens
                except Exception as exc:
                    logger.warning("Native Gemini runner failed: %s", exc)
                    if self._try_next_provider(exc):
                        continue
                    raise
            else:
                # LangChain model path
                try:
                    return self._lc_loop(query, model)
                except Exception as exc:
                    logger.warning("LC provider %s failed: %s", name, exc)
                    if self._try_next_provider(exc):
                        continue
                    raise

        raise ProviderError("All agent providers exhausted.")

    def _lc_loop(self, query: str, model: Any) -> tuple[str, list[dict], int]:
        """LangChain ReAct loop implementation."""
        tool_desc = "\n".join(f"- {n}: {d}" for n, d in TOOL_DESCRIPTIONS.items())
        system_msgs = build_agent_system_prompt(tool_desc)
        messages: list[BaseMessage] = [
            SystemMessage(content=system_msgs[0]["content"]),
            HumanMessage(content=query),
        ]

        max_iter = settings.AGENT_MAX_ITERATIONS
        steps: list[dict] = []
        step_index = 0
        total_tokens = 0

        for iteration in range(max_iter):
            ai_msg: AIMessage = model.invoke(messages)
            messages.append(ai_msg)
            if getattr(ai_msg, "usage_metadata", None):
                um = ai_msg.usage_metadata or {}
                total_tokens += int(um.get("total_tokens", 0) or 0)

            tool_calls = getattr(ai_msg, "tool_calls", None) or []
            if not tool_calls:
                return _extract_text_content(ai_msg.content), steps, total_tokens

            for tc in tool_calls:
                step_index += 1
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {}) or {}
                result = self._execute_tool(tool_name, tool_args)
                steps.append({
                    "step_index": step_index,
                    "tool_name": tool_name,
                    "tool_input": tool_args,
                    "tool_output": result.output if result.success else "",
                    "success": result.success,
                    "error_message": result.error,
                    "latency_ms": result.latency_ms,
                    "tokens": 0,
                })
                content = result.output if result.success else json.dumps({"error": result.error})
                messages.append(ToolMessage(
                    content=str(content),
                    tool_call_id=tc.get("id", f"call_{step_index}"),
                ))

        # Hit iteration cap — force synthesis
        try:
            ai_msg = model.invoke(
                messages + [HumanMessage(content="Produce your best final answer now based on evidence gathered.")]
            )
            return _extract_text_content(ai_msg.content), steps, total_tokens
        except Exception:
            return "The agent reached its step limit.", steps, total_tokens

    def _execute_tool(self, name: str, args: dict[str, Any]):
        """Validate args, run the tool with a timeout, return ToolResult."""
        from .reliability import ToolResult
        tool = self._tool_map.get(name)
        if tool is None:
            return ToolResult(output=None, success=False, error=f"Unknown tool: {name}")
        try:
            validated = tool.args_schema(**args)
        except Exception as exc:
            return ToolResult(output=None, success=False, error=f"Invalid parameters for {name}: {exc}")
        return with_timeout_and_retry(lambda: tool._run(**validated.model_dump()))


def get_agent_executor(ctx: AgentContext) -> AgentExecutor:
    return AgentExecutor(ctx)
