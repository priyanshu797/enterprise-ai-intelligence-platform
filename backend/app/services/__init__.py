"""Business-logic services for the platform.

Services are intentionally framework-agnostic where possible so they can be
unit-tested in isolation and swapped (e.g. vector store / LLM provider)
without touching the API layer.
"""
