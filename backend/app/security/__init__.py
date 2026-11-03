"""Security utilities: authentication + prompt-injection defense.

This package covers two distinct concerns:
  * `auth`        — password hashing (passlib) and JWT issuance/validation.
  * `injection`   — treating retrieved document text as UNTRUSTED data and
                    neutralising embedded instruction attacks (Challenge 2).
"""
from __future__ import annotations

from .auth import (
    create_access_token,
    create_refresh_token,
    decode_access_token,
    get_current_user,
    get_current_active_user,
    hash_password,
    verify_password,
)
from .injection import (
    INJECTION_PATTERNS,
    InjectionResult,
    analyze_text_for_injection,
    sanitize_retrieved_context,
    strip_prompt_injection,
)

__all__ = [
    "create_access_token",
    "create_refresh_token",
    "decode_access_token",
    "get_current_user",
    "get_current_active_user",
    "hash_password",
    "verify_password",
    "INJECTION_PATTERNS",
    "InjectionResult",
    "analyze_text_for_injection",
    "sanitize_retrieved_context",
    "strip_prompt_injection",
]
