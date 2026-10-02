"""Small local Ollama adapter; invalid responses never escape as raw JSON."""
import json
import logging
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class OllamaError(RuntimeError):
    """The local model was unavailable or failed structured validation."""


class OllamaClient:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    async def model_digest(self) -> str | None:
        """Optional provenance metadata; its absence is never a fabricated version."""
        if not self.settings.llm_enabled:
            return None
        try:
            async with httpx.AsyncClient(timeout=min(3, self.settings.request_timeout), trust_env=False) as client:
                response = await client.get(self.settings.ollama_url.rstrip("/") + "/api/tags")
                response.raise_for_status()
                for model in response.json().get("models", []):
                    if model.get("name") == self.settings.ollama_model or model.get("model") == self.settings.ollama_model:
                        return model.get("digest")
        except (httpx.HTTPError, ValueError, TypeError):
            pass
        return None

    async def structured(self, prompt: str, schema: type[T], *, system: str = "", context_tokens: int | None = None) -> T:
        if not self.settings.llm_enabled:
            raise OllamaError("Local model is disabled (LLM_ENABLED=false)")
        messages = [
            {"role": "system", "content": system + "\nReturn only JSON matching this schema:\n" + json.dumps(schema.model_json_schema(), ensure_ascii=False)},
            {"role": "user", "content": prompt},
        ]
        payload = {
            "model": self.settings.ollama_model,
            "messages": messages,
            "format": schema.model_json_schema(),
            "stream": False,
            "options": {"temperature": 0, "num_predict": 4096},
        }
        if context_tokens is not None:
            payload["options"]["num_ctx"] = context_tokens
        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=self.settings.ollama_timeout, trust_env=False) as client:
            for attempt in range(self.settings.ollama_retries + 1):
                try:
                    response = await client.post(self.settings.ollama_url.rstrip("/") + "/api/chat", json=payload)
                    response.raise_for_status()
                    content = response.json()["message"]["content"]
                    return schema.model_validate_json(content)
                except (httpx.HTTPError, ValueError, TypeError, KeyError, ValidationError) as exc:
                    last_error = exc
                    # Do not log full response bodies, prompts, or credentials.
                    logger.warning("Ollama attempt %d failed: %s", attempt + 1, type(exc).__name__)
        raise OllamaError(f"Ollama failed after {self.settings.ollama_retries + 1} attempts ({type(last_error).__name__})") from last_error
