"""Cross-platform LLM & VLM client: Ollama local-first, OpenAI-compatible cloud, or offline fallback."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

import httpx

from spoolgenie.config import SpoolGenieSettings, get_settings
from spoolgenie.doctor import pick_best_ollama_models, probe_ollama


class UnifiedLLMClient:
    """Routes text and multimodal calls to local Ollama or any OpenAI-compatible API."""

    def __init__(self, settings: SpoolGenieSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.provider: str = "fallback"
        self.chat_model: str | None = None
        self.vlm_model: str | None = None
        self._resolve_provider()

    def _resolve_provider(self) -> None:
        pref = self.settings.llm_provider.lower()
        ollama_ok, models = probe_ollama(self.settings.ollama_base_url)
        rec_llm, rec_vlm = pick_best_ollama_models(models)

        if pref in ("auto", "ollama") and ollama_ok and models:
            self.provider = "ollama"
            self.chat_model = (
                rec_llm if self.settings.ollama_model == "auto" else self.settings.ollama_model
            )
            self.vlm_model = (
                rec_vlm if self.settings.ollama_vlm_model == "auto" else self.settings.ollama_vlm_model
            )
            return

        if pref in ("auto", "openai") and self.settings.openai_api_key:
            self.provider = "openai"
            self.chat_model = self.settings.openai_model
            self.vlm_model = self.settings.openai_vlm_model
            return

        self.provider = "fallback"

    @property
    def available(self) -> bool:
        return self.provider in ("ollama", "openai")

    def release_vram(self) -> None:
        """Ask local Ollama to evict loaded models from GPU VRAM before heavy local 2D/3D inference."""
        if self.provider != "ollama":
            return
        for model in {m for m in (self.chat_model, self.vlm_model) if m}:
            try:
                httpx.post(
                    f"{self.settings.ollama_base_url.rstrip('/')}/api/generate",
                    json={"model": model, "keep_alive": 0},
                    timeout=5.0,
                )
            except Exception:
                pass

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        json_mode: bool = False,
        timeout: float = 120.0,
    ) -> str | None:
        if self.provider == "ollama" and self.chat_model:
            payload: dict[str, Any] = {
                "model": self.chat_model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            }
            if json_mode:
                payload["format"] = "json"
            try:
                r = httpx.post(
                    f"{self.settings.ollama_base_url.rstrip('/')}/api/chat",
                    json=payload,
                    timeout=timeout,
                )
                r.raise_for_status()
                return r.json()["message"]["content"]
            except Exception:
                return None

        if self.provider == "openai" and self.chat_model:
            payload = {
                "model": self.chat_model,
                "messages": messages,
                "temperature": temperature,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            try:
                r = httpx.post(
                    f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
                    json=payload,
                    timeout=timeout,
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception:
                return None

        return None

    def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        timeout: float = 120.0,
    ) -> dict[str, Any] | None:
        raw = self.chat(messages, temperature=temperature, json_mode=True, timeout=timeout)
        if not raw:
            return None
        return extract_json_object(raw)

    def analyze_image(
        self,
        image_path: str | Path,
        prompt: str,
        timeout: float = 120.0,
    ) -> str | None:
        path = Path(image_path)
        if not path.exists():
            return None
        img_b64 = base64.b64encode(path.read_bytes()).decode("ascii")

        if self.provider == "ollama" and self.vlm_model:
            payload = {
                "model": self.vlm_model,
                "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
                "stream": False,
                "options": {"temperature": 0.2},
            }
            try:
                r = httpx.post(
                    f"{self.settings.ollama_base_url.rstrip('/')}/api/chat",
                    json=payload,
                    timeout=timeout,
                )
                r.raise_for_status()
                return r.json()["message"]["content"]
            except Exception:
                return None

        if self.provider == "openai" and self.vlm_model:
            suffix = path.suffix.lower().lstrip(".") or "png"
            mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
            payload = {
                "model": self.vlm_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{img_b64}"}},
                        ],
                    }
                ],
                "temperature": 0.2,
            }
            try:
                r = httpx.post(
                    f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
                    json=payload,
                    timeout=timeout,
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception:
                return None

        return None


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Robustly pull the first valid JSON object out of an LLM response."""
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Strip fenced markdown blocks
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        try:
            obj = json.loads(fenced.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Fallback: match outermost braces
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None
