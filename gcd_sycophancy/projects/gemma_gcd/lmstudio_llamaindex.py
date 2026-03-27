from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Optional
from urllib import request
from urllib.error import HTTPError


@dataclass
class CompatibleCompletion:
    text: str


@dataclass
class CompatibleRequestOutput:
    prompt: str
    outputs: list[CompatibleCompletion]
    response_id: str | None = None


@dataclass
class LMStudioNativeChatResult:
    text: str
    response_id: str | None
    raw: dict[str, Any]


class LMStudioLlamaIndexAdapter:
    """Expose LM Studio native `/api/v1/chat` through a small evaluator-friendly API."""

    supports_stateful_chat = True

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str = "http://localhost:1234",
        request_timeout: float = 120.0,
        temperature: float | None = None,
        additional_kwargs: dict[str, Any] | None = None,
        api_token: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.temperature = temperature
        self.additional_kwargs = additional_kwargs or {}
        self.api_token = api_token

    def generate(
        self,
        prompts: Iterable[str],
        sampling_params: Any,
    ) -> list[CompatibleRequestOutput]:
        outputs: list[CompatibleRequestOutput] = []
        for prompt in prompts:
            response = self.chat(
                prompt,
                sampling_params=sampling_params,
                store=False,
            )
            outputs.append(
                CompatibleRequestOutput(
                    prompt=prompt,
                    outputs=[CompatibleCompletion(text=response.text)],
                    response_id=response.response_id,
                )
            )
        return outputs

    def chat(
        self,
        user_input: str,
        *,
        previous_response_id: Optional[str] = None,
        sampling_params: Any = None,
        store: bool = True,
    ) -> LMStudioNativeChatResult:
        payload = {
            "model": self.model_name,
            "input": user_input,
            "store": store,
        }
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
        payload.update(self._sampling_params_to_kwargs(sampling_params))
        payload.update(self.additional_kwargs)
        raw = self._post_json("/api/v1/chat", payload)
        return LMStudioNativeChatResult(
            text=self._extract_text(raw),
            response_id=raw.get("response_id"),
            raw=raw,
        )

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={
                "Content-Type": "application/json",
                **(
                    {"Authorization": f"Bearer {self.api_token}"}
                    if self.api_token
                    else {}
                ),
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.request_timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"LM Studio native API request failed with HTTP {exc.code}: {details}"
            ) from exc

    @staticmethod
    def _extract_text(raw: dict[str, Any]) -> str:
        chunks = []
        for item in raw.get("output", []):
            if item.get("type") == "message" and item.get("content"):
                chunks.append(item["content"])
        return "\n".join(chunks).strip()

    @staticmethod
    def _sampling_params_to_kwargs(sampling_params: Any) -> dict[str, Any]:
        if sampling_params is None:
            return {}
        kwargs: dict[str, Any] = {}
        for source_attr, target_key in (
            ("max_tokens", "max_output_tokens"),
            ("temperature", "temperature"),
            ("top_p", "top_p"),
        ):
            value = getattr(sampling_params, source_attr, None)
            if value is not None:
                kwargs[target_key] = value
        return kwargs
