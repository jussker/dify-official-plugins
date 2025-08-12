import json
from collections.abc import Generator
from typing import Optional, Union

from dify_plugin import OAICompatLargeLanguageModel
from dify_plugin.entities.model import AIModelEntity, ModelFeature
from dify_plugin.entities.model.llm import (
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
)
from dify_plugin.entities.model.message import PromptMessage, PromptMessageTool


class OpenRouterLargeLanguageModel(OAICompatLargeLanguageModel):
    def _update_credential(self, model: str, credentials: dict):
        credentials["endpoint_url"] = "https://openrouter.ai/api/v1"
        credentials["mode"] = self.get_model_mode(model).value
        schema = self.get_model_schema(model, credentials)
        if schema and {
            ModelFeature.TOOL_CALL,
            ModelFeature.MULTI_TOOL_CALL,
        }.intersection(schema.features or []):
            credentials["function_calling_type"] = "tool_call"

        # Add OpenRouter specific headers for rankings on openrouter.ai
        credentials["extra_headers"] = {
            "HTTP-Referer": "https://dify.ai/",
            "X-Title": "Dify",
        }

    def _preprocess_model_parameters(self, model: str, model_parameters: dict) -> None:
        """Adapt parameters for OpenAI o*/gpt-5 reasoning series via OpenRouter."""
        if model.startswith("openai/o") or model.startswith("openai/gpt-5"):
            if (
                "max_tokens" in model_parameters
                and "max_completion_tokens" not in model_parameters
            ):
                model_parameters["max_completion_tokens"] = model_parameters.pop(
                    "max_tokens"
                )
        # gpt-5 系列 verbosity 需放入 text.verbosity
        if model.startswith("openai/gpt-5") and "verbosity" in model_parameters:
            verbosity = model_parameters.pop("verbosity")
            text_cfg = model_parameters.get("text") or {}
            # 仅当未设置时覆盖
            if "verbosity" not in text_cfg:
                text_cfg["verbosity"] = verbosity
            model_parameters["text"] = text_cfg

        # provider 路由开关：默认不启用。启用时透传 provider；未启用则移除 provider
        enable_provider = model_parameters.pop("enable_provider", None)
        if not enable_provider and "provider" in model_parameters:
            # 未开启时，避免发送 provider 字段
            model_parameters.pop("provider", None)
        elif enable_provider:
            # 开启时若 provider 为字符串（来自表单），尝试解析为 JSON 对象
            pv = model_parameters.get("provider")
            if isinstance(pv, str):
                try:
                    model_parameters["provider"] = json.loads(pv)
                except Exception:
                    # 解析失败则移除，避免发送无效字符串
                    model_parameters.pop("provider", None)

        # 合并 response_format 与 json_schema（若提供）
        rf = model_parameters.get("response_format")
        js = model_parameters.pop("json_schema", None)
        if js is not None:
            # 允许两种形态：直接 schema 对象 或 { name, schema }
            if isinstance(js, dict) and ("name" in js and "schema" in js):
                js_payload = js
            else:
                js_payload = {"name": "dify_response", "schema": js}
            if isinstance(rf, dict):
                # 强制类型为 json_schema
                rf_type = rf.get("type")
                if rf_type != "json_schema":
                    rf["type"] = "json_schema"
                # 仅当未提供时填充
                rf.setdefault("json_schema", js_payload)
                model_parameters["response_format"] = rf
            else:
                model_parameters["response_format"] = {
                    "type": "json_schema",
                    "json_schema": js_payload,
                }

        # 组合关系：top_logprobs 仅在 logprobs 启用时有效
        if "top_logprobs" in model_parameters:
            try:
                n = int(model_parameters["top_logprobs"])  # 规格要求 0-20
                if n < 0:
                    n = 0
                if n > 20:
                    n = 20
                model_parameters["top_logprobs"] = n
            except Exception:
                # 无法解析为整数则移除，避免无效值
                model_parameters.pop("top_logprobs", None)
            else:
                # 若提供了 top_logprobs，则强制打开 logprobs
                if not model_parameters.get("logprobs"):
                    model_parameters["logprobs"] = True
        else:
            # 未提供 top_logprobs 时无需改动 logprobs
            pass

        enable_stream = model_parameters.pop("enable_stream", None)
        if enable_stream is not None:
            model_parameters["__force_stream"] = bool(enable_stream)

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        self._update_credential(model, credentials)
        # reasoning params mapping (priority: effort > max_tokens)
        reasoning = {}
        reasoning_effort = model_parameters.pop("reasoning_effort", None)
        reasoning_budget = model_parameters.pop("reasoning_budget", None)
        enable_thinking = model_parameters.pop("enable_thinking", None)
        exclude_reasoning_tokens = model_parameters.pop(
            "exclude_reasoning_tokens", None
        )

        if enable_thinking == "dynamic":
            reasoning["enabled"] = True
        else:
            if reasoning_effort is not None:
                reasoning["effort"] = reasoning_effort
            elif reasoning_budget is not None:
                reasoning["max_tokens"] = reasoning_budget
            elif enable_thinking:  # 普通 True 打开推理
                reasoning["enabled"] = True

        # gpt-5 系列在启用 enable_thinking 时添加 reasoning.summary = "auto"
        if enable_thinking and model.startswith("openai/gpt-5"):
            reasoning["summary"] = "auto"

        if exclude_reasoning_tokens is not None:
            reasoning["exclude"] = bool(exclude_reasoning_tokens)

        if reasoning:
            model_parameters["reasoning"] = reasoning

        # preprocess for gpt-5 / o* series (max_tokens -> max_completion_tokens, enable_stream)
        self._preprocess_model_parameters(model, model_parameters)
        force_stream = model_parameters.pop("__force_stream", None)
        if force_stream is not None:
            stream = force_stream
        return self._generate(
            model,
            credentials,
            prompt_messages,
            model_parameters,
            tools,
            stop,
            stream,
            user,
        )

    def validate_credentials(self, model: str, credentials: dict) -> None:
        self._update_credential(model, credentials)
        return super().validate_credentials(model, credentials)

    def _generate(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        self._update_credential(model, credentials)
        return super()._generate(
            model,
            credentials,
            prompt_messages,
            model_parameters,
            tools,
            stop,
            stream,
            user,
        )

    def _wrap_thinking_by_reasoning_content(
        self, delta: dict, is_reasoning: bool
    ) -> tuple[str, bool]:
        """
        If the reasoning response is from delta.get("reasoning") or delta.get("reasoning_content"),
        we wrap it with HTML think tag.

        :param delta: delta dictionary from LLM streaming response
        :param is_reasoning: is reasoning
        :return: tuple of (processed_content, is_reasoning)
        """

        content = delta.get("content") or ""
        # NOTE(hzw): OpenRouter uses "reasoning" instead of "reasoning_content".
        reasoning_content = delta.get("reasoning") or delta.get("reasoning_content")

        if reasoning_content:
            if not is_reasoning:
                content = "<think>\n" + reasoning_content
                is_reasoning = True
            else:
                content = reasoning_content
        elif is_reasoning and content:
            content = "\n</think>" + content
            is_reasoning = False
        return content, is_reasoning

    def _generate_block_as_stream(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        user: Optional[str] = None,
    ) -> Generator:
        resp = super()._generate(
            model,
            credentials,
            prompt_messages,
            model_parameters,
            tools,
            stop,
            False,
            user,
        )
        yield LLMResultChunk(
            model=model,
            prompt_messages=prompt_messages,
            delta=LLMResultChunkDelta(
                index=0,
                message=resp.message,
                usage=self._calc_response_usage(
                    model=model,
                    credentials=credentials,
                    prompt_tokens=resp.usage.prompt_tokens,
                    completion_tokens=resp.usage.completion_tokens,
                ),
                finish_reason="stop",
            ),
        )

    def get_customizable_model_schema(
        self, model: str, credentials: dict
    ) -> AIModelEntity:
        return super().get_customizable_model_schema(model, credentials)

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        self._update_credential(model, credentials)
        return super().get_num_tokens(model, credentials, prompt_messages, tools)
