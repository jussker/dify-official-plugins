import json
import codecs
import re
from collections.abc import Generator
from typing import Optional, Union, Any

import requests
from pydantic import TypeAdapter, ValidationError

from dify_plugin.entities.model import AIModelEntity, ModelFeature
from dify_plugin.entities.model.llm import LLMResult, LLMResultChunk, LLMResultChunkDelta, LLMMode
from dify_plugin.entities.model.message import (
    PromptMessage,
    PromptMessageTool,
    UserPromptMessage,
    TextPromptMessageContent,
    ImagePromptMessageContent,
    PromptMessageContentType,
    AssistantPromptMessage,
    PromptMessageContent,
)
from dify_plugin import OAICompatLargeLanguageModel
from dify_plugin.errors.model import InvokeError
from dify_plugin.interfaces.model.openai_compatible.llm import _increase_tool_call


IMAGE_GENERATION_MODELS = {
    "google/gemini-2.5-flash-image-preview",
    "google/gemini-2.5-flash-image-preview:free",
}


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

    def _convert_files_to_text(self, messages: list[PromptMessage]) -> list[PromptMessage]:
        """
        Convert any file content in messages to text descriptions to avoid validation issues
        """
        converted_messages = []
        
        for message in messages:
            if isinstance(message, UserPromptMessage) and isinstance(message.content, list):
                # Process multimodal content
                text_parts = []
                for content in message.content:
                    if isinstance(content, TextPromptMessageContent):
                        text_parts.append(content.data)
                    elif isinstance(content, ImagePromptMessageContent):
                        # Convert image to text description
                        if hasattr(content, 'url') and content.url:
                            text_parts.append(f"[Image file uploaded]: {content.url}")
                        else:
                            text_parts.append("[Image file uploaded]")
                    elif hasattr(content, 'type') and content.type == PromptMessageContentType.DOCUMENT:
                        # Handle document files like PDF
                        if hasattr(content, 'url') and content.url:
                            text_parts.append(f"[Document file uploaded]: {content.url}")
                        else:
                            text_parts.append("[Document file uploaded]")
                    else:
                        # Handle any other content types
                        if hasattr(content, 'url'):
                            text_parts.append(f"[File uploaded]: {content.url}")
                        else:
                            text_parts.append(str(content))
                
                # Create new text-only message
                converted_message = UserPromptMessage(content=" ".join(text_parts))
                converted_messages.append(converted_message)
            else:
                # Keep non-multimodal messages as is
                converted_messages.append(message)
        
        return converted_messages

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
        # 仅在不支持视觉的模型上，将文件/图像内容转为文本描述，避免校验问题
        model_schema = self.get_model_schema(model, credentials)
        if not (model_schema and ModelFeature.VISION in (model_schema.features or [])):
            prompt_messages = self._convert_files_to_text(prompt_messages)
        # reasoning
        reasoning_params = {}
        reasoning_budget = model_parameters.pop('reasoning_budget', None)
        enable_thinking = model_parameters.pop('enable_thinking', None)
        if enable_thinking == 'dynamic':
            reasoning_budget = -1
        if reasoning_budget is not None:
            reasoning_params['max_tokens'] = reasoning_budget
        reasoning_effort = model_parameters.pop('reasoning_effort', None)
        if reasoning_effort is not None:
            reasoning_params['effort'] = reasoning_effort
        exclude_reasoning_tokens = model_parameters.pop('exclude_reasoning_tokens', None)
        if exclude_reasoning_tokens is not None:
            reasoning_params['exclude'] = exclude_reasoning_tokens
        if reasoning_params:
            model_parameters['reasoning'] = reasoning_params
        return self._generate(model, credentials, prompt_messages, model_parameters, tools, stop, stream, user)

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

    def _parse_image_part(self, choice: dict, stream: bool) -> list[PromptMessageContent]:
        contents: list[PromptMessageContent] = []

        if stream:
            message = choice
        else:
            message = choice.get("message", {})
        if message.get("images"):
            for image in message["images"]:
                image_url = image["image_url"]["url"]  # data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABAAA...
                match = re.match(r"data:(.*?);base64,(.*?)", image_url)
                if match:
                    mime_type = match.group(1)
                else:
                    raise InvokeError("receive image with no mime_type")
                if mime_type.startswith("image/"):
                    base64_data = image_url.split(";")[1].split(",")[1]
                    mime_subtype = mime_type.split("/", maxsplit=1)[-1]
                    contents.append(
                        ImagePromptMessageContent(
                            format=mime_subtype,
                            base64_data=base64_data,
                            mime_type=mime_type,
                            detail=ImagePromptMessageContent.DETAIL.HIGH,
                        )
                    )
        return contents

    def _handle_generate_response(
        self,
        model: str,
        credentials: dict,
        response: requests.Response,
        prompt_messages: list[PromptMessage],
    ) -> LLMResult:
        response_json: dict = response.json()

        completion_type = LLMMode.value_of(credentials["mode"])

        output = response_json["choices"][0]
        message_id = response_json.get("id")

        response_content = ""
        tool_calls = None
        function_calling_type = credentials.get("function_calling_type", "no_call")
        if completion_type is LLMMode.CHAT:
            response_content = output.get("message", {})["content"]
            if function_calling_type == "tool_call":
                tool_calls = output.get("message", {}).get("tool_calls")
            elif function_calling_type == "function_call":
                tool_calls = output.get("message", {}).get("function_call")

        elif completion_type is LLMMode.COMPLETION:
            response_content = output["text"]

        if model not in IMAGE_GENERATION_MODELS:
            assistant_message = AssistantPromptMessage(content=response_content, tool_calls=[])
        else:
            contents: list[PromptMessageContent] = [TextPromptMessageContent(data=response_content)]
            contents.extend(
                self._parse_image_part(output, stream=False)
            )

            assistant_message = AssistantPromptMessage(
                content=contents, tool_calls=[]
            )

        if tool_calls:
            if function_calling_type == "tool_call":
                assistant_message.tool_calls = self._extract_response_tool_calls(tool_calls)
            elif function_calling_type == "function_call":
                assistant_message.tool_calls = [self._extract_response_function_call(tool_calls)]

        usage = response_json.get("usage")
        if usage:
            # transform usage
            prompt_tokens = usage["prompt_tokens"]
            completion_tokens = usage["completion_tokens"]
        else:
            # calculate num tokens
            assert prompt_messages[0].content is not None
            prompt_tokens = self._num_tokens_from_string(model, prompt_messages[0].content)
            assert assistant_message.content is not None
            completion_tokens = self._num_tokens_from_string(model, assistant_message.content)

        # transform usage
        usage = self._calc_response_usage(model, credentials, prompt_tokens, completion_tokens)

        # transform response
        result = LLMResult(
            id=message_id,
            model=response_json["model"],
            message=assistant_message,
            usage=usage,
        )

        return result

    def _handle_generate_stream_response(
        self, model: str, credentials: dict, response: requests.Response, prompt_messages: list[PromptMessage]
    ) -> Generator:
        """
        Handle llm stream response

        :param model: model name
        :param credentials: model credentials
        :param response: streamed response
        :param prompt_messages: prompt messages
        :return: llm response chunk generator
        """
        chunk_index = 0
        full_assistant_content = ""
        tools_calls: list[AssistantPromptMessage.ToolCall] = []
        finish_reason = None
        usage = None
        is_reasoning_started = False
        # delimiter for stream response, need unicode_escape
        delimiter = credentials.get("stream_mode_delimiter", "\n\n")
        delimiter = codecs.decode(delimiter, "unicode_escape")
        for chunk in response.iter_lines(decode_unicode=True, delimiter=delimiter):
            chunk = chunk.strip()
            if chunk:
                # ignore sse comments
                if chunk.startswith(":"):
                    continue
                decoded_chunk = chunk.strip().removeprefix("data:").lstrip()
                if decoded_chunk == "[DONE]":  # Some provider returns "data: [DONE]"
                    continue

                try:
                    chunk_json: dict = TypeAdapter(dict[str, Any]).validate_json(decoded_chunk)
                # stream ended
                except ValidationError:
                    yield self._create_final_llm_result_chunk(
                        index=chunk_index + 1,
                        message=AssistantPromptMessage(content=""),
                        finish_reason="Non-JSON encountered.",
                        usage=usage,
                        model=model,
                        credentials=credentials,
                        prompt_messages=prompt_messages,
                        full_content=full_assistant_content,
                    )
                    break
                # handle the error here. for issue #11629
                if chunk_json.get("error") and chunk_json.get("choices") is None:
                    raise ValueError(chunk_json.get("error"))

                if chunk_json:  # noqa: SIM102
                    if u := chunk_json.get("usage"):
                        usage = u
                if not chunk_json or len(chunk_json["choices"]) == 0:
                    continue

                choice = chunk_json["choices"][0]
                finish_reason = chunk_json["choices"][0].get("finish_reason")
                chunk_index += 1

                if "delta" in choice:
                    delta = choice["delta"]
                    delta_content, is_reasoning_started = self._wrap_thinking_by_reasoning_content(
                        delta, is_reasoning_started
                    )
                    if model in IMAGE_GENERATION_MODELS:
                        if delta_content:
                            delta_content = [TextPromptMessageContent(data=delta_content)]
                        else:
                            delta_content = []
                        delta_content.extend(
                            self._parse_image_part(delta, stream=True)  # type: ignore
                        )

                    assistant_message_tool_calls = None

                    if "tool_calls" in delta and credentials.get("function_calling_type", "no_call") == "tool_call":
                        assistant_message_tool_calls = delta.get("tool_calls", None)
                    elif (
                        "function_call" in delta
                        and credentials.get("function_calling_type", "no_call") == "function_call"
                    ):
                        assistant_message_tool_calls = [
                            {"id": "tool_call_id", "type": "function", "function": delta.get("function_call", {})}
                        ]

                    # extract tool calls from response
                    if assistant_message_tool_calls:
                        tool_calls = self._extract_response_tool_calls(assistant_message_tool_calls)
                        _increase_tool_call(tool_calls, tools_calls)

                    if not delta_content:
                        continue

                    # transform assistant message to prompt message
                    assistant_prompt_message = AssistantPromptMessage(
                        content=delta_content,
                    )

                    if isinstance(delta_content, str):
                        full_assistant_content += delta_content
                    else:
                        full_assistant_content += "".join([content.data for content in delta_content])
                elif "text" in choice:
                    choice_text = choice.get("text", "")
                    if choice_text == "":
                        continue

                    # transform assistant message to prompt message
                    assistant_prompt_message = AssistantPromptMessage(content=choice_text)
                    full_assistant_content += choice_text
                else:
                    continue

                yield LLMResultChunk(
                    model=model,
                    delta=LLMResultChunkDelta(
                        index=chunk_index,
                        message=assistant_prompt_message,
                    ),
                )

            chunk_index += 1

        if tools_calls:
            yield LLMResultChunk(
                model=model,
                delta=LLMResultChunkDelta(
                    index=chunk_index,
                    message=AssistantPromptMessage(tool_calls=tools_calls, content=""),
                ),
            )

        yield self._create_final_llm_result_chunk(
            index=chunk_index,
            message=AssistantPromptMessage(content=""),
            finish_reason=finish_reason,
            usage=usage,
            model=model,
            credentials=credentials,
            prompt_messages=prompt_messages,
            full_content=full_assistant_content,
        )
