#!/usr/bin/env python3
"""
Fetch OpenRouter available models list and convert each into a Dify model YAML file.
- Reads from OpenRouter Models API (or a local JSON file for offline testing)
- Generates YAMLs into a temporary output directory
- Prints a summary and output path

Usage examples:
  # Fetch from API (default)
  python scripts/openrouter_models_to_dify_yaml.py

  # Use local JSON mock
  python scripts/openrouter_models_to_dify_yaml.py --input-json models/openrouter/openrouter_list-available-models.json

Options:
  --api-url URL         Override OpenRouter models API URL.
  --input-json PATH     Use a local JSON file instead of calling the API.
  --tmp-dir PATH        Output directory (default: mkdtemp under /tmp).
  --provider openrouter Optional provider key used in generated "models/<provider>/models/llm/..." paths in YAML.
  --dry-run             Only print summary, do not write files.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests  # type: ignore
except Exception:
    requests = None

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


DEFAULT_API_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_PROVIDER = "openrouter"


@dataclass
class ModelEntry:
    id: str
    name: Optional[str]
    context_length: Optional[int]
    architecture: Optional[Dict[str, Any]]
    pricing: Optional[Dict[str, Any]]
    top_provider: Optional[Dict[str, Any]]
    supported_parameters: Optional[List[str]]
    input_modalities: Optional[List[str]]


def slugify(value: str) -> str:
    v = value.strip().lower()
    v = re.sub(r"[^a-z0-9\-_/]+", "-", v)
    v = re.sub(r"/+", "/", v)
    v = re.sub(r"-+", "-", v)
    return v.strip("-/")


def infer_features(entry: ModelEntry) -> List[str]:
    supported = set((entry.supported_parameters or []))
    modalities = set((entry.input_modalities or (entry.architecture or {}).get("input_modalities") or []))

    features: List[str] = []

    # agent-thought: supported_parameters contains reasoning
    if "reasoning" in supported:
        features.append("agent-thought")

    # vision: input_modalities contains image
    if "image" in modalities:
        features.append("vision")

    # tool-call: supported_parameters contains tools
    if "tools" in supported:
        features.append("tool-call")
        # stream-tool-call: OpenRouter models generally support stream; if tools then expose stream-tool-call
        features.append("stream-tool-call")

    # multi-tool-call: need both tools and tool_choice
    if "tools" in supported and "tool_choice" in supported:
        features.append("multi-tool-call")

    # document: input_modalities contains file
    if "file" in modalities:
        features.append("document")

    # audio: input_modalities contains audio
    if "audio" in modalities:
        features.append("audio")

    # video: input_modalities contains video
    if "video" in modalities:
        features.append("video")

    return features


def gen_parameter_rules(supported: List[str] | None, max_completion_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    # Base common rules
    rules: List[Dict[str, Any]] = []

    def add(name: str, use_template: Optional[str] = None, extra: Optional[Dict[str, Any]] = None):
        item: Dict[str, Any] = {"name": name}
        if use_template:
            item["use_template"] = use_template
        if extra:
            item.update(extra)
        rules.append(item)

    # type 枚举值 ['float', 'int', 'boolean', 'string', 'text']
    # OpenRouter common sampling params
    add("temperature", "temperature") if not supported or "temperature" in supported else None
    add("top_p", "top_p") if not supported or "top_p" in supported else None
    # Some models support top_k
    if not supported or "top_k" in supported:
        add(
            "top_k",
            None,
            {
                "label": {"zh_Hans": "取样数量", "en_US": "Top k"},
                "type": "int",
                "help": {
                    "zh_Hans": "仅从每个后续标记的前 K 个选项中采样。",
                    "en_US": "Only sample from the top K options for each subsequent token.",
                },
                "required": False,
            },
        )

    add("presence_penalty", "presence_penalty") if not supported or "presence_penalty" in supported else None
    add("frequency_penalty", "frequency_penalty") if not supported or "frequency_penalty" in supported else None

    # repetition_penalty (6)
    if not supported or "repetition_penalty" in supported:
        add(
            "repetition_penalty",
            None,
            {
                "required": False,
                "type": "float",
                "default": 1.0,
                "min": 0.0,
                "max": 2.0,
                "label": {"zh_Hans": "重复惩罚", "en_US": "Repetition penalty"},
                "help": {
                    "zh_Hans": "用于控制模型生成时的重复度。提高 repetition_penalty 时可以降低重复。1.0 表示不做惩罚。",
                    "en_US": "Controls repetition during generation. Higher repetition_penalty reduces repetition. 1.0 means no penalty.",
                },
            },
        )

    # min_p (7)
    if not supported or "min_p" in supported:
        add(
            "min_p",
            None,
            {
                "required": False,
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 1.0,
                "label": {"zh_Hans": "最小概率 (Min-P)", "en_US": "Min P"},
                "help": {
                    "zh_Hans": "相对最可能标记的最小概率阈值。仅考虑概率不低于该阈值的标记。",
                    "en_US": "Minimum probability relative to the most likely token; filters out tokens below this threshold.",
                },
            },
        )

    # top_a (8)
    if not supported or "top_a" in supported:
        add(
            "top_a",
            None,
            {
                "required": False,
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 1.0,
                "label": {"zh_Hans": "Top-A", "en_US": "Top A"},
                "help": {
                    "zh_Hans": "基于最可能标记的概率进行自适应筛选，类似动态 Top-P。",
                    "en_US": "Adaptive filtering based on the probability of the most likely token; similar to dynamic Top-P.",
                },
            },
        )

    # max_tokens is frequently supported (10)
    if not supported or "max_tokens" in supported:
        base_max = int(max_completion_limit) if max_completion_limit is not None else 16384
        target = base_max // 3
        aligned = (target // 128) * 128
        if aligned < 128:
            aligned = 128 if base_max >= 128 else max(16, target)
        if aligned > base_max:
            aligned = (base_max // 128) * 128 or (base_max if base_max < 128 else 128)
        # Ensure default is at least 16 but not exceeding base_max
        if aligned < 16:
            aligned = 16 if base_max >= 16 else base_max
        param_name = "max_completion_tokens" if max_completion_limit is not None else "max_tokens"
        add(
            param_name,
            "max_tokens",
            {
                "default": int(aligned),
                "min": 1,
                "max": int(base_max),
            },
        )

    # seed (9)
    if not supported or "seed" in supported:
        add(
            "seed",
            None,
            {
                "label": {"zh_Hans": "Seed", "en_US": "Seed"},
                "type": "int",
                "required": False,
                "help": {
                    "zh_Hans": "指定后将尽可能使采样具有确定性；同一 seed 与参数应产生相同结果（部分模型不保证）。",
                    "en_US": "If specified, sampling will be as deterministic as possible; same seed and parameters should return the same result for some models.",
                },
            },
        )

    # logit_bias (11)
    if not supported or "logit_bias" in supported:
        add(
            "logit_bias",
            None,
            {
                "label": {"zh_Hans": "Logit 偏置", "en_US": "Logit Bias"},
                "type": "string",
                "required": False,
                "help": {
                    "zh_Hans": "JSON 对象：tokenId -> 偏置（-100 到 100）。用于在采样前调整 logits。",
                    "en_US": "JSON object mapping token IDs to bias (-100 to 100). Applied to logits prior to sampling.",
                },
            },
        )

    # logprobs (12)
    if not supported or "logprobs" in supported:
        add(
            "logprobs",
            None,
            {
                "label": {"zh_Hans": "返回对数概率", "en_US": "Logprobs"},
                "type": "boolean",
                "required": False,
                "default": False,
                "help": {
                    "zh_Hans": "是否返回输出 token 的对数概率。",
                    "en_US": "Whether to return log probabilities of output tokens.",
                },
            },
        )

    # top_logprobs (13)
    if not supported or "top_logprobs" in supported:
        add(
            "top_logprobs",
            None,
            {
                "label": {"zh_Hans": "Top 对数概率数量", "en_US": "Top logprobs"},
                "type": "int",
                "required": False,
                "min": 0,
                "max": 20,
                "help": {
                    "zh_Hans": "每个位置返回的最可能 token 数量（需开启 logprobs）。",
                    "en_US": "Number of most likely tokens to return at each position (requires logprobs).",
                },
            },
        )

    # response_format / structured outputs (14, 15)
    if not supported or ("response_format" in supported or "structured_outputs" in supported):
        include_json_schema = (supported is None) or ("structured_outputs" in supported)
        options = ["text", "json_object"] + (["json_schema"] if include_json_schema else [])
        # 改为对象结构，包含 type 字段；如选 json_schema，可配合下方 json_schema 提供 schema
        add(
            "response_format",
            "response_format",
            {
                "label": {"zh_Hans": "回复格式", "en_US": "Response Format"},
                "type": "string",
                "required": False,
                "options": options,
                "help": {
                    "zh_Hans": "强制输出特定格式。例如 {type: text|json_object|json_schema}。当选择 json_schema 时，请提供下方 JSON Schema。",
                    "en_US": "Force specific output format, e.g. {type: text|json_object|json_schema}. When using json_schema, also provide the JSON Schema below.",
                },
            },
        )
        # 仅当支持 structured_outputs 时提供 json_schema 模板，便于 Dify UI 输入并在调用侧组装
        if include_json_schema:
            add("json_schema", "json_schema")

    # stop sequences (16) Dify 默认强制提供该参数配置
    # if not supported or "stop" in supported:
    #     add(
    #         "stop",
    #         None,
    #         {
    #             "label": {"zh_Hans": "停止序列", "en_US": "Stop Sequences"},
    #             "type": "text",
    #             "required": False,
    #             "help": {
    #                 "zh_Hans": "以逗号分隔的停止序列（UI 将在发送到模型前拆分为数组）。",
    #                 "en_US": "Comma-separated stop sequences (will be split into an array before sending).",
    #             },
    #         },
    #     )

    # tool_choice (18)
    if not supported or "tool_choice" in supported:
        add(
            "tool_choice",
            None,
            {
                "label": {"zh_Hans": "工具选择", "en_US": "Tool Choice"},
                "type": "string",
                "required": False,
                "options": ["none", "auto", "required"],
                "help": {
                    "zh_Hans": "控制是否以及如何调用工具。可选 none、auto、required。",
                    "en_US": "Controls if/which tool is called: none, auto, or required.",
                },
            },
        )

    # parallel_tool_calls (19)
    if not supported or "parallel_tool_calls" in supported:
        add(
            "parallel_tool_calls",
            None,
            {
                "label": {"zh_Hans": "并行工具调用", "en_US": "Parallel Tool Calls"},
                "type": "boolean",
                "required": False,
                "default": True,
                "help": {
                    "zh_Hans": "启用时模型可同时调用多个函数。",
                    "en_US": "When enabled, the model can call multiple functions simultaneously.",
                },
            },
        )

    # enable_stream: toggle streaming responses
    add(
        "enable_stream",
        None,
        {
            "label": {"zh_Hans": "启用流式输出", "en_US": "Enable Stream"},
            "type": "boolean",
            "required": False,
            "default": True,
            "help": {
                "zh_Hans": "是否以流式方式返回模型输出。关闭后将一次性返回完整响应。",
                "en_US": "Whether to stream model outputs. When disabled, a single complete response is returned.",
            },
        },
    )

    # reasoning-related params
    if (not supported) or ("reasoning" in supported) or ("include_reasoning" in supported):
        # enable_thinking: only special value 'dynamic' is needed; leaving unset means disabled
        add(
            "enable_thinking",
            None,
            {
                "label": {"zh_Hans": "启用思考模式", "en_US": "Enable Thinking"},
                "type": "string",
                "required": False,
                "options": ["dynamic", "manual"],
                "help": {
                    "zh_Hans": "选择 dynamic 强制开启思考（忽略 reasoning_effort/reasoning_budget，等价 reasoning.enabled=true）。选择 manual 可与 reasoning_effort 或 reasoning_budget 搭配；若都不填则仅开启默认思考。",
                    "en_US": "Choose 'dynamic' to force reasoning.enabled=true (ignores reasoning_effort/reasoning_budget). Choose 'manual' to optionally use reasoning_effort or reasoning_budget; if neither is set, it just enables default reasoning.",
                },
            },
        )
        # reasoning_effort: OpenAI/Grok style effort control
        add(
            "reasoning_effort",
            None,
            {
                "label": {"zh_Hans": "思考力度（effort）", "en_US": "Reasoning Effort"},
                "type": "string",
                "required": False,
                "options": ["low", "medium", "high"],
                "help": {
                    "zh_Hans": "控制思考强度：low/medium/high。与 reasoning_budget 互斥，优先级更高。",
                    "en_US": "Controls reasoning strength: low/medium/high. Mutually exclusive with reasoning_budget and takes precedence.",
                },
            },
        )
        # reasoning_budget: Anthropic/Gemini style token budget
        add(
            "reasoning_budget",
            None,
            {
                "label": {"zh_Hans": "思考 Token 预算", "en_US": "Reasoning Token Budget"},
                "type": "int",
                "required": False,
                "min": 1,
                **({"max": int(max_completion_limit)} if max_completion_limit is not None else {}),
                "help": {
                    "zh_Hans": "为思考分配的最大 token 数（部分模型支持）。与 effort 互斥。",
                    "en_US": "Maximum tokens allocated for reasoning (supported by some models). Mutually exclusive with effort.",
                },
            },
        )
        # exclude_reasoning_tokens: don't return reasoning in the response
        add(
            "exclude_reasoning_tokens",
            None,
            {
                "label": {"zh_Hans": "隐藏思考内容", "en_US": "Exclude Reasoning Tokens"},
                "type": "boolean",
                "required": False,
                "default": False,
                "help": {
                    "zh_Hans": "启用后仍会使用思考，但不会在响应中包含 reasoning 字段。",
                    "en_US": "When enabled, model still uses reasoning but won't include it in the response.",
                },
            },
        )
        # verbosity (20) - reasoning-only
        add(
            "verbosity",
            None,
            {
                "label": {"zh_Hans": "详细程度", "en_US": "Verbosity"},
                "type": "string",
                "required": False,
                "default": "medium",
                "options": ["low", "medium", "high"],
                "help": {
                    "zh_Hans": "控制模型响应的详细程度：low/medium/high。",
                    "en_US": "Controls response verbosity: low/medium/high.",
                },
            },
        )

    # provider routing toggle & preferences
    add(
        "enable_provider",
        None,
        {
            "label": {"zh_Hans": "启用 Provider 路由", "en_US": "Enable Provider Routing"},
            "type": "boolean",
            "required": False,
            "default": False,
            "help": {
                "zh_Hans": "默认关闭。开启后可在“provider”对象中配置 OpenRouter Provider 路由偏好（order、only、ignore、allow_fallbacks、require_parameters、data_collection、quantizations、sort、max_price 等）",
                "en_US": "Disabled by default. When enabled, you can configure OpenRouter provider preferences via the 'provider' object (order, only, ignore, allow_fallbacks, require_parameters, data_collection, quantizations, sort, max_price, etc).",
            },
        },
    )
    add(
        "provider",
        None,
        {
            "label": {"zh_Hans": "Provider 路由配置", "en_US": "Provider Preferences"},
            "type": "text",
            "required": False,
            "help": {
                "zh_Hans": "仅当启用“启用 Provider 路由”后生效。请输入 JSON 字符串，系统会在调用前解析为对象并按 OpenRouter 规范透传，例如 {order:[\"anthropic\",\"openai\"], allow_fallbacks:true, sort:\"price\", only:[], ignore:[], require_parameters:false, data_collection:\"allow|deny\", quantizations:[\"fp16\"], max_price:{prompt:1, completion:2}}。",
                "en_US": "Effective only when 'Enable Provider Routing' is on. Enter a JSON string; it will be parsed into an object before request and passed per OpenRouter spec, e.g. {order:[\"anthropic\",\"openai\"], allow_fallbacks:true, sort:\"price\", only:[], ignore:[], require_parameters:false, data_collection:\"allow|deny\", quantizations:[\"fp16\"], max_price:{prompt:1, completion:2}}. See openrouter.ai/docs/features/provider-routing",
            },
        },
    )

    return rules


def build_model_yaml(entry: ModelEntry, provider: str) -> Dict[str, Any]:
    def to_per_million(v: Any) -> str:
        try:
            return str(float(v) * 1_000_000)
        except Exception:
            return "0"

    # 使用模型 id 按斜线分组后的最后一段作为 label（如 ai21/jamba-mini-1.7 -> jamba-mini-1.7）
    label_base = entry.id.rsplit("/", 1)[-1] if entry.id else (entry.name or entry.id)

    data: Dict[str, Any] = {
        "model": entry.id,  # use raw id in YAML
        "label": {"en_US": label_base, "zh_Hans": label_base},
        "model_type": "llm",
        "features": infer_features(entry),
        "model_properties": {
            "mode": "chat",
            "context_size": int(entry.context_length or entry.top_provider.get("context_length", 128000) if entry.top_provider else 128000),
        },
        "parameter_rules": gen_parameter_rules(entry.supported_parameters, (entry.top_provider or {}).get("max_completion_tokens") if entry.top_provider else None),
        "pricing": {
            # Convert per-token prices to per-million-token prices expected by unit=0.000001
            "input": to_per_million(entry.pricing.get("prompt")) if entry.pricing else "0",
            "output": to_per_million(entry.pricing.get("completion")) if entry.pricing else "0",
            "unit": "0.000001",
            "currency": "USD",
        },
    }

    # Ensure features at least empty list (no null)
    if not data["features"]:
        data["features"] = []

    return data


def dump_yaml(obj: Dict[str, Any]) -> str:
    # Avoid requiring PyYAML; write simple YAML via safe dumper if available else manual
    if yaml is not None:
        return yaml.safe_dump(obj, allow_unicode=True, sort_keys=False)
    # Fallback: minimal manual YAML (not perfect but sufficient)
    try:
        import ruamel.yaml  # type: ignore
        ry = ruamel.yaml.YAML()
        from io import StringIO
        s = StringIO()
        ry.dump(obj, s)
        return s.getvalue()
    except Exception:
        # Extremely minimal fallback
        return json.dumps(obj, ensure_ascii=False, indent=2)


def load_models(input_json: Optional[str], api_url: str) -> List[ModelEntry]:
    data_json: Dict[str, Any]
    if input_json:
        with open(input_json, "r", encoding="utf-8") as f:
            data_json = json.load(f)
    else:
        if requests is None:
            raise RuntimeError("requests not available; install requests or use --input-json")
        resp = requests.get(api_url, timeout=30)
        resp.raise_for_status()
        data_json = resp.json()

    models: List[ModelEntry] = []
    for item in data_json.get("data", []):
        mid = item.get("id")
        if not isinstance(mid, str) or not mid:
            continue
        models.append(
            ModelEntry(
                id=mid,
                name=item.get("name"),
                context_length=item.get("context_length"),
                architecture=item.get("architecture"),
                pricing=item.get("pricing"),
                top_provider=item.get("top_provider"),
                supported_parameters=item.get("supported_parameters"),
                input_modalities=(item.get("architecture") or {}).get("input_modalities"),
            )
        )
    return models


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert OpenRouter models to Dify YAMLs")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--input-json")
    parser.add_argument("--tmp-dir")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)

    try:
        models = load_models(args.input_json, args.api_url)
    except Exception as e:
        print(f"Failed to load models: {e}", file=sys.stderr)
        return 1

    if not models:
        print("No models found in response.")
        return 0

    out_dir = args.tmp_dir or tempfile.mkdtemp(prefix="dify-openrouter-models-")
    os.makedirs(out_dir, exist_ok=True)

    written: List[Tuple[str, str]] = []  # (model_id, file_path)

    for m in models:
        try:
            doc = build_model_yaml(m, args.provider)
        except Exception as e:
            print(f"Skip {m.id}: build error: {e}", file=sys.stderr)
            continue

        safe_model_name = slugify(m.id).replace("/", "-")
        file_name = f"{safe_model_name}.yaml"
        file_path = os.path.join(out_dir, file_name)

        if args.dry_run:
            print(f"Would write: {file_path}")
            # Show a short preview
            preview = dump_yaml(doc)
            print(preview[:400] + ("...\n" if len(preview) > 400 else ""))
        else:
            content = dump_yaml(doc)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            written.append((m.id, file_path))

    # Write _position.yaml with list of models (by id) in current order
    position_path = os.path.join(out_dir, "_position.yaml")
    position_models = [m.id for m in models]
    if args.dry_run:
        print(f"Would write: {position_path}")
        if yaml is not None:
            pos_preview = yaml.safe_dump(position_models, allow_unicode=True, sort_keys=False)
        else:
            pos_preview = "\n".join(f"- {mid}" for mid in position_models) + "\n"
        print(pos_preview[:400] + ("...\n" if len(pos_preview) > 400 else ""))
    else:
        if yaml is not None:
            pos_content = yaml.safe_dump(position_models, allow_unicode=True, sort_keys=False)
        else:
            pos_content = "\n".join(f"- {mid}" for mid in position_models) + "\n"
        with open(position_path, "w", encoding="utf-8") as f:
            f.write(pos_content)

    print(f"Converted {len(written) or len(models)} models. Output dir: {out_dir}")
    if not args.dry_run:
        # Print a small sample of outputs
        for i, (mid, path) in enumerate(written[:10]):
            print(f" - {mid} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
