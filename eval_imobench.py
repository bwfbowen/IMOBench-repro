#!/usr/bin/env python3
"""
Standalone IMO-Bench evaluator.

This script is intentionally separate from RL reward / validation code. It supports:
  - IMO-AnswerBench: local solver generation + exact answer scoring
  - IMO-ProofBench: local solver generation + deterministic Gemini proof grading
  - IMO-GradingBench: deterministic Gemini grading calibration

Default comparison pair:
  - deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  - hbx/JustRL-DeepSeek-1.5B
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Protocol, Sequence

os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

import torch


ANSWERBENCH_URL = "https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/answerbench_v2.csv"
PROOFBENCH_URL = "https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/proofbench.csv"
GRADINGBENCH_URL = "https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/gradingbench.csv"

DEFAULT_MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "hbx/JustRL-DeepSeek-1.5B",
]
DEFAULT_LOCAL_JUDGE_MODELS = [
    "Qwen/QwQ-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
]

FOUR_WAY_LABELS = ("Correct", "Almost", "Partial", "Incorrect")
BATCH_TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "BATCH_STATE_SUCCEEDED",
    "BATCH_STATE_FAILED",
    "BATCH_STATE_CANCELLED",
    "BATCH_STATE_EXPIRED",
}
BATCH_SUCCESS_STATES = {
    "JOB_STATE_SUCCEEDED",
    "BATCH_STATE_SUCCEEDED",
}


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "na"


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_text(url: str, timeout: int = 60) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def download_if_missing(url: str, path: Path) -> Path:
    if path.exists():
        return path
    ensure_dir(path.parent)
    text = download_text(url)
    path.write_text(text, encoding="utf-8")
    return path


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def stratified_sample_by_points(
    rows: Sequence[Dict[str, str]],
    per_point: int,
    seed: int,
) -> List[Dict[str, str]]:
    if per_point <= 0:
        return list(rows)
    grouped: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[safe_int(row.get("Points", 0))].append(row)

    rng = random.Random(seed)
    selected: List[Dict[str, str]] = []
    for points in sorted(grouped):
        bucket = list(grouped[points])
        rng.shuffle(bucket)
        selected.extend(bucket[:per_point])
    selected.sort(key=lambda row: row.get("Grading ID", ""))
    return selected


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def existing_ids(path: Path, key: str) -> set[str]:
    ids = set()
    for row in load_jsonl(path):
        value = row.get(key)
        if value is not None:
            ids.add(str(value))
    return ids


def existing_key_tuples(path: Path, keys: Sequence[str]) -> set[tuple[str, ...]]:
    values = set()
    for row in load_jsonl(path):
        item = []
        skip = False
        for key in keys:
            if key not in row:
                skip = True
                break
            item.append(str(row[key]))
        if not skip:
            values.add(tuple(item))
    return values


def split_batches(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def parse_csv_list(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def slurm_gpu_allocated() -> bool:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return bool(visible and visible not in {"NoDevFiles", "-1"})


def visible_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or visible in {"NoDevFiles", "-1"}:
        return 0
    return len([gpu for gpu in visible.split(",") if gpu.strip()])


def proofbench_split(problem_id: str) -> str:
    if problem_id.startswith("PB-Basic-"):
        return "basic"
    if problem_id.startswith("PB-Advanced-"):
        return "advanced"
    return "unknown"


def points_to_label(points: int) -> str:
    if points >= 7:
        return "Correct"
    if points == 6:
        return "Almost"
    if points <= 0:
        return "Incorrect"
    return "Partial"


def label_to_paper_points(label: str) -> int:
    normalized = label.strip().capitalize()
    return {"Correct": 7, "Almost": 6, "Partial": 1, "Incorrect": 0}[normalized]


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def macro_f1(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> float:
    per_label = []
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_label.append(f1)
    return sum(per_label) / len(per_label) if per_label else 0.0


def ensure_boxed_answer_prompt(problem: str) -> str:
    return (
        "Solve the following olympiad-style mathematics problem. "
        "Reason step by step, and put the final answer inside \\\\boxed{}.\n\n"
        f"Problem:\n{problem}\n"
    )


def proof_prompt(problem: str) -> str:
    return (
        "Solve the following olympiad-style mathematics problem. "
        "Provide a rigorous and complete proof.\n\n"
        f"Problem:\n{problem}\n"
    )


def gradingbench_prompt(problem: str, proposed_solution: str) -> str:
    return f"""
Carefully analyze the given problem statement and the proposed solution, and then write
out your analysis regarding the correctness of the proposed solution.
After the analysis, you must provide a score based on the following criteria:
• incorrect: The solution is completely incorrect or irrelevant.
• partial: The solution is partially correct but has significant errors or omissions.
• almost: The solution is almost correct but contains minor errors or inaccuracies.
• correct: The solution is fully correct and complete.
The very last part of your response must be only one of the following words: incorrect,
partial, almost, or correct.
Problem:{problem} Solution:{proposed_solution}
""".strip()


class TextGenerator(Protocol):
    def generate(
        self,
        prompts: Sequence[str],
        max_new_tokens: int,
        batch_size: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> List[str]: ...


@dataclass
class HFGenerator:
    model_name: str
    max_input_length: int = 8192
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._uses_gpu = slurm_gpu_allocated()
        try:
            if self._uses_gpu:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code,
                    dtype=torch.bfloat16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code,
                )
        except Exception as err:  # noqa: BLE001
            if self._uses_gpu:
                print(
                    f"[HFGenerator] GPU load failed for {self.model_name}; falling back to CPU. Error: {err}",
                    file=sys.stderr,
                )
                self._uses_gpu = False
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code,
                )
            else:
                raise
        self.model.eval()

    def _input_device(self) -> torch.device | None:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return None

    def _render_prompt(self, prompt: str) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def _generation_config_for_call(
        self,
        do_sample: bool,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> Any | None:
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is None:
            return None
        generation_config = copy.deepcopy(generation_config)
        generation_config.do_sample = bool(do_sample)
        generation_config.max_new_tokens = max_new_tokens
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        if do_sample:
            generation_config.temperature = temperature
            generation_config.top_p = top_p
        else:
            # Some pretrained models ship sampling-only flags in generation_config.json.
            # Reset them for deterministic decoding so transformers does not warn.
            reset_values = {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 50,
                "typical_p": 1.0,
                "min_p": None,
                "top_h": None,
                "epsilon_cutoff": 0.0,
                "eta_cutoff": 0.0,
            }
            for attr, value in reset_values.items():
                if hasattr(generation_config, attr):
                    setattr(generation_config, attr, value)
        return generation_config

    def generate(
        self,
        prompts: Sequence[str],
        max_new_tokens: int,
        batch_size: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> List[str]:
        rendered_prompts = [self._render_prompt(prompt) for prompt in prompts]
        outputs: List[str] = []

        with torch.inference_mode():
            for prompt_batch in split_batches(rendered_prompts, batch_size):
                enc = self.tokenizer(
                    list(prompt_batch),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_input_length,
                )
                input_device = self._input_device()
                if input_device is not None and input_device.type != "cpu":
                    enc = {k: v.to(input_device) for k, v in enc.items()}

                generate_kwargs: Dict[str, Any] = {
                    **enc,
                }
                generation_config = self._generation_config_for_call(
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                )
                if generation_config is not None:
                    generate_kwargs["generation_config"] = generation_config
                else:
                    generate_kwargs["max_new_tokens"] = max_new_tokens
                    generate_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
                    generate_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
                    generate_kwargs["do_sample"] = do_sample
                    if do_sample:
                        generate_kwargs["temperature"] = temperature
                        generate_kwargs["top_p"] = top_p

                generated = self.model.generate(**generate_kwargs)

                input_lengths = enc["attention_mask"].sum(dim=1).tolist()
                for seq, prompt_len in zip(generated, input_lengths):
                    new_tokens = seq[int(prompt_len) :]
                    outputs.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

        return outputs


@dataclass
class VLLMGenerator:
    model_name: str
    max_input_length: int = 8192
    trust_remote_code: bool = False
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.9

    def __post_init__(self) -> None:
        from transformers import AutoTokenizer

        try:
            from vllm import LLM
        except ImportError as err:  # noqa: BLE001
            raise RuntimeError(
                "vLLM local engine requested, but the 'vllm' package is not installed. "
                "Install it in a separate environment or this one, keeping in mind it may replace "
                "the current torch/transformers versions."
            ) from err

        if not slurm_gpu_allocated():
            raise RuntimeError(
                "vLLM local engine requires CUDA_VISIBLE_DEVICES to point at one or more GPUs."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        llm_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "trust_remote_code": self.trust_remote_code,
            "dtype": "bfloat16",
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }
        if self.max_model_len:
            llm_kwargs["max_model_len"] = self.max_model_len
        gpu_count = visible_gpu_count()
        if gpu_count > 1:
            llm_kwargs["tensor_parallel_size"] = gpu_count

        self.llm = LLM(**llm_kwargs)

    def _render_prompt(self, prompt: str) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def generate(
        self,
        prompts: Sequence[str],
        max_new_tokens: int,
        batch_size: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> List[str]:
        from vllm import SamplingParams

        rendered_prompts = [self._render_prompt(prompt) for prompt in prompts]
        outputs: List[str] = []
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
            skip_special_tokens=True,
        )
        for prompt_batch in split_batches(rendered_prompts, batch_size):
            generated = self.llm.generate(list(prompt_batch), sampling_params, use_tqdm=False)
            for request_output in generated:
                candidates = getattr(request_output, "outputs", None) or []
                if not candidates:
                    outputs.append("")
                    continue
                outputs.append(candidates[0].text.strip())
        return outputs


def resolve_local_max_model_len(
    max_input_length: int,
    max_new_tokens: int,
    explicit_max_model_len: int = 0,
) -> int | None:
    if explicit_max_model_len > 0:
        return explicit_max_model_len
    if max_input_length <= 0 or max_new_tokens <= 0:
        return None
    return max_input_length + max_new_tokens


def build_local_generator(
    *,
    engine: str,
    model_name: str,
    max_input_length: int,
    trust_remote_code: bool,
    max_model_len: int | None = None,
    vllm_gpu_memory_utilization: float = 0.9,
) -> TextGenerator:
    if engine == "transformers":
        return HFGenerator(
            model_name=model_name,
            max_input_length=max_input_length,
            trust_remote_code=trust_remote_code,
        )
    if engine == "vllm":
        return VLLMGenerator(
            model_name=model_name,
            max_input_length=max_input_length,
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
        )
    raise ValueError(f"Unknown local engine: {engine}")


class GeminiJudge:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: int = 600,
        retries: int = 5,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.temperature = temperature
        self.top_p = top_p

    def _build_generation_config(self, response_mime_type: str | None = None) -> Dict[str, Any]:
        generation_config: Dict[str, Any] = {
            "temperature": self.temperature,
            "topP": self.top_p,
        }
        if response_mime_type is not None:
            generation_config["responseMimeType"] = response_mime_type
        return generation_config

    def _build_text_request(self, prompt: str, response_mime_type: str | None = None) -> Dict[str, Any]:
        return {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": self._build_generation_config(response_mime_type=response_mime_type),
        }

    @staticmethod
    def _extract_text_from_generate_response(data: Dict[str, Any]) -> str:
        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return ""
        first_candidate = candidates[0]
        if not isinstance(first_candidate, dict):
            return ""
        content = first_candidate.get("content")
        if not isinstance(content, dict):
            return ""
        parts = content.get("parts")
        if not isinstance(parts, list):
            return ""
        return "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()

    def _batch_request(
        self,
        method: str,
        path: str,
        payload: Dict[str, Any] | None = None,
        *,
        timeout: int = 60,
        download: bool = False,
        alt_media: bool = False,
    ) -> Any:
        base_url = "https://generativelanguage.googleapis.com/download/v1beta" if download else "https://generativelanguage.googleapis.com/v1beta"
        url = f"{base_url}/{path}"
        if alt_media:
            url += "?" + urllib.parse.urlencode({"alt": "media"})

        headers = {"x-goog-api-key": self.api_key}
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()

        if alt_media:
            return raw
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    @staticmethod
    def batch_state(batch_job: Dict[str, Any]) -> str:
        metadata = batch_job.get("metadata")
        if isinstance(metadata, dict) and metadata.get("state"):
            return str(metadata["state"])
        if batch_job.get("state"):
            return str(batch_job["state"])
        return "JOB_STATE_SUCCEEDED" if batch_job.get("done") else "JOB_STATE_PENDING"

    @staticmethod
    def batch_inline_responses(batch_job: Dict[str, Any]) -> List[Dict[str, Any]]:
        response = batch_job.get("response")
        if isinstance(response, dict):
            inlined = response.get("inlinedResponses")
            if isinstance(inlined, list):
                return [item for item in inlined if isinstance(item, dict)]
            if isinstance(inlined, dict):
                nested = inlined.get("inlinedResponses")
                if isinstance(nested, list):
                    return [item for item in nested if isinstance(item, dict)]
        dest = batch_job.get("dest")
        if isinstance(dest, dict):
            inlined = dest.get("inlinedResponses")
            if isinstance(inlined, list):
                return [item for item in inlined if isinstance(item, dict)]
            if isinstance(inlined, dict):
                nested = inlined.get("inlinedResponses")
                if isinstance(nested, list):
                    return [item for item in nested if isinstance(item, dict)]
        return []

    @staticmethod
    def batch_responses_file(batch_job: Dict[str, Any]) -> str | None:
        response = batch_job.get("response")
        if isinstance(response, dict) and response.get("responsesFile"):
            return str(response["responsesFile"])
        dest = batch_job.get("dest")
        if isinstance(dest, dict) and dest.get("fileName"):
            return str(dest["fileName"])
        return None

    def create_batch_job(self, requests: Sequence[Dict[str, Any]], display_name: str) -> Dict[str, Any]:
        payload = {
            "batch": {
                "display_name": display_name,
                "input_config": {
                    "requests": {
                        "requests": list(requests),
                    }
                },
            }
        }
        return self._batch_request(
            "POST",
            f"models/{self.model}:batchGenerateContent",
            payload=payload,
            timeout=60,
        )

    def get_batch_job(self, batch_name: str) -> Dict[str, Any]:
        return self._batch_request("GET", batch_name, timeout=60)

    def download_batch_result_file(self, file_name: str) -> bytes:
        return self._batch_request(
            "GET",
            f"{file_name}:download",
            timeout=120,
            download=True,
            alt_media=True,
        )

    def _request(self, prompt: str, response_mime_type: str | None = None) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = self._build_text_request(prompt, response_mime_type=response_mime_type)
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})

        last_err: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return self._extract_text_from_generate_response(data)
            except Exception as err:  # noqa: BLE001
                last_err = err
                if attempt < self.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Gemini judge request failed after {self.retries} attempts: {last_err}") from last_err

    def _call_json(self, prompt: str) -> Dict[str, Any]:
        return self._parse_json_response(self._request(prompt, response_mime_type="application/json"))

    def _call_text(self, prompt: str) -> str:
        return self._request(prompt, response_mime_type=None)

    @staticmethod
    def _parse_json_response(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    @staticmethod
    def _extract_boxed_grade(text: str) -> str | None:
        match = re.search(r"\\boxed\{(Correct|Incorrect)\}", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).capitalize()
        return None

    @staticmethod
    def _extract_points_score(text: str) -> int | None:
        match = re.search(r"<points>\s*([0167])\s*out of 7\s*</points>", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r"([0167])\s*out of 7", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _extract_label_from_text(text: str) -> str | None:
        text = text.strip().lower()
        if not text:
            return None
        last_token_match = re.search(r"(incorrect|partial|almost|correct)\s*$", text)
        if last_token_match:
            return last_token_match.group(1)
        return None

    def judge_answer(self, problem: str, model_solution: str, golden_answer: str) -> Dict[str, Any]:
        prompt = f"""
# System Role: Deterministic Mathematical Autograder
You are a precise, automated grading system. Your sole function is
to determine if the final answer provided in the Model Solution is
mathematically equivalent to the Golden Answer. You must NOT grade
the reasoning or steps, only the final result.

# 1. Grading Guidelines (Equivalence Rules)
Equivalence is mandatory for a correct grade. You must rigorously
verify if the answers represent the exact same mathematical value
or expression, even if the format differs.
* **Algebraic Equivalence:** e.g., 'n(n+1)/2' is equivalent to
'n^2/2 + n/2'. You must verify the algebra.
* **Numerical Equivalence:** e.g., '1/2' is equivalent to '0.5';
'sqrt(2)/2' is equivalent to '1/sqrt(2)'.
* **Set/List Equivalence:** Unless specified as an ordered
tuple/vector, the order of elements does not matter.
* **Partial Credit:** No partial credit is allowed. If the answer
is incomplete or partially incorrect, it is incorrect.
* **No Answers:** If no clear, unambiguous final answer can be
extracted, the solution must be graded as incorrect.

# 3. Output Protocol (Strict Compliance Required)
You must execute the task using a two-part structure.
**Part 1: Analysis (Chain-of-Thought)**
You MUST perform your analysis within <thinking></thinking> tags.
Make your thinking concise. This section details your reasoning
process and must follow these steps sequentially:
1. **Golden Answer:** State the Golden Answer.
2. **Extracted Model Answer:** State the extracted answer. If none found, state "No clear final answer found."
3. **Equivalence Analysis:** Compare the two answers using the grading guidelines.
4. **Conclusion:** State the final determination ("Correct" or "Incorrect").
**Part 2: Final Grade**
Immediately following the closing </thinking> tag, output **ONLY**
the final grade.
* If Correct: \\boxed{{Correct}}
* If Incorrect: \\boxed{{Incorrect}}

# 4. Input Data
Here is the problem, model solution, and golden answer to grade:
Problem: {problem}
Model Solution: {model_solution}
Golden Answer: {golden_answer}
""".strip()

        response = self._call_text(prompt)
        grade = self._extract_boxed_grade(response)
        if grade is None:
            raise RuntimeError("AnswerAutoGrader response did not contain a parseable boxed grade.")
        return {
            "correct": grade == "Correct",
            "judge_grade": grade,
            "judge_response": response,
        }

    def _build_proof_prompt(
        self,
        problem: str,
        candidate_solution: str,
        reference_solution: str,
        grading_guidelines: str,
    ) -> str:
        return f"""
You are an expert grader for the International Mathematics Olympiad
(IMO). Your task is to evaluate a proposed solution strictly and
rigorously. Keep in mind the standards at the IMO are extremely
high: only arguments that are logically sound, complete, and
precise should be rewarded.

### General Scoring Rubric
Scores are assigned on a 0-7 scale. The general guidelines are:
* **7 Points (Correct):** The solution is complete, correct, and
fully rigorous. If the submission contains incorrect attempts or
lines of reasoning but ultimately presents a complete and correct
solution, it should still be awarded full points; the presence of
earlier, discarded work does not detract from the final correct
proof.
* **6 Points (Almost Correct):** The solution is almost correct with
a sound core argument, but contains minor errors in calculation
or small gaps in logic. Missing proofs for major components,
unjustified claims, or sketchy arguments are **not** eligible
for 6 points.
* **1 Point (Partial Progress):** The solution demonstrates
substantial progress explicitly mentioned in the grading
guidelines. Initial observations, reformulating the problem
without making substantive headway, or proving partial results
not mentioned in the grading guidelines are generally **not**
eligible for this score.
* **0 Points (Incorrect):** The solution doesn't make substantial
progress that is a key step in the full solution or is
fundamentally flawed. All partial progress without key results
or lacking rigor also fall in this category.

### Input Data and Interpretation
You are provided with the following:
1. **Problem Statement:** The IMO problem.
2. **Ground Truth Solution:** A reference solution. Assume this
solution is correct. It demonstrates one valid approach.
3. **Specific Grading Guidelines:** Criteria for awarding credit for
this specific problem. These guidelines take precedence over
the General Scoring Rubric, especially for partial credit.
4. **Proposed Solution:** The student submission.

### Evaluation Process
You must follow this structured process:
1. **Analyze References:** Meticulously read and understand the
problem and Ground Truth Solution check the Specific Grading
Guidelines. Identify the key steps for a complete solution and
the criteria for partial credit.
2. **Step-by-Step Verification:** Verify the logical validity and
rigor of every step. Identify all flaws, gaps, assumptions, and
errors. **Make sure you fully understand every piece of logic
behind each step of the proposed solution, you must be careful
for solutions that 'pretend' to be correct.**
3. **Assess Progress:** Determine the extent of non-trivial progress
made.
4. **Score Determination:** Compare the findings against the
Specific Grading Guidelines and the General Rubric to determine
the final score.

### Output Requirements
You must provide your final score in the format <points>N out of
7</points>. Ensure the '<points>' block is used **only once**, as
your answer will be parsed based on the first <points></points>
block that appears in your whole response.

**PROBLEM STATEMENT**
{problem}
**GROUND-TRUTH SOLUTION**
{reference_solution}
**SPECIFIC GRADING GUIDELINES**
{grading_guidelines}
**PROPOSED SOLUTION**
{candidate_solution}
Present your detailed thought process and formal justification
based on the scoring rubric and grading guidelines, and finally
present your final score in the format below.
[Select one of the following options]
<points>7 out of 7</points>
<points>6 out of 7</points>
<points>1 out of 7</points>
<points>0 out of 7</points>
""".strip()

    def build_proof_request(
        self,
        problem: str,
        candidate_solution: str,
        reference_solution: str,
        grading_guidelines: str,
    ) -> Dict[str, Any]:
        return self._build_text_request(
            self._build_proof_prompt(
                problem=problem,
                candidate_solution=candidate_solution,
                reference_solution=reference_solution,
                grading_guidelines=grading_guidelines,
            )
        )

    def _proof_grade_from_response_text(self, response: str) -> Dict[str, Any]:
        score = self._extract_points_score(response)
        if score is None:
            raise RuntimeError("ProofAutoGrader response did not contain a parseable <points> score.")
        return {
            "score_0_7": score,
            "label_4way": points_to_label(score),
            "judge_response": response,
        }

    def judge_proof(
        self,
        problem_id: str,
        problem: str,
        candidate_solution: str,
        reference_solution: str,
        grading_guidelines: str,
    ) -> Dict[str, Any]:
        del problem_id
        prompt = self._build_proof_prompt(
            problem=problem,
            candidate_solution=candidate_solution,
            reference_solution=reference_solution,
            grading_guidelines=grading_guidelines,
        )
        response = self._call_text(prompt)
        return self._proof_grade_from_response_text(response)

    def extract_gradingbench_label(self, model_response: str) -> str:
        prompt = f"""
## Instructions for Extracting Final Scores
**Objective:** Given an response of an evaluation prompt, extract
the final score presented within the response and format it
specifically.

**Process:**
1. **Analyze the response:** Scan the response to identify the final
score provided by the evaluator.
2. **Extract and format the final answer:** Present the extracted
score on a new line, preceded exactly by "Final answer: ".

**Formatting Rules:**
* **Evaluation Categories:** The expected output must be one
of the following categories: 'correct', 'partial', 'almost',
'incorrect', or 'not_found'.
* **Score Identification:** The extraction is based on identifying
the keyword used by the evaluator to summarize their conclusion.
* **incorrect:** The evaluator concluded that the solution is
completely incorrect or irrelevant.
* **partial:** The evaluator concluded that the solution is
partially correct but has significant errors or omissions.
* **almost:** The evaluator concluded that the solution is almost
correct but contains minor errors or inaccuracies.
* **correct:** The evaluator concluded that the solution is fully
correct and complete.
* **not_found:** The evaluation response does not clearly contain
one of the four explicit scores listed above.
* **Extraction:** Determine the provided score from the response
and extract the category ('correct', 'partial', 'almost', or
'incorrect'). If a score cannot be reliably identified within
the text, the output must be 'not_found'.
**Note:** No additional markings or explanations are needed beyond
"Final answer: " and the extracted answer.
Below is the response:
{model_response}
""".strip()

        response = self._call_text(prompt)
        match = re.search(r"Final answer:\s*(correct|partial|almost|incorrect|not_found)", response, flags=re.IGNORECASE)
        if not match:
            return "not_found"
        return match.group(1).lower()

    def judge_gradingbench(
        self,
        grading_id: str,
        problem_id: str,
        problem: str,
        proposed_solution: str,
    ) -> Dict[str, Any]:
        del grading_id, problem_id
        prompt = gradingbench_prompt(problem=problem, proposed_solution=proposed_solution)
        response = self._call_text(prompt)
        return self._parse_gradingbench_response(response)

    def _parse_gradingbench_response(self, response: str) -> Dict[str, Any]:
        label = self._extract_label_from_text(response)
        if label is None:
            label = self.extract_gradingbench_label(response)
        if label == "not_found":
            label = "incorrect"
        label = label.capitalize()
        return {
            "score_0_7": label_to_paper_points(label),
            "label_4way": label,
            "judge_response": response,
        }


class MathVerifyAnswerJudge:
    def __init__(self) -> None:
        try:
            from math_verify import ExprExtractionConfig, LatexExtractionConfig, LatexNormalizationConfig, parse, verify
        except ImportError as err:  # noqa: BLE001
            raise RuntimeError(
                "Math-Verify is not installed. Install `math-verify[antlr4_13_2]` to use "
                "`--answer-grader-backend math_verify`."
            ) from err

        self._parse = parse
        self._verify = verify
        self._gold_cache: Dict[str, Any] = {}
        self._gold_extraction_config = [
            LatexExtractionConfig(),
            ExprExtractionConfig(),
        ]
        # Prefer boxed math when present, and use stricter normalization for reward-style checking.
        self._prediction_extraction_config = [
            LatexExtractionConfig(
                boxed_match_priority=0,
                normalization_config=LatexNormalizationConfig(
                    basic_latex=True,
                    units=True,
                    malformed_operators=False,
                    nits=False,
                    boxed="all",
                    equations=False,
                ),
            ),
            ExprExtractionConfig(),
        ]

    def _parse_gold(self, golden_answer: str) -> Any:
        cached = self._gold_cache.get(golden_answer)
        if cached is None and golden_answer not in self._gold_cache:
            cached = self._parse(golden_answer, extraction_config=self._gold_extraction_config)
            self._gold_cache[golden_answer] = cached
        return self._gold_cache[golden_answer]

    def _parse_prediction(self, text: str) -> Any:
        return self._parse(text, extraction_config=self._prediction_extraction_config)

    @staticmethod
    def _extract_boxed_contents(text: str) -> List[str]:
        needle = r"\boxed{"
        contents: List[str] = []
        start = 0
        while True:
            idx = text.find(needle, start)
            if idx < 0:
                break
            i = idx + len(needle)
            depth = 1
            buf: List[str] = []
            while i < len(text) and depth > 0:
                ch = text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                if depth > 0:
                    buf.append(ch)
                i += 1
            if depth == 0:
                contents.append("".join(buf).strip())
                start = i
            else:
                break
        return contents

    @classmethod
    def _cleanup_text(cls, text: str) -> str:
        text = text.strip()
        text = text.replace("$", "")
        text = text.replace(r"\left", "")
        text = text.replace(r"\right", "")
        text = re.sub(r"\\(?:text|mathrm|operatorname)\{([^{}]*)\}", r"\1", text)
        text = text.replace("{", "").replace("}", "")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @classmethod
    def _normalize_text_answer(cls, text: str) -> str:
        text = cls._cleanup_text(text).lower()
        text = re.sub(r"\s*([(),])\s*", r"\1", text)
        return text.strip(" .,:;")

    @classmethod
    def _prediction_candidates(cls, model_solution: str) -> List[str]:
        candidates: List[str] = []
        seen = set()

        def add(text: str) -> None:
            normalized = text.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                candidates.append(normalized)

        for boxed in reversed(cls._extract_boxed_contents(model_solution)[-3:]):
            add(boxed)
        add(model_solution)
        lines = [line.strip() for line in model_solution.splitlines() if line.strip()]
        if lines:
            add(lines[-1])
            if len(lines) > 1:
                add(lines[-2])
        add(model_solution.strip()[-500:])
        return candidates

    @classmethod
    def _split_function_relation(cls, text: str) -> tuple[str, str, str] | None:
        cleaned = cls._cleanup_text(text)
        match = re.fullmatch(r"([A-Za-z]\w*)\s*\(\s*([A-Za-z]\w*)\s*\)\s*=\s*(.+)", cleaned)
        if not match:
            return None
        name, arg, rhs = match.groups()
        return name.lower(), arg.lower(), rhs.strip()

    def _compare_function_relation(self, golden_answer: str, candidate: str) -> bool:
        gold_relation = self._split_function_relation(golden_answer)
        candidate_relation = self._split_function_relation(candidate)
        if gold_relation is None or candidate_relation is None:
            return False
        if gold_relation[:2] != candidate_relation[:2]:
            return False

        try:
            from sympy import simplify, sympify

            gold_rhs = sympify(gold_relation[2].replace("^", "**"))
            candidate_rhs = sympify(candidate_relation[2].replace("^", "**"))
            return bool(simplify(gold_rhs - candidate_rhs) == 0)
        except Exception:  # noqa: BLE001
            pass

        gold_rhs = self._parse(gold_relation[2], extraction_config=self._gold_extraction_config)
        candidate_rhs = self._parse(candidate_relation[2], extraction_config=self._prediction_extraction_config)
        if gold_rhs and candidate_rhs:
            return bool(self._verify(gold_rhs, candidate_rhs, allow_set_relation_comp=True))
        return self._normalize_text_answer(golden_answer) == self._normalize_text_answer(candidate)

    def _matches_text_fallback(self, golden_answer: str, candidate: str) -> bool:
        gold_norm = self._normalize_text_answer(golden_answer)
        candidate_norm = self._normalize_text_answer(candidate)
        if not gold_norm or not candidate_norm:
            return False
        if gold_norm == candidate_norm:
            return True
        if self._compare_function_relation(golden_answer, candidate):
            return True
        if gold_norm in {"no solutions", "no solution"}:
            return "no solution" in candidate_norm
        if gold_norm == "no positive integers":
            return "no positive integer" in candidate_norm
        if gold_norm == "all positive integers":
            return "all positive integers" in candidate_norm or "every positive integer" in candidate_norm
        if gold_norm == "taking the empty card":
            return "empty card" in candidate_norm
        if gold_norm.endswith(" is prime"):
            subject = gold_norm[: -len(" is prime")].strip()
            return "prime" in candidate_norm and (not subject or subject in candidate_norm)
        return False

    @staticmethod
    def _result_payload(
        correct: bool,
        golden_answer: str,
        parsed_gold: bool,
        parsed_prediction: bool,
        matched_candidate: str | None,
    ) -> str:
        return json.dumps(
            {
                "backend": "math_verify",
                "correct": correct,
                "golden_answer": golden_answer,
                "gold_parsed": parsed_gold,
                "prediction_parsed": parsed_prediction,
                "matched_candidate": matched_candidate,
            },
            ensure_ascii=False,
        )

    def judge_answer(self, problem: str, model_solution: str, golden_answer: str) -> Dict[str, Any]:
        del problem

        parsed_gold = self._parse_gold(golden_answer)
        prediction_candidates = self._prediction_candidates(model_solution)
        prediction_parsed = False

        if parsed_gold:
            for candidate in prediction_candidates:
                parsed_prediction = self._parse_prediction(candidate)
                if parsed_prediction:
                    prediction_parsed = True
                    if self._verify(parsed_gold, parsed_prediction, allow_set_relation_comp=True):
                        return {
                            "correct": True,
                            "judge_grade": "Correct",
                            "judge_response": self._result_payload(
                                True,
                                golden_answer,
                                True,
                                True,
                                candidate,
                            ),
                        }

        for candidate in prediction_candidates:
            if self._matches_text_fallback(golden_answer, candidate):
                return {
                    "correct": True,
                    "judge_grade": "Correct",
                    "judge_response": self._result_payload(
                        True,
                        golden_answer,
                        bool(parsed_gold),
                        prediction_parsed,
                        candidate,
                    ),
                }

        return {
            "correct": False,
            "judge_grade": "Incorrect",
            "judge_response": self._result_payload(
                False,
                golden_answer,
                bool(parsed_gold),
                prediction_parsed,
                prediction_candidates[0] if prediction_candidates else None,
            ),
        }

class LocalHFJudge(GeminiJudge):
    def __init__(
        self,
        model_name: str,
        max_input_length: int = 8192,
        max_new_tokens: int = 256,
        trust_remote_code: bool = False,
        engine: str = "transformers",
        batch_size: int = 1,
        vllm_max_model_len: int = 0,
        vllm_gpu_memory_utilization: float = 0.9,
    ) -> None:
        self.model = model_name
        self.max_new_tokens = max_new_tokens
        self.batch_size = max(1, batch_size)
        self.generator = build_local_generator(
            engine=engine,
            model_name=model_name,
            max_input_length=max_input_length,
            trust_remote_code=trust_remote_code,
            max_model_len=resolve_local_max_model_len(
                max_input_length=max_input_length,
                max_new_tokens=max_new_tokens,
                explicit_max_model_len=vllm_max_model_len,
            ),
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        )

    def _call_text(self, prompt: str) -> str:
        outputs = self.generator.generate(
            [prompt],
            max_new_tokens=self.max_new_tokens,
            batch_size=self.batch_size,
            do_sample=False,
        )
        return outputs[0]

    def judge_gradingbench_batch(self, rows: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
        prompts = [
            gradingbench_prompt(problem=row["Problem"], proposed_solution=row["Response"])
            for row in rows
        ]
        outputs = self.generator.generate(
            prompts,
            max_new_tokens=self.max_new_tokens,
            batch_size=self.batch_size,
            do_sample=False,
        )
        return [self._parse_gradingbench_response(output) for output in outputs]


def summarize_answerbench(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    run_groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        run_groups[safe_int(row.get("run_idx", 0))].append(row)

    def mean_std(values: Sequence[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        mean = sum(values) / len(values)
        var = sum((x - mean) ** 2 for x in values) / len(values)
        return mean, math.sqrt(var)

    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    per_run_accuracies = []
    for run_rows in run_groups.values():
        per_run_accuracies.append(sum(1 for row in run_rows if row["correct"]) / len(run_rows) if run_rows else 0.0)
    accuracy_mean, accuracy_std = mean_std(per_run_accuracies)

    by_category: Dict[str, Dict[str, float]] = {}
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("category", "unknown"))].append(row)
    for category, category_rows in grouped.items():
        category_run_groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for row in category_rows:
            category_run_groups[safe_int(row.get("run_idx", 0))].append(row)
        category_run_accuracies = [
            sum(1 for row in run_rows if row["correct"]) / len(run_rows) if run_rows else 0.0
            for run_rows in category_run_groups.values()
        ]
        category_mean, category_std = mean_std(category_run_accuracies)
        by_category[category] = {
            "count": len({str(row["problem_id"]) for row in category_rows}),
            "runs": len(category_run_groups),
            "accuracy_mean": category_mean,
            "accuracy_std": category_std,
            "pass@1": category_mean,
        }
    return {
        "count": len({str(row["problem_id"]) for row in rows}),
        "num_evals": total,
        "runs": len(run_groups),
        "correct": correct,
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,
        "pass@1": accuracy_mean,
        "by_category": by_category,
    }


def summarize_proofbench(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def build(split_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(split_rows)
        total_points = sum(int(row["score_0_7"]) for row in split_rows)
        return {
            "count": total,
            "mean_score_0_7": total_points / total if total else 0.0,
            "normalized_percent": (100.0 * total_points / (7 * total)) if total else 0.0,
        }

    basic = [row for row in rows if row["split"] == "basic"]
    advanced = [row for row in rows if row["split"] == "advanced"]
    return {
        "overall": build(rows),
        "basic": build(basic),
        "advanced": build(advanced),
    }


def summarize_gradingbench(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    gold = [str(row["gold_label_4way"]) for row in rows]
    pred = [str(row["pred_label_4way"]) for row in rows]
    accuracy = sum(1 for g, p in zip(gold, pred) if g == p) / len(rows) if rows else 0.0

    points_gold = [int(row["gold_points"]) for row in rows]
    points_pred = [int(row["pred_points"]) for row in rows]
    mae_raw = sum(abs(g - p) for g, p in zip(points_gold, points_pred)) / len(rows) if rows else 0.0
    mae_percent = 100.0 * mae_raw / 7.0 if rows else 0.0
    points_accuracy = sum(1 for g, p in zip(points_gold, points_pred) if g == p) / len(rows) if rows else 0.0
    off_by_one_accuracy = sum(1 for g, p in zip(points_gold, points_pred) if abs(g - p) <= 1) / len(rows) if rows else 0.0
    catastrophic_overgrade_rate = (
        sum(1 for g, p in zip(points_gold, points_pred) if g <= 1 and p >= 6) / len(rows) if rows else 0.0
    )

    confusion: Dict[str, Dict[str, int]] = {}
    for gold_label in FOUR_WAY_LABELS:
        confusion[gold_label] = {pred_label: 0 for pred_label in FOUR_WAY_LABELS}
    for g, p in zip(gold, pred):
        confusion[g][p] += 1

    return {
        "count": len(rows),
        "accuracy": accuracy,
        "mae_raw": mae_raw,
        "mae_percent": mae_percent,
        "macro_f1": macro_f1(gold, pred, FOUR_WAY_LABELS),
        "points_accuracy": points_accuracy,
        "off_by_one_accuracy": off_by_one_accuracy,
        "catastrophic_overgrade_rate": catastrophic_overgrade_rate,
        "confusion": confusion,
    }


def compare_models(
    model_summaries: Dict[str, Dict[str, Any]],
    grading_summary: Dict[str, Any],
    grading_by_judge: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    rows = []
    for model_name, summary in model_summaries.items():
        answer = summary.get("answerbench", {})
        proof = summary.get("proofbench", {})
        rows.append(
            {
                "model": model_name,
                "answerbench_accuracy_mean": answer.get("accuracy_mean"),
                "answerbench_accuracy_std": answer.get("accuracy_std"),
                "answerbench_runs": answer.get("runs"),
                "answerbench_pass@1": answer.get("pass@1"),
                "proofbench_mean_score_0_7": proof.get("overall", {}).get("mean_score_0_7"),
                "proofbench_basic_percent": proof.get("basic", {}).get("normalized_percent"),
                "proofbench_advanced_percent": proof.get("advanced", {}).get("normalized_percent"),
            }
        )
    judge_rows = []
    for judge_name, summary in (grading_by_judge or {}).items():
        judge_rows.append(
            {
                "judge_model": judge_name,
                "gradingbench_accuracy": summary.get("accuracy"),
                "gradingbench_mae_percent": summary.get("mae_percent"),
                "gradingbench_macro_f1": summary.get("macro_f1"),
                "gradingbench_points_accuracy": summary.get("points_accuracy"),
                "gradingbench_off_by_one_accuracy": summary.get("off_by_one_accuracy"),
                "gradingbench_catastrophic_overgrade_rate": summary.get("catastrophic_overgrade_rate"),
                "count": summary.get("count"),
            }
        )
    return {
        "models": model_summaries,
        "gradingbench": grading_summary,
        "gradingbench_by_judge": grading_by_judge or {},
        "comparison_rows": rows,
        "judge_comparison_rows": judge_rows,
    }


def save_summary_json_csv(path_prefix: Path, summary: Dict[str, Any]) -> None:
    path_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    flat_rows = summary.get("comparison_rows")
    if not flat_rows:
        return
    csv_path = path_prefix.with_suffix(".csv")
    fieldnames = list(flat_rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def run_answerbench(
    model_name: str,
    generator: TextGenerator,
    judge: Any,
    rows: Sequence[Dict[str, str]],
    output_path: Path,
    max_new_tokens: int,
    batch_size: int,
    answer_runs: int,
    answer_temperature: float,
    answer_top_p: float,
) -> List[Dict[str, Any]]:
    done = existing_key_tuples(output_path, ("problem_id", "run_idx"))
    do_sample = answer_runs > 1
    for run_idx in range(answer_runs):
        pending = [row for row in rows if (str(row["Problem ID"]), str(run_idx)) not in done]
        if not pending:
            continue
        prompts = [ensure_boxed_answer_prompt(row["Problem"]) for row in pending]
        outputs = generator.generate(
            prompts,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            do_sample=do_sample,
            temperature=answer_temperature,
            top_p=answer_top_p,
        )
        result_rows = []
        for row, output in zip(pending, outputs):
            grade = judge.judge_answer(
                problem=row["Problem"],
                model_solution=output,
                golden_answer=row["Short Answer"],
            )
            result_rows.append(
                {
                    "benchmark": "answerbench",
                    "model": model_name,
                    "run_idx": run_idx,
                    "problem_id": row["Problem ID"],
                    "problem": row["Problem"],
                    "gold_answer": row["Short Answer"],
                    "category": row["Category"],
                    "subcategory": row["Subcategory"],
                    "source": row["Source"],
                    "output": output,
                    "correct": bool(grade["correct"]),
                    "judge_grade": grade["judge_grade"],
                    "judge_response": grade["judge_response"],
                    "solver_do_sample": do_sample,
                    "solver_temperature": answer_temperature,
                    "solver_top_p": answer_top_p,
                }
            )
        append_jsonl(output_path, result_rows)
    return load_jsonl(output_path)


def run_proofbench(
    model_name: str,
    generator: TextGenerator | None,
    judge: GeminiJudge,
    rows: Sequence[Dict[str, str]],
    prediction_path: Path,
    judged_path: Path,
    max_new_tokens: int,
    batch_size: int,
    judge_mode: str = "realtime",
    judge_concurrency: int = 1,
    batch_poll_seconds: int = 30,
) -> List[Dict[str, Any]]:
    pred_done = existing_ids(prediction_path, "problem_id")
    pending_predictions = [row for row in rows if row["Problem ID"] not in pred_done]
    if pending_predictions:
        if generator is None:
            raise RuntimeError("ProofBench generation requested without an initialized generator.")
        prompts = [proof_prompt(row["Problem"]) for row in pending_predictions]
        outputs = generator.generate(prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
        result_rows = []
        for row, output in zip(pending_predictions, outputs):
            result_rows.append(
                {
                    "benchmark": "proofbench",
                    "model": model_name,
                    "problem_id": row["Problem ID"],
                    "split": proofbench_split(row["Problem ID"]),
                    "problem": row["Problem"],
                    "reference_solution": row["Solution"],
                    "grading_guidelines": row["Grading guidelines"],
                    "category": row["Category"],
                    "level": row["Level"],
                    "short_answer": row["Short Answer"],
                    "source": row["Source"],
                    "output": output,
                }
            )
        append_jsonl(prediction_path, result_rows)

    predictions = load_jsonl(prediction_path)
    judged_done = existing_ids(judged_path, "problem_id")
    pending_judged = [row for row in predictions if row["problem_id"] not in judged_done]
    if pending_judged:
        error_path = judged_path.with_name("proofbench_judge_errors.jsonl")
        batch_job_path = judged_path.with_name("proofbench_batch_job.json")

        def render_batch_error(error_payload: Any) -> str:
            if isinstance(error_payload, str):
                return error_payload
            try:
                return json.dumps(error_payload, ensure_ascii=False, sort_keys=True)
            except TypeError:
                return str(error_payload)

        def inline_response_key(item: Dict[str, Any]) -> str | None:
            metadata = item.get("metadata")
            if isinstance(metadata, dict) and metadata.get("key") is not None:
                return str(metadata["key"])
            if item.get("key") is not None:
                return str(item["key"])
            return None

        def batch_display_name(rows_for_batch: Sequence[Dict[str, Any]]) -> str:
            digest_source = "\n".join(
                [judge.model, model_name, *(str(row["problem_id"]) for row in rows_for_batch)]
            ).encode("utf-8")
            digest = hashlib.sha1(digest_source).hexdigest()[:12]
            return f"proofbench-{slugify(model_name)[:32]}-{digest}"

        if judge_mode == "batch":
            pending_by_problem_id = {str(row["problem_id"]): row for row in pending_judged}
            current_pending_ids = list(pending_by_problem_id.keys())
            manifest = load_json(batch_job_path)
            batch_job: Dict[str, Any] | None = None
            submitted_ids: List[str] = []

            if manifest is not None:
                manifest_backend = str(manifest.get("backend", "")).strip()
                manifest_judge_model = str(manifest.get("judge_model", "")).strip()
                manifest_solver_model = str(manifest.get("solver_model", "")).strip()
                if (
                    manifest_backend != "gemini_batch"
                    or manifest_judge_model != judge.model
                    or manifest_solver_model != model_name
                ):
                    raise RuntimeError(
                        "Existing ProofBench batch manifest targets a different backend or model combination. "
                        f"Use a fresh output root or remove {batch_job_path} before switching judge models."
                    )
                submitted_ids = [str(problem_id) for problem_id in manifest.get("submitted_problem_ids", [])]
                batch_name = str(manifest.get("batch_name", "")).strip()
                submitted_id_set = set(submitted_ids)
                current_id_set = set(current_pending_ids)

                if batch_name and current_id_set.issubset(submitted_id_set):
                    batch_job = judge.get_batch_job(batch_name)
                    manifest["last_state"] = judge.batch_state(batch_job)
                    manifest["last_checked_at"] = datetime.now().isoformat()
                    write_json(batch_job_path, manifest)
                elif not batch_name:
                    manifest = None
                else:
                    previous_state = str(manifest.get("last_state", ""))
                    if previous_state and previous_state not in BATCH_TERMINAL_STATES:
                        raise RuntimeError(
                            "Existing active ProofBench batch job does not match the current pending proof set. "
                            f"Inspect {batch_job_path} before submitting another batch."
                        )
                    manifest = None

            if manifest is None or batch_job is None:
                request_plan = [{"key": str(row["problem_id"]), "problem_id": str(row["problem_id"])} for row in pending_judged]
                created_batch = judge.create_batch_job(
                    requests=[
                        {
                            "request": judge.build_proof_request(
                                problem=row["problem"],
                                candidate_solution=row["output"],
                                reference_solution=row["reference_solution"],
                                grading_guidelines=row["grading_guidelines"],
                            ),
                            "metadata": {"key": str(row["problem_id"])},
                        }
                        for row in pending_judged
                    ],
                    display_name=batch_display_name(pending_judged),
                )
                batch_name = str(created_batch.get("name", "")).strip()
                if not batch_name:
                    raise RuntimeError("Gemini Batch API did not return a batch job name.")
                manifest = {
                    "backend": "gemini_batch",
                    "benchmark": "proofbench",
                    "judge_model": judge.model,
                    "solver_model": model_name,
                    "batch_name": batch_name,
                    "display_name": batch_display_name(pending_judged),
                    "submitted_problem_ids": current_pending_ids,
                    "requests": request_plan,
                    "created_at": datetime.now().isoformat(),
                    "last_checked_at": datetime.now().isoformat(),
                    "last_state": judge.batch_state(created_batch),
                }
                write_json(batch_job_path, manifest)
                batch_job = created_batch
            else:
                request_plan = [
                    {
                        "key": str(item.get("key", item.get("problem_id", ""))),
                        "problem_id": str(item.get("problem_id", item.get("key", ""))),
                    }
                    for item in manifest.get("requests", [])
                    if item.get("problem_id") or item.get("key")
                ]

            batch_name = str(manifest["batch_name"])
            while judge.batch_state(batch_job) not in BATCH_TERMINAL_STATES:
                current_state = judge.batch_state(batch_job)
                print(f"[ProofBench] Waiting on Gemini batch {batch_name}: {current_state}", file=sys.stderr)
                time.sleep(max(1, batch_poll_seconds))
                batch_job = judge.get_batch_job(batch_name)
                manifest["last_state"] = judge.batch_state(batch_job)
                manifest["last_checked_at"] = datetime.now().isoformat()
                write_json(batch_job_path, manifest)

            final_state = judge.batch_state(batch_job)
            manifest["last_state"] = final_state
            manifest["last_checked_at"] = datetime.now().isoformat()

            if final_state not in BATCH_SUCCESS_STATES:
                manifest["error"] = batch_job.get("error")
                write_json(batch_job_path, manifest)
                raise RuntimeError(
                    f"Gemini ProofBench batch {batch_name} finished with state {final_state}: "
                    f"{render_batch_error(batch_job.get('error'))}"
                )

            inline_results = judge.batch_inline_responses(batch_job)
            if not inline_results:
                responses_file = judge.batch_responses_file(batch_job)
                if responses_file:
                    file_content = judge.download_batch_result_file(responses_file).decode("utf-8")
                    inline_results = [json.loads(line) for line in file_content.splitlines() if line.strip()]
                else:
                    raise RuntimeError(
                        f"Gemini ProofBench batch {batch_name} succeeded but returned no inline responses or result file."
                    )

            request_order = [str(item["key"]) for item in request_plan if item.get("key")]
            result_by_key: Dict[str, Dict[str, Any]] = {}
            for idx, result in enumerate(inline_results):
                key = inline_response_key(result)
                if key is None and idx < len(request_order):
                    key = request_order[idx]
                if key is not None:
                    result_by_key[key] = result

            judged_rows: List[Dict[str, Any]] = []
            error_rows: List[Dict[str, Any]] = []
            for request_item in request_plan:
                key = str(request_item["key"])
                problem_id = str(request_item["problem_id"])
                row = pending_by_problem_id.get(problem_id)
                if row is None:
                    continue

                result = result_by_key.get(key)
                if result is None:
                    error_rows.append(
                        {
                            **row,
                            "judge_error": "Gemini batch completed without returning a result for this proof.",
                            "error_type": "MissingBatchResult",
                            "batch_request_key": key,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    continue

                if result.get("error") is not None:
                    error_rows.append(
                        {
                            **row,
                            "judge_error": render_batch_error(result.get("error")),
                            "error_type": "BatchRequestError",
                            "batch_request_key": key,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    continue

                response_payload = result.get("response")
                if not isinstance(response_payload, dict) and isinstance(result.get("candidates"), list):
                    response_payload = result
                if not isinstance(response_payload, dict):
                    error_rows.append(
                        {
                            **row,
                            "judge_error": "Gemini batch result did not include a GenerateContent response payload.",
                            "error_type": "MissingBatchResponse",
                            "batch_request_key": key,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    continue

                try:
                    response_text = judge._extract_text_from_generate_response(response_payload)
                    if not response_text:
                        raise RuntimeError("Gemini batch result contained no text response.")
                    grade = judge._proof_grade_from_response_text(response_text)
                except Exception as err:  # noqa: BLE001
                    error_rows.append(
                        {
                            **row,
                            "judge_error": str(err),
                            "error_type": type(err).__name__,
                            "batch_request_key": key,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    continue

                judged_rows.append({**row, **grade})

            if judged_rows:
                append_jsonl(judged_path, judged_rows)
            if error_rows:
                append_jsonl(error_path, error_rows)

            manifest["result_applied_at"] = datetime.now().isoformat()
            manifest["judged_row_count"] = len(judged_rows)
            manifest["error_row_count"] = len(error_rows)
            write_json(batch_job_path, manifest)
            return load_jsonl(judged_path)

        max_workers = max(1, judge_concurrency)

        def judge_row(row: Dict[str, Any]) -> Dict[str, Any]:
            return judge.judge_proof(
                problem_id=row["problem_id"],
                problem=row["problem"],
                candidate_solution=row["output"],
                reference_solution=row["reference_solution"],
                grading_guidelines=row["grading_guidelines"],
            )

        if max_workers == 1:
            for row in pending_judged:
                try:
                    grade = judge_row(row)
                except Exception as err:  # noqa: BLE001
                    append_jsonl(
                        error_path,
                        [
                            {
                                **row,
                                "judge_error": str(err),
                                "error_type": type(err).__name__,
                                "timestamp": datetime.now().isoformat(),
                            }
                        ],
                    )
                    print(
                        f"[ProofBench] Judge failed for {row['problem_id']}: {err}",
                        file=sys.stderr,
                    )
                    continue
                append_jsonl(judged_path, [{**row, **grade}])
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_row = {executor.submit(judge_row, row): row for row in pending_judged}
                for future in as_completed(future_to_row):
                    row = future_to_row[future]
                    try:
                        grade = future.result()
                    except Exception as err:  # noqa: BLE001
                        append_jsonl(
                            error_path,
                            [
                                {
                                    **row,
                                    "judge_error": str(err),
                                    "error_type": type(err).__name__,
                                    "timestamp": datetime.now().isoformat(),
                                }
                            ],
                        )
                        print(
                            f"[ProofBench] Judge failed for {row['problem_id']}: {err}",
                            file=sys.stderr,
                        )
                        continue
                    append_jsonl(judged_path, [{**row, **grade}])
    return load_jsonl(judged_path)


def run_gradingbench(
    judge: GeminiJudge,
    rows: Sequence[Dict[str, str]],
    output_path: Path,
) -> List[Dict[str, Any]]:
    done = existing_ids(output_path, "grading_id")
    pending = [row for row in rows if row["Grading ID"] not in done]
    if pending:
        batch_judge = getattr(judge, "judge_gradingbench_batch", None)
        if callable(batch_judge):
            batch_size = max(1, safe_int(getattr(judge, "batch_size", 1), default=1))
            for row_batch in split_batches(pending, batch_size):
                judged_rows = []
                grades = batch_judge(row_batch)
                for row, grade in zip(row_batch, grades):
                    judged_rows.append(
                        {
                            "benchmark": "gradingbench",
                            "grading_id": row["Grading ID"],
                            "problem_id": row["Problem ID"],
                            "problem_source": row["Problem Source"],
                            "gold_points": safe_int(row["Points"]),
                            "gold_label_4way": row["Reward"],
                            "pred_points": grade["score_0_7"],
                            "pred_label_4way": grade["label_4way"],
                            "judge_response": grade["judge_response"],
                        }
                    )
                append_jsonl(output_path, judged_rows)
        else:
            judged_rows = []
            for row in pending:
                grade = judge.judge_gradingbench(
                    grading_id=row["Grading ID"],
                    problem_id=row["Problem ID"],
                    problem=row["Problem"],
                    proposed_solution=row["Response"],
                )
                judged_rows.append(
                    {
                        "benchmark": "gradingbench",
                        "grading_id": row["Grading ID"],
                        "problem_id": row["Problem ID"],
                        "problem_source": row["Problem Source"],
                        "gold_points": safe_int(row["Points"]),
                        "gold_label_4way": row["Reward"],
                        "pred_points": grade["score_0_7"],
                        "pred_label_4way": grade["label_4way"],
                        "judge_response": grade["judge_response"],
                    }
                )
            append_jsonl(output_path, judged_rows)
    return load_jsonl(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local checkpoints / public models on IMO-Bench.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="One or more Hugging Face causal LM model ids to compare.",
    )
    parser.add_argument(
        "--benchmarks",
        default="answerbench,proofbench,gradingbench",
        help="Comma-separated subset of answerbench,proofbench,gradingbench",
    )
    parser.add_argument(
        "--answer-grader-backend",
        choices=("gemini", "math_verify"),
        default="gemini",
        help="AnswerBench grader backend. Use gemini for paper-faithful AnswerAutoGrader, "
        "math_verify for a cheaper local final-answer verifier.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path.cwd() / "imobench_data"),
        help="Directory to cache downloaded IMO-Bench CSVs.",
    )
    parser.add_argument(
        "--output-root",
        default=str(Path.cwd() / "imobench_runs" / now_tag()),
        help="Directory to store model artifacts and summaries.",
    )
    parser.add_argument(
        "--local-engine",
        choices=("transformers", "vllm"),
        default="transformers",
        help="Runtime for local open-weight models used for solver generation and local_hf judges.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Local generation batch size.")
    parser.add_argument("--judge-batch-size", type=int, default=1, help="Local GradingBench judge batch size.")
    parser.add_argument("--answer-max-new-tokens", type=int, default=2048)
    parser.add_argument("--proof-max-new-tokens", type=int, default=8192)
    parser.add_argument(
        "--answer-runs",
        type=int,
        default=8,
        help="Number of solver runs for AnswerBench. Paper reports most public-model results averaged over 8 runs.",
    )
    parser.add_argument(
        "--answer-temperature",
        type=float,
        default=1.0,
        help="Solver temperature for AnswerBench when answer-runs > 1. Paper does not disclose a public universal decode setting; this is an explicit house choice.",
    )
    parser.add_argument(
        "--answer-top-p",
        type=float,
        default=1.0,
        help="Solver top-p for AnswerBench when answer-runs > 1.",
    )
    parser.add_argument("--limit-answerbench", type=int, default=0, help="Optional cap for smoke tests.")
    parser.add_argument("--limit-proofbench", type=int, default=0, help="Optional cap for smoke tests.")
    parser.add_argument("--limit-gradingbench", type=int, default=0, help="Optional cap for smoke tests.")
    parser.add_argument(
        "--gradingbench-per-point",
        type=int,
        default=0,
        help="Optional stratified GradingBench sample size per human score bucket (0-7). Useful for calibration slices.",
    )
    parser.add_argument("--gradingbench-seed", type=int, default=0, help="Seed for stratified GradingBench sampling.")
    parser.add_argument("--gemini-model", default="gemini-2.5-pro")
    parser.add_argument("--gemini-api-env", default="GEMINI_API_KEY")
    parser.add_argument("--gemini-timeout", type=int, default=600, help="Per-request Gemini read timeout in seconds.")
    parser.add_argument("--gemini-retries", type=int, default=5, help="Number of retries for Gemini API requests.")
    parser.add_argument(
        "--proof-judge-mode",
        choices=("realtime", "batch"),
        default="realtime",
        help="ProofBench Gemini judging mode. Use batch to submit one Gemini Batch API job per model and poll for completion.",
    )
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=1,
        help="Concurrent realtime ProofBench judge requests to run in parallel. Ignored when --proof-judge-mode=batch.",
    )
    parser.add_argument(
        "--batch-poll-seconds",
        type=int,
        default=30,
        help="Polling interval in seconds while waiting for a Gemini Batch API ProofBench job to finish.",
    )
    parser.add_argument(
        "--judge-backend",
        choices=("gemini", "local_hf"),
        default="gemini",
        help="Judge backend. Use gemini for paper-faithful runs, local_hf for cheaper local judge comparisons.",
    )
    parser.add_argument(
        "--judge-models-csv",
        default=",".join(DEFAULT_LOCAL_JUDGE_MODELS),
        help="Comma-separated local HF judge models to compare when --judge-backend=local_hf.",
    )
    parser.add_argument("--judge-max-input-length", type=int, default=8192, help="Max prompt length for local HF judges.")
    parser.add_argument("--judge-max-new-tokens", type=int, default=256, help="Max new tokens for local HF judges.")
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=0,
        help="Optional vLLM max_model_len override. Defaults to max_input_length + max_new_tokens for the active task.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Target fraction of visible GPU memory for vLLM's KV cache and weights.",
    )
    parser.add_argument("--judge-temperature", type=float, default=0.0, help="Gemini grader temperature.")
    parser.add_argument("--judge-top-p", type=float, default=1.0, help="Gemini grader top-p.")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmarks = {name.strip() for name in args.benchmarks.split(",") if name.strip()}
    output_root = ensure_dir(Path(args.output_root))
    data_dir = ensure_dir(Path(args.data_dir))

    answerbench_path = download_if_missing(ANSWERBENCH_URL, data_dir / "answerbench_v2.csv")
    proofbench_path = download_if_missing(PROOFBENCH_URL, data_dir / "proofbench.csv")
    gradingbench_path = download_if_missing(GRADINGBENCH_URL, data_dir / "gradingbench.csv")

    answerbench_rows = read_csv_rows(answerbench_path)
    proofbench_rows = read_csv_rows(proofbench_path)
    gradingbench_rows = read_csv_rows(gradingbench_path)

    if args.limit_answerbench > 0:
        answerbench_rows = answerbench_rows[: args.limit_answerbench]
    if args.limit_proofbench > 0:
        proofbench_rows = proofbench_rows[: args.limit_proofbench]
    if args.limit_gradingbench > 0:
        gradingbench_rows = gradingbench_rows[: args.limit_gradingbench]
    if args.gradingbench_per_point > 0:
        gradingbench_rows = stratified_sample_by_points(
            gradingbench_rows,
            per_point=args.gradingbench_per_point,
            seed=args.gradingbench_seed,
        )

    gemini_key = os.environ.get(args.gemini_api_env) or os.environ.get("GOOGLE_API_KEY")
    if args.judge_backend == "local_hf" and benchmarks - {"gradingbench"}:
        raise RuntimeError("local_hf judge backend currently supports GradingBench-only runs.")

    model_summaries: Dict[str, Dict[str, Any]] = {}
    answer_judge: Any = None
    proof_judge: GeminiJudge | None = None
    for model_name in args.models:
        model_dir = ensure_dir(output_root / slugify(model_name))
        generator = None
        summary: Dict[str, Any] = {"model": model_name}
        solver_max_new_tokens = 0
        if "answerbench" in benchmarks:
            solver_max_new_tokens = max(solver_max_new_tokens, args.answer_max_new_tokens)
        if "proofbench" in benchmarks:
            solver_max_new_tokens = max(solver_max_new_tokens, args.proof_max_new_tokens)

        if "answerbench" in benchmarks:
            if answer_judge is None:
                if args.answer_grader_backend == "gemini":
                    if not gemini_key:
                        raise RuntimeError(
                            f"Missing Gemini API key. Set {args.gemini_api_env} (or GOOGLE_API_KEY) in the environment."
                        )
                    answer_judge = GeminiJudge(
                        api_key=gemini_key,
                        model=args.gemini_model,
                        timeout=args.gemini_timeout,
                        retries=args.gemini_retries,
                        temperature=args.judge_temperature,
                        top_p=args.judge_top_p,
                    )
                else:
                    answer_judge = MathVerifyAnswerJudge()
            if generator is None:
                generator = build_local_generator(
                    engine=args.local_engine,
                    model_name=model_name,
                    max_input_length=8192,
                    trust_remote_code=args.trust_remote_code,
                    max_model_len=resolve_local_max_model_len(
                        max_input_length=8192,
                        max_new_tokens=solver_max_new_tokens,
                        explicit_max_model_len=args.vllm_max_model_len,
                    ),
                    vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                )
            answer_rows = run_answerbench(
                model_name=model_name,
                generator=generator,
                judge=answer_judge,
                rows=answerbench_rows,
                output_path=model_dir / "answerbench_predictions.jsonl",
                max_new_tokens=args.answer_max_new_tokens,
                batch_size=args.batch_size,
                answer_runs=args.answer_runs,
                answer_temperature=args.answer_temperature,
                answer_top_p=args.answer_top_p,
            )
            summary["answerbench"] = summarize_answerbench(answer_rows)

        if "proofbench" in benchmarks:
            if proof_judge is None:
                if not gemini_key:
                    raise RuntimeError(
                        f"Missing Gemini API key. Set {args.gemini_api_env} (or GOOGLE_API_KEY) in the environment."
                    )
                proof_judge = GeminiJudge(
                    api_key=gemini_key,
                    model=args.gemini_model,
                    timeout=args.gemini_timeout,
                    retries=args.gemini_retries,
                    temperature=args.judge_temperature,
                    top_p=args.judge_top_p,
                )
            proof_prediction_path = model_dir / "proofbench_predictions.jsonl"
            needs_proof_generation = len(existing_ids(proof_prediction_path, "problem_id")) < len(proofbench_rows)
            if generator is None and needs_proof_generation:
                generator = build_local_generator(
                    engine=args.local_engine,
                    model_name=model_name,
                    max_input_length=8192,
                    trust_remote_code=args.trust_remote_code,
                    max_model_len=resolve_local_max_model_len(
                        max_input_length=8192,
                        max_new_tokens=solver_max_new_tokens,
                        explicit_max_model_len=args.vllm_max_model_len,
                    ),
                    vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                )
            proof_rows = run_proofbench(
                model_name=model_name,
                generator=generator,
                judge=proof_judge,
                rows=proofbench_rows,
                prediction_path=proof_prediction_path,
                judged_path=model_dir / "proofbench_judged.jsonl",
                max_new_tokens=args.proof_max_new_tokens,
                batch_size=args.batch_size,
                judge_mode=args.proof_judge_mode,
                judge_concurrency=args.judge_concurrency,
                batch_poll_seconds=args.batch_poll_seconds,
            )
            summary["proofbench"] = summarize_proofbench(proof_rows)

        summary_path = model_dir / "summary"
        summary_path.with_suffix(".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        csv_summary_rows = [
            {
                "model": model_name,
                "answerbench_accuracy_mean": summary.get("answerbench", {}).get("accuracy_mean"),
                "answerbench_accuracy_std": summary.get("answerbench", {}).get("accuracy_std"),
                "answerbench_pass@1": summary.get("answerbench", {}).get("pass@1"),
                "proofbench_mean_score_0_7": summary.get("proofbench", {}).get("overall", {}).get("mean_score_0_7"),
                "proofbench_basic_percent": summary.get("proofbench", {}).get("basic", {}).get("normalized_percent"),
                "proofbench_advanced_percent": summary.get("proofbench", {}).get("advanced", {}).get("normalized_percent"),
            }
        ]
        with summary_path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_summary_rows)

        model_summaries[model_name] = summary
        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    grading_summary: Dict[str, Any] = {}
    grading_by_judge: Dict[str, Dict[str, Any]] = {}
    if "gradingbench" in benchmarks:
        judge_specs: List[tuple[str, GeminiJudge | LocalHFJudge]] = []
        if args.judge_backend == "gemini":
            if not gemini_key:
                raise RuntimeError(
                    f"Missing Gemini API key. Set {args.gemini_api_env} (or GOOGLE_API_KEY) in the environment."
                )
            judge_specs.append(
                (
                    args.gemini_model,
                    GeminiJudge(
                        api_key=gemini_key,
                        model=args.gemini_model,
                        temperature=args.judge_temperature,
                        top_p=args.judge_top_p,
                    ),
                )
            )
        else:
            for judge_model in parse_csv_list(args.judge_models_csv):
                judge_specs.append(
                    (
                        judge_model,
                        LocalHFJudge(
                            model_name=judge_model,
                            max_input_length=args.judge_max_input_length,
                            max_new_tokens=args.judge_max_new_tokens,
                            trust_remote_code=args.trust_remote_code,
                            engine=args.local_engine,
                            batch_size=args.judge_batch_size,
                            vllm_max_model_len=args.vllm_max_model_len,
                            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                        ),
                    )
                )

        for judge_name, judge in judge_specs:
            judge_dir = ensure_dir(output_root / "judge_models" / slugify(judge_name))
            grading_rows = run_gradingbench(
                judge=judge,
                rows=gradingbench_rows,
                output_path=judge_dir / "gradingbench_judged.jsonl",
            )
            summary = summarize_gradingbench(grading_rows)
            grading_by_judge[judge_name] = summary
            if not grading_summary:
                grading_summary = summary
            (judge_dir / "gradingbench_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            summary_csv_rows = [
                {
                    "benchmark": "gradingbench",
                    "judge_model": judge_name,
                    "accuracy": summary.get("accuracy"),
                    "mae_percent": summary.get("mae_percent"),
                    "macro_f1": summary.get("macro_f1"),
                    "points_accuracy": summary.get("points_accuracy"),
                    "off_by_one_accuracy": summary.get("off_by_one_accuracy"),
                    "catastrophic_overgrade_rate": summary.get("catastrophic_overgrade_rate"),
                    "count": summary.get("count"),
                }
            ]
            with (judge_dir / "gradingbench_summary.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_csv_rows)

            del judge
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    comparison = compare_models(model_summaries, grading_summary, grading_by_judge)
    save_summary_json_csv(output_root / "comparison_summary", comparison)

    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
