"""Text-output VoiceBench runner and paired-result summarizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
BASE_REVISION = "f75b40e3da2003cdd6e1829b1f420ca70797c34e"
VOICEBENCH_REVISION = "6992cf4fc51d0426c52c4805b5002e0aae49118a"
DATASET_ID = "hlt-lab/voicebench"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
MANIFEST_FIELD = "run_manifest"
ID_FIELD = "voicebench_id"


@dataclass(frozen=True)
class RunSettings:
    model_mode: str
    data: str
    split: str
    modality: str
    seed: int = 17
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_new_tokens: int = 256
    max_silent_chunks: int = 4
    dtype: str = "bfloat16"
    adapter: Path | None = None
    allow_download: bool = False

    def validate(self) -> None:
        if self.model_mode not in {"base", "duplex"}:
            raise ValueError("model_mode must be base or duplex.")
        if self.modality not in {"audio", "text"}:
            raise ValueError("modality must be audio or text.")
        if self.max_new_tokens <= 0 or self.max_silent_chunks < 0:
            raise ValueError("Generation limits are invalid.")
        if self.dtype != "bfloat16":
            raise ValueError("This runner supports only bfloat16.")
        if self.model_mode == "duplex" and self.adapter is None:
            raise ValueError("Duplex mode requires --checkpoint or --adapter.")
        if self.model_mode == "base" and self.adapter is not None:
            raise ValueError("Base mode must use the untouched checkpoint.")


class Backend(Protocol):
    def generate_audio(self, audio: object, *, example_id: str) -> str: ...
    def generate_text(self, prompt: str, *, example_id: str) -> str: ...


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_example_id(item: Mapping[str, Any], *, data: str, split: str, index: int) -> str:
    source = {key: value for key, value in item.items() if key != "audio"}
    value = {"dataset": DATASET_ID, "data": data, "split": split, "index": index, "row": source}
    digest = hashlib.sha256(_canonical(value).encode()).hexdigest()[:24]
    return f"{data}:{split}:{index}:{digest}"


def _adapter_identity(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    weights = resolved / "adapter_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Adapter directory has no adapter_model.safetensors: {resolved}")
    digest = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors", "duplex_config.yaml"):
        candidate = resolved / name
        if candidate.is_file():
            digest.update(name.encode())
            with candidate.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    return {"path": str(resolved), "sha256": digest.hexdigest()}


def _hardware() -> dict[str, Any]:
    result: dict[str, Any] = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        import torch
        result.update(torch=torch.__version__, cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            result.update(cuda_device=torch.cuda.get_device_name(0), cuda_capability=list(torch.cuda.get_device_capability(0)))
    except ImportError:
        result.update(torch=None, cuda_available=False)
    return result


def build_manifest(settings: RunSettings) -> dict[str, Any]:
    settings.validate()
    timing = ("ordinary_text_generation" if settings.modality == "text" else
              "official_half_duplex" if settings.model_mode == "base" else
              "fixed_2s_streaming_silent_tail_until_stop")
    return {
        "schema_version": 1,
        "dataset": {"id": DATASET_ID, "subset": settings.data, "split": settings.split},
        "modality": settings.modality,
        "model": {"mode": settings.model_mode, "base_id": MODEL_ID,
                  "base_revision": BASE_REVISION, "adapter": _adapter_identity(settings.adapter),
                  "dtype": settings.dtype},
        "seed": settings.seed,
        "system_prompt": settings.system_prompt,
        "decoding": {"max_new_tokens": settings.max_new_tokens, "do_sample": False,
                     "max_silent_chunks": settings.max_silent_chunks},
        "upstream": {"repository": "https://github.com/MatthewCYM/VoiceBench",
                     "revision": VOICEBENCH_REVISION},
        "hardware": _hardware(),
        "timing_mode": timing,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(settings: RunSettings) -> object:
    from datasets import Audio, load_dataset as hf_load_dataset
    data = hf_load_dataset(DATASET_ID, settings.data, split=settings.split)
    return data.cast_column("audio", Audio(decode=False))


def _audio_values(audio: object) -> tuple[object, int]:
    if isinstance(audio, Mapping) and "array" in audio:
        return audio["array"], int(audio["sampling_rate"])
    if isinstance(audio, Mapping) and (audio.get("bytes") is not None or audio.get("path")):
        import io
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
        source = io.BytesIO(audio["bytes"]) if audio.get("bytes") is not None else audio["path"]
        values, rate = sf.read(source, dtype="float32", always_2d=False)
        if values.ndim == 2:
            values = values.mean(axis=1)
        if rate != 16_000:
            divisor = math.gcd(int(rate), 16_000)
            values = resample_poly(values, 16_000 // divisor, int(rate) // divisor).astype(np.float32)
            rate = 16_000
        return values, int(rate)
    getter = getattr(audio, "get_all_samples", None)
    if callable(getter):
        samples = getter()
        values, rate = samples.data, int(samples.sample_rate)
        if getattr(values, "ndim", 1) == 2 and values.shape[0] == 1:
            values = values[0]
        return values, rate
    raise TypeError("Unsupported decoded VoiceBench audio value.")


def _messages(system: str, content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": content}]


def _chat_ids(processor: object, system: str, prompt: str) -> object:
    return processor.apply_chat_template(
        _messages(system, [{"type": "text", "text": prompt}]),
        tokenize=True, add_generation_prompt=True, return_tensors="pt")


class BaseQwenBackend:
    """Official half-duplex Qwen path with Talker disabled."""

    def __init__(self, settings: RunSettings) -> None:
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        common = {"revision": BASE_REVISION, "local_files_only": not settings.allow_download}
        self.torch, self.settings = torch, settings
        self.processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID, **common)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_ID, **common, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
            attn_implementation="sdpa", low_cpu_mem_usage=True)
        self.model.disable_talker()
        self.model.eval()

    def _decode(self, inputs: object) -> str:
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        length = inputs.input_ids.shape[1]
        with self.torch.inference_mode():
            ids = self.model.thinker.generate(
                **inputs, max_new_tokens=self.settings.max_new_tokens,
                do_sample=False)[:, length:]
        return self.processor.batch_decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def generate_audio(self, audio: object, *, example_id: str) -> str:
        values, rate = _audio_values(audio)
        if rate != 16_000:
            raise ValueError(f"VoiceBench audio must be 16 kHz, got {rate}.")
        messages = _messages(self.settings.system_prompt, [{"type": "audio", "audio": values}])
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self._decode(self.processor(
            text=text, audio=[values], return_tensors="pt", padding=True,
            use_audio_in_video=False))

    def generate_text(self, prompt: str, *, example_id: str) -> str:
        messages = _messages(self.settings.system_prompt, [{"type": "text", "text": prompt}])
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self._decode(self.processor(text=text, return_tensors="pt", padding=True))


class DuplexQwenBackend:
    """Adapter-active ordinary text path and explicit two-second audio path."""

    def __init__(self, settings: RunSettings) -> None:
        from duplex.training import build_training_model, load_adapter_weights, load_training_config
        assert settings.adapter is not None
        config = load_training_config(settings.adapter / "training_config.yaml")
        config["model"]["local_files_only"] = not settings.allow_download
        self.model, self.processor, _ = build_training_model(config)
        load_adapter_weights(self.model, settings.adapter)
        self.model.eval()
        self.settings = settings

    def generate_audio(self, audio: object, *, example_id: str) -> str:
        from duplex.streaming import QwenDuplexStreamer

        class CappedStreamer(QwenDuplexStreamer):
            def __init__(self, *args, lexical_cap: int, **kwargs):
                super().__init__(*args, **kwargs)
                self.lexical_cap = lexical_cap
                self.lexical_count = 0

            def _select_event(self, logits, state):
                if self.lexical_count >= self.lexical_cap:
                    selected = self.model.control_tokens.stop
                    raw = int(logits.argmax().item())
                    return selected, raw, raw != selected
                selected, raw, changed = super()._select_event(logits, state)
                controls = self.model.control_tokens
                if selected not in {controls.idle, controls.start, controls.stop}:
                    self.lexical_count += 1
                return selected, raw, changed

        values, rate = _audio_values(audio)
        context = _chat_ids(self.processor, self.settings.system_prompt, "")
        context = context.tolist() if hasattr(context, "tolist") else context
        context = context[0] if context and isinstance(context[0], list) else context
        result = CappedStreamer(
            self.model, self.processor, max_silent_chunks=self.settings.max_silent_chunks,
            sample=False, lexical_cap=self.settings.max_new_tokens).run(
                values, sample_id=example_id, context_token_ids=[int(x) for x in context],
                sample_rate_hz=rate)
        token_ids = [row.event_id for row in result.trace if row.event_type == "TEXT"]
        return self.processor.tokenizer.decode(
            token_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)

    def generate_text(self, prompt: str, *, example_id: str) -> str:
        import torch
        device = self.model.base_thinker.get_input_embeddings().weight.device
        ids = _chat_ids(self.processor, self.settings.system_prompt, prompt).to(device)
        with torch.inference_mode():
            generated = self.model.thinker.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=self.settings.max_new_tokens, do_sample=False)[:, ids.shape[1]:]
        return self.processor.tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def load_backend(settings: RunSettings) -> Backend:
    settings.validate()
    return BaseQwenBackend(settings) if settings.model_mode == "base" else DuplexQwenBackend(settings)


def _read_existing(path: Path, manifest: Mapping[str, Any]) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        example_id = record.get(ID_FIELD)
        if not isinstance(example_id, str):
            raise ValueError(f"{path}:{number} has no string {ID_FIELD}.")
        if example_id in seen:
            raise ValueError(f"Duplicate {ID_FIELD} {example_id!r} in {path}.")
        if record.get(MANIFEST_FIELD) != manifest:
            raise ValueError(f"Run manifest mismatch in {path}:{number}.")
        seen.add(example_id)
    return seen


def run_benchmark(
    settings: RunSettings, *, output: Path, limit: int | None = None, start_index: int = 0,
    dataset_loader: Callable[[RunSettings], object] = load_dataset,
    backend_loader: Callable[[RunSettings], Backend] = load_backend,
) -> dict[str, Any]:
    settings.validate()
    if start_index < 0 or (limit is not None and limit < 0):
        raise ValueError("start_index and limit must be non-negative.")
    manifest = build_manifest(settings)
    seen = _read_existing(output, manifest)
    data = dataset_loader(settings)
    stop = len(data) if limit is None else min(len(data), start_index + limit)
    selected = list(range(min(start_index, len(data)), stop))
    pending = []
    for index in selected:
        item = data[index]
        example_id = stable_example_id(item, data=settings.data, split=settings.split, index=index)
        if example_id not in seen:
            pending.append((item, example_id))
    backend = backend_loader(settings) if pending else None
    output.parent.mkdir(parents=True, exist_ok=True)
    started, written = time.perf_counter(), 0
    with output.open("a", encoding="utf-8") as handle:
        for item, example_id in pending:
            assert backend is not None
            response = (backend.generate_audio(item["audio"], example_id=example_id)
                        if settings.modality == "audio" else
                        backend.generate_text(str(item["prompt"]), example_id=example_id))
            if not isinstance(response, str) or not response.strip():
                raise RuntimeError(f"Model returned an empty response for {example_id}.")
            record = {key: value for key, value in item.items() if key != "audio"}
            record.update(response=response, voicebench_id=example_id, run_manifest=manifest)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
    return {"output": str(output), "selected": len(selected), "written": written,
            "resumed": written != len(selected), "elapsed_seconds": time.perf_counter() - started,
            "manifest": manifest}


PAIR_FIELDS = (
    ("dataset",), ("modality",), ("seed",), ("system_prompt",), ("decoding",),
    ("upstream", "revision"), ("model", "base_id"),
    ("model", "base_revision"), ("model", "dtype"),
)


def _nested(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = mapping
    for key in path:
        value = value[key]
    return value


def _read_records(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    records, manifest = {}, None
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        example_id, current = record.get(ID_FIELD), record.get(MANIFEST_FIELD)
        if not isinstance(example_id, str) or not isinstance(current, dict):
            raise ValueError(f"Invalid benchmark result schema in {path}:{number}.")
        if example_id in records:
            raise ValueError(f"Duplicate {ID_FIELD} {example_id!r} in {path}.")
        if manifest is not None and current != manifest:
            raise ValueError(f"Mixed run manifests in {path}.")
        records[example_id], manifest = record, current
    if manifest is None:
        raise ValueError(f"No records in {path}.")
    return records, manifest


EVALUATORS = {
    "alpacaeval": ("open", True), "alpacaeval_full": ("open", True),
    "commoneval": ("open", True), "wildvoice": ("open", True),
    "sd-qa": ("qa", True), "ifeval": ("ifeval", False),
    "advbench": ("harm", False), "openbookqa": ("mcq", False),
    "mmsu": ("mcq", False), "bbh": ("bbh", False),
}


def _upstream_commands(path: Path, subset: str, upstream_dir: Path) -> list[str]:
    if subset not in EVALUATORS:
        raise ValueError(f"No pinned VoiceBench evaluator mapping for subset {subset!r}.")
    evaluator, judge = EVALUATORS[subset]
    workdir, scripts, name = path.resolve().parent, upstream_dir.resolve(), path.name
    prefix = f"cd {shlex.quote(str(workdir))} && python "
    commands = []
    if judge:
        commands.append(prefix + f"{shlex.quote(str(scripts / 'api_judge.py'))} --src_file {shlex.quote(name)}")
        name = "result-" + name
    commands.append(prefix + f"{shlex.quote(str(scripts / 'evaluate.py'))} --src_file {shlex.quote(name)} --evaluator {evaluator}")
    return commands


def summarize_pair(base: Path, duplex: Path, *, upstream_dir: Path) -> dict[str, Any]:
    base_rows, base_manifest = _read_records(base)
    duplex_rows, duplex_manifest = _read_records(duplex)
    if base_manifest["model"]["mode"] != "base" or duplex_manifest["model"]["mode"] != "duplex":
        raise ValueError("Pair inputs must be base then duplex outputs.")
    mismatches = [".".join(path) for path in PAIR_FIELDS
                  if _nested(base_manifest, path) != _nested(duplex_manifest, path)]
    if mismatches:
        raise ValueError(f"Paired run manifests differ in required settings: {mismatches}.")
    missing_base = sorted(set(duplex_rows) - set(base_rows))
    missing_duplex = sorted(set(base_rows) - set(duplex_rows))
    if missing_base or missing_duplex:
        raise ValueError(f"Missing paired examples: base={missing_base[:5]}, duplex={missing_duplex[:5]}.")
    subset = base_manifest["dataset"]["subset"]
    return {
        "schema_version": 1, "pair_count": len(base_rows), "example_ids": sorted(base_rows),
        "base_output": str(base), "duplex_output": str(duplex),
        "settings": {"dataset": base_manifest["dataset"], "seed": base_manifest["seed"],
                     "system_prompt": base_manifest["system_prompt"],
                     "decoding": base_manifest["decoding"]},
        "upstream_revision": VOICEBENCH_REVISION,
        "commands": {"base": _upstream_commands(base, subset, upstream_dir),
                     "duplex": _upstream_commands(duplex, subset, upstream_dir)},
        "warning": "Prepared commands only; no external judge was called and scores are not interpreted.",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--model-mode", required=True, choices=["base", "duplex"])
    adapter = run.add_mutually_exclusive_group()
    adapter.add_argument("--checkpoint", type=Path)
    adapter.add_argument("--adapter", type=Path)
    run.add_argument("--data", required=True)
    run.add_argument("--split", default="test")
    run.add_argument("--modality", required=True, choices=["audio", "text"])
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--limit", type=int)
    run.add_argument("--start-index", type=int, default=0)
    run.add_argument("--seed", type=int, default=17)
    run.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    run.add_argument("--max-new-tokens", type=int, default=256)
    run.add_argument("--max-silent-chunks", type=int, default=4)
    run.add_argument("--allow-download", action="store_true")
    summary = sub.add_parser("summarize")
    summary.add_argument("--base", required=True, type=Path)
    summary.add_argument("--duplex", required=True, type=Path)
    summary.add_argument("--upstream-dir", required=True, type=Path)
    summary.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "summarize":
        result = summarize_pair(args.base, args.duplex, upstream_dir=args.upstream_dir)
        rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return
    settings = RunSettings(
        model_mode=args.model_mode, data=args.data, split=args.split, modality=args.modality,
        seed=args.seed, system_prompt=args.system_prompt, max_new_tokens=args.max_new_tokens,
        max_silent_chunks=args.max_silent_chunks, adapter=args.adapter or args.checkpoint,
        allow_download=args.allow_download)
    _seed_everything(settings.seed)
    print(json.dumps(run_benchmark(
        settings, output=args.output, limit=args.limit, start_index=args.start_index),
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
