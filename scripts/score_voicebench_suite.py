"""Validate the paired artifacts and run the pinned official VoiceBench scorers."""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import hashlib
import importlib
import io
import json
from pathlib import Path
import random
import sys
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from transformers import Qwen2_5OmniProcessor
from prepare_voicebench import CONFIGS, OFFLINE_CONFIGS


def read(path):
    return [json.loads(s) for s in path.read_text().split("\n") if s.strip()]


def official_modules(upstream):
    # Load the scorer modules unchanged; avoid the aggregate __init__ importing
    # the optional PEDANT model. SD-QA here uses the requested GPT-4o metric.
    package = ModuleType("voicebench_official")
    package.__path__ = [str((upstream / "src/evaluator").resolve())]
    sys.modules[package.__name__] = package
    return {name: importlib.import_module(f"voicebench_official.{name}")
            for name in ("open", "harm", "ifeval", "mcq", "bbh")}


class RandomFallback(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/voicebench-response-gain003"))
    parser.add_argument("--data", type=Path, default=Path("data/VoiceBench-eval"))
    parser.add_argument("--upstream", type=Path, default=Path("outputs/voicebench-upstream"))
    parser.add_argument("--offline-only", action="store_true")
    args = parser.parse_args()
    configs = OFFLINE_CONFIGS if args.offline_only else CONFIGS
    import nltk
    nltk.data.path.insert(0, str((args.output / "nltk_data").resolve()))
    from langdetect import DetectorFactory
    DetectorFactory.seed = 17
    official = official_modules(args.upstream)
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B",
        revision="f75b40e3da2003cdd6e1829b1f420ca70797c34e", local_files_only=True)
    meta = json.loads((args.output / "dataset-metadata.json").read_text())["cardData"]["dataset_info"]
    expected = {(d["config_name"], s["name"]): s["num_examples"] for d in meta for s in d["splits"]}
    report = {"models": {}, "paired_audio_identity": True, "decoded_token_identity": True,
              "upstream_files": {str(p.relative_to(args.upstream)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (args.upstream / "src/evaluator").rglob("*.py")}}
    paired_ids = {}
    for mode in ("base", "duplex"):
        if not (args.output / mode / "complete.json").exists():
            raise ValueError(f"{mode} inference is incomplete")
        model_results = {}
        for config in configs:
            data = []
            split_rows = {}
            for folder in sorted((args.data / config).iterdir()):
                if not folder.is_dir():
                    continue
                split = folder.name
                source = read(folder / "manifest.jsonl")
                records = [r for p in sorted((args.output / mode).glob(f"{config}--{split}*.jsonl")) for r in read(p)]
                records.sort(key=lambda r: int(r["voicebench_id"].rsplit(":", 1)[1]))
                if len(records) != len(source) or len(records) != expected[config, split]:
                    raise ValueError(f"Incorrect sample count for {mode} {config}/{split}")
                for row, result in zip(source, records):
                    if row["voicebench_id"] != result["voicebench_id"]:
                        raise ValueError("Missing, duplicate, or substituted example")
                    if row["pcm_sha256"] != result["pcm_sha256"]:
                        raise ValueError("Different waveform used for paired inference")
                    if any(result[k] != v for k, v in row["source"].items()):
                        raise ValueError("Benchmark source fields were changed")
                    tokens = np.load(result["tokens_path"])
                    decoded = processor.tokenizer.decode(tokens["token_ids"].tolist(),
                        skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    if decoded != result["response"]:
                        raise ValueError("Scored answer differs from actual generated tokens")
                    if mode == "duplex":
                        whole = processor.tokenizer.decode(tokens["event_ids"].tolist(),
                            skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        if whole != result["response"]:
                            raise ValueError("Scored answer differs from the whole event sequence")
                        controls = {151643, 151644, 151645}
                        lexical = [int(x) for x in tokens["event_ids"] if x not in controls]
                        if lexical != tokens["token_ids"].tolist():
                            raise ValueError("Duplex text does not match event trace")
                    pair = (result["pcm_sha256"], {k: result[k] for k in row["source"]})
                    if mode == "base":
                        paired_ids[result["voicebench_id"]] = pair
                    elif paired_ids[result["voicebench_id"]] != pair:
                        raise ValueError("Paired examples differ")
                split_rows[split] = len(records)
                data.extend(records)
            result = {"n": len(data), "splits": split_rows,
                "empty_responses": sum(not r["response"].strip() for r in data),
                "finish_reasons": dict(Counter(r["finish_reason"] for r in data)),
                "mean_tokens": float(np.mean([r["token_count"] for r in data]))}
            judged_files = list((args.output / "judged" / mode).glob(f"{config}--*.jsonl"))
            if config in {"commoneval", "alpacaeval_full", "wildvoice", "sd-qa"}:
                judged = [r for p in judged_files for r in read(p)]
                indexed = {r["voicebench_id"]: r for r in judged}
                if len(indexed) != len(judged):
                    raise ValueError("Duplicate judge answers")
                for row in data:
                    if row["voicebench_id"] in indexed:
                        j = indexed[row["voicebench_id"]]
                        if j["response"] != row["response"] or j["pcm_sha256"] != row["pcm_sha256"]:
                            raise ValueError("Judge scored a different answer")
                        if len(j["score"]) != 3 or not j["judge_model"].startswith("gpt-4o"):
                            raise ValueError("Incorrect judge settings")
                result["judged"] = len(judged)
                if len(judged) == len(data):
                    if config == "sd-qa":
                        def majority(scores):
                            scores = [s.lower() for s in scores]
                            return max(set(scores), key=scores.count) == "yes"
                        correct = [majority(r["score"]) for r in judged]
                        result["score"] = float(np.mean(correct) * 100)
                        result["metric"] = "GPT-4o majority-vote accuracy (%)"
                        normalized = [sum(s.strip().strip(".*` ").lower() == "yes" for s in r["score"]) >= 2 for r in judged]
                        result["normalized_vote_accuracy"] = float(np.mean(normalized) * 100)
                        result["noncanonical_votes"] = sum(s.lower() not in {"yes", "no"} for r in judged for s in r["score"])
                    else:
                        result["score"] = float(official["open"].OpenEvaluator().evaluate(judged)["gpt"])
                        result["metric"] = "GPT-4o rating (1–5)"
                else:
                    result.update(score=None, metric="GPT-4o unavailable or incomplete")
            elif config in {"openbookqa", "mmsu"}:
                evaluator = official["mcq"].MCQEvaluator()
                random.seed(17)
                values = evaluator.evaluate(data)
                predictions = [evaluator.extract_answer(r["response"]) for r in data]
                result.update(score=float(values["acc"]), metric="Official accuracy (%)",
                    extraction_fail_percent=float(values["fail"]),
                    strict_accuracy=float(np.mean([p == r["reference"] for p, r in zip(predictions, data)]) * 100))
            elif config == "bbh":
                evaluator = official["bbh"].BBHEvaluator()
                random.seed(17)
                with contextlib.redirect_stdout(io.StringIO()):
                    result.update(score=float(evaluator.evaluate(data)["acc"]), metric="Official accuracy (%)")
                choice = random.choice
                def reject(_):
                    raise RandomFallback()
                random.choice = reject
                predictions = []
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        for row in data:
                            try:
                                predictions.append(evaluator.extract_answer(row["response"], row["id"]))
                            except RandomFallback:
                                predictions.append(None)
                finally:
                    random.choice = choice
                truth = {"yes": 1, "no": 0, "(a)": 0, "(b)": 1}
                result.update(extraction_fail_percent=sum(p is None for p in predictions) / len(data) * 100,
                    strict_accuracy=float(np.mean([p == truth[r["reference"].lower()] for p, r in zip(predictions, data)]) * 100))
            elif config == "ifeval":
                random.seed(17)
                values = official["ifeval"].IFEvaluator().evaluate(data)
                result.update(score=float(values["final"] * 100), metric="Mean strict/loose instruction/prompt accuracy (%)",
                    submetrics={k: float(v * 100) for k, v in values.items() if k != "final"})
            elif config == "advbench":
                result.update(score=float(official["harm"].HarmEvaluator().evaluate(data)["refusal_rate"] * 100),
                    metric="Official keyword refusal rate (%)")
            model_results[config] = result
            print(mode, config, json.dumps(result), flush=True)
        report["models"][mode] = model_results
    report["paired_samples"] = len(paired_ids)
    (args.output / "scores.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    lines = ["| Benchmark | Samples | Original 3B | Full-epoch duplex | Difference |", "|---|---:|---:|---:|---:|"]
    for config in configs:
        base, duplex = (report["models"][m][config] for m in ("base", "duplex"))
        a, b = base["score"], duplex["score"]
        lines.append(f"| {config} | {base['n']} | {a:.2f} | {b:.2f} | {b-a:+.2f} |" if a is not None and b is not None
                     else f"| {config} | {base['n']} | N/A | N/A | N/A |")
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
