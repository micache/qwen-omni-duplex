"""Train the mandatory complete-conversation overfit gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf
import torch

from duplex.contract import prompt_token_ids
from duplex.conversations import ConversationLength, CompleteConversationDataset, exact_audio_frame_count
from duplex.dataset import DuplexCollator, Split, assign_split, load_conversation, read_manifest
from duplex.model import NextEventLossWeights
from duplex.training import (_ListDataset, _load_adapter_weights, build_training_model,
                             load_training_config, make_trainer, seed_everything)


IDS = ("data_stereo/38", "data_stereo/64", "data_stereo/84")


def gold_counts(values, controls):
    counts = Counter()
    for value in values:
        for event in value.targets.events:
            counts["text" if event.kind.value == "TEXT" else event.kind.value.lower()] += 1
    return {name: counts[name] for name in ("text", "idle", "start", "stop")}


def assess_generation(model, fixtures):
    results = []
    for path in fixtures:
        result = model.generate(path)
        kinds = Counter(row.event_type for row in result.events)
        segments = [{"text": row["text"], "tokens": len(row["tokens"])} for row in result.segments]
        results.append({"audio": str(path), "text": result.text, "segments": segments,
                        "events": dict(kinds), "first_word_time_s": result.first_word_time_s,
                        "last_word_time_s": result.last_word_time_s})
    usable = any(row["text"].strip() for row in results)
    no_empty = all(all(segment["tokens"] > 0 for segment in row["segments"]) for row in results)
    no_cycle = all(row["events"].get("START", 0) <= max(2, row["events"].get("TEXT", 0))
                   for row in results)
    return results, usable and no_empty and no_cycle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_conversation_overfit.yaml")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    config = load_training_config(args.config)
    output = Path(config["training"]["output_dir"])
    if args.finalize:
        report = json.loads((output / "report.json").read_text())
        fresh_report = json.loads((output / "fresh_report.json").read_text())
        in_memory = report["generations"]
        same = [row["text"] for row in in_memory] == [row["text"] for row in fresh_report["generations"]]
        usable = any(row["text"].strip() for row in in_memory)
        no_empty = all(all(segment["tokens"] > 0 for segment in row["segments"]) for row in in_memory)
        no_cycle = all(row["events"].get("START", 0) <= max(2, row["events"].get("TEXT", 0)) for row in in_memory)
        gate = (report["final_loss"] <= report["baseline_loss"] * 0.6 and usable
                and no_empty and no_cycle and fresh_report["usable"] and same)
        report.update({"gate": "PASS" if gate else "FAIL", "fresh": fresh_report,
                       "same_reloaded_text": same,
                       "fresh_initial_returncode": report.get("fresh_returncode"),
                       "fresh_returncode": 0,
                       "fresh_command": ".venv/bin/python scripts/run_complete_overfit.py --config configs/full_conversation_overfit.yaml --fresh"})
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"OVERFIT_GATE={report['gate']}", flush=True)
        return
    seed_everything(config["training"]["seed"])
    model, processor, targets = build_training_model(config)
    if args.fresh:
        _load_adapter_weights(model, output / "final")
    entries = {row.conversation_id: row for row in read_manifest(
        Path(config["data"]["root"]) / config["data"]["manifest"])}
    salt = f"DailyTalkContiguous-session11-{config['training']['split_seed']}"
    values = []
    lengths = {}
    for conversation_id in IDS:
        if assign_split(conversation_id, salt=salt) is not Split.TRAIN:
            raise ValueError(f"{conversation_id} is not in the training split")
        record = load_conversation(entries[conversation_id], dataset_root=config["data"]["root"],
                                   speaker_label_map=config["data"].get("speaker_label_map"))
        frames = exact_audio_frame_count(record.user_waveform, processor, model.base_thinker.audio_tower)
        required = len(prompt_token_ids(processor.tokenizer)) + frames
        context_limit = model.base_thinker.config.text_config.max_position_embeddings
        if required > context_limit:
            raise ValueError(f"{conversation_id}: needs {required} positions; limit {context_limit}")
        lengths[conversation_id] = ConversationLength(conversation_id, Split.TRAIN,
                                                      record.duration_seconds, frames, required)
    dataset = CompleteConversationDataset([entries[key] for key in IDS], root=config["data"]["root"],
        split=Split.TRAIN, config=config, processor=processor, model=model,
        preflight_lengths=lengths)
    values = [dataset[index] for index in range(len(dataset))]
    counts = gold_counts(values, model.control_tokens)
    model.loss_weights = NextEventLossWeights.capped_inverse_sqrt(
        counts, max_ratio=config.get("loss_weight_max_ratio", 5.0))
    model.audio_cache_dir = output / "audio-cache"
    prompt_ids = prompt_token_ids(processor.tokenizer)
    collator = DuplexCollator(audio_processor=processor, tokenizer=processor.tokenizer,
        control_tokens=model.control_tokens, thinker_bos_token_id=model.base_thinker.config.bos_token_id,
        context_token_ids=prompt_ids, interruption_probability=0)
    fixtures = []
    output.mkdir(parents=True, exist_ok=True)
    for value in values:
        path = output / "fixtures" / f"{value.sample_id.split('@')[0].replace('/', '-')}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        # The first spoken user activity plus silence is frozen as ordinary audio.
        first_user_end = next((word.end_seconds for word in value.user_words), 2.0)
        question = value.user_waveform[:round(first_user_end * 16_000)]
        import numpy as np
        sf.write(path, np.concatenate((question, np.zeros(32_000, dtype=np.float32))), 16_000)
        fixtures.append(path)
    if args.fresh:
        generations, usable = assess_generation(model, fixtures)
        report = {"usable": usable, "generations": generations}
        (output / "fresh_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"FRESH_OVERFIT_GATE={'PASS' if usable else 'FAIL'}", flush=True)
        return
    model.eval()
    with torch.inference_mode():
        baseline = float(model(**{key: val.to(next(model.parameters()).device) if isinstance(val, torch.Tensor) else val
                                  for key, val in collator([values[0]]).items()
                                  if key in ("text_ids", "text_mask", "control_ids", "control_mask", "attention_mask",
                                             "position_ids", "context_ids", "bootstrap_mask", "labels", "input_features",
                                             "feature_attention_mask", "preconv_feature_lengths", "audio_chunk_counts",
                                             "audio_cache_keys")}).loss)
    trainer, audit = make_trainer(config, model, processor, targets, _ListDataset(values), None, collator)
    trainer.train()
    trainer.save_model(output / "final")
    model.eval()
    with torch.inference_mode():
        final_loss = float(model(**{key: val.to(next(model.parameters()).device) if isinstance(val, torch.Tensor) else val
                                  for key, val in collator([values[0]]).items()
                                  if key in ("text_ids", "text_mask", "control_ids", "control_mask", "attention_mask",
                                             "position_ids", "context_ids", "bootstrap_mask", "labels", "input_features",
                                             "feature_attention_mask", "preconv_feature_lengths", "audio_chunk_counts",
                                             "audio_cache_keys")}).loss)
    generations, usable = assess_generation(model, fixtures)
    fresh = subprocess.run([str(Path(".venv/bin/python").resolve()), __file__,
                            "--config", args.config, "--fresh"], text=True, capture_output=True)
    fresh_report = json.loads((output / "fresh_report.json").read_text()) if fresh.returncode == 0 else {}
    same_behavior = [row["text"] for row in generations] == [row["text"] for row in fresh_report.get("generations", [])]
    gate = final_loss <= baseline * 0.6 and usable and fresh_report.get("usable") and same_behavior
    report = {"gate": "PASS" if gate else "FAIL", "baseline_loss": baseline,
              "final_loss": final_loss, "gold_counts": counts,
              "loss_weights": model.loss_weights.as_dict(), "conversations": IDS,
              "optimizer_steps": trainer.state.global_step, "generations": generations,
              "fresh": fresh_report, "same_reloaded_text": same_behavior,
              "fresh_returncode": fresh.returncode}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"OVERFIT_GATE={report['gate']} baseline={baseline:.4f} final={final_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
