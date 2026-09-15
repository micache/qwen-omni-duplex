"""Fresh-process load and three fixed Session 15 interaction probes."""

import json
from pathlib import Path

from duplex.dataset import Split
from duplex.streaming import QwenDuplexStreamer, write_trace_jsonl
from duplex.training import _load_adapter_weights, build_training_model, load_training_config
from generate import _context_token_ids
from scripts.run_session11_diagnostic import _load_spans, _span_waveform


def main():
    config = load_training_config("configs/session15_main.yaml")
    output = Path(config["training"]["output_dir"])
    selected = output / "selected"
    expected = {"adapter_model.safetensors", "adapter_config.json", "duplex_config.yaml", "training_config.yaml", "dependency_versions.json"}
    if {path.name for path in selected.iterdir()} != expected:
        raise RuntimeError("Selected output contains unexpected files or lacks adapter metadata")
    model, processor, _ = build_training_model(config)
    _load_adapter_weights(model, selected)
    model.eval()
    spans, _ = _load_spans(config, tokenizer=processor.tokenizer, controls=model.control_tokens,
                          bos_token_id=model.base_thinker.config.bos_token_id)
    by_id = {span.conversation_id: span for span in spans if span.split is Split.VALIDATION}
    generation = config["generation"]
    context = _context_token_ids(processor.tokenizer, generation["system"], "")
    streamer = QwenDuplexStreamer(model, processor, max_silent_chunks=generation["max_silent_chunks"],
                                  sample=generation["sample"], temperature=generation["temperature"],
                                  top_k=generation["top_k"])
    cases = (("wait_user_utterance", "data_stereo/67", False),
             ("start_after_completed_turn", "data_stereo/173", False),
             ("stop_on_interruption", "data_stereo/173", True))
    results = []
    for name, conversation_id, interrupted in cases:
        span = by_id[conversation_id]
        result = streamer.run(_span_waveform(span, interrupted=interrupted), sample_id=name,
                              context_token_ids=context, sample_rate_hz=span.record.sample_rate_hz)
        trace = output / "fresh_verify" / f"{name}.jsonl"
        write_trace_jsonl(trace, result.trace)
        events = [{"frame": index, "event": record.event_type} for index, record in enumerate(result.trace)
                  if record.event_type in ("START", "STOP", "TEXT")]
        onset = None
        if interrupted:
            onset = span.interrupted_chunk_index * 50 + span.interrupted.interruption.cut_frame
        user_activity = [(word.start_seconds - span.span_start_seconds,
                          word.end_seconds - span.span_start_seconds) for word in span.record.user_words
                         if span.span_start_seconds <= word.start_seconds < span.span_start_seconds + 8]
        results.append({"case": name, "conversation_id": conversation_id, "interruption_onset_frame": onset,
                        "user_activity_spans_seconds": user_activity, "events": events,
                        "text": result.text, "stop_reason": result.stop_reason,
                        "timed_out": result.timed_out, "trace": str(trace)})
    report = {"fresh_load": "PASS", "adapter": str(selected), "generation": generation, "cases": results}
    (output / "fresh_verify" / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
