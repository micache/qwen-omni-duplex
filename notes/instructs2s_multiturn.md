# Post-STOP frames and continuous multi-turn inference

The user asked why the displayed generation ended at STOP and requested a
multi-turn audio-only run. The prior 224-event rectangle trace was complete for
the single-turn inference call: `QwenDuplexStreamer.run` stops immediately on STOP
inside an appended EOF silent chunk. It does not predict the remaining frames
of that chunk. This is a runtime termination rule, not evidence that the model
cannot emit IDLE after STOP.

Training initializes the entire timeline to IDLE, then inserts START, contiguous
answer tokens, and STOP. Any remaining frames stay IDLE. The InstructS2S subset
uses text-sized synthetic assistant silence rounded to two-second chunks;
unlike the older TASTE view, it does not preserve original assistant speech
duration. No assistant speech is supplied to the model.

## Complete rectangle timeline

With the same raw question padded to eight seconds followed by two seconds of
provided silence, inference consumes all 250 frames. The first 224 emitted IDs
exactly match the earlier CLI result, followed by **26 predicted IDLE events**.
START is at frame 180, STOP at 223, and frames 224–249 are IDLE. The correct
rectangle answer is unchanged. All 250 events are saved, not padded artificially
after generation. There are zero grammar-mask substitutions.

## One continuous three-turn recording

The deterministic selection is the first source-order conversation in the
already downloaded Parquet with rounds 1, 2, 3 and an ID absent from both prepared
manifests: `instruct_en_20`. Only its current-question audio bytes are used;
later-row text histories and reference answers are never passed to inference.
The three questions are DNA, RNA and ATP expansion. Each question is padded to
a full two-second boundary and followed by a fixed four seconds of silence,
including after the final question. Silence duration is independent of reference
answer length. A single `run` call retains KV cache, generated text/control
history and response state across all three turns. Context is empty; no START
or STOP is forced and no reference assistant speech is played.

The recording is 20 seconds / 500 frames. It yields 454 IDLE, three START,
40 TEXT, and three STOP events; no raw argmax is changed by the grammar mask.

| Spoken question | Actual generated text | START / STOP frame | Following IDLE frames |
| --- | --- | --- | ---: |
| What does DNA stand for? | DNA stands for Deoxyribonucleic | 79 / 89 | 154 |
| What does RNA stand for? | Zarren stands for Zara, a popular fashion brand. | 244 / 258 | 141 |
| What does ATP stand for? | ATP stands for Adenosine Triphosphate, a crucial energy molecule in cells. | 400 / 419 | 80 |

DNA is truncated (missing Acid), RNA is incorrect, ATP is correct. There are
79 IDLE events before the first START. This demonstrates continued listening
and multiple response episodes, not reliable multi-turn semantic performance.
The checkpoint was trained for only 200 updates on first turns. These questions
also do not require coreference, so the test does not establish conversational
memory or history-dependent reasoning.

Isolated follow-ups reset the cache and use the same current-question recordings
and four-second silence. RNA still fails: "The term 'Czar' is a historical title
used to describe the rulers of the Russian Empire, specifically the Tsars who
reigned from the thirteenth century to the twentieth century." ATP correctly
answers "ATP stands for Adenosine Triphosphate." Thus retained history is not
the sole cause of the RNA failure. No claim about the source audio's quality is
made from these model outputs alone.

All original events, full raw special-token strings, input FLACs and reports are
under `outputs/instructs2s-response-gain003-pilot/multiturn-inspection/` and
`multiturn-isolated-controls/`. `multiturn-raw-wrapped.txt` adds display line breaks
only; removing them gives the complete original raw string exactly.

```bash
OMP_NUM_THREADS=4 .venv/bin/python scripts/evaluate_instructs2s_multiturn.py --output outputs/instructs2s-response-gain003-pilot/multiturn-inspection
OMP_NUM_THREADS=4 .venv/bin/python scripts/evaluate_instructs2s_multiturn.py --isolated-followups --output outputs/instructs2s-response-gain003-pilot/multiturn-isolated-controls
```

Runtime checks assert every provided 25 Hz frame was processed. Source data,
model, adapter and input policy are the previous experiment's pinned versions.
No training or default single-turn stopping behavior changed for this diagnostic.
