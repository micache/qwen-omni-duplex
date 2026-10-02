# VoiceBench: original 3B versus the completed duplex epoch

Completed: 5,394 paired audio examples, five offline subsets. All duplex samples were rerun with mandatory appended silence. GPT-4o judging is excluded.

Authoritative artifacts: `outputs/voicebench-response-gain003-silence/`.

## Duplex-only output review, 2026-10-02

The completed offline review below is preserved. The README displays original
official IFEval/OpenBookQA/MMSU/BBH scores and reviewed AdvBench, with their
scoring sources stated. Only duplex responses were re-scored; all original
baseline figures and saved generations are unchanged.
These are custom reviewed scores, not a replacement run of the official metrics.

| Subset | Original baseline | Reviewed duplex | Correct duplex answers | Recovered / removed versus official |
|---|---:|---:|---:|---:|
| IFEval | 42.21 | 4.74 | 14/345 strict prompts | 0 / 33 strict prompts |
| AdvBench | 99.42 | 83.08 | 432/520 | 45 / 7 |
| OpenBookQA | 74.73 | 61.32 | 279/455 | 1 / 11 |
| MMSU | 48.31 | 41.77 | 1,284/3,074 | 5 / 151 |
| BBH | 57.60 | 40.00 | 400/1,000 | 6 / 147 |

Screened every saved duplex answer, then reviewed changed/ambiguous extractions
and candidates whose answer text did not contain the referenced choice verbatim.
Reviewed 20 OpenBookQA, 591 MMSU and 325 BBH cases, plus all 102 IFEval responses
receiving any original instruction credit. The completed AdvBench review supplies
520 duplex decisions. This was offline inspection in this session, with no paid
or external judge. It does not claim independent human adjudication of all 5,394
responses. Decisions bind the entire original prompt and response by SHA-256;
references, denominators and generated text are never edited.

For single-choice tasks, require an unambiguous A–D/A–B selection; for binary
tasks require Yes/No. Accept ordinary punctuation variants and repeated consistent
prefixes. Do not enforce a new literal-colon rule. Reject conflicting choices,
multiple selected options, a correct letter naming a different answer/value,
unusable fragments/loops and answers to a different question. No random guesses
receive credit. A correct selection with a consistent paraphrase remains correct;
the review does not require every incidental reasoning statement to be perfect.
Clear content-only answers lacking the requested choice/binary form remain wrong.

IFEval retains its mean of strict/loose prompt/instruction accuracy. Whole
unusable, wrong-task or structurally invalid responses get zero on all checks;
otherwise retain the official per-instruction checks, including partial credit
for separate keyword/length constraints. The resulting prompt submetrics are
14/345 (4.06%) each; instruction submetrics are 27/498 (5.42%) each. Their mean
is 4.74%, not the strict prompt percentage. Restored seed 17 and the upstream
order of all strict checks followed by all loose checks to reproduce the original
18.106774925790113% before applying review masks.

Examples: OpenBookQA `test:152` says `B is correct. Evaporation...`, which the
old parser misses. MMSU `business:171` and `philosophy:133` use valid `A is...`
declarations. BBH `test:551` describes the correct A phrase and explicitly says
`The answer is A.`; the upstream extractor instead selects B. Conversely, MMSU
`biology:95` labels C but names geotropism instead of thigmotropism, and BBH
`test:499` contains only `Yes, ifNo, you`. Neither receives credit.

Local audit files: `review/corrected-scores.json`, `review/duplex-decisions.json`,
`review/{config}-decisions.jsonl` and the existing `review/advbench-review.jsonl`.
`scores.json` is the preserved official report. Its SHA-256 is
`30f9043fbfa967a8b68a4a93e185f62180c67b1e81f610010d8438fb31f305bb`.
Reproduce the review with the local decision ledger and pinned upstream checkout,
running each config sequentially:

```bash
.venv/bin/python scripts/rescore_voicebench_duplex.py --config advbench --finalize
.venv/bin/python scripts/rescore_voicebench_duplex.py --config openbookqa --finalize
.venv/bin/python scripts/rescore_voicebench_duplex.py --config mmsu --finalize
.venv/bin/python scripts/rescore_voicebench_duplex.py --config bbh --finalize
.venv/bin/python scripts/rescore_voicebench_duplex.py --config ifeval --finalize
```

All original official totals must reproduce before corrected scores are written;
finalization rejects ambiguous cases without matching decisions. Raw results and
decision ledgers stay local, as requested; no evidence bundle is uploaded.
The stricter duplex review widens the observed gaps. Retaining useful knowledge
and learning duplex structure does not establish unchanged intelligence or isolate
weight forgetting from inference/timing/formatting failures.

## Original official results (preserved)

All primary values are percentages; differences are percentage points. IFEval is the mean of strict/loose prompt/instruction accuracy; AdvBench is the official keyword refusal rate. The other rows use official accuracy.

| Benchmark | Samples | Original Qwen2.5-Omni-3B | Full-epoch duplex | Difference |
|---|---:|---:|---:|---:|
| ifeval | 345 | 42.21 | 18.11 | -24.11 |
| advbench | 520 | 99.42 | 75.77 | -23.65 |
| openbookqa | 455 | 74.73 | 63.52 | -11.21 |
| mmsu | 3,074 | 48.31 | 46.52 | -1.79 |
| bbh | 1,000 | 57.60 | 54.10 | -3.50 |

Official MCQ/BBH evaluators randomly guess when answer extraction fails. Scores use seed 17 and canonical source-index order within each split; provisional scores in generation order can differ. Strict scores count extraction failures as wrong.

| Benchmark | Base strict accuracy | Duplex strict accuracy | Base extraction failures | Duplex extraction failures |
|---|---:|---:|---:|---:|
| openbookqa | 74.73 | 62.86 | 1.32 | 2.20 |
| mmsu | 44.05 | 45.51 | 14.51 | 4.03 |
| bbh | 56.70 | 50.30 | 1.60 | 7.60 |

The largest regressions are instruction following and keyword refusal. MMSU is close on the official metric, and duplex has slightly higher strict accuracy; parser behavior materially affects this comparison. This measures native audio chat against the trained streaming interface, so it does not isolate weight forgetting.

## Fixed protocol

- Adapter: `outputs/instructs2s-response-gain003-one-epoch/final`, complete 3,750-step / 44,000-conversation epoch, response audio gain 0.03. Adapter identity SHA-256: `5cc97caece42b59c378eacdceac0202211291893d64e25bad47cd92876bd1b7e`.
- Baseline: untouched Qwen/Qwen2.5-Omni-3B Thinker weights, revision `f75b40e3da2003cdd6e1829b1f420ca70797c34e`. Our measured 3B baseline; no Talker is constructed or evaluated.
- Pinned VoiceBench repository: `b56154172f2a57a43d29005de7d0471d748d70d1`; HF hlt-lab/voicebench Parquet conversion revision `de69a22f41561676635bef0b31681df4b866ec07`. Scorer source hashes are in scores.json.
- IFEval 345; AdvBench 520; OpenBookQA 455; all 12 MMSU subjects 3,074; BBH 1,000. Every example is retained. Judge-dependent subsets and optional metrics are excluded.
- Both models receive the same SHA-256-checked 16 kHz waveform. PCM-24 preparation restores original amplitude; audited quantization error is below 2e-7 times the per-clip storage scale. No question transcript, reference or answer hint enters either model.
- Duplex uses trained empty text context, fixed padded 2-second chunks, validated 25 Hz, FP32 frozen audio encoder, BF16 decoder and additive fusion. Evaluation is clean, without training noise or synthetic interruptions.
- After every input, consume at least one additional full 2-second zero-audio chunk. Continue unfinished text until actual STOP, the 2,048 lexical-token limit, or 42 extra chunks (84 seconds). No forced STOP.
- Native baseline uses VoiceBench audio chat with the system text You are a helpful assistant, original BF16 audio/text weights and native multimodal positions. Greedy decoding, 2,048-token limit.
- Score the complete generated event sequence with skip_special_tokens=True and clean_up_tokenization_spaces=False. Save actual lexical, event and raw-argmax IDs in compressed NPZ files.

## Validation and timing

All 5,394 IDs per model are present exactly once. Official scoring checks every source field, paired waveform hash, actual generated-token decode, and duplex event/text identity. Every duplex trace consumed all input frames plus at least 50 silent frames. Independently recomputed strict MCQ scores and checked 15 prepared waveforms. Repeated scoring produced a byte-identical scores.json. All 18 local correctness tests pass.

No empty answers: base 0, duplex 0. Token-capped answers remain in the denominator: base 72, duplex 21. There are no silence-budget timeouts.

Timing problems remain in the generated output: 2,724/3,074 MMSU examples contain multiple nonempty response segments, and 2,717 begin text more than two seconds before source audio ends. Full sequence scoring includes these fragments. Audio-end timing is measured from the supplied clip, without a separate speech endpoint detector. See validation.json for every subset.

## Execution and historical corrections

Equal-work 64-question / 64-token diagnostics: one native copy 14.31 s, two 11.86 s, four 12.00 s. Four fit but did not outperform two, so production used two independent native processes with source-index modulo sharding. Diagnostic outputs are excluded from scoring.

Duplex used batch maximum 128 and a 300,000-frame budget with packed static KV, variable-length FlashAttention and CUDA graphs. Native ended at maximum 64 per copy and 120,000 frames per copy after finished-row compaction freed memory. OOM retries halve batches without discarding examples. BF16 changes in batch shape can change near-tied tokens; no bitwise equality across all batch sizes is claimed. Weights remain unmerged.

Native generation removes completed rows between 64-token blocks, retaining full prefixes, native cache and RoPE deltas. B=1 block continuation matched 256 native reference tokens; all 26 production-shaped initial 64-token prefixes matched. Only an unpadded single-row tail uses its validated cached graph: 512-token trials, a 320-token capped case and a 166-token EOS case matched. A padded fast-path test failed strict parity, so every padded row retains ordinary native generation. No scored output used the rejected padded path.

The earlier duplex run in outputs/voicebench-response-gain003 had 4,107 traces that ended at question EOF after an earlier STOP, consuming no extra silent chunk. Its decode audit passed but missed the appended-silence contract. All duplex examples were rerun in the authoritative folder; the earlier comparison is superseded. Native outputs remain valid and are reused by symlink. Earlier judging artifacts are historical only; the runtime API credential was removed. Intentional inference restarts appear as KeyboardInterrupt / return code -2 in old logs; no samples were lost.

## Artifacts and reproduction

- Final scores/table/audit: authoritative folder scores.json, comparison.md, validation.json.
- Duplex response files: authoritative folder duplex/{config}--{split}.jsonl; original IDs under duplex/tokens/.
- Duplex log: authoritative folder inference.log; official scoring logs scoring.log and scoring-recheck.log.
- Base response files: authoritative folder base/ symlink; native logs remain outputs/voicebench-response-gain003/offline-base-{0,1}.log.
- Dataset/model/protocol manifests and pinned metadata remain with the runs.

With this workspace's prepared audio, pinned upstream checkout, baseline outputs and completed adapter:

```bash
.venv/bin/python scripts/run_voicebench_offline.py
.venv/bin/python scripts/score_voicebench_suite.py --offline-only
.venv/bin/python -m pytest -q
```

The launcher resumes the corrected folder, reuses completed native results and invokes offline scoring. The corrected duplex rerun took 1,659.58 s (27.66 minutes).

Official protocol: [VoiceBench](https://github.com/MatthewCYM/VoiceBench#evaluation).
