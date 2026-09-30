# One-epoch TASTE diagnosis — 2026-09-30

The investigation found two distinct failures. The inference defaults added a
chat prefix that training never used; this reproduced all-IDLE output and is
now fixed. Independently, the trained adapter suppresses native chat behavior
while its duplex text generation remains imperfect. Very early stopping keeps
the tested native abilities but leaves duplex text severely undertrained.
There is no evidence here that simply shortening the run solves both problems.

## Controlled experiments

Started from commit `8c319a5`, the saved one-epoch adapter and step-1,500
checkpoint, on the same A100 40 GB. No new model or dataset was downloaded.
All inference was greedy, with noise/interruption disabled. Native probes use
Qwen's chat template and explicitly set Thinker EOS to 151645; the direct
Thinker's locally loaded generation config does not supply that EOS itself.
Duplex probes use the training context (empty), fixed 2-second audio chunks
and the existing 25 Hz additive-fusion model.

The main comparison used four train rows and 32 dev rows: the first 12 dev
rows plus 20 evenly spaced rows. The first 12 dev rows also received streamed
inference. Twelve native text probes cover arithmetic, factual questions,
translation, ordering, copying and simple deduction; three original dev audio
clips were also tested through native audio chat. These are small diagnostic
sets, not comprehensive intelligence or full-duplex benchmarks.

Two additional runs started from identical base weights, seeds, data order,
augmentation settings and frame-budget batches. Each ran only 200 updates,
at learning rates 2e-4 and 5e-5, saving early checkpoints. The 2e-4 pilot's
logged losses at steps 1, 50, 100 and 200 exactly match the original run.
This reproduces its early trajectory rather than comparing a new random run.
Pilot inference used 16 dev rows and streamed the first six. The table below
restricts every model to the **same first six dev rows and first six native
questions**. Native correctness means the first answer is correct, allowing
case differences and subsequent explanation; duplex exact match uses the
dataset reference. Native generation caps were 64 tokens in the main probe
and 16 in the pilot; the final adapter was empty even with the larger cap.

| Adapter / updates | Native answers correct | Duplex text NLL | Duplex lexical accuracy | Streamed dev exact |
| --- | ---: | ---: | ---: | ---: |
| Base, adapter disabled | 6/6 | 12.804 | 0.0% | Not tested |
| 2e-4, 50 | 6/6 | 6.743 | 0.0% | 0/6 |
| 2e-4, 200 | 6/6 | 4.210 | 15.1% | 0/6 |
| 5e-5, 50 | 6/6 | 6.756 | 3.8% | 0/6 |
| 5e-5, 200 | 6/6 | 4.959 | 9.4% | 0/6 |
| 2e-4, 1,500 | 0/6 | 1.890 | 54.7% | 1/6 |
| 2e-4, 1,919 | 0/6 | 1.654 | 58.5% | 1/6 |

The high-rate 200-update model generated `BaseBaseBaseBase,,,,,` for sports,
`Dog,,,,,,` for animals and `Knife,,,,,,,,` for kitchen tools. At lower-rate
200 updates, five of these six streamed dev runs had empty text. Thus the
tested early checkpoints do not yet supply good duplex responses.

## 1. Confirmed inference-prefix mismatch

`TurnPackedCollator` supplies an empty context during training. Before this
fix, both `generate.py` and `QwenDuplexThinker.generate()` inserted a
system/chat prefix by default. This changes the decoder's KV history and all
timeline positions before any audio arrives.

With the old default prefix, **all three original dev clips emitted only
IDLE and timed out**. With empty context, the same adapter produced their
previously reported text and STOP. That explains an all-IDLE failure in the
normal CLI even though the original diagnostic evaluator produced text.
It does not explain the malformed text from the empty-context runs.

The CLI and model method now default to an empty system string, and
`prompt_token_ids(..., "")` returns no tokens instead of wrapping an empty
string in chat markup. Explicit system prompts remain supported but are not
part of this checkpoint's training distribution. A real run of the fixed CLI
on the saved color-selection WAV returned `Red and green.` and STOP, without
timeout. Seven local tests pass, including the empty/explicit-prefix contract.

## 2. Confirmed adapter interference with native chat

With adapters disabled, all 12 native probes begin with the correct answer.
At steps 1,500 and 1,919, all 12 return empty decoded text. At the final
native prompt's first output position, controls receive a mean **99.916%**
of probability mass; the raw argmax is the IDLE token `<|endoftext|>` on
all 12 prompts. The PAD/BOS/EOS rows are being used as attended event controls
in duplex training, which differs from their native chat roles.

Suppressing IDLE/START and forcing a first non-EOS token does not recover the
answers: outputs remain punctuation, whitespace or fragments. This is more
than a stop-token selection issue.

Scaling the final LoRA update to 0.25 restores native responses: 10/12 start
with the expected literal answer; the remaining sequence and sorting answers
contain the correct answer with different wording/spacing. But on the original
three dev examples, duplex text NLL worsens from 2.328 to 7.003, and two
streams are empty. A half-strength adapter gives only 1/12 correct native
answer prefixes and also damages duplex behavior. Scaling is not a solution.

The base weights are frozen. These observations show a large, reversible
change in the **adapter-enabled function**, not erasure of the base weights.
Native-chat failure alone also does not prove that knowledge is absent within
the intended duplex interface. The evidence supports interference between
native and event-stream behavior; it does not isolate all malformed duplex
answers as catastrophic forgetting.

This interpretation is consistent with research showing that fine-tuning can
shift task inference and suppress previously accessible behavior, rather than
necessarily destroying the underlying capability.
[Kotha et al., ICLR 2024](https://arxiv.org/abs/2309.10105).

## 3. Short data is a limitation, not missing meaning in these clips

Scanning all local Parquet text and audio headers found:

- 44,000 train conversations; 43,890 unique prompts and 21,338 unique responses.
- 347,793 response tokens before augmentation; the trained interrupted view
  retained 322,164 text targets versus 5,851,586 IDLE targets.
- Median response: 7 tokens; 65.7% have at most 8; 99th percentile: 24 tokens.
- Median user audio: 3.007 seconds; maximum: 7.86 seconds in train.
- Only 29/4,000 exact dev prompts occur in train; none of the original three do.

This is a small amount of lexical supervision for adapting the language
decoder to a substantially different input representation. It contains short
single-turn instructions rather than long explanations or multi-turn dialogue.
It cannot establish retained general reasoning ability.

However, these individual recordings have enough meaning to ask their simple
questions. Through native audio chat with adapters disabled, the base model
gave a coherent kitchen-tools answer containing knife, cutting board and spoon.
Other native answers had tone interpretation/refusal issues, so the native
path is not perfect either. We have no evidence that extending these particular
recordings would repair the malformed subwords.

## 4. Training loss hides the hard part of the task

Across the same 32 dev rows, text NLL improves **1.267 -> 1.003** from step
1,500 to 1,919, and lexical accuracy improves **70.7% -> 77.3%**. Both
checkpoints exactly match 2/12 streamed references. One additional final
answer differs only in punctuation (`This is disappointing!`) and is content
correct. Thus exact match should not be equated with semantic failure.
The other reported malformed/repeated words are substantive errors.

In a separate 31-row decomposition, alphanumeric-containing target tokens
(including function words, not only nouns) have NLL **1.540**, accuracy
**65.4%**. Separators have NLL **0.180**, accuracy **95.6%**. Easy punctuation
and list structure make aggregate text metrics look better than content-word
generation. First-token accuracy is 77.4%, close to 76.4% for remaining
tokens: this is not exclusively a first-token/start-boundary problem.

Full dev text loss 1.072 and control losses 0.031/0.054/0.260 are different
prediction problems. Repeated IDLE and two boundary classes are easier than
selecting lexical tokens from the vocabulary. Despite its low count, text
accounts for **82.6% of the final dev weighted-loss numerator**, so its high
loss is not evidence that late training ignored text because of the weights.

Teacher forcing supplies correct earlier words. Free generation feeds errors
back as inputs, and does not penalize future repetition until a rollout is
evaluated. Likelihood-trained models can retain repetitive generation despite
reasonable token losses; this is documented in
[Welleck et al., ICLR 2020](https://arxiv.org/abs/1908.04319).

## 5. No reproduced cache bug; fusion calibration remains a design concern

Using identical audio vectors, gold previous events and position IDs, cached
and full-forward argmax agree at **34/34** lexical positions in the three
original dev examples. Maximum BF16 logit difference is 0.1875. Both paths
make the same content mistakes; this check does not cover every possible input.
Forcing correct START/STOP timing still produces malformed and repeated text.
Forcing just the correct first token does not repair the remaining response.

For example, even with gold previous `B`, the animals model favors `ions`
over `ears`; its first-token preferences favor `T`/`L`. Malformed words are
already visible in the learned conditional distributions, not merely in a
tokenizer display routine. Removing previous lexical inputs worsens NLL, so
the model does use text history, but that history is insufficient to constrain
its word choices reliably.

Audio silence is not a zero embedding. At response token positions in these
examples, audio feature norms are **35.9–39.5**, versus **0.958–0.979** for
active previous-text embeddings. Unlike Qwen's native chat sequence, the
duplex path adds this vector at each text step and uses a different prefix.
The base model's duplex text NLL is 11.875 over the 32-row cohort even though
its native answers are coherent. There is substantial interface learning to do
before training, independent of later forgetting.

Changing only response audio scaling at inference to 0.5 slightly improves
the three text losses but does not repair their predicted content. Reducing
it to 0.025 worsens two of three; zeroing it worsens all three to NLL above 5.
The adapter has adapted to this representation. Large norm ratios are a
measured design concern, not proof that normalization alone fixes the model.
A controlled training ablation would be needed to establish that cause.

## Practical next experiments

Keep the corrected empty-context default. Choose checkpoints using streamed
semantic correctness and a native retention gate alongside text/control NLL.
The untested 200–1,500 interval may contain a better tradeoff; checkpoints
around 400/800/1,200 would resolve that rather than assuming one exists.
Test a smaller/decaying learning rate with those gates, fewer decoder LoRA
projections, and calibrated audio fusion during training as separate ablations.
Oversampling the existing longer TASTE responses can test whether richer
lexical supervision improves conditional word continuity. Keep the Thinker,
fixed chunk/frame contract, additive fusion, weighted next-event objective,
synthetic interruptions and text-only output throughout.

These changes were not applied to the completed one-epoch weights. Only the
confirmed inference-prefix bug was fixed. The experiments do not establish a
new reliable checkpoint, an optimal training duration, or a fix from a single
learning-rate change.

## Reproduction and artifacts

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -u scripts/diagnose_taste_one_epoch.py --output outputs/one-epoch-diagnosis-eos
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -u scripts/run_taste_diagnostic_pilot.py --output outputs/diagnostic-pilot-lr2e4 --learning-rate 0.0002 --steps 200
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -u scripts/run_taste_diagnostic_pilot.py --output outputs/diagnostic-pilot-lr5e5 --learning-rate 0.00005 --steps 200
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -u scripts/diagnose_taste_one_epoch.py --compact --dev-samples 16 --stream-samples 6 --output outputs/one-epoch-diagnosis-pilots --adapter outputs/diagnostic-pilot-lr2e4/checkpoint-50 --adapter outputs/diagnostic-pilot-lr2e4/final --adapter outputs/diagnostic-pilot-lr5e5/checkpoint-50 --adapter outputs/diagnostic-pilot-lr5e5/final
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -u scripts/probe_duplex_fusion.py --output outputs/one-epoch-fusion-probe-complete
.venv/bin/python generate.py --config configs/turn_packed_one_epoch_a100.yaml --audio outputs/one-epoch-a100/train-read_aloud_047388-user.wav --trace outputs/one-epoch-diagnosis-eos/fixed-cli-trace.jsonl
.venv/bin/python -m pytest -q
```

Output directories already exist; choose fresh ones when reproducing.
Raw JSONL probes/logs and early adapters are saved in the named ignored output
folders. `outputs/one-epoch-diagnosis-eos/research-summary.json` joins the matched
comparisons, prefix results and token decomposition; `data-statistics.json`
records the full data scan; `fixed-cli.log` and `fixed-cli-trace.jsonl` record
the corrected real CLI run. Partial exploratory runs in the older diagnosis
and fusion folders were superseded by the completed folders named above.
