# Design

## Session 01 boundary

The experiment treats duplex behavior as text next-event prediction over fixed
two-second windows. Each valid window contains 50 frames at 25 Hz. The eventual
model path will use only the `Qwen/Qwen2.5-Omni-3B` Thinker and add audio, text,
and control representations before prediction. Controls are exactly `IDLE`,
`START`, and `STOP`; output remains text.

The training view is named DailyTalkContiguous because examples must preserve
conversation continuity. Synthetic interruption supplies the overlap behavior
that ordinary turn-by-turn dialogue lacks. The objective is a weighted
next-event loss so sparse control boundaries can receive deliberate weight.

No tensor shapes, injection layer, weighting schedule, or serialization format
is fixed in Session 01. Those decisions require a model compatibility probe and
real sample inspection in later sessions.

## Session 02 timeline primitive

Control token IDs are read from the supplied Qwen configuration at
`talker_config.tts_text_pad_token_id`, `tts_text_start_token_id`, and
`tts_text_end_token_id`. They are accepted only when they are distinct,
non-negative integers inside `thinker_config.text_config.vocab_size`; the code
has no fallback control IDs.

Targets are represented independently as an event sequence whose kinds are
`IDLE`, `START`, `TEXT`, `STOP`, and `PADDING`. Validation tracks inactive and
active response state. `START` enters the active state, `TEXT` is valid only in
that state, and `STOP` leaves it. `IDLE` is valid in either state and therefore
can occur between text tokens without ending a response. Padding is trailing
only. Errors name the sample and frame.

Encoding produces aligned Python lists for text and control IDs and masks,
target labels and event types, attention, and optional frame times. Position
zero receives the supplied Thinker BOS token. Each later real position receives
only the preceding target event in either the text or control input stream.
Padding uses label `-100` and disables text, control, and attention masks. Event
types describe targets for supervision/inspection; they are not causal model
inputs.

This session does not define tensor shapes beyond one-dimensional event streams,
model injection, loss weights, or an on-disk serialization format.

## Session 04 word/token alignment and frame targets

A Session 04 sample is one `WindowMetadata` crop of a validated
`ConversationRecord`. The returned waveform is the crop of the record's
right/user channel only. The caller passes the nominal frame rate; existing
timeline configuration validation accepts only 25 Hz, yielding 50 target
frames for the fixed two-second crop.

Assistant utterances are maximal assistant speaker runs in the full annotated
conversation. Turn IDs are stable conversation-local IDs, and word provenance
uses the original sidecar alignment index. Only complete assistant runs are
supervised: if a crop begins or ends inside a run, that partial run contributes
`IDLE` targets. This avoids inventing a `START` or `STOP` outside the crop and
keeps every emitted state sequence legal.

Each complete assistant utterance is reconstructed and tokenized as a whole,
not word by word, so punctuation and leading-space tokenization retain their
context. Punctuation is attached deterministically; otherwise one separator
space is inserted between adjacent word annotations, and every such insertion
is recorded on the utterance alignment. Explicit whitespace and Unicode are
preserved. Special tokens are disabled. The complete token sequence must decode
exactly to the reconstructed target with tokenizer cleanup disabled; failures
name the sample, turn, and word indices.

Fast-tokenizer character offsets are preferred. If offsets are unavailable,
alignment uses decoded prefixes of the already-created full-utterance token
sequence. Prefixes are committed only when they are exact target prefixes, so
incomplete multi-byte token groups remain deterministic. No token is discarded,
and the final exact round trip is still mandatory. Each aligned token retains
its ID, inspection decode, character span, source word index and time span, and
source turn ID.

Token events are initially placed from their source-word timing. Collisions are
resolved by a deterministic, strictly increasing schedule over the full valid
assistant-turn frame range. This permits extra BPE pieces to use otherwise free
assistant-time frames. `START` occupies the frame immediately before the first
lexical event and `STOP` the frame immediately after the last. If the turn does
not have enough frames for every token plus those controls, construction raises
an `InvalidTimelineError` with sample, turn, word, time, and capacity details;
events are never overwritten.

All other frames are `IDLE`. Frame inspection metadata retains frame index,
absolute aligned audio time, user-word activity, target event, token decode,
post-event response state, source word index, and source turn ID. Model inputs
are produced exclusively by passing the constructed `TargetEventSequence` to
the Session 02 `encode_causal_timeline` shift, preserving the one-event causal
boundary and preventing target leakage.

## Session 05 synthetic interruption and collation

Synthetic interruption operates on a completed window timeline. Eligible
assistant turns must be followed directly by an annotated user turn and must
leave the configured minimum number of active response frames before a legal
cut. The later user suffix is shifted to that cut, the interrupted assistant
suffix is removed from targets and all alignment/provenance views, and `STOP`
is emitted on the first frame containing shifted user activity. The remainder
of the fixed window is zero/idle filled. All causal streams are regenerated
from the spliced target sequence. Randomness is supplied by the caller;
probability zero and ineligible samples retain object identity.

The training collator accepts explicit conversation/window pairs or completed
timelines. It builds and optionally augments timelines, pads event streams with
ignored labels, and calls only the feature extractor of the Qwen processor
provided by the caller. It returns the extractor's raw feature attention mask
and its pre-convolution lengths. It does not infer restored/flattened audio
lengths and never invokes the audio tower; that conversion remains deferred to
`model.py` until the real interface probe.
