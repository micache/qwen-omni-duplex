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
