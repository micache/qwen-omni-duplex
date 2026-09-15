from __future__ import annotations

from duplex.dataset import SpanMetadata, Split
from duplex.timeline import EventKind, build_window_timeline, augment_synthetic_interruption, InterruptionConfig
from test_timeline_alignment import CharacterTokenizer, assistant, user, record, controls
import random
import torch
from test_model import wrapper, no_audio_inputs


def build(words, users=()):
    source = record(assistant_words=words, user_words=users, duration=10.0)
    metadata = SpanMetadata(source.conversation_id, Split.TRAIN, 0.0, 8.0)
    return build_window_timeline(source, metadata, tokenizer=CharacterTokenizer(),
                                 control_tokens=controls(), thinker_bos_token_id=11,
                                 frame_rate_hz=25)


def test_internal_boundary_keeps_response_once_and_one_bootstrap():
    value = build((assistant("Hello", 1.7, 2.3),), (user("prompt", 0.2, 1.0),))
    kinds = [event.kind for event in value.targets.events]
    assert len(kinds) == 200
    assert kinds.count(EventKind.START) == kinds.count(EventKind.STOP) == 1
    assert kinds.count(EventKind.TEXT) == 5
    assert value.causal.bootstrap_mask.count(True) == 1
    assert value.causal.bootstrap_mask[0]
    assert not any(value.causal.bootstrap_mask[index] for index in (50, 100, 150))
    assert value.causal.control_ids[50] == controls().idle


def test_outer_boundary_omits_partial_response():
    value = build((assistant("Hello", 7.8, 8.2),))
    assert all(event.kind is EventKind.IDLE for event in value.targets.events)


def test_synthetic_interruption_updates_full_sequence():
    value = build((assistant("Hello friend", 1.8, 3.8),),
                  (user("prompt", 0.2, 1.0), user("interrupt", 5.0, 5.5)))
    interrupted = augment_synthetic_interruption(
        value, config=InterruptionConfig(probability=1.0, min_assistant_frames=4),
        rng=random.Random(12), control_tokens=controls(), thinker_bos_token_id=11)
    assert interrupted.interruption is not None
    cut = interrupted.interruption.cut_frame
    assert interrupted.targets.events[cut].kind is EventKind.STOP
    assert interrupted.frames[cut].user_active
    assert interrupted.causal.labels[cut] == controls().stop
    assert len(interrupted.targets.events) == 200


def test_context_positions_are_excluded_from_logits_and_loss():
    value = wrapper()
    inputs = no_audio_inputs(4)
    labels = torch.tensor([[1, 2, 3, 4]])
    baseline = value(**inputs, labels=labels)
    contextual = value(**inputs, labels=labels, context_ids=torch.tensor([[5, 5]]),
                       bootstrap_mask=torch.tensor([[True, False, False, False]]))
    assert contextual.logits.shape == (1, 4, 9)
    assert torch.equal(contextual.logits, baseline.logits)
    assert torch.equal(contextual.loss, baseline.loss)
    assert value.thinker.model.last_call["inputs_embeds"].shape[1] == 6
    assert value.thinker.model.last_call["attention_mask"].shape[1] == 6
