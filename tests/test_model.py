from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from duplex import model
from duplex.model import NextEventLossWeights, QwenDuplexThinker
from duplex.timeline import IGNORE_LABEL, ControlTokenIds


VOCAB_SIZE = 9
HIDDEN_SIZE = 3
IDLE = 6
START = 7
STOP = 8


def qwen_config(*, seconds: int = 2, positions_per_second: int = 25):
    return SimpleNamespace(
        thinker_config=SimpleNamespace(
            seconds_per_chunk=seconds,
            position_id_per_seconds=positions_per_second,
            text_config=SimpleNamespace(vocab_size=VOCAB_SIZE),
        ),
        talker_config=SimpleNamespace(
            tts_text_pad_token_id=IDLE,
            tts_text_start_token_id=START,
            tts_text_end_token_id=STOP,
        ),
    )


class TinyAudioTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output_lengths = None
        self.register_buffer("flattened", torch.empty(0, HIDDEN_SIZE))
        self.calls = 0

    def _get_feat_extract_output_lengths(self, input_lengths):
        output_lengths = (
            input_lengths
            if self.output_lengths is None
            else self.output_lengths.to(input_lengths.device)
        )
        return input_lengths, output_lengths

    def forward(self, input_features, *, feature_attention_mask):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=self.flattened)


class TinyCausalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_call = None

    def forward(self, **kwargs):
        self.last_call = kwargs
        return SimpleNamespace(
            last_hidden_state=kwargs["inputs_embeds"],
            past_key_values="next-cache",
            hidden_states=(kwargs["inputs_embeds"],),
            attentions=None,
        )


class TinyThinker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(vocab_size=VOCAB_SIZE)
        )
        self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.audio_tower = TinyAudioTower()
        self.model = TinyCausalModel()
        self.lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.arange(VOCAB_SIZE * HIDDEN_SIZE, dtype=torch.float32).reshape(
                    VOCAB_SIZE, HIDDEN_SIZE
                )
                / 10
            )
            self.lm_head.weight.copy_(
                torch.arange(VOCAB_SIZE * HIDDEN_SIZE, dtype=torch.float32).reshape(
                    VOCAB_SIZE, HIDDEN_SIZE
                )
                / 20
            )

    def get_input_embeddings(self):
        return self.embedding

    def get_audio_features(
        self, *, input_features, feature_attention_mask, return_dict
    ):
        assert return_dict is True
        return self.audio_tower(
            input_features, feature_attention_mask=feature_attention_mask
        )


def wrapper(*, weights=None) -> QwenDuplexThinker:
    return QwenDuplexThinker(
        TinyThinker(), qwen_config=qwen_config(), loss_weights=weights
    )


def no_audio_inputs(length: int = 4):
    return {
        "text_ids": torch.tensor([[1, 2, 3, 4]])[:, :length],
        "text_mask": torch.tensor([[True, True, True, True]])[:, :length],
        "control_ids": torch.zeros((1, length), dtype=torch.long),
        "control_mask": torch.zeros((1, length), dtype=torch.bool),
        "attention_mask": torch.ones((1, length), dtype=torch.bool),
    }


def test_model_scope_is_frozen() -> None:
    assert model.MODEL_ID == "Qwen/Qwen2.5-Omni-3B"
    assert model.MODEL_COMPONENT == "thinker"
    assert model.FUSION == "additive_audio_text_control"
    assert model.OUTPUT == "text_only"


def test_control_ids_and_timeline_constants_come_from_qwen_config() -> None:
    duplex = wrapper()

    assert duplex.control_tokens.idle == IDLE
    assert duplex.control_tokens.start == START
    assert duplex.control_tokens.stop == STOP
    assert duplex.timeline.chunk_seconds == 2
    assert duplex.timeline.frame_rate_hz == 25
    assert duplex.frames_per_chunk == 50
    assert duplex.supports_last_position_only

    with pytest.raises(ValueError, match="25 Hz"):
        QwenDuplexThinker(TinyThinker(), qwen_config=qwen_config(positions_per_second=50))


def test_existing_thinker_control_ids_can_override_talker_ids() -> None:
    controls = ControlTokenIds(3, 4, 5, thinker_vocab_size=VOCAB_SIZE)
    duplex = QwenDuplexThinker(
        TinyThinker(), qwen_config=qwen_config(), control_tokens=controls
    )

    assert duplex.control_tokens == controls


def test_exact_additive_fusion_masks_and_batch_audio_restoration() -> None:
    duplex = wrapper()
    flattened = torch.tensor(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
            [40.0, 41.0, 42.0],
            [50.0, 51.0, 52.0],
        ]
    )
    duplex.thinker.audio_tower.flattened = flattened
    text_ids = torch.tensor([[1, 2, 3], [4, 5, 1]])
    text_mask = torch.tensor([[True, False, False], [False, True, False]])
    control_ids = torch.tensor([[IDLE, START, STOP], [STOP, IDLE, START]])
    control_mask = torch.tensor([[False, True, False], [True, False, True]])
    attention_mask = torch.tensor([[True, True, False], [True, True, True]])

    output = duplex(
        text_ids=text_ids,
        text_mask=text_mask,
        control_ids=control_ids,
        control_mask=control_mask,
        attention_mask=attention_mask,
        input_features=torch.zeros(2, 1, 3),
        feature_attention_mask=attention_mask,
        preconv_feature_lengths=torch.tensor([2, 3]),
    )

    restored_audio = torch.tensor(
        [
            [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [0.0, 0.0, 0.0]],
            [[30.0, 31.0, 32.0], [40.0, 41.0, 42.0], [50.0, 51.0, 52.0]],
        ]
    )
    embedding = duplex.thinker.embedding
    expected = (
        embedding(text_ids) * text_mask.unsqueeze(-1)
        + restored_audio
        + embedding(control_ids) * control_mask.unsqueeze(-1)
    )
    assert torch.equal(duplex.thinker.model.last_call["inputs_embeds"], expected)
    assert torch.equal(output.lexical_hidden_states, expected)
    assert duplex.thinker.audio_tower.calls == 1
    assert output.loss is None
    assert output.group_losses == {}


def test_weighted_unshifted_loss_matches_manual_calculation() -> None:
    weights = NextEventLossWeights(text=2.0, idle=0.5, start=3.0, stop=4.0)
    duplex = wrapper(weights=weights)
    inputs = no_audio_inputs()
    labels = torch.tensor([[1, IDLE, START, STOP]])

    output = duplex(**inputs, labels=labels)

    unreduced = F.cross_entropy(
        output.logits.transpose(1, 2), labels, reduction="none"
    )
    applied = torch.tensor([[2.0, 0.5, 3.0, 4.0]])
    expected = (unreduced * applied).sum() / applied.sum()
    assert torch.allclose(output.loss, expected)
    assert output.loss_weight_sum.item() == pytest.approx(9.5)


def test_all_padding_has_differentiable_zero_loss_and_zero_accounting() -> None:
    duplex = wrapper()
    inputs = no_audio_inputs()
    inputs["text_mask"].zero_()
    inputs["attention_mask"].zero_()
    labels = torch.full((1, 4), IGNORE_LABEL, dtype=torch.long)

    output = duplex(**inputs, labels=labels)

    assert output.loss.item() == 0.0
    assert output.loss.requires_grad
    assert output.loss_weight_sum.item() == 0.0
    assert {name: value.item() for name, value in output.group_losses.items()} == {
        "text": 0.0,
        "idle": 0.0,
        "start": 0.0,
        "stop": 0.0,
    }
    assert sum(value.item() for value in output.target_counts.values()) == 0
    assert sum(value.item() for value in output.prediction_counts.values()) == 0


def test_group_losses_and_target_prediction_counts_cover_valid_positions() -> None:
    duplex = wrapper()
    labels = torch.tensor([[1, 2, IDLE, START, STOP, IGNORE_LABEL]])
    inputs = {
        "text_ids": torch.tensor([[1, 2, 3, 4, 5, 0]]),
        "text_mask": torch.tensor([[True, True, True, True, True, False]]),
        "control_ids": torch.zeros((1, 6), dtype=torch.long),
        "control_mask": torch.zeros((1, 6), dtype=torch.bool),
        "attention_mask": torch.tensor([[True, True, True, True, True, False]]),
    }

    output = duplex(**inputs, labels=labels)
    unreduced = F.cross_entropy(
        output.logits.transpose(1, 2),
        labels,
        reduction="none",
        ignore_index=IGNORE_LABEL,
    )

    assert {name: count.item() for name, count in output.target_counts.items()} == {
        "text": 2,
        "idle": 1,
        "start": 1,
        "stop": 1,
    }
    assert sum(count.item() for count in output.prediction_counts.values()) == 5
    assert torch.allclose(output.group_losses["text"], unreduced[0, :2].mean())
    assert torch.allclose(output.group_losses["idle"], unreduced[0, 2])
    assert torch.allclose(output.group_losses["start"], unreduced[0, 3])
    assert torch.allclose(output.group_losses["stop"], unreduced[0, 4])


def test_cache_position_ids_and_use_cache_are_forwarded() -> None:
    duplex = wrapper()
    past = object()
    position_ids = torch.tensor([[4]])
    inputs = no_audio_inputs(length=1)
    inputs["attention_mask"] = torch.ones((1, 5), dtype=torch.bool)

    output = duplex(
        **inputs,
        position_ids=position_ids,
        past_key_values=past,
        use_cache=True,
        last_position_only=True,
    )

    call = duplex.thinker.model.last_call
    assert call["past_key_values"] is past
    assert call["use_cache"] is True
    assert call["position_ids"] is position_ids
    assert call["attention_mask"] is inputs["attention_mask"]
    assert output.past_key_values == "next-cache"
    assert output.logits.shape == (1, 1, VOCAB_SIZE)
    assert output.lexical_hidden_states.shape == (1, 1, HIDDEN_SIZE)


@pytest.mark.parametrize(
    ("output_lengths", "flattened_rows", "message"),
    [
        ([1, 3], 4, "cannot be aligned"),
        ([2, 3], 4, "Flattened Qwen audio output length"),
    ],
)
def test_unexplained_audio_lengths_fail(output_lengths, flattened_rows, message) -> None:
    duplex = wrapper()
    duplex.thinker.audio_tower.output_lengths = torch.tensor(output_lengths)
    duplex.thinker.audio_tower.flattened = torch.zeros(flattened_rows, HIDDEN_SIZE)
    attention_mask = torch.tensor([[True, True, False], [True, True, True]])

    with pytest.raises(ValueError, match=message):
        duplex(
            text_ids=torch.zeros((2, 3), dtype=torch.long),
            text_mask=torch.zeros((2, 3), dtype=torch.bool),
            control_ids=torch.zeros((2, 3), dtype=torch.long),
            control_mask=torch.zeros((2, 3), dtype=torch.bool),
            attention_mask=attention_mask,
            input_features=torch.zeros(2, 1, 3),
            feature_attention_mask=attention_mask,
            preconv_feature_lengths=torch.tensor([2, 3]),
        )


def test_preconvolution_lengths_and_fusion_masks_are_checked() -> None:
    duplex = wrapper()
    duplex.thinker.audio_tower.flattened = torch.zeros(2, HIDDEN_SIZE)
    inputs = no_audio_inputs(length=2)

    with pytest.raises(ValueError, match="do not match"):
        duplex(
            **inputs,
            input_features=torch.zeros(1, 1, 2),
            feature_attention_mask=torch.ones((1, 2), dtype=torch.bool),
            preconv_feature_lengths=torch.tensor([1]),
        )

    inputs = no_audio_inputs(length=2)
    inputs["text_mask"][0, 1] = True
    inputs["attention_mask"][0, 1] = False
    with pytest.raises(ValueError, match="text_mask enables"):
        duplex(**inputs)


def test_last_position_only_rejects_training_labels() -> None:
    duplex = wrapper()
    inputs = no_audio_inputs(length=2)
    with pytest.raises(ValueError, match="inference-only"):
        duplex(**inputs, labels=torch.tensor([[1, 2]]), last_position_only=True)


def test_last_position_only_keeps_full_lexical_states_and_exact_final_logits() -> None:
    duplex = wrapper()
    inputs = no_audio_inputs(length=3)

    full = duplex(**inputs)
    reduced = duplex(**inputs, last_position_only=True)

    assert reduced.lexical_hidden_states.shape == (1, 3, HIDDEN_SIZE)
    assert reduced.logits.shape == (1, 1, VOCAB_SIZE)
    torch.testing.assert_close(reduced.logits, full.logits[:, -1:, :])
