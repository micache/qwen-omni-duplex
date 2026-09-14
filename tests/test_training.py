from pathlib import Path
import pytest
import torch
from torch import nn

from duplex.training import (
    PROJECTION_SUFFIXES,
    assert_only_allowed_lora_trainable,
    discover_text_decoder_lora_targets,
    load_training_config,
)


class TinyProjectionGroup(nn.Module):
    def __init__(self, names):
        super().__init__()
        for name in names:
            setattr(self, name, nn.Linear(2, 2, bias=False))


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyProjectionGroup(PROJECTION_SUFFIXES[:4])
        self.mlp = TinyProjectionGroup(PROJECTION_SUFFIXES[4:])


class TinyThinker(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLayer(), TinyLayer()])
        self.audio_tower = nn.Module()
        self.audio_tower.q_proj = nn.Linear(2, 2)


def test_discovery_uses_full_text_decoder_paths_and_excludes_audio() -> None:
    targets = discover_text_decoder_lora_targets(TinyThinker())

    assert len(targets) == 14
    assert targets[0] == "model.layers.0.self_attn.q_proj"
    assert targets[-1] == "model.layers.1.mlp.down_proj"
    assert not any("audio" in target for target in targets)


class TinyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].self_attn = nn.Module()
        self.model.layers[0].self_attn.q_proj = nn.Module()
        self.model.layers[0].self_attn.q_proj.lora_A = nn.ModuleDict(
            {"default": nn.Linear(2, 1, bias=False)}
        )
        self.model.layers[0].self_attn.q_proj.lora_B = nn.ModuleDict(
            {"default": nn.Linear(1, 2, bias=False)}
        )
        self.frozen_base = nn.Parameter(torch.ones(1), requires_grad=False)


def test_trainable_allowlist_accepts_only_targeted_lora_parameters() -> None:
    adapter = TinyAdapter()
    names = assert_only_allowed_lora_trainable(
        adapter, ["model.layers.0.self_attn.q_proj"]
    )
    assert names == (
        "model.layers.0.self_attn.q_proj.lora_A.default.weight",
        "model.layers.0.self_attn.q_proj.lora_B.default.weight",
    )

    adapter.frozen_base.requires_grad_(True)
    with pytest.raises(RuntimeError, match="outside the LoRA allowlist"):
        assert_only_allowed_lora_trainable(
            adapter, ["model.layers.0.self_attn.q_proj"]
        )


@pytest.mark.parametrize(
    ("name", "method", "load_in_4bit", "label"),
    [
        ("train_lora.yaml", "lora", False, "LoRA"),
        ("train_qlora_16gb.yaml", "qlora", True, "QLoRA"),
        ("debug.yaml", "lora", False, "LoRA"),
    ],
)
def test_checked_in_training_configs_are_explicit(
    name, method, load_in_4bit, label
) -> None:
    path = Path(__file__).parents[1] / "configs" / name
    config = load_training_config(path)

    assert config["training"]["method"] == method
    assert config["training"]["run_label"] == label
    assert config["model"]["load_in_4bit"] is load_in_4bit
    assert config["lora"]["rank"] == 16
    assert config["training"]["auto_find_batch_size"] is False


def test_qlora_cannot_be_activated_by_lora_config() -> None:
    path = Path(__file__).parents[1] / "configs" / "debug.yaml"
    import yaml

    config = yaml.safe_load(path.read_text())
    config["model"]["load_in_4bit"] = True
    temporary = path.parent / ".invalid-session08-test.yaml"
    temporary.write_text(yaml.safe_dump(config))
    try:
        with pytest.raises(ValueError, match="requires explicit"):
            load_training_config(temporary)
    finally:
        temporary.unlink()
