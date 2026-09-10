from duplex import model


def test_model_scope_is_frozen() -> None:
    assert model.MODEL_ID == "Qwen/Qwen2.5-Omni-3B"
    assert model.MODEL_COMPONENT == "thinker"
    assert model.FUSION == "additive_audio_text_control"
    assert model.OUTPUT == "text_only"
