"""Regression cases for actual VoiceBench extraction errors."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "duplex_rescore", Path(__file__).resolve().parents[1] / "scripts/rescore_voicebench_duplex.py")
rescore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rescore)


@pytest.mark.parametrize("text, expected", [
    ("The answer is a boiling hot mug of tea.", None),
    ("The answer is B, Arctic.", "B"),
    ("The answer is D, B.F. Skinner.", "D"),
    ("A. The correct answer is D.", None),
    ("The correct answer is A, B, and C.", None),
    ("The correct answers are A and D.", None),
    ("B is correct. Evaporation.", "B"),
    ("The answer is **(C)**.", "C"),
    (r"The answer is \boxed{D}.", "D"),
    ("The answer is AThe", None),
    ("", None),
])
def test_choices_are_unambiguous_and_not_articles_or_name_initials(text, expected):
    assert rescore.mcq_answer({"response": text})[0] == expected


@pytest.mark.parametrize("text, expected", [
    ("The answer is: yes.", "yes"),
    ("The answer is The answer is no.", "no"),
    ("Yes, if you follow these instructionsNo, you do not return.", None),
    ("No, you do not return. The answer is yes.", None),
    ("Please provide the instructions.", None),
])
def test_binary_answers_do_not_pick_a_convenient_conflicting_prefix(text, expected):
    assert rescore.bbh_answer({"id": "navigate_1", "response": text})[0] == expected


def test_review_identity_binds_both_question_and_whole_response():
    original = {"prompt": "Question\u2028second line", "response": "The answer is B."}
    assert rescore.identity(original) == rescore.identity(dict(original))
    assert rescore.identity(original) != rescore.identity({**original, "response": "The answer is A."})
    assert rescore.identity(original) != rescore.identity({**original, "prompt": "Another question"})
