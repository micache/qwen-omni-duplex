"""Correct saved duplex answers offline; keep original baseline scores untouched."""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
import re

def identity(row):
    return hashlib.sha256(json.dumps([row["prompt"], row["response"]],
        ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def choice_labels(response, letters="ABCD"):
    """Find explicit selections without treating the article 'a' as choice A."""
    text = response.replace("**", "").replace("`", "")
    label = rf"([{letters}])"
    end = r"(?![A-Za-z])"
    patterns = [
        rf"\b(?i:answer|choice|option)(?:\s+(?i:to\s+(?:the\s+)?(?:question|multiple.choice question)))?\s*(?i:is|would be|should be|=|:)\s*:?\s*(?i:option\s*)?[\[(' \" ]*{label}{end}",
        rf"\b(?i:select|choose|pick|selecting|choosing|selected)\s+(?i:option\s+)?[\[(' \" ]*{label}{end}",
        rf"\b(?i:is|be)\s+(?i:therefore\s+)?(?i:option)\s*[\[(' \" ]*{label}{end}",
        rf"\b(?i:option)\s*[\[(' \" ]*{label}{end}[\])]*\s+(?i:is\s+(?:the\s+)?(?:correct|best|right))",
        rf"\\boxed\s*\{{\s*(?:\\(?:text|mathrm)\s*\{{\s*)?{label}\s*\}}",
        rf"\b(?i:(?:final\s+)?answer)\s*:\s*[\[(' \" ]*{label}{end}",
        rf"\b{label}\s+(?i:is\s+(?:the\s+)?(?:correct|best|right)(?:\s+answer)?)\b",
    ]
    selected = [m.group(1) for p in patterns for m in re.finditer(p, text)]
    # A selected list is not a single-choice answer. Do not confuse an
    # explanatory list of options elsewhere in the response with this list.
    list_end = end + r"(?!\.[A-Z])"  # B.F. Skinner is answer text, not another choice.
    for m in re.finditer(rf"\b(?i:answers?|choices?|options?)\s+(?i:is|are)\s*:?\s*({label}{list_end}(?:\s*(?:,|(?i:and|or))\s*{label}{list_end})+)", text):
        selected.extend(re.findall(label, m.group(1)))
    start = re.match(rf"\s*[\[(' \" ]*{label}{end}(?:[\]).:,]|\s*$)", text)
    if start:
        selected.append(start.group(1))
    final = text.strip().split("\n\n")[-1]
    last = re.match(rf"\s*[\[(' \" ]*{label}{end}(?:[\]).:,]|\s*$)", final)
    if last:
        selected.append(last.group(1))
    direct = re.findall(rf"\b{label}{end}(?=[.,:])", text)
    if len(text) < 500 and len(set(direct)) == 1:
        selected.extend(direct)
    for m in re.finditer(rf"\({label}\)(?:[.!]?\s*$)", text):
        selected.append(m.group(1))
    lower = re.fullmatch(rf"\s*[([]?([{letters.lower()}])[)\]]?[.!]?\s*", text)
    if lower:
        selected.append(lower.group(1).upper())
    return sorted(set(selected))


def mcq_answer(row):
    labels = choice_labels(row["response"])
    if len(labels) > 1:
        return None, "conflicting_answer_labels"
    if not labels:
        return None, "missing_valid_choice_label"
    return labels[0], "explicit_choice"


def bbh_answer(row):
    text = row["response"].replace("**", "").replace("`", "")
    if "hyperbaton" in row["id"]:
        labels = choice_labels(text, "AB")
        return (labels[0].lower(), "explicit_choice") if len(labels) == 1 else (None, "missing_or_conflicting_choice")
    # Ordinary punctuation variants remain valid. Opposing answer declarations
    # in the same output are not resolved by picking the first or last one.
    matches = re.findall(r"\b(?:the\s+)?(?:final\s+)?answer\s+(?:is|would be)\s*:?\s*(yes|no)\b", text, re.I)
    lead = re.match(r"\s*(yes|no)\b", text, re.I)
    if lead:
        matches.append(lead.group(1))
    matches.extend(re.findall(r"(?:\b|(?<=[a-z])(?=[YN]))((?i:yes|no))\s*,\s*(?:you|the sentence|it|[A-Z][a-z]+)\b", text))
    answers = set(s.lower() for s in matches)
    if len(answers) != 1:
        return None, "missing_or_conflicting_yes_no"
    return next(iter(answers)), "explicit_yes_no"


def main():
    from score_voicebench_suite import RandomFallback, official_modules, read

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=("openbookqa", "mmsu", "bbh", "ifeval", "advbench"))
    parser.add_argument("--output", type=Path, default=Path("outputs/voicebench-response-gain003-silence"))
    parser.add_argument("--upstream", type=Path, default=Path("outputs/voicebench-upstream"))
    parser.add_argument("--finalize", action="store_true", help="Require hash-bound decisions for every ambiguous scoring case")
    args = parser.parse_args()
    root, config = args.output, args.config
    import nltk
    nltk.data.path.insert(0, str((root / "nltk_data").resolve()))
    from langdetect import DetectorFactory
    DetectorFactory.seed = 17
    mods = official_modules(args.upstream)
    original = json.loads((root / "scores.json").read_text())
    review = root / "review"
    review.mkdir(exist_ok=True)
    decision_path = review / "duplex-decisions.json"
    decisions = json.loads(decision_path.read_text()) if decision_path.exists() else {}
    rows = [r for f in sorted((root / "duplex").glob(config + "--*.jsonl")) for r in read(f)]
    rows.sort(key=lambda r: (r["voicebench_id"].split(":")[1], int(r["voicebench_id"].split(":")[-1])))
    assert len(rows) == len({r["voicebench_id"] for r in rows}) == original["models"]["duplex"][config]["n"]
    results, queue = [], []
    fallback = random.Random(17)
    ifeval_checks = {}
    if config == "ifeval":
        # The official implementation runs every strict check before any loose
        # check. Some descriptions use seeded random defaults; preserve order.
        random.seed(17)
        inputs = mods["ifeval"].read_prompt_list(rows)
        mapping = {r["prompt"]: r["response"] for r in rows}
        strict_outputs = [mods["ifeval"].test_instruction_following_strict(inp, mapping) for inp in inputs]
        loose_outputs = [mods["ifeval"].test_instruction_following_loose(inp, mapping) for inp in inputs]
        ifeval_checks = {r["voicebench_id"]: {"strict": s.follow_instruction_list, "loose": l.follow_instruction_list}
                        for r, s, l in zip(rows, strict_outputs, loose_outputs)}
    adv = {r["voicebench_id"]: r for r in read(review / "advbench-review.jsonl") if r["model"] == "duplex"} if config == "advbench" else {}
    for row in rows:
        ident = row["voicebench_id"]
        checks = None
        if config == "advbench":
            item = adv[ident]
            assert item["prompt_response_sha256"] == identity(row)
            correct, reason, prediction = item["reviewed_correct"], item["reason"], None
            old = item["official_keyword_correct"]
            official_correct = old
        elif config in {"mmsu", "openbookqa"}:
            old_pred = mods["mcq"].MCQEvaluator().extract_answer(row["response"])
            old = old_pred == row["reference"]
            official_correct = (old_pred if old_pred is not None else fallback.choice("ABCD")) == row["reference"]
            prediction, reason = mcq_answer(row)
            correct = prediction == row["reference"]
            if prediction != old_pred or reason == "conflicting_answer_labels":
                queue.append({**row, "old_prediction": old_pred, "prediction": prediction, "reason": reason})
        elif config == "bbh":
            choice = random.choice
            random.choice = lambda _: (_ for _ in ()).throw(RandomFallback())
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    try:
                        old_pred = mods["bbh"].BBHEvaluator().extract_answer(row["response"], row["id"])
                    except RandomFallback:
                        old_pred = None
            finally:
                random.choice = choice
            truth = row["reference"].strip("()").lower()
            old = old_pred == {"yes": 1, "no": 0, "a": 0, "b": 1}[truth]
            official_correct = (old_pred if old_pred is not None else fallback.choice([0, 1])) == {"yes": 1, "no": 0, "a": 0, "b": 1}[truth]
            prediction, reason = bbh_answer(row)
            correct = prediction == truth
            if prediction is None or old != correct:
                queue.append({**row, "old_prediction": old_pred, "prediction": prediction, "reason": reason})
        else:
            checks = ifeval_checks[ident]
            old = all(checks["strict"])
            correct, reason, prediction = old, "official_instruction_checks", None
            official_correct = old
            if any(checks["strict"]) or any(checks["loose"]):
                queue.append({**row, "strict": checks["strict"], "loose": checks["loose"]})
        original_checks = checks
        if ident in decisions:
            d = decisions[ident]
            assert d["sha256"] == identity(row), f"Decision for a different answer: {ident}"
            correct, reason = d["correct"], d["reason"]
            if checks is not None and not d["correct"] and d.get("invalid_response"):
                checks = {k: [False] * len(v) for k, v in checks.items()}
        results.append({"voicebench_id": ident, "sha256": identity(row), "prediction": prediction,
            "correct": bool(correct), "reason": reason, "old_no_guess_correct": bool(old), "checks": checks,
            "official_correct": bool(official_correct),
            "original_checks": original_checks,
            "reviewed": ident in decisions or config == "advbench"})
    result = {"n": len(rows), "correct": sum(r["correct"] for r in results),
        "score": 100 * sum(r["correct"] for r in results) / len(rows),
        "official_score": original["models"]["duplex"][config]["score"],
        "recovered_vs_no_guess": sum(r["correct"] and not r["old_no_guess_correct"] for r in results),
        "removed_vs_no_guess": sum(not r["correct"] and r["old_no_guess_correct"] for r in results),
        "recovered_vs_official": sum(r["correct"] and not r["official_correct"] for r in results),
        "removed_vs_official": sum(not r["correct"] and r["official_correct"] for r in results),
        "reasons": dict(Counter(r["reason"] for r in results)), "semantic_decisions": sum(r["reviewed"] for r in results)}
    if config == "ifeval":
        original_sub = [sum(all(r["original_checks"][kind]) for r in results) / len(results) for kind in ("strict", "loose")]
        original_sub += [sum(sum(r["original_checks"][kind]) for r in results) / sum(len(r["original_checks"][kind]) for r in results) for kind in ("strict", "loose")]
        assert abs(100 * sum(original_sub) / 4 - result["official_score"]) < 1e-8, "Official IFEval reproduction failed"
        sub = {f"{kind}-prompt": sum(all(r["checks"][kind]) for r in results) / len(results) for kind in ("strict", "loose")}
        sub.update({f"{kind}-instruction": sum(sum(r["checks"][kind]) for r in results) / sum(len(r["checks"][kind]) for r in results) for kind in ("strict", "loose")})
        result["submetrics"] = sub
        result["score"] = 100 * sum(sub.values()) / len(sub)
    else:
        assert abs(100 * sum(r["official_correct"] for r in results) / len(rows) - result["official_score"]) < 1e-8, "Official score reproduction failed"
    unresolved = [r["voicebench_id"] for r in queue if r["voicebench_id"] not in decisions]
    result["unresolved"] = unresolved
    result["status"] = "verified" if args.finalize and not unresolved else "screened"
    if args.finalize and unresolved:
        raise ValueError(f"Unreviewed cases: {unresolved}")
    report_path = review / "corrected-scores.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"baseline_unchanged": True, "duplex": {}}
    report["duplex"][config] = result
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    (review / (config + "-decisions.jsonl")).write_text("".join(json.dumps(r) + "\n" for r in results))
    (review / (config + "-queue.jsonl")).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in queue))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
