"""VoiceBench's three-vote rubric, using the requested GPT-4o judge.

The key is read from an owner-only runtime file and never written to artifacts.
Only the benchmark transcription and the decoded answer reach the judge.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.request

JUDGED = {"commoneval", "alpacaeval_full", "wildvoice", "sd-qa"}


def rubric_constants(path):
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {"meta_prompt_open", "meta_prompt_qa"}:
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "strip":
                result[node.targets[0].id] = ast.literal_eval(value.func.value).strip()
            else:
                result[node.targets[0].id] = ast.literal_eval(value)
    if len(result) != 2:
        raise ValueError("Cannot locate pinned upstream rubrics")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/voicebench-response-gain003"))
    parser.add_argument("--upstream", type=Path, default=Path("outputs/voicebench-upstream"))
    parser.add_argument("--key-file", type=Path, default=Path("/tmp/qwen-voicebench-openai-key"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-judge-tokens", type=int, default=16)
    parser.add_argument("--tokens-per-minute", type=int, default=27000)
    args = parser.parse_args()
    if args.key_file.stat().st_mode & 0o077:
        raise PermissionError("The runtime key file must be owner-only")
    key = args.key_file.read_text().strip()
    rubrics = rubric_constants(args.upstream / "api_judge.py")
    (args.output / "judge-protocol.json").write_text(json.dumps({"requested_model": "gpt-4o",
        "temperature": 0.5, "top_p": 0.95, "n": 3, "max_tokens": args.max_judge_tokens,
        "upstream_max_tokens": 1024, "tokens_per_minute_target": args.tokens_per_minute,
        "rubric_sha256": hashlib.sha256(json.dumps(rubrics, sort_keys=True).encode()).hexdigest(),
        "upstream_revision": "b56154172f2a57a43d29005de7d0471d748d70d1",
        "system": "You are a helpful assistant who tries to help answer the user's question."}, indent=2))
    rate_lock, error_lock = threading.Lock(), threading.Lock()
    next_request = time.monotonic()

    def pace(messages):
        nonlocal next_request
        # Conservative byte estimate, plus all three one-word completions.
        estimated = len(json.dumps(messages, ensure_ascii=False).encode()) / 3 + 3 * args.max_judge_tokens
        with rate_lock:
            scheduled = max(time.monotonic(), next_request)
            next_request = scheduled + estimated * 60 / args.tokens_per_minute
        time.sleep(max(0, scheduled - time.monotonic()))

    def error_log(item, attempt, status):
        with error_lock, (args.output / "judge-errors.jsonl").open("a") as handle:
            handle.write(json.dumps({"voicebench_id": item["voicebench_id"], "attempt": attempt,
                                    **status}) + "\n")

    def request(item):
        qa = "reference" in item
        prompt = rubrics["meta_prompt_qa" if qa else "meta_prompt_open"]
        prompt = prompt.replace("{prompt}", item["prompt"])
        if qa:
            prompt = prompt.replace("{reference}", item["reference"])
        prompt = prompt.replace("{response}", item["response"])
        payload = {"model": "gpt-4o", "messages": [
            {"role": "system", "content": "You are a helpful assistant who tries to help answer the user's question."},
            {"role": "user", "content": prompt}], "max_tokens": args.max_judge_tokens, "frequency_penalty": 0,
            "presence_penalty": 0, "stop": None, "temperature": 0.5, "top_p": 0.95, "n": 3}
        for attempt in range(10):
            try:
                pace(payload["messages"])
                call = urllib.request.Request("https://api.openai.com/v1/chat/completions",
                    data=json.dumps(payload).encode(), headers={"Authorization": "Bearer " + key,
                    "Content-Type": "application/json"})
                with urllib.request.urlopen(call, timeout=90) as response:
                    body = json.load(response)
                    request_id = response.headers.get("x-request-id")
                scores = [c["message"]["content"].strip() for c in body["choices"]]
                if len(scores) != 3:
                    raise ValueError("Incorrect number of judge votes")
                if qa:
                    # Upstream retains raw QA votes, including punctuation or
                    # explanations; its literal majority scorer handles them.
                    valid = all(bool(s) for s in scores)
                else:
                    def rating(s):
                        try:
                            return float(s)
                        except ValueError:
                            match = re.search(r"\[\[(\d+)\]\]", s)
                            return float(match[1]) if match else float("nan")
                    valid = all(1 <= rating(s) <= 5 for s in scores)
                if not valid:
                    raise ValueError("Invalid judge vote format")
                return {**item, "score": scores, "judge_model": body["model"],
                    "judge_usage": body["usage"], "judge_request_id": request_id,
                    "judge_max_tokens": args.max_judge_tokens}
            except urllib.error.HTTPError as error:
                try:
                    code = json.loads(error.read())["error"].get("code")
                except Exception:
                    code = "unavailable"
                error_log(item, attempt, {"http_status": error.code, "error_code": code})
                # Deliberately exclude server message text and headers from errors/logs.
                if error.code in {401, 403} or code in {"insufficient_quota", "invalid_api_key"}:
                    return {"judge_unavailable": True, "http_status": error.code, "error_code": code}
                if error.code not in {429, 500, 502, 503, 504}:
                    return {"judge_failed": True, "http_status": error.code, "error_code": code}
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as error:
                error_log(item, attempt, {"error_type": type(error).__name__})
            time.sleep(min(2 ** attempt, 30))
        return {"judge_failed": True, "error_code": "retry_limit"}

    count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while True:
            pending = []
            for mode in ("duplex", "base"):
                destination = args.output / "judged" / mode
                destination.mkdir(parents=True, exist_ok=True)
                for source in sorted((args.output / mode).glob("*.jsonl")):
                    if source.name.split("--")[0] not in JUDGED:
                        continue
                    target = destination / source.name
                    seen = {json.loads(line)["voicebench_id"] for line in target.read_text().split("\n") if line.strip()} if target.exists() else set()
                    # Inference may be appending a large UTF-8 record right now.
                    # Consume only physically newline-terminated complete records.
                    for line in source.read_bytes().split(b"\n")[:-1]:
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        if item["voicebench_id"] not in seen:
                            pending.append((target, item))
            if not pending:
                if all((args.output / mode / "complete.json").exists() for mode in ("base", "duplex")):
                    break
                time.sleep(10)
                continue
            # Keep the request queue bounded while inference continues producing answers.
            for start in range(0, len(pending), 64):
                futures = {pool.submit(request, item): target for target, item in pending[start:start + 64]}
                failure = None
                for future in as_completed(futures):
                    result = future.result()
                    if result.get("judge_unavailable") or result.get("judge_failed"):
                        (args.output / "judge-status.json").write_text(json.dumps(result))
                        print(json.dumps(result), flush=True)
                        failure = result
                        continue
                    with futures[future].open("a") as handle:
                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    count += 1
                if failure:
                    return
                print(f"judged {count} new answers", flush=True)
                (args.output / "judge-status.json").write_text(json.dumps({"available": True, "new_answers": count}))
    (args.output / "judge-complete.json").write_text(json.dumps({"new_answers": count, "model": "gpt-4o"}))
    print("GPT-4o judging complete", flush=True)


if __name__ == "__main__":
    main()
