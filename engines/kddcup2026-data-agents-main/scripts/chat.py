"""Terminal chat client for the local Qwen endpoint.

Reads the same env vars the eval container uses (`MODEL_API_URL`,
`MODEL_API_KEY`, `MODEL_NAME`), so pointing it at the DGX Spark vLLM is just:

    export MODEL_API_URL=http://<dgx-lan-ip>:8000/v1
    uv run python scripts/chat.py

Modes
-----
One-shot (positional prompt or stdin pipe):

    uv run python scripts/chat.py "list 5 colors"
    cat prompt.txt | uv run python scripts/chat.py

Interactive REPL (no prompt → drops into a multi-turn loop with history):

    uv run python scripts/chat.py
    > hello
    pong
    > /system You are a CSV expert.
    > /json on
    > {"q": "..."}
    > /exit

Slash commands (REPL only):
    /exit | /quit            leave
    /clear                   wipe chat history (keep system prompt)
    /system <text>           replace system prompt (empty → clear)
    /temp <float>            set temperature
    /max <int>               set max tokens
    /json on|off|toggle      toggle response_format=json_object
    /show                    print current settings + history length
    /help                    list commands
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

try:
    from openai import APIError, OpenAI
except ImportError:  # pragma: no cover
    print("openai package required: pip install 'openai>=1.40' (or `uv run`)", file=sys.stderr)
    raise


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_API_KEY = "local"
DEFAULT_MODEL = "qwen3.5-35b-a3b"


def make_client() -> tuple[OpenAI, str, str]:
    base_url = os.environ.get("MODEL_API_URL", DEFAULT_BASE_URL)
    api_key = os.environ.get("MODEL_API_KEY", DEFAULT_API_KEY)
    model = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    return OpenAI(base_url=base_url, api_key=api_key), model, base_url


def stream_completion(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> tuple[str, dict[str, Any] | None]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    pieces: list[str] = []
    usage: dict[str, Any] | None = None
    stream = client.chat.completions.create(**kwargs)
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            piece = chunk.choices[0].delta.content
            pieces.append(piece)
            sys.stdout.write(piece)
            sys.stdout.flush()
        if getattr(chunk, "usage", None):
            usage = chunk.usage.model_dump()
    sys.stdout.write("\n")
    sys.stdout.flush()
    return "".join(pieces), usage


def one_shot(
    prompt: str,
    *,
    system: str | None,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    quiet: bool,
) -> int:
    client, model, base_url = make_client()
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    if not quiet:
        print(
            f"[chat] endpoint={base_url} model={model} json={json_mode} temp={temperature}",
            file=sys.stderr,
        )
    t0 = time.perf_counter()
    try:
        _, usage = stream_completion(
            client,
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
    except APIError as exc:
        print(f"[chat] API error: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - t0
    if not quiet:
        print(f"[chat] {elapsed:.2f}s · usage={usage}", file=sys.stderr)
    return 0


_HELP_TEXT = (
    "/exit | /quit             leave\n"
    "/clear                    wipe history (keep system prompt)\n"
    "/system <text>            replace system prompt (empty → clear)\n"
    "/temp <float>             set temperature\n"
    "/max <int>                set max tokens\n"
    "/json on|off|toggle       toggle response_format=json_object\n"
    "/show                     print current settings + history length\n"
    "/help                     this listing"
)


def repl(
    *,
    system: str | None,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> int:
    client, model, base_url = make_client()
    print(f"[chat] endpoint={base_url} model={model}", file=sys.stderr)
    print("[chat] type /help for commands, ctrl+d to exit", file=sys.stderr)

    history: list[dict[str, str]] = []
    if system:
        history.append({"role": "system", "content": system})

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input[1:].split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("exit", "quit", "q"):
                return 0
            if cmd == "help":
                print(_HELP_TEXT, file=sys.stderr)
                continue
            if cmd == "clear":
                history = [m for m in history if m["role"] == "system"]
                print("[chat] history cleared", file=sys.stderr)
                continue
            if cmd == "system":
                history = [m for m in history if m["role"] != "system"]
                if arg:
                    history.insert(0, {"role": "system", "content": arg})
                    print(f"[chat] system prompt set ({len(arg)} chars)", file=sys.stderr)
                else:
                    print("[chat] system prompt cleared", file=sys.stderr)
                continue
            if cmd == "temp":
                try:
                    temperature = float(arg)
                    print(f"[chat] temperature={temperature}", file=sys.stderr)
                except ValueError:
                    print("[chat] usage: /temp <float>", file=sys.stderr)
                continue
            if cmd == "max":
                try:
                    max_tokens = int(arg)
                    print(f"[chat] max_tokens={max_tokens}", file=sys.stderr)
                except ValueError:
                    print("[chat] usage: /max <int>", file=sys.stderr)
                continue
            if cmd == "json":
                arg_lower = arg.lower()
                if arg_lower in ("on", "true", "1"):
                    json_mode = True
                elif arg_lower in ("off", "false", "0"):
                    json_mode = False
                elif arg_lower in ("", "toggle"):
                    json_mode = not json_mode
                else:
                    print("[chat] usage: /json on|off|toggle", file=sys.stderr)
                    continue
                print(f"[chat] json_mode={json_mode}", file=sys.stderr)
                continue
            if cmd == "show":
                sys_msg = next((m["content"] for m in history if m["role"] == "system"), None)
                turns = sum(1 for m in history if m["role"] != "system")
                print(
                    f"[chat] model={model} temp={temperature} max_tokens={max_tokens} "
                    f"json={json_mode} turns={turns} system={sys_msg!r}",
                    file=sys.stderr,
                )
                continue
            print(f"[chat] unknown command: /{cmd} (try /help)", file=sys.stderr)
            continue

        history.append({"role": "user", "content": user_input})
        try:
            t0 = time.perf_counter()
            content, usage = stream_completion(
                client,
                model,
                history,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
            elapsed = time.perf_counter() - t0
        except APIError as exc:
            history.pop()  # don't keep an unanswered turn
            print(f"[chat] API error: {exc}", file=sys.stderr)
            continue

        history.append({"role": "assistant", "content": content})
        print(f"[chat] {elapsed:.2f}s · usage={usage}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Terminal chat for an OpenAI-compatible Qwen endpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Env (matches the eval contract):\n"
            f"  MODEL_API_URL  default {DEFAULT_BASE_URL}\n"
            f"  MODEL_API_KEY  default {DEFAULT_API_KEY}\n"
            f"  MODEL_NAME     default {DEFAULT_MODEL}\n"
        ),
    )
    ap.add_argument("prompt", nargs="*", help="One-shot prompt (omit for REPL).")
    ap.add_argument("--system", help="System prompt (one-shot or REPL initial).")
    ap.add_argument(
        "--json",
        action="store_true",
        help="Force response_format={'type':'json_object'}.",
    )
    ap.add_argument("--temperature", "-t", type=float, default=0.0)
    ap.add_argument("--max-tokens", "-m", type=int, default=1024)
    ap.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="One-shot only: suppress the [chat] header/footer (just stream the answer).",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    prompt = " ".join(args.prompt).strip()
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()

    if prompt:
        return one_shot(
            prompt,
            system=args.system,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            json_mode=args.json,
            quiet=args.quiet,
        )

    return repl(
        system=args.system,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        json_mode=args.json,
    )


if __name__ == "__main__":
    raise SystemExit(main())
