import json
from pathlib import Path


def _strip_json_comments(text: str) -> str:
    """Remove // and /* */ comments while preserving string literals."""
    result: list[str] = []
    in_string = False
    string_quote = ""
    escape = False
    i = 0
    length = len(text)

    while i < length:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < length else ""

        if in_string:
            result.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == string_quote:
                in_string = False
            i += 1
            continue

        if ch in ('"', "'"):
            in_string = True
            string_quote = ch
            result.append(ch)
            i += 1
            continue

        if ch == "/" and nxt == "/":
            i += 2
            while i < length and text[i] not in "\r\n":
                i += 1
            continue

        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < length and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2 if i + 1 < length else 0
            continue

        result.append(ch)
        i += 1

    return "".join(result)


def load_jsonc(path: str | Path):
    with open(path, "r") as f:
        return json.loads(_strip_json_comments(f.read()))
