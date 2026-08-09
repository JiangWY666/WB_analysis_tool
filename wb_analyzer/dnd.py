"""Drag-and-drop path parsing helpers."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


def decode_drop_token(token: object) -> str:
    if isinstance(token, Path):
        return str(token)
    if isinstance(token, str):
        return token.strip().strip("{}")
    if isinstance(token, (bytes, bytearray)):
        for encoding in ("utf-8", "gbk", "mbcs", "latin1"):
            try:
                return bytes(token).decode(encoding).strip().strip("{}")
            except UnicodeDecodeError:
                continue
    return str(token).strip().strip("{}")


def parse_drop_paths(data: str) -> list[Path]:
    if not data:
        return []
    text = data.strip()
    found: list[str] = []
    found.extend(re.findall(r"\{([^}]+)\}", text))
    found.extend(re.findall(r"file:///[^\s]+|file://[^\s]+", text))
    if not found:
        found.extend(re.findall(r"[^\s]+", text))

    paths: list[Path] = []
    for item in found:
        cleaned = decode_drop_token(item)
        cleaned = unquote(cleaned)
        if cleaned.lower().startswith("file:///"):
            cleaned = cleaned[8:]
        elif cleaned.lower().startswith("file://"):
            cleaned = cleaned[7:]
        if re.match(r"^/[A-Za-z]:/", cleaned):
            cleaned = cleaned[1:]
        if sys.platform.startswith("win") and re.match(r"^[A-Za-z]:/", cleaned):
            cleaned = cleaned.replace("/", "\\")
        if cleaned:
            paths.append(Path(cleaned))
    return paths
