"""Deterministic, dependency-free repository audit used by CI."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".env", ".ipynb", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
SECRET_PATTERNS = {
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "OpenAI-style token": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "Groq token": re.compile(r"gsk_[A-Za-z0-9]{20,}"),
}


def files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and ".runtime" not in path.parts
        and "__pycache__" not in path.parts
    ]


def secret_errors(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.name == ".env.example":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label} found in {path.relative_to(ROOT)}")
    committed_env = ROOT / ".env"
    if committed_env.exists():
        errors.append("A root .env file is present; commit only .env.example")
    return errors


def notebook_errors(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        if path.suffix != ".ipynb":
            continue
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            errors.append(f"Invalid notebook {path.relative_to(ROOT)}: {exc}")
            continue
        if not isinstance(notebook, dict) or "cells" not in notebook:
            errors.append(f"Notebook schema missing cells: {path.relative_to(ROOT)}")
    return errors


def duplicate_pdfs(paths: list[Path]) -> list[list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        if path.suffix.lower() == ".pdf":
            groups[hashlib.sha256(path.read_bytes()).hexdigest()].append(path)
    return [group for group in groups.values() if len(group) > 1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-duplicates", action="store_true")
    args = parser.parse_args()

    paths = files()
    errors = secret_errors(paths) + notebook_errors(paths)
    required = ["README.md", "requirements.txt", ".env.example", "SECURITY.md"]
    errors.extend(f"Missing required file: {name}" for name in required if not (ROOT / name).is_file())

    duplicates = duplicate_pdfs(paths)
    for group in duplicates:
        names = ", ".join(str(path.relative_to(ROOT)) for path in group)
        print(f"WARNING duplicate PDF content: {names}")
    if args.strict_duplicates and duplicates:
        errors.append("Duplicate PDF content found")

    if errors:
        for error in errors:
            print(f"ERROR {error}")
        return 1
    print(f"Repository audit passed ({len(paths)} files; {len(duplicates)} duplicate-PDF groups reported).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
