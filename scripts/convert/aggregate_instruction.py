#!/usr/bin/env python3
"""
Aggregate language instructions across many JSON metadata files.

Example input JSON (per file) includes fields like:
- uuid: str
- current_task: str  (or language_instruction, instruction, task, etc.)

Output:
{
  "<uuid>": {
    "language_instruction1": "...",
    "language_instruction2": "...",
    ...
  },
  ...
}
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


INSTRUCTION_KEYS_CANDIDATES = [
    # Your example uses current_task
    "current_task",
    # Common alternatives
    "language_instruction",
    "language",
    "instruction",
    "task",
    "goal",
]


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        # Skip unreadable/invalid JSON files, but keep going.
        print(f"[WARN] Skipping {path} (failed to read JSON): {e}", file=sys.stderr)
        return None


def _extract_uuid(data: Dict[str, Any]) -> Optional[str]:
    u = data.get("uuid")
    if isinstance(u, str) and u.strip():
        return u.strip()
    return None


def _extract_instruction(data: Dict[str, Any]) -> Optional[str]:
    for k in INSTRUCTION_KEYS_CANDIDATES:
        v = data.get(k, None)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # Sometimes instruction lives nested; add lightweight fallbacks here if needed.
    # Example patterns (uncomment if your data uses them):
    # v = data.get("metadata", {}).get("current_task")
    # if isinstance(v, str) and v.strip():
    #     return v.strip()

    return None


def _dedup_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def aggregate(root: Path, strict_uuid: bool = True) -> Dict[str, Dict[str, str]]:
    """
    Returns:
      dict[uuid] -> dict["language_instruction{i}"] -> instruction
    """
    per_uuid_instructions: Dict[str, List[str]] = defaultdict(list)

    json_paths = list(root.rglob("*.json"))
    print(f"[INFO] Found {len(json_paths)} JSON files under {root}")

    for p in json_paths:
        data = _safe_load_json(p)
        if data is None:
            continue
        if not isinstance(data, dict):
            print(f"[WARN] Skipping {p} (top-level JSON is not an object)", file=sys.stderr)
            continue

        uuid = _extract_uuid(data)
        if uuid is None:
            msg = f"[WARN] Skipping {p} (missing/invalid 'uuid')"
            if strict_uuid:
                print(msg, file=sys.stderr)
                continue
            else:
                uuid = f"__missing_uuid__::{p.as_posix()}"

        instr = _extract_instruction(data)
        if instr is None:
            print(f"[WARN] Skipping {p} (no instruction field found)", file=sys.stderr)
            continue

        per_uuid_instructions[uuid].append(instr)

    # Build requested output schema and de-duplicate within each uuid
    aggregated: Dict[str, Dict[str, str]] = {}
    for uuid, instr_list in per_uuid_instructions.items():
        instr_list = _dedup_preserve_order(instr_list)
        aggregated[uuid] = {
            f"language_instruction{i+1}": s for i, s in enumerate(instr_list)
        }

    return aggregated


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root directory to scan recursively for *.json")
    ap.add_argument("--out", type=str, required=True, help="Output aggregated JSON path")
    ap.add_argument(
        "--strict-uuid",
        action="store_true",
        help="If set, skip files missing uuid (default behavior).",
    )
    ap.add_argument(
        "--no-strict-uuid",
        dest="strict_uuid",
        action="store_false",
        help="If set, keep files missing uuid by assigning a synthetic key.",
    )
    ap.set_defaults(strict_uuid=True)

    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root is not a directory: {root}")

    aggregated = aggregate(root=root, strict_uuid=args.strict_uuid)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False, sort_keys=True)

    # Quick summary
    num_uuids = len(aggregated)
    num_instr = sum(len(v) for v in aggregated.values())
    print(f"[INFO] Wrote {out_path}")
    print(f"[INFO] UUIDs: {num_uuids} | Total instructions: {num_instr}")


if __name__ == "__main__":
    main()
