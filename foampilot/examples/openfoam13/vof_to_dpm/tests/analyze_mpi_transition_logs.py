#!/usr/bin/env python3
"""Analyse des logs MPI de transition VOF -> DPM.

Le script accepte les lignes de diagnostic suivantes, entre autres :

    [rank 1] parcelCreated cloud=spray timeIndex=42 fragmentId=123 ownerProc=1 count=1 mass=2e-6
    rank=1 confirmed cloud=spray timeIndex=42 fragmentId=123 ownerProc=1 parcelsAdded=1 massAdded=2e-6 expectedMass=2e-6 success=1

Code retour :
    0 : aucun invariant violé
    1 : doublon, owner incohérent, confirmation manquante ou bilan incorrect
    2 : erreur de lecture/arguments
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

KEY_VALUE = re.compile(r"(?P<key>[A-Za-z][A-Za-z0-9_]*)=(?P<value>[^\s,;]+)")
RANK_PREFIX = re.compile(r"(?:\[?rank\s*=?|Pstream::myProcNo\(\)=)(?P<rank>-?\d+)", re.I)
ID_WORD = re.compile(r"\b(?:fragmentId|fragmentID|id)\s*[=:]?\s*(?P<id>\d+)", re.I)
TIME_WORD = re.compile(r"\btimeIndex\s*[=:]?\s*(?P<time>-?\d+)", re.I)


@dataclass(frozen=True)
class Event:
    path: str
    line: int
    kind: str
    fragment_id: int
    time_index: int | None
    rank: int | None
    owner: int | None
    count: int | None
    mass: float | None
    expected_mass: float | None
    success: bool | None
    raw: str


@dataclass
class Issue:
    kind: str
    message: str
    events: list[dict]


def parse_scalar(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def parse_bool(value: str) -> bool | None:
    value = value.lower()
    if value in {"1", "true", "yes", "ok"}:
        return True
    if value in {"0", "false", "no", "fail", "failed"}:
        return False
    return None


def fields(line: str) -> dict[str, str]:
    return {m.group("key"): m.group("value") for m in KEY_VALUE.finditer(line)}


def parse_event(path: Path, line_number: int, line: str) -> Event | None:
    lower = line.lower()
    if not any(token in lower for token in
               ("parcelcreated", "parcel_created", "confirmed", "confirmation",
                "parcelcreated", "fragment")):
        return None

    kv = fields(line)
    id_match = ID_WORD.search(line)
    if "fragmentid" in kv:
        fragment_id = parse_int(kv["fragmentid"])
    elif "fragmentID" in kv:
        fragment_id = parse_int(kv["fragmentID"])
    elif id_match:
        fragment_id = parse_int(id_match.group("id"))
    else:
        return None
    if fragment_id is None:
        return None

    time_match = TIME_WORD.search(line)
    time_index = parse_int(kv.get("timeIndex", "")) if "timeIndex" in kv else (
        parse_int(time_match.group("time")) if time_match else None
    )

    rank_match = RANK_PREFIX.search(line)
    rank = parse_int(kv.get("rank", "")) if "rank" in kv else (
        parse_int(rank_match.group("rank")) if rank_match else None
    )

    owner = parse_int(kv.get("ownerProc", kv.get("owner", "")))
    count = parse_int(kv.get("count", kv.get("parcelsAdded", "")))
    mass = parse_scalar(kv.get("mass", kv.get("massAdded", "")))
    expected = parse_scalar(kv.get("expectedMass", ""))
    success = parse_bool(kv["success"]) if "success" in kv else None

    if "parcelcreated" in lower or "parcel_created" in lower:
        kind = "created"
    elif "confirmation" in lower or "confirmed" in lower:
        kind = "confirmed"
    else:
        kind = "fragment"

    return Event(
        str(path), line_number, kind, fragment_id, time_index, rank, owner,
        count, mass, expected, success, line.rstrip("\n")
    )


def read_events(paths: Iterable[Path]) -> list[Event]:
    events: list[Event] = []
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                event = parse_event(path, line_number, line)
                if event is not None:
                    events.append(event)
    return events


def event_key(event: Event) -> tuple[int | None, int]:
    return event.time_index, event.fragment_id


def event_dict(event: Event) -> dict:
    return asdict(event)


def analyze(events: list[Event], tolerance: float) -> tuple[list[Issue], dict]:
    issues: list[Issue] = []
    grouped: dict[tuple[int | None, int], list[Event]] = defaultdict(list)
    for event in events:
        grouped[event_key(event)].append(event)

    duplicate_created = 0
    duplicate_confirmed = 0
    wrong_owner = 0
    rejected = 0
    mass_expected = 0.0
    mass_confirmed = 0.0

    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        created = [e for e in group if e.kind == "created"]
        confirmed = [e for e in group if e.kind == "confirmed"]

        if len(created) > 1:
            duplicate_created += 1
            issues.append(Issue(
                "duplicate_creation",
                f"{key}: {len(created)} créations pour le même fragment",
                [event_dict(e) for e in created],
            ))

        if len(confirmed) > 1:
            duplicate_confirmed += 1
            issues.append(Issue(
                "duplicate_confirmation",
                f"{key}: {len(confirmed)} confirmations pour le même fragment",
                [event_dict(e) for e in confirmed],
            ))

        owners = {e.owner for e in group if e.owner is not None}
        if len(owners) > 1:
            wrong_owner += 1
            issues.append(Issue(
                "inconsistent_owner",
                f"{key}: plusieurs ownerProc {sorted(owners)}",
                [event_dict(e) for e in group],
            ))

        if confirmed:
            confirmation = confirmed[-1]
            if confirmation.success is False:
                rejected += 1
            if confirmation.expected_mass is not None:
                mass_expected += confirmation.expected_mass
            if confirmation.mass is not None and confirmation.success:
                mass_confirmed += confirmation.mass

            if confirmation.count is not None and confirmation.count != 1 and confirmation.success:
                issues.append(Issue(
                    "invalid_parcel_count",
                    f"{key}: parcelsAdded/count={confirmation.count}, attendu 1",
                    [event_dict(confirmation)],
                ))

            if (confirmation.success
                    and confirmation.mass is not None
                    and confirmation.expected_mass is not None
                    and abs(confirmation.mass - confirmation.expected_mass)
                    > tolerance * max(abs(confirmation.expected_mass), 1.0)):
                issues.append(Issue(
                    "mass_mismatch",
                    f"{key}: masse créée {confirmation.mass} != attendue {confirmation.expected_mass}",
                    [event_dict(confirmation)],
                ))

    if mass_expected and abs(mass_confirmed - mass_expected) > tolerance * max(abs(mass_expected), 1.0):
        issues.append(Issue(
            "global_mass_mismatch",
            f"masse confirmée {mass_confirmed} != masse attendue {mass_expected}",
            [],
        ))

    summary = {
        "events": len(events),
        "fragments": len(grouped),
        "createdEvents": sum(1 for e in events if e.kind == "created"),
        "confirmationEvents": sum(1 for e in events if e.kind == "confirmed"),
        "duplicateCreations": duplicate_created,
        "duplicateConfirmations": duplicate_confirmed,
        "inconsistentOwners": wrong_owner,
        "rejectedConfirmations": rejected,
        "expectedMass": mass_expected,
        "confirmedMass": mass_confirmed,
        "issues": len(issues),
    }
    return issues, summary


def print_report(issues: list[Issue], summary: dict, as_json: bool) -> None:
    report = {"summary": summary, "issues": [asdict(issue) for issue in issues]}
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print("MPI transition log audit")
    for key in (
        "events", "fragments", "createdEvents", "confirmationEvents",
        "duplicateCreations", "duplicateConfirmations", "inconsistentOwners",
        "rejectedConfirmations", "expectedMass", "confirmedMass", "issues",
    ):
        print(f"  {key}: {summary[key]}")

    for issue in issues:
        print(f"ERROR [{issue.kind}] {issue.message}")
        for event in issue.events:
            print(f"  {event['path']}:{event['line']}: {event['raw']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="logs MPI à analyser")
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--json", action="store_true", help="sortie JSON")
    parser.add_argument("--follow", action="store_true", help="réanalyser périodiquement les logs")
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    for path in args.logs:
        if not path.is_file():
            print(f"Fichier introuvable: {path}", file=sys.stderr)
            return 2

    while True:
        try:
            events = read_events(args.logs)
            issues, summary = analyze(events, args.tolerance)
            print_report(issues, summary, args.json)
        except OSError as error:
            print(f"Erreur de lecture: {error}", file=sys.stderr)
            return 2

        if not args.follow:
            return 1 if issues else 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
