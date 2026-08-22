"""Embed OpenFOAM source dictionaries as Python string templates."""

from pathlib import Path
import sys


def py_string(text: str) -> str:
    return repr(text)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: embed_case_templates.py SOURCE_SYSTEM OUTPUT_PY")
    source = Path(sys.argv[1])
    output = Path(sys.argv[2])
    files = sorted(p for p in source.iterdir() if p.is_file())
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '"""Dictionaries for the Tobias Holzmann tutorial case.',
        "Generated from the downloaded full case archive; written through",
        "FoamPilot's raw dictionary writer by the case runner.",
        '"""',
        "",
        "DICTIONARIES = {",
    ]
    for path in files:
        lines.append(f"    {path.name!r}: {py_string(path.read_text(encoding='utf-8'))},")
    lines.extend(["}", ""])
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Embedded {len(files)} dictionaries into {output}")


if __name__ == "__main__":
    main()
