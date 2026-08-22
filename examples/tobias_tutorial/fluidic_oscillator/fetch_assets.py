"""Download the large public assets required by the Tobias case."""

from pathlib import Path
import tarfile
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
URL = "https://holzmann-cfd.com/OpenFOAMCases/020_fluidicOscillator/fluidicOscillator-12.tar.gz"


def fetch() -> None:
    archive = ROOT / "fluidicOscillator-12.tar.gz"
    with urlopen(URL) as response, archive.open("wb") as stream:
        stream.write(response.read())
    with tarfile.open(archive, "r:gz") as source:
        member_root = "cases-12/fluidicOscillator/"
        for member in source.getmembers():
            if not member.name.startswith(member_root):
                continue
            relative = Path(member.name[len(member_root):])
            if not relative.parts or relative.parts[0] not in {"cad", "constant"}:
                continue
            target = ROOT / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as input_stream, target.open("wb") as output:
                    output.write(input_stream.read())
    archive.unlink()


if __name__ == "__main__":
    fetch()
