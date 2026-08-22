"""Download the large public UNV asset for Falling Droplets."""
from pathlib import Path
import tarfile
from urllib.request import urlopen
ROOT = Path(__file__).resolve().parent
URL = "https://holzmann-cfd.com/OpenFOAMCases/019_fallingDroplets/fallingDroplets-12.tar.gz"
def fetch():
    archive = ROOT / "fallingDroplets-12.tar.gz"
    with urlopen(URL) as response, archive.open("wb") as stream:
        stream.write(response.read())
    with tarfile.open(archive, "r:gz") as source:
        prefix = "cases-12/fallingDroplets/"
        for member in source.getmembers():
            if member.name.startswith(prefix) and member.name[len(prefix):].startswith("cad/"):
                rel = Path(member.name[len(prefix):])
                target = ROOT / rel
                if member.isdir(): target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as inp, target.open("wb") as out: out.write(inp.read())
    archive.unlink()
if __name__ == "__main__": fetch()
