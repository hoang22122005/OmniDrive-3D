"""Build the portable source attachment consumed by kaggle/OmniShield.ipynb."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from omnishield.io import atomic_file, file_sha256


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "dist/omnishield-src.zip")
    args = parser.parse_args()
    paths = [ROOT / "pyproject.toml", ROOT / "README.md", ROOT / "kaggle/OmniShield.ipynb",
             ROOT / "docs/kaggle_training.md", ROOT / "docs/omnishield_x_implementation_spec.md",
             ROOT / "docs/phenikaa_mvp.md"]
    for directory, extension in (("omnishield", ".py"), ("configs", ".json"), ("kaggle", ".json")):
        paths.extend(path for path in (ROOT / directory).rglob(f"*{extension}") if path.is_file())

    def write(file):
        with ZipFile(file, "w", compression=ZIP_DEFLATED) as archive:
            for path in sorted(paths):
                archive.write(path, path.relative_to(ROOT).as_posix())

    atomic_file(args.out, write)
    print(f"{args.out}\nsha256 {file_sha256(args.out)}")


if __name__ == "__main__":
    main()
