"""Create a local Kaggle input archive for the indexed PHENIKAA MVP."""
import argparse
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

from omnishield.data.acquire import read_catalog
from omnishield.data.manifest import load_manifest, lut_path, sample_path
from omnishield.data.prepare import audit
from omnishield.io import atomic_file, child_path, load_json


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "dist/phenikaa-mvp.zip")
    args = parser.parse_args()
    raw, prepared = args.raw.resolve(), args.prepared.resolve()
    index, splits, selected, source_files = read_catalog(ROOT / "configs/data/phenikaa_mvp/index.json")
    receipt = load_json(raw / "acquired.json")
    if receipt["frames"] != selected or receipt["splits"] != splits:
        raise ValueError("Only the complete two-clip acquisition can be bundled")
    for relative, (_, size) in source_files.items():
        path = child_path(raw, relative)
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Raw file missing or incomplete: {relative}")
    config = load_json(ROOT / "configs/model/edge_mvp.json")
    path = prepared / "manifest.json"
    manifest = load_manifest(path, config)
    actual_frames = {name: [row["frame_id"] for row in manifest["frames"] if row["drive_id"] == name]
                     for name in selected}
    if actual_frames != selected or len(manifest["frames"]) != sum(map(len, selected.values())):
        raise ValueError("Prepared frame IDs differ from the published acquisition index")
    report = audit(raw, prepared, manifest, config, write=False)
    if report != load_json(prepared / "audit.json"):
        raise ValueError("Prepared data changed after its audit")
    for split in ("train", "val"):
        if min(report["occupancy_classes"][split]) <= 0 or not all(
            report["coverage"][split][name] for name in ("depth", "semantics")
        ):
            raise ValueError(f"{split} is missing training or validation targets")
    targets = [path, prepared / "audit.json"]
    targets += [lut_path(prepared, name) for name in manifest["rigs"]]
    targets += [sample_path(prepared, row) for row in manifest["frames"]]

    def pack(file):
        with ZipFile(file, "w", compression=ZIP_STORED) as bundle:
            for relative in (*source_files, "acquired.json"):
                bundle.write(child_path(raw, relative), f"raw/{relative}")
            for target in targets:
                bundle.write(target, "prepared/" + target.relative_to(prepared).as_posix())
            bundle.writestr("ATTRIBUTION.txt", "PHENIKAA Open Dataset · " + index["license"] + "\n"
                            + index["release_url"] + "\nhttps://github.com/phenikaa-dataset/phenikaa-devkit\n"
                            + "OmniShield-derived labels use camera-depth free-space proxies.\n")

    atomic_file(args.out, pack)
    print(f"{args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
