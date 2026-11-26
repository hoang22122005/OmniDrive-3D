import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from omnishield.io import atomic_file, child_path, load_json, write_json


def read_catalog(path, max_frames=None):
    index = load_json(path)
    if index["version"] != 1:
        raise ValueError("Unsupported PHENIKAA catalog version")
    catalog = {}
    for name, file_id, size in index["calibration"]:
        catalog[f"calibration/{name}"] = (file_id, size)
    split_clips = {split: [] for split in ("train", "val", "test")}
    selected = {}
    for entry in index["clips"]:
        clip = load_json(child_path(Path(path).parent, entry))
        frames = sorted(clip["frames"])
        if max_frames is not None and len(frames) > max_frames:
            if max_frames == 1:
                frames = [frames[len(frames) // 2]]
            else:
                frames = [frames[round(i * (len(frames) - 1) / (max_frames - 1))] for i in range(max_frames)]
        split_clips[clip["split"]].append(clip["name"])
        selected[clip["name"]] = frames
        for frame in frames:
            for kind, (file_id, size) in zip(clip["kinds"], clip["frames"][frame], strict=True):
                if kind.startswith("CAM_F_"):
                    relative = f"{clip['name']}/camera/{kind}/{frame}.jpg"
                else:
                    ext = "laz" if kind == "lidar" else "txt"
                    relative = f"{clip['name']}/{kind}/{frame}.{ext}"
                catalog[relative] = (file_id, size)
    if not split_clips["train"] or not split_clips["val"]:
        raise ValueError("Catalog needs train and validation clips")
    return index, split_clips, selected, catalog


def download_file(path, file_id, size):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", file_id) or not 0 < size < 16_000_000:
        raise ValueError(f"Invalid catalog metadata for {path.name}")
    if path.is_file() and path.stat().st_size == size:
        return False
    query = urlencode({"id": file_id, "export": "download", "confirm": "t"})
    request = Request("https://drive.usercontent.google.com/download?" + query,
                      headers={"User-Agent": "OmniShield-PHENIKAA-MVP/1"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=25) as server:
                data = server.read(size + 1)
            if len(data) != size or data.lstrip()[:15].lower().startswith((b"<!doctype html", b"<html")):
                raise ValueError(f"Drive content differs from the listed byte count ({size} expected)")
            signature = {".jpg": b"\xff\xd8", ".laz": b"LASF", ".json": b"{"}.get(path.suffix)
            if signature and not data.lstrip().startswith(signature):
                raise ValueError("File signature does not match the expected format")
            atomic_file(path, lambda file: file.write_bytes(data))
            return True
        except (OSError, ValueError) as error:
            if attempt == 2:
                raise RuntimeError(f"Download failed for {path}: {error}") from error
            time.sleep(1 + attempt)


def acquire_phenikaa(catalog_path, raw, workers, max_frames=None):
    if not 1 <= workers <= 12 or (max_frames is not None and max_frames < 1):
        raise ValueError("Use 1–12 download workers and a positive frame limit")
    source, splits, selected, files = read_catalog(catalog_path, max_frames)
    raw = Path(raw)
    print(f"{len(files)} files, {sum(size for _,size in files.values()) / 1e6:.1f} MB; "
          f"frames by clip: { {key: len(value) for key, value in selected.items()} }", flush=True)

    def get(item):
        relative, (file_id, size) = item
        try:
            return download_file(child_path(raw, relative), file_id, size), None
        except RuntimeError as error:
            return False, str(error)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        fetched = 0
        errors = []
        for done, (changed, error) in enumerate(pool.map(get, files.items()), 1):
            fetched += changed
            if error:
                errors.append(error)
            if done % 50 == 0:
                print(f"Checked {done}/{len(files)}, fetched {fetched}", flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} files failed. Re-run to resume. " + " | ".join(errors[:3]))
    write_json(raw / "acquired.json", {
        "source": source["source"], "license": source["license"], "url": source["release_url"],
        "splits": splits, "frames": selected, "files": len(files), "bytes": sum(size for _, size in files.values()),
    })
