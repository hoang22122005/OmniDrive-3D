from contextlib import contextmanager
from pathlib import Path

import torch

from omnishield.io import atomic_file


@contextmanager
def run_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".running"
    try:
        with lock.open("x"):
            pass
    except FileExistsError as error:
        raise ValueError(f"Run directory has a writer or stale .running lock: {directory}") from error
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema") != 1:
        raise ValueError(f"Invalid checkpoint schema: {path}")
    return state


def save_checkpoint(path, state):
    atomic_file(path, lambda file: torch.save(state, file))
