"""Content-addressed base responses. Atomic JSON writes and cross-process locks."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time

CACHE_ROOT = Path(
    os.getenv(
        "GEMMA_BASE_CACHE_DIR",
        Path(__file__).resolve().parents[1] / ".cache/base_responses",
    )
)
_LOCKS = [threading.Lock() for _ in range(64)]


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def tensor_fingerprint(value):
    import torch

    tensor = value.detach().cpu().contiguous()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    }


@contextmanager
def cache_lock(root, key):
    root.mkdir(parents=True, exist_ok=True)
    with _LOCKS[int(key[:8], 16) % len(_LOCKS)]:
        with (root / f"{key}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def get_or_generate(request, generate, root=None):
    """Only accepts BASE generation; steered code never calls this function."""
    root = Path(root) if root is not None else CACHE_ROOT
    key = hashlib.sha256(canonical(request).encode()).hexdigest()
    with cache_lock(root, key):
        path = root / f"{key}.json"
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if (
                isinstance(saved, dict)
                and saved.get("request") == request
                and isinstance(saved.get("answer"), str)
            ):
                return saved["answer"], True
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            pass
        answer = generate()
        if not isinstance(answer, str):
            raise ValueError("Base generation must return text.")
        payload = {
            "format_version": 1,
            "created_unix": time.time(),
            "request": request,
            "answer": answer,
        }
        name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=root, suffix=".tmp", delete=False
            ) as file:
                name = file.name
                json.dump(payload, file, indent=2, ensure_ascii=False, allow_nan=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(name, path)
        finally:
            if name and os.path.exists(name):
                os.unlink(name)
        return answer, False
