"""Content-addressed base responses. Atomic JSON writes and cross-process locks."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time

import portalocker


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
    """Acquire an inter-thread and inter-process lock for a cache key."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # Thread-level lock avoids unnecessary contention between threads
    # in the same process.
    with _LOCKS[int(key[:8], 16) % len(_LOCKS)]:
        lock_path = root / f"{key}.lock"

        with lock_path.open("a+") as lock_file:
            portalocker.lock(lock_file, portalocker.LOCK_EX)

            try:
                yield
            finally:
                portalocker.unlock(lock_file)


def get_or_generate(request, generate, root=None):
    """Only accepts BASE generation; steered code never calls this function."""
    root = Path(root) if root is not None else CACHE_ROOT
    root.mkdir(parents=True, exist_ok=True)

    key = hashlib.sha256(canonical(request).encode("utf-8")).hexdigest()
    path = root / f"{key}.json"

    with cache_lock(root, key):
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))

            if (
                isinstance(saved, dict)
                and saved.get("request") == request
                and isinstance(saved.get("answer"), str)
            ):
                return saved["answer"], True

        except (
            FileNotFoundError,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ):
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

        temp_path = None

        try:
            # Create the temporary file in the same directory as the
            # destination. This is important because os.replace() is
            # atomic only when source and destination are on the same
            # filesystem/volume.
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=root,
                prefix=f".{key}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temp_path = Path(file.name)

                json.dump(
                    payload,
                    file,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )

                file.flush()
                os.fsync(file.fileno())

            # Atomic replacement on Windows, macOS and Linux.
            os.replace(temp_path, path)

        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass

        return answer, False