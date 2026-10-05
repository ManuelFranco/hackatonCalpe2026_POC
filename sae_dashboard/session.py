"""Private session state. No model weights, global experiment state, or disk writes."""

from dataclasses import dataclass
import copy
import threading
import uuid
from typing import Any
from .manifests import Manifest, Pair


@dataclass(frozen=True)
class Settings:
    seed: int = 0
    temperature: float = 0.0
    max_new_tokens: int = 256
    token_scope: str = "all"
    aggregation: str = "mean"

    def validate(self):
        from .model_runtime import validate_generation_settings
        from research.code_regions import TOKEN_SCOPES

        validate_generation_settings(self.max_new_tokens, self.temperature, self.seed)
        if self.token_scope not in TOKEN_SCOPES:
            raise ValueError("Choose a supported profile-token scope.")
        if self.aggregation not in {"mean", "max"}:
            raise ValueError("Choose mean or max for aggregation.")

    @property
    def capture_key(self):
        return self.token_scope, self.aggregation


class Session:
    def __init__(self):
        self.id = uuid.uuid4().hex
        self.save_enabled = False
        self.manifests: tuple[Manifest, ...] = ()
        self.manual_pairs: tuple[Pair, ...] = ()
        self.manual_next_id = 1
        self.profile: dict[int, Any] = {}
        self.profile_id = None
        self.capture_key = None
        self.capture_details: list[dict] = []
        self.vectors: dict[int, Any] = {}
        self.vector_id = None
        self.results: list[dict] = []
        self.benchmark_sample = None
        self.benchmark_base = []
        self.benchmark_settings = None
        self.benchmark_result = None
        self.transfer_sources: dict[str, Any] = {}
        self.transfer_plan = None
        self.transfer_result = None
        self.code_experiment_job = None
        self.lock = threading.RLock()

    def __deepcopy__(self, memo):
        # Gradio deep-copies initial state. Locks and identities cannot be shared.
        clone = Session()
        memo[id(self)] = clone
        for key, value in vars(self).items():
            if key not in {"lock", "id"}:
                setattr(clone, key, copy.deepcopy(value, memo))
        return clone

    def invalidate_profile(self):
        self.code_experiment_job = None
        self.profile = {}
        self.profile_id = None
        self.capture_key = None
        self.capture_details = []
        self.vectors = {}
        self.vector_id = None
        self.results = []
        # Frozen transfer vectors survive loading another calibration family.
        self.transfer_plan = None
        self.transfer_result = None
