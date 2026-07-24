from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class SourceConfig:
    site: str
    dump_path: str


@dataclass
class EvalCarveoutConfig:
    target_size: int
    score_bucket_edges: list[int]


@dataclass
class LabelingConfig:
    top_quantile: float
    bottom_quantile: float


@dataclass
class DedupConfig:
    similarity_threshold: float
    top_k_neighbors: int


@dataclass
class PathsConfig:
    interim: str
    eval: str
    models_classifier: str


@dataclass
class AlloyConfig:
    seed: int
    source: SourceConfig
    eval_carveout: EvalCarveoutConfig
    labeling: LabelingConfig
    dedup: DedupConfig
    paths: PathsConfig

    @classmethod
    def load(cls, path: str | Path) -> "AlloyConfig":
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            seed=raw["project"]["seed"],
            source=SourceConfig(**raw["source"]),
            eval_carveout=EvalCarveoutConfig(**raw["eval_carveout"]),
            labeling=LabelingConfig(**raw["labeling"]),
            dedup=DedupConfig(**raw["dedup"]),
            paths=PathsConfig(**raw["paths"]),
        )