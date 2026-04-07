"""Pydantic schemas for experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from policy_interp.constants import DEFAULT_SEED, PROXY_COLUMNS


class DatasetConfig(BaseModel):
    base_dir: Path
    segments_csv: str = "AGORA Data/segments.csv"
    documents_csv: str = "AGORA Data/documents.csv"
    authorities_csv: str = "AGORA Data/authorities.csv"
    collections_csv: str = "AGORA Data/collections.csv"
    fulltext_dir: str = "AGORA Data/fulltext"
    working_dir: str = "artifacts"
    prepared_segments_name: str = "prepared_segments.parquet"
    prepared_documents_name: str = "prepared_documents.parquet"
    proxy_columns: dict[str, str] = Field(default_factory=lambda: PROXY_COLUMNS.copy())

    @property
    def segments_path(self) -> Path:
        return self.base_dir / self.segments_csv

    @property
    def documents_path(self) -> Path:
        return self.base_dir / self.documents_csv

    @property
    def authorities_path(self) -> Path:
        return self.base_dir / self.authorities_csv

    @property
    def collections_path(self) -> Path:
        return self.base_dir / self.collections_csv

    @property
    def fulltext_path(self) -> Path:
        return self.base_dir / self.fulltext_dir

    @property
    def artifacts_path(self) -> Path:
        return self.base_dir / self.working_dir


class SplitsConfig(BaseModel):
    train_ratio: float = 0.68
    dev_ratio: float = 0.16
    test_ratio: float = 0.16
    seed: int = DEFAULT_SEED

    @model_validator(mode="after")
    def _validate_ratios(self) -> "SplitsConfig":
        total = self.train_ratio + self.dev_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        return self


class MatchingWeights(BaseModel):
    text: float = 0.45
    authority: float = 0.10
    jurisdiction: float = 0.10
    form: float = 0.10
    domain: float = 0.10
    year: float = 0.05
    length: float = 0.10


class MatchingConfig(BaseModel):
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    max_length: int = 512
    weights: MatchingWeights = Field(default_factory=MatchingWeights)


class BackboneConfig(BaseModel):
    model_name: str = "google/gemma-2-2b"
    device: str = "cuda"
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    max_length: int = 512
    trust_remote_code: bool = False


class SaeConfig(BaseModel):
    release: str = "gemma-scope-2b-pt-res-canonical"
    sae_id: str = "layer_24/width_16k/canonical"
    device: str = "cuda"


class ExtractConfig(BaseModel):
    layers: list[int] = Field(default_factory=lambda: [12, 16, 20, 24])
    max_segments_per_shard: int = 64
    max_tokens_per_shard: int = 4096
    max_layers_per_pass: int = 1
    flush_every_shard: bool = True
    pooling_method: Literal["max", "mean"] = "max"
    residual_pooling_method: Literal["max", "mean"] = "mean"
    store_segment_top_features: bool = True
    segment_top_feature_count: int = 32
    segment_top_token_positions_per_feature: int = 5
    top_contexts_per_feature: int = 5
    context_window: int = 12
    store_positive_mean_stats: bool = True
    store_tail_activation_stats: bool = True
    store_token_span_text: bool = True
    store_catalog_rich_cache: bool = True
    catalog_activation_threshold: float = 0.0


class DiscoveryConfig(BaseModel):
    graph_metric: Literal["jaccard"] = "jaccard"
    active_feature_source: Literal["segment_top_features"] = "segment_top_features"
    top_ngrams_per_module: int = 15
    ngram_range: tuple[int, int] = (1, 3)


class ValidityConfig(BaseModel):
    graph_metric: Literal["jaccard"] = "jaccard"
    edge_thresholds: list[float] = Field(default_factory=lambda: [0.05, 0.10, 0.15])
    community_method: Literal["leiden"] = "leiden"
    resolutions: list[float] = Field(default_factory=lambda: [0.5, 1.0, 1.5])
    canonical_threshold: float = 0.10
    canonical_resolution: float = 1.0
    min_module_size: int = 3
    max_module_size: int = 50
    min_presence_configs: int = 5
    min_best_match_jaccard: float = 0.5
    enrichment_test_method: Literal["permutation"] = "permutation"
    enrichment_random_trials: int = 1000
    enrichment_alpha: float = 0.05
    enrichment_correction: Literal["bh_fdr"] = "bh_fdr"
    enrichment_match_bins: int = 10

    @field_validator("edge_thresholds", "resolutions")
    @classmethod
    def _validate_non_empty(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("Threshold and resolution lists must be non empty")
        return value


class LabelingConfig(BaseModel):
    template_enabled: bool = True
    llm_hook_enabled: bool = True
    generator_model: str = "google/gemma-2-2b-it"
    num_top_contexts: int = 5
    num_exemplars: int = 5
    num_negative_controls: int = 5
    negative_snippet_strategy: Literal["metadata_matched_low_activation"] = "metadata_matched_low_activation"
    max_new_tokens: int = 120


class FeatureCatalogConfig(BaseModel):
    ranking_families: list[Literal["global_dominance", "policy_specific", "layer_unique"]] = Field(
        default_factory=lambda: ["global_dominance", "policy_specific", "layer_unique"]
    )
    top_n_per_family: int = 100
    top_contexts_per_feature: int = 5
    top_exemplars_per_feature: int = 10
    store_decoder_vectors: bool = True
    store_logit_attribution: bool = True
    logit_top_k: int = 50
    store_catalog_segment_scores: bool = True
    layer_unique_candidate_pool: int = 300
    cross_layer_similarity_threshold: float = 0.35
    evidence_backfill_enabled: bool = True
    evidence_backfill_ranking_families: list[Literal["global_dominance", "policy_specific", "layer_unique"]] = Field(
        default_factory=lambda: ["policy_specific", "layer_unique"]
    )
    evidence_backfill_top_n_per_family: int = 10
    evidence_backfill_min_positive_examples: int = 5

    @field_validator("ranking_families")
    @classmethod
    def _validate_ranking_families(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("Feature ranking families must be non empty")
        return value

    @field_validator("evidence_backfill_ranking_families")
    @classmethod
    def _validate_backfill_ranking_families(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("Feature evidence backfill families must be non empty")
        return value


class AutoInterpConfig(BaseModel):
    enabled: bool = False
    ranking_families: list[Literal["policy_specific", "layer_unique", "global_dominance"]] = Field(
        default_factory=lambda: ["policy_specific", "layer_unique"]
    )
    top_n_per_family: int = 10
    generation_model: str = "google/gemma-2-2b-it"
    simulation_model: str | None = None
    num_train_positive: int = 3
    num_train_negative: int = 3
    num_holdout_positive: int = 4
    num_holdout_negative: int = 4
    low_activation_quantile: float = 0.2
    max_new_tokens: int = 160
    use_proxy_overlay_context: bool = True

    @field_validator("ranking_families")
    @classmethod
    def _validate_autointerp_families(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("AutoInterp ranking families must be non empty")
        return value


class BaselinesConfig(BaseModel):
    sentence_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    logistic_c: float = 1.0
    max_iter: int = 1000
    sparse_binarize_threshold: float = 0.0
    individual_feature_top_k: int = 64
    compute_feature_module_overlap: bool = True


class MaskingConfig(BaseModel):
    anchor_top_k: int = 10
    mask_token: str = "[MASK]"


class AblationConfig(BaseModel):
    target_proxies: list[str] = Field(default_factory=lambda: ["privacy", "bias"])
    candidate_set_sizes: list[int] = Field(default_factory=lambda: [1, 3, 5])
    random_control_trials: int = 100
    dense_control_type: Literal["pca_subspace"] = "pca_subspace"
    dense_control_rank_policy: Literal["match_sparse_cardinality"] = "match_sparse_cardinality"
    individual_probe_top_k: int = 3
    include_autointerp_single_feature_targets: bool = True
    include_autointerp_feature_set_targets: bool = True
    autointerp_target_ranking_families: list[Literal["policy_specific", "layer_unique", "global_dominance"]] = Field(
        default_factory=lambda: ["policy_specific", "layer_unique"]
    )
    autointerp_top_features_per_layer: int = 1
    autointerp_feature_set_sizes: list[int] = Field(default_factory=lambda: [3, 5])
    autointerp_min_faithfulness: float = 0.5
    evaluation_split: Literal["test", "dev"] = "test"
    use_matched_negative_evaluation: bool = True
    bootstrap_iterations: int = 2000
    ci_level: float = 0.95

    @field_validator("autointerp_target_ranking_families")
    @classmethod
    def _validate_autointerp_target_families(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("AutoInterp ablation ranking families must be non empty")
        return value

    @field_validator("autointerp_feature_set_sizes")
    @classmethod
    def _validate_feature_set_sizes(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("AutoInterp feature set sizes must be non empty")
        if any(item <= 1 for item in value):
            raise ValueError("AutoInterp feature set sizes must be greater than 1")
        return value


class SteeringConfig(BaseModel):
    enabled: bool = True
    alpha_grid: list[float] = Field(default_factory=lambda: [0.5, 1.0, 2.0])
    target_layers: list[int] = Field(default_factory=lambda: [24])
    min_dev_positives_for_sweep: int = 10
    default_alpha_if_small_dev: float = 1.0
    report_overlap_with_sae: bool = True


class AuditFamilyConfig(BaseModel):
    family_id: str
    family_label: str
    document_ids: list[int]
    max_cases: int = 5

    @field_validator("family_id", "family_label")
    @classmethod
    def _validate_non_empty_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Audit family fields must be non empty")
        return cleaned

    @field_validator("document_ids")
    @classmethod
    def _validate_document_ids(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("Audit family document_ids must be non empty")
        return value


class AuditConfig(BaseModel):
    enabled: bool = True
    families: list[AuditFamilyConfig] = Field(default_factory=list)
    preferred_splits: list[str] = Field(default_factory=lambda: ["validated", "test", "dev", "train"])
    segment_length_min: int = 180
    segment_length_max: int = 1400
    overall_top_k: int = 15
    policy_specific_top_k: int = 15
    global_anchor_count: int = 1
    perturbations: list[Literal["heading_removal", "lexical_anchor_masking", "sentence_compression"]] = Field(
        default_factory=lambda: ["heading_removal", "lexical_anchor_masking", "sentence_compression"]
    )
    perturbation_anchor_top_k: int = 3
    sentence_compression_keep_sentences: int = 2
    causal_concentration_cases_per_family: int = 2
    causal_concentration_random_trials: int = 3
    causal_concentration_bootstrap_iterations: int = 200
    flat_proxy_margin_threshold: float = 0.02
    near_zero_kl_threshold: float = 0.001
    near_zero_paired_threshold: float = 0.01

    @field_validator("families")
    @classmethod
    def _validate_families(cls, value: list[AuditFamilyConfig]) -> list[AuditFamilyConfig]:
        if not value:
            raise ValueError("Audit families must be non empty")
        return value

    @field_validator("preferred_splits", "perturbations")
    @classmethod
    def _validate_non_empty_lists(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("Audit lists must be non empty")
        return value


class ReportConfig(BaseModel):
    export_dir: str = "paper_exports"
    dossier_format: Literal["md", "html"] = "md"
    heatmap_dpi: int = 180
    export_feature_catalog: bool = True
    export_layer_overlap: bool = True
    export_causal_feature_tables: bool = True
    export_batch_scorer_reports: bool = True
    export_autointerp_reports: bool = True
    export_autointerp_causal_reports: bool = True
    export_audit_reports: bool = True
    autointerp_top_features_per_layer: int = 5
    autointerp_high_faithfulness_threshold: float = 2.0 / 3.0


class PilotConfig(BaseModel):
    enabled: bool = False
    source_run_name: str
    sample_size: int = 200
    include_proxies: list[str] = Field(default_factory=lambda: ["privacy", "bias"])
    include_splits: list[str] = Field(default_factory=lambda: ["train", "dev", "test"])
    include_matched_negatives: bool = True
    positive_target_size: int = 120
    matched_negative_target_size: int = 40
    random_fill_seed: int = DEFAULT_SEED
    stratify_fill_by_split: bool = True

    @model_validator(mode="after")
    def _validate_targets(self) -> "PilotConfig":
        if self.sample_size <= 0:
            raise ValueError("Pilot sample size must be positive")
        if self.positive_target_size < 0 or self.matched_negative_target_size < 0:
            raise ValueError("Pilot target sizes must be non negative")
        if self.positive_target_size + self.matched_negative_target_size > self.sample_size:
            raise ValueError("Pilot positive and matched negative targets cannot exceed sample size")
        return self


class ExperimentConfig(BaseModel):
    name: str
    dataset: DatasetConfig
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    backbone: BackboneConfig = Field(default_factory=BackboneConfig)
    sae: SaeConfig = Field(default_factory=SaeConfig)
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    validity: ValidityConfig = Field(default_factory=ValidityConfig)
    labeling: LabelingConfig = Field(default_factory=LabelingConfig)
    feature_catalog: FeatureCatalogConfig = Field(default_factory=FeatureCatalogConfig)
    autointerp: AutoInterpConfig = Field(default_factory=AutoInterpConfig)
    baselines: BaselinesConfig = Field(default_factory=BaselinesConfig)
    masking: MaskingConfig = Field(default_factory=MaskingConfig)
    ablation: AblationConfig = Field(default_factory=AblationConfig)
    steering: SteeringConfig = Field(default_factory=SteeringConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    pilot: PilotConfig | None = None

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, value: str) -> str:
        return value.strip().replace(" ", "_")

    @property
    def run_root(self) -> Path:
        return self.dataset.artifacts_path / self.name
