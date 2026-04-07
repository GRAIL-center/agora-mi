"""Typer CLI for the Policy Interp pipeline."""

from __future__ import annotations

import typer

from policy_interp.agora import load_and_prepare_agora
from policy_interp.audit_eval import run_audit_evaluation
from policy_interp.autointerp import run_autointerp
from policy_interp.baselines import run_baselines
from policy_interp.batch_scorer import run_batch_scorer
from policy_interp.config import load_experiment_config
from policy_interp.discovery import run_module_discovery
from policy_interp.extract import run_extraction
from policy_interp.feature_catalog import build_feature_catalog, run_feature_labeling
from policy_interp.interventions import run_interventions
from policy_interp.io import read_parquet, write_parquet
from policy_interp.labeling import run_labeling
from policy_interp.masking import run_masking_retention
from policy_interp.matching import build_matched_negatives, build_text_embeddings
from policy_interp.pilot import build_pilot_subset
from policy_interp.reproducibility import build_artifact_hash_report
from policy_interp.reporting import export_reports
from policy_interp.utils import ensure_dir

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("validate-config")
def validate_config(config_path: str) -> None:
    config = load_experiment_config(config_path)
    typer.echo(f"Config OK: {config.name}")


@app.command("prepare-agora")
def prepare_agora(config_path: str) -> None:
    config = load_experiment_config(config_path)
    paths = load_and_prepare_agora(config)
    typer.echo(f"Prepared segments: {paths.segments}")
    typer.echo(f"Prepared documents: {paths.documents}")
    typer.echo(f"Split manifest: {paths.splits}")


@app.command("build-matches")
def build_matches(config_path: str) -> None:
    config = load_experiment_config(config_path)
    prepared_segments_path = config.run_root / config.dataset.prepared_segments_name
    if not prepared_segments_path.exists():
        raise typer.BadParameter("Prepared segments not found. Run prepare-agora first.")
    segments = read_parquet(prepared_segments_path)
    embeddings = build_text_embeddings(segments, config)
    matches = build_matched_negatives(segments, embeddings, config)
    matches_root = ensure_dir(config.run_root / "matching")
    embeddings_path = matches_root / "segment_text_embeddings.parquet"
    matches_path = matches_root / "matched_negatives.parquet"
    write_parquet(embeddings, embeddings_path)
    write_parquet(matches, matches_path)
    typer.echo(f"Embeddings: {embeddings_path}")
    typer.echo(f"Matches: {matches_path}")


@app.command("build-pilot")
def build_pilot(config_path: str) -> None:
    config = load_experiment_config(config_path)
    outputs = build_pilot_subset(config)
    typer.echo(f"Pilot prepared segments: {outputs['prepared_segments']}")
    typer.echo(f"Pilot prepared documents: {outputs['prepared_documents']}")
    typer.echo(f"Pilot split manifest: {outputs['split_manifest']}")


@app.command("extract-activations")
def extract_activations(config_path: str) -> None:
    config = load_experiment_config(config_path)
    artifacts = run_extraction(config)
    typer.echo(f"Extraction complete: {artifacts.layer_feature_summary_paths}")


@app.command("discover-modules")
def discover_modules(config_path: str) -> None:
    config = load_experiment_config(config_path)
    artifacts = run_module_discovery(config)
    typer.echo(f"Module candidates: {artifacts.module_candidates_path}")
    typer.echo(f"Stable modules: {artifacts.module_stability_path}")


@app.command("run-baselines")
def run_baseline_stage(config_path: str) -> None:
    config = load_experiment_config(config_path)
    artifacts = run_baselines(config)
    typer.echo(f"Baseline comparison: {artifacts.comparison_path}")
    typer.echo(f"Sparse feature selection: {artifacts.sparse_feature_selection_path}")
    typer.echo(f"Feature module overlap: {artifacts.sparse_feature_module_overlap_path}")


@app.command("run-masking")
def run_masking_stage(config_path: str) -> None:
    config = load_experiment_config(config_path)
    path = run_masking_retention(config)
    typer.echo(f"Masking retention: {path}")


@app.command("run-interventions")
def run_intervention_stage(config_path: str) -> None:
    config = load_experiment_config(config_path)
    artifacts = run_interventions(config)
    typer.echo(f"Ablation table: {artifacts.ablation_path}")
    typer.echo(f"Proxy effects: {artifacts.proxy_effects_path}")
    typer.echo(f"Steering table: {artifacts.steering_path}")
    typer.echo(f"Feature causal summary: {artifacts.feature_summary_path}")
    typer.echo(f"Feature causal per segment: {artifacts.feature_per_segment_path}")


@app.command("label-modules")
def label_modules(config_path: str) -> None:
    config = load_experiment_config(config_path)
    artifacts = run_labeling(config)
    typer.echo(f"Module labels: {artifacts.labels_path}")


@app.command("build-feature-catalog")
def build_feature_catalog_stage(config_path: str) -> None:
    config = load_experiment_config(config_path)
    artifacts = build_feature_catalog(config)
    typer.echo(f"Feature summary: {artifacts.summary_path}")
    typer.echo(f"Feature catalog: {artifacts.catalog_path}")
    typer.echo(f"Feature overlay: {artifacts.proxy_overlay_path}")


@app.command("label-features")
def label_features(config_path: str) -> None:
    config = load_experiment_config(config_path)
    artifacts = run_feature_labeling(config)
    typer.echo(f"Feature labels: {artifacts.labels_path}")


@app.command("run-autointerp")
def run_autointerp_stage(config_path: str) -> None:
    config = load_experiment_config(config_path)
    artifacts = run_autointerp(config)
    typer.echo(f"AutoInterp candidates: {artifacts.candidates_path}")
    typer.echo(f"AutoInterp simulation: {artifacts.simulation_path}")
    typer.echo(f"AutoInterp scores: {artifacts.scores_path}")
    typer.echo(f"AutoInterp labels: {artifacts.labels_path}")


@app.command("run-batch-scorer")
def run_batch_scorer_stage(
    config_path: str,
    input_path: str,
    output_name: str = typer.Option("batch_scorer", help="Output directory name under features/batch_scorer."),
    segment_mode: str = typer.Option("paragraph", help="One of paragraph, line, or whole."),
) -> None:
    config = load_experiment_config(config_path)
    artifacts = run_batch_scorer(
        config=config,
        input_path=input_path,
        output_name=output_name,
        segment_mode=segment_mode,
    )
    typer.echo(f"Document feature scores: {artifacts.document_feature_scores_path}")
    typer.echo(f"Top feature evidence: {artifacts.top_feature_evidence_path}")
    typer.echo(f"Layer profile summary: {artifacts.layer_profile_summary_path}")
    typer.echo(f"Proxy overlay summary: {artifacts.proxy_overlay_summary_path}")
    typer.echo(f"Causal notes: {artifacts.causal_notes_path}")
    typer.echo(f"Report: {artifacts.report_path}")


@app.command("run-audit-eval")
def run_audit_eval_stage(config_path: str) -> None:
    config = load_experiment_config(config_path)
    artifacts = run_audit_evaluation(config)
    typer.echo(f"Audit case manifest: {artifacts.case_manifest_path}")
    typer.echo(f"Audit discriminativeness: {artifacts.discriminativeness_summary_path}")
    typer.echo(f"Audit robustness: {artifacts.robustness_summary_path}")
    typer.echo(f"AutoInterp validation: {artifacts.autointerp_validation_path}")
    typer.echo(f"Failure transparency: {artifacts.failure_transparency_path}")


@app.command("export-reports")
def export_report_stage(config_path: str) -> None:
    config = load_experiment_config(config_path)
    outputs = export_reports(config)
    for key, value in outputs.items():
        typer.echo(f"{key}: {value}")


@app.command("hash-artifacts")
def hash_artifacts(config_path: str) -> None:
    config = load_experiment_config(config_path)
    report = build_artifact_hash_report(config.run_root)
    typer.echo(f"Artifact hashes: {report}")


@app.command("run-all")
def run_all(config_path: str) -> None:
    config = load_experiment_config(config_path)
    load_and_prepare_agora(config)
    prepared_segments_path = config.run_root / config.dataset.prepared_segments_name
    segments = read_parquet(prepared_segments_path)
    embeddings = build_text_embeddings(segments, config)
    matches = build_matched_negatives(segments, embeddings, config)
    matches_root = ensure_dir(config.run_root / "matching")
    write_parquet(embeddings, matches_root / "segment_text_embeddings.parquet")
    write_parquet(matches, matches_root / "matched_negatives.parquet")
    run_extraction(config)
    run_module_discovery(config)
    run_baselines(config)
    run_masking_retention(config)
    run_interventions(config)
    run_labeling(config)
    build_feature_catalog(config)
    run_feature_labeling(config)
    if config.autointerp.enabled:
        run_autointerp(config)
    if config.audit.enabled:
        run_audit_evaluation(config)
    export_reports(config)
    build_artifact_hash_report(config.run_root)
    typer.echo("Pipeline complete")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
