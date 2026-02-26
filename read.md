# RUNBOOK — Mapping Safety vs Innovation Circuits (Gemma-2-2B + AGORA + SAEs)

Owner: Seunghyun Yoo  
Goal: Build a reproducible pipeline that (1) constructs contrastive corpora (Dsafe/Dinnov) from AGORA tags, (2) extracts SAE features from Gemma-2-2B residual activations, (3) finds polarized features via contrastive stats, (4) runs causal interventions (ablation/clamp), and (5) optionally constructs circuit graphs and compares topology.

This runbook is written for Codex to implement as a clean, CLI-driven research repo.

---

## 0) High-level Targets (MVP first)

### MVP (Today)
1) Build `Dsafe_dev.jsonl` and `Dinnov_dev.jsonl` (≈200 each), document-level split.
2) Smoke test: load Gemma-2-2B; run forward; extract residual stream at chosen layer.
3) Smoke test: load Gemma Scope SAE; encode residuals; print top active features.
4) Compute polarization ∆ for dev set; save top-k features (k=64) to CSV.
5) Pilot ablation: ablate top safety features on Dsafe prompts; measure Δlogprob for target label token.

### Next (This Week)
6) Scale to full train/dev/test; add sanity checks (length confound, simple AUC).
7) Implement bootstrap CIs and permutation tests; add FDR correction.
8) Implement interference clamp experiment.
9) Optional: circuit-tracer graph extraction and topology metrics.

---

## 1) Repo Layout (Create exactly)

