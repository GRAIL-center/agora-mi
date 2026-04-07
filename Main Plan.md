# Experimental design plan for revising a mechanistic interpretability paper on policy text analysis

**The core revision strategy should pivot from a proxy-label-driven pipeline to a bottom-up mechanistic approach that discovers, validates, and causally tests policy-relevant features in Gemma 2 using Gemma Scope SAEs—grounding every claim in causal evidence rather than correlational proxy performance.** This plan draws on 60+ papers from 2020–2026, spanning circuit tracing, sparse feature circuits, activation oracles, SAEBench evaluations, and domain-specific SAE work. The revision addresses the paper's three critical weaknesses: the CoreScore gap (0.530 vs. 0.617), the lack of causal replication beyond Privacy, and the vulnerability to the "strong feature hypothesis" critique. Below is a complete experimental design organized into concrete phases, with specific methods, figures, statistics, and appendix content.

## Phase 1: Bottom-up feature discovery replaces proxy-label-driven search

The existing pipeline starts from proxy labels and searches for SAE features that correlate with them. This approach is vulnerable to the critique that discovered features are statistical artifacts of the label distribution rather than genuine mechanistic units. The revised pipeline inverts this: start from SAE activations on AGORA text, discover structure in those activations, and only then evaluate alignment with proxy labels.

**Experiment 1.1 — Unsupervised SAE feature profiling across AGORA.** Run all AGORA policy segments through Gemma 2 2B and 9B using Gemma Scope JumpReLU SAEs (Lieberum et al., 2024, arXiv:2408.05147) at three widths: **16K, 65K, and 131K** (for 9B). Extract token-level SAE activations at the residual stream site for layers 12, 16, 20, and 24 (2B) and layers 18, 24, 30, and 36 (9B), chosen because middle-to-late layers encode more semantic rather than syntactic features. For each feature, compute: activation frequency across AGORA, mean activation magnitude, and max-activating token contexts. This produces a **feature activation matrix** of shape (n_documents × n_features) that is entirely label-free.

**Experiment 1.2 — Feature clustering and co-activation graph construction.** Following Li et al. (2025, "The Geometry of Concepts," *Entropy* 27(4)) and the sparse feature co-activation work (arXiv:2506.18141, 2025), build a co-activation graph where nodes are SAE features and edges represent co-occurrence frequency above a threshold (e.g., Jaccard similarity > 0.1 across AGORA segments). Apply community detection (Louvain or spectral clustering) to identify **feature modules**—groups of features that consistently co-activate on policy text. Visualize these modules and label them post hoc using AutoInterp (Paulo et al., "Automatically Interpreting Millions of Features in Large Language Models," ICML 2025, arXiv:2410.13928) to generate natural language descriptions. This produces the first bottom-up map of what Gemma 2 "sees" in policy text, independent of any proxy labels.

**Experiment 1.3 — Feature-to-proxy alignment analysis.** Only after Experiments 1.1–1.2 are complete, measure how discovered feature modules align with the six AGORA proxy tasks (Bias, Discrimination, Privacy, Rights violation, Transparency, Interpretability). For each module, compute mutual information and AUROC against each proxy label. Report a **module-proxy alignment matrix** as a heatmap figure. The key hypothesis is that some modules will align cleanly with single proxies (validating the proxy), while others will span multiple proxies or represent concepts absent from the proxy taxonomy entirely (revealing the taxonomy's limitations).

**Figure 1** (main text): Heatmap of feature module vs. proxy label alignment scores, with dendrogram showing module clustering. **Figure 2** (main text): Co-activation graph with community coloring and AutoInterp-generated module labels.

**Appendix A**: Full AutoInterp descriptions for top-50 most activated features per module. Include activation examples (token-level highlighting) from Neuronpedia-style displays.

---

## Phase 2: Causal validation through multi-method convergence

The original paper's causal evidence rests on ablation alone, and only Privacy replicates across 2B and 9B. The revision must establish causal validity through converging evidence from at least three independent methods. This addresses the Lewis Smith critique (2024, "The 'strong' feature hypothesis could be wrong," Alignment Forum) and the Paulo & Belrose (2025, arXiv:2501.16615) finding that only **30% of SAE features are shared across random seeds**.

**Experiment 2.1 — Cross-seed feature stability for policy features.** Train 3 additional SAEs on the same Gemma 2 2B layer-20 residual stream activations using different random seeds, matching Gemma Scope hyperparameters (JumpReLU, width 65K). For each policy-relevant feature identified in Phase 1, compute cosine similarity with the nearest feature in each re-trained SAE. Report the fraction of policy-relevant features that replicate (cosine > 0.7) across seeds. Paulo & Belrose found ~30% replication at this threshold in general; the hypothesis is that policy-relevant features replicate at a higher rate because they represent genuine, high-frequency concepts in the model's representation. **If replication is below 50%, this is a critical negative result that must be reported honestly.**

**Experiment 2.2 — Sparse feature circuit discovery.** Apply the Sparse Feature Circuits method (Marks et al., "Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models," ICLR 2025, arXiv:2403.19647) to each proxy task. For Privacy (the best-replicated case), construct a full sparse feature circuit by computing attribution patching through SAE features. This method computes approximate indirect effects for each SAE feature and prunes features below a threshold. The circuit should reveal not just which features are important, but how they connect—which attention-output features feed into which MLP-output features, and how information flows from token-level pattern detection to document-level classification.

Use SHIFT (Sparse Human-Interpretable Feature Trimming) from the same paper to ablate features judged task-irrelevant by the circuit analysis, then measure whether classification generalizes better. If SHIFT improves OOD generalization on held-out AGORA documents, this provides strong evidence that the circuit captures genuine policy-processing mechanisms rather than spurious correlations.

**Experiment 2.3 — Contrastive activation analysis with steering vectors.** Following Panickssery et al. ("Steering Llama 2 via Contrastive Activation Addition," ACL 2024, arXiv:2312.06681), compute difference-in-means steering vectors for each proxy task. For Privacy: take the mean residual-stream activation across all Privacy-positive AGORA segments and subtract the mean activation across Privacy-negative segments, at each layer. This produces a **privacy direction** in activation space. Test its causal role by adding/subtracting this vector during inference on held-out policy text and measuring the change in proxy-relevant model behavior (e.g., next-token prediction shifts, classification logit changes).

Critically, decompose the steering vector into its SAE feature components using the SAE decoder matrix. If the privacy steering vector aligns strongly with the SAE features identified in Experiment 2.2, this provides converging causal evidence from two independent methods. Use the Feature-Guided Activation Additions (FGAA) method (Soo et al., 2025) to construct interpretable steering vectors in SAE latent space and compare their effectiveness.

**Experiment 2.4 — Integrated gradients attribution.** Apply integrated gradients (Sundararajan et al., "Axiomatic Attribution for Deep Networks," ICML 2017, arXiv:1703.01365) through the SAE-augmented model to compute token-level attribution scores for each proxy classification. Compare the features identified by IG against those from Experiment 2.2 (attribution patching) and 2.3 (steering vectors). Compute a **method agreement matrix**: for each pair of methods, what fraction of the top-k features overlap?

**Figure 3** (main text): Sparse feature circuit diagram for Privacy, showing the computational graph from input tokens through SAE features to output logits. **Figure 4** (main text): Three-way Venn diagram showing feature overlap across attribution patching, steering vectors, and integrated gradients. **Table 1** (main text): Method agreement statistics (Jaccard similarity of top-k feature sets) across all three causal methods, for each proxy task.

**Appendix B**: Full circuit diagrams for all 6 proxy tasks. Steering vector decomposition into SAE features with cosine similarity scores.

---

## Phase 3: Layer-wise feature evolution and the mechanistic story

One of the strongest contributions the revised paper can make—absent from most SAE-for-classification work—is tracing how policy concepts develop through the transformer's layers. Gemma Scope provides SAEs at every layer and sublayer, enabling this analysis out of the box.

**Experiment 3.1 — Tuned lens prediction depth analysis.** Apply the tuned lens (Belrose et al., "Eliciting Latent Predictions from Transformers with the Tuned Lens," arXiv:2303.08112, 2023) to AGORA text in Gemma 2 2B and 9B. For each token, compute the layer at which the model's prediction "crystallizes" (prediction depth). Compare prediction depth for policy-relevant tokens (e.g., "privacy," "bias," "transparency") vs. generic tokens. The hypothesis: **policy-relevant concept tokens have deeper prediction trajectories** because they require more contextual processing to interpret correctly in a regulatory context.

**Experiment 3.2 — Layer-wise SAE feature tracking.** For the top policy-relevant features identified in Phase 1, track their activation patterns across all layers using Gemma Scope's per-layer SAEs. Following Laptev et al. ("Systematically Mapping Features Across Consecutive Layers," ICML 2025), use cosine similarity between SAE decoder directions at consecutive layers to map feature evolution. Identify where policy features first emerge, where they peak, and where they merge or split. This answers the mechanistic question: **at what depth does Gemma 2 develop policy-relevant representations?**

Also apply the sparse crosscoders method (Lindsey et al., "Sparse Crosscoders for Cross-Layer Feature Discovery," Transformer Circuits Thread, 2024) to track the same features across layers without relying on post-hoc mapping. If crosscoders and per-layer tracking agree on when features emerge, this provides converging evidence.

**Experiment 3.3 — Attention vs. MLP contribution decomposition.** For each policy-relevant feature identified in Phase 2, decompose its activation into contributions from attention heads vs. MLP layers using Gemma Scope's site-specific SAEs (attention output, MLP output, and residual stream). Following Geva et al. ("Transformer Feed-Forward Layers Are Key-Value Memories," EMNLP 2021), test whether MLP layers store policy-relevant factual associations (e.g., "EPA requires..." → regulatory compliance pattern) while attention layers compose them.

**Figure 5** (main text): Layer-wise activation heatmap for top-10 policy features across all layers of Gemma 2 2B, with phase annotations (emergence, peak, stabilization). **Figure 6** (main text): Tuned lens prediction depth distribution for policy-relevant vs. generic tokens, showing that policy concepts require deeper processing.

**Appendix C**: Full layer-wise tracking for all policy-relevant features. Attention head vs. MLP contribution breakdown tables.

---

## Phase 4: Rigorous comparison against baselines closes the CoreScore gap narrative

The current paper's weakest point is that SAE features lose to sentence embeddings on aggregate benchmarks (CoreScore 0.530 vs. 0.617). The revision should reframe this gap as an expected consequence of the interpretability-performance tradeoff, supported by quantitative evidence from the literature, while also showing where SAE features provide unique value that embeddings cannot.

**Experiment 4.1 — Dense probe vs. SAE probe head-to-head.** Following Smith et al. (2025, "Negative Results for Sparse Autoencoders on Downstream Tasks," GDM Safety Research), train dense linear probes on the same Gemma 2 layer activations used for SAE feature extraction. Compare: (a) 1-sparse SAE probes (single best feature), (b) k-sparse SAE probes (top-k features), (c) full SAE feature logistic regression, (d) dense linear probe on raw activations, (e) sentence embedding baseline (e.g., all-MiniLM-L6-v2 + logistic regression). Report AUROC, F1, and calibration for each proxy task. The GDM results predict dense probes will outperform SAE probes, but the gap should narrow with k-sparse probes.

Crucially, also measure **out-of-distribution generalization**: train on AGORA documents from one jurisdiction and test on another, or train on pre-2023 documents and test on 2023–2025 documents. The Goodfire/Rakuten (2025) result that SAE probes generalize better from synthetic to real data suggests SAE features may show an advantage on distribution shift. Also follow Gallifant et al. ("Sparse Autoencoder Features for Classifications and Transferability," EMNLP 2025) who found SAE-derived features achieve macro F1 > 0.8 and demonstrate cross-model transfer from Gemma 2 2B to 9B-IT.

**Experiment 4.2 — SAE feature binarization and pooling strategies.** Following Gallifant et al. (EMNLP 2025), systematically test binarization of continuous SAE activations (feature active/inactive) and different pooling strategies (max-pooling, mean-pooling, CLS-position) for document-level classification. Their work found binarization is competitive with continuous activations and much more interpretable.

**Experiment 4.3 — Interpretability advantage demonstration.** For cases where SAE features and dense probes disagree on classification, use AutoInterp descriptions and activation examples to show that SAE features provide **explanatory transparency** that dense probes cannot. Create a side-by-side comparison: for a misclassified AGORA document, show which SAE features activated (with human-readable descriptions) vs. the opaque dense probe decision. This is the core value proposition.

**Table 2** (main text): Head-to-head classification performance across all methods and proxy tasks, with OOD generalization column. **Table 3** (main text): Interpretability scores (automated and human-judged) for SAE features vs. dense probe directions, following Bills et al. (2023, "Language Models Can Explain Neurons in Language Models," OpenAI) methodology.

**Appendix D**: Full classification results with confidence intervals. Pooling strategy ablation. OOD generalization curves.

---

## Phase 5: Width sensitivity and feature splitting analysis

Addressing the SAE hyperparameter sensitivity concern head-on strengthens the paper's methodological rigor. SAEBench (Karvonen et al., "SAEBench: A Comprehensive Benchmark for Sparse Autoencoders," ICML 2025, arXiv:2503.09532) showed that no single SAE configuration is optimal across all tasks.

**Experiment 5.1 — Width sensitivity analysis.** Repeat the Phase 1 feature discovery at three Gemma Scope widths: 16K, 65K, and 131K (9B only). For each policy-relevant feature at width 65K, identify its parent feature at 16K and child features at 131K using decoder cosine similarity. Document feature splitting patterns: does "privacy" at 16K split into "data privacy," "medical privacy," and "surveillance privacy" at 131K? This follows the Gemma Scope "feature-splitting suite" design and connects to the meta-SAE analysis of Leask et al. ("Sparse Autoencoders Do Not Find Canonical Units of Analysis," ICLR 2025, arXiv:2502.04878).

**Experiment 5.2 — Feature absorption detection.** Apply the Chanin et al. absorption metric ("A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders," arXiv:2409.14507, 2024) to policy-relevant features. Feature absorption—where a general "privacy" feature fails to fire on privacy-relevant text because more specific child features absorb its activation—is a known failure mode that worsens with width. If absorption is detected for policy features, test whether Matryoshka SAEs (arXiv:2503.17547, 2025) mitigate it, since SAEBench showed Matryoshka SAEs substantially outperform on feature disentanglement.

**Experiment 5.3 — Causal stability across widths.** For the sparse feature circuits discovered in Experiment 2.2, test whether the same circuit structure holds at different SAE widths. A genuine mechanistic circuit should be robust to feature granularity—the circuit at 131K may have more nodes but should preserve the same connectivity pattern. If the circuit fundamentally changes with width, this is a significant concern.

**Figure 7** (main text): Feature splitting tree for "Privacy" concept across 16K → 65K → 131K widths. **Table 4** (main text): Causal ablation effect sizes at each width, showing stability (or instability) of mechanistic findings.

**Appendix E**: Full width sensitivity tables. Feature absorption rates per proxy task. Matryoshka SAE comparison.

---

## Phase 6: Cross-model replication and model diffing

The original paper's finding that Privacy is the only proxy that replicates across 2B and 9B is a weakness. The revision should investigate why other proxies fail to replicate and use model diffing to understand the 2B–9B differences.

**Experiment 6.1 — Crosscoder analysis between 2B and 9B.** Following Minder et al. ("Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning," arXiv:2504.02922, NeurIPS 2025 Spotlight), train a BatchTopK crosscoder on matched layer activations from Gemma 2 2B and 9B processing the same AGORA text. Identify shared features (present in both models) and model-exclusive features (present in only one). For policy-relevant features, determine: which are shared (robust, likely genuine) and which are model-specific (potentially artifacts of model size or training)?

**Experiment 6.2 — Circuit comparison across model sizes.** For the Privacy circuit from Experiment 2.2, replicate the analysis in both 2B and 9B. Map corresponding circuit components using the crosscoder's shared feature dictionary. If the Privacy circuit has a similar structure in both models, this provides strong evidence for mechanism-level replication. For proxies that fail to replicate (e.g., Bias, Transparency), the crosscoder analysis can reveal whether the failure is due to different internal representations or simply different SAE decompositions of similar representations.

**Experiment 6.3 — Base vs. instruction-tuned comparison.** Gemma Scope includes SAEs for both pre-trained and instruction-tuned 9B models. Following Kissane et al. ("SAEs (Usually) Transfer Between Base and Chat Models," 2024) and their finding that transfer is imperfect, compare policy feature activations between base and IT models. Policy concepts may be represented differently in IT models that have been trained to follow regulatory-analysis instructions.

**Figure 8** (main text): Crosscoder visualization showing shared vs. model-exclusive policy features between 2B and 9B. **Table 5** (main text): Circuit-level replication rates across model sizes for each proxy task.

**Appendix F**: Full crosscoder training details. Feature-by-feature comparison tables. Base vs. IT activation distribution comparisons.

---

## Phase 7: Activation oracle evaluation and the auditing connection

The AuditBench results (Anthropic, 2026, arXiv:2602.22755) show that activation oracles are the best white-box interpretability tool for model auditing. Connecting the paper to this evaluation framework strengthens its policy-relevance motivation.

**Experiment 7.1 — Activation oracle for policy feature discovery.** Following Karvonen et al. ("Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers," arXiv:2512.15674, 2025/2026), use an activation oracle to query Gemma 2's activations on AGORA text with natural language questions: "Is this text about privacy regulation?", "Does this document discuss algorithmic bias?", "What governance topic does this text address?" Compare oracle answers against (a) SAE feature-based classification and (b) proxy labels. If the oracle identifies policy-relevant information that SAE features miss—or vice versa—this illuminates the complementary strengths of each approach.

**Experiment 7.2 — Tool-to-analyst gap analysis.** Inspired by AuditBench's "tool-to-agent gap" finding, evaluate whether SAE features can actually be used by a policy analyst to audit model behavior. Create a structured evaluation: present 20 AGORA segments with their SAE feature activations and AutoInterp descriptions to 5 policy domain experts. Ask them to (a) identify the governance topic, (b) rate feature descriptions for usefulness, and (c) identify any policy-relevant aspects the features miss. This measures the practical value of SAE interpretability for the intended use case.

Wang et al. ("Persona Features Control Emergent Misalignment," arXiv:2506.19823, OpenAI, 2025) demonstrated that SAE-based model diffing can identify specific latents responsible for misaligned behavior—including a single "toxic persona" latent whose ablation drops misalignment from ~80% to ~12%. Their Δ-attribution methodology provides a template for identifying the most causally important policy-relevant features in Gemma 2.

**Table 6** (main text): Activation oracle vs. SAE feature-based classification comparison on AGORA proxy tasks. **Appendix G**: Full activation oracle query results. Policy analyst evaluation rubric and results.

---

## Phase 8: Addressing the strong feature hypothesis directly

Rather than avoiding the critique, the revision should engage with it head-on, treating it as a strength: the paper explicitly tests and reports the limits of SAE-based mechanistic interpretability for domain-specific text analysis.

**Experiment 8.1 — Meta-SAE decomposition of policy features.** Following Leask et al. (ICLR 2025, arXiv:2502.04878), train a meta-SAE on the decoder matrix of the 65K Gemma Scope SAE. For each policy-relevant feature, decompose it into meta-latent components. A "privacy" feature might decompose into "personal data" + "legal requirement" + "digital context" meta-latents. If meta-latents are more interpretable and more stable across seeds than the original features, this suggests the original features are compositional rather than atomic—supporting a weaker but still useful version of the feature hypothesis.

**Experiment 8.2 — Feature geometry analysis for policy concepts.** Following Engels et al. ("Not All Language Model Features Are One-Dimensionally Linear," ICLR 2025, arXiv:2405.14860), analyze the geometric structure of policy-relevant SAE features. Are features for related proxy tasks (e.g., Bias and Discrimination) arranged in a meaningful geometric pattern in the decoder space? If policy features form clusters, gradients, or manifold structures rather than isolated directions, this provides evidence for multi-dimensional policy representations that SAEs decompose into 1D projections.

**Experiment 8.3 — Causal scrubbing validation.** Following Chan et al. ("Causal Scrubbing," Redwood Research, 2022), formalize the mechanistic hypothesis from Phase 2's circuit analysis and test it via causal scrubbing. This is the most rigorous test: if resampling activations according to the hypothesis preserves model performance, the hypothesis is validated. If performance degrades significantly, the hypothesis is incomplete.

**Figure 9** (main text): Meta-SAE decomposition tree for top policy features. **Figure 10** (main text): Feature geometry visualization (PCA/UMAP) of policy-relevant SAE decoder directions, colored by proxy task.

**Appendix H**: Full meta-SAE decomposition tables. Causal scrubbing performance curves. Feature geometry analysis details.

---

## Comprehensive reference architecture for all experiments

All experiments use the following shared infrastructure and should be reported with these standard details:

**Models**: Gemma 2 2B (pre-trained), Gemma 2 9B (pre-trained and instruction-tuned). **SAEs**: Gemma Scope JumpReLU SAEs at widths 16K, 65K, 131K; residual stream, attention output, and MLP output sites. **Dataset**: AGORA AI Governance and Regulatory Archive (Arnold et al., "Introducing the AI Governance and Regulatory Archive," AIES 2024, DOI:10.1609/aies.v7i1.31615), 950+ documents with thematic annotations. **Evaluation**: SAEBench metrics (loss recovered, sparse probing, automated interpretability, feature absorption, SCR, TPP) applied to all SAEs used. **Baselines**: Dense linear probes, sentence embeddings (all-MiniLM-L6-v2), full-model fine-tuning, LIME/SHAP post-hoc explanations. **Compute**: All SAE training on TPUv3 or equivalent; activation extraction parallelized across documents.

---

## Statistical reporting standards for all experiments

Every experiment should report: **effect sizes** with 95% confidence intervals (bootstrap, 10,000 resamples), **multiple comparison corrections** (Benjamini-Hochberg FDR for proxy-task-level comparisons), and **practical significance** alongside statistical significance. For feature stability (Experiment 2.1), report the full distribution of cosine similarities, not just the fraction above threshold. For classification (Experiment 4.1), report calibration curves alongside discrimination metrics. For causal experiments (Experiments 2.2–2.4), report both the faithfulness (fraction of behavior explained by the circuit) and completeness (fraction of behavior captured) metrics from Marks et al. (2025).

---

## Key papers that must be cited in the revision

The following papers form the methodological backbone and should be cited with the details below:

**SAE foundations**: Bricken et al. (2023), "Towards Monosemanticity," Transformer Circuits Thread. Templeton et al. (2024), "Scaling Monosemanticity," Transformer Circuits Thread. Cunningham et al. (2024), "Sparse Autoencoders Find Highly Interpretable Features," ICLR 2024, arXiv:2309.08600.

**Gemma Scope**: Lieberum et al. (2024), "Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2," BlackboxNLP at EMNLP 2024, arXiv:2408.05147.

**SAE architectures**: Rajamanoharan et al. (2024), "Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders," ICLR 2025, arXiv:2407.14435. Gao et al. (2024), "Scaling and Evaluating Sparse Autoencoders," ICLR 2025, arXiv:2406.04093.

**SAE evaluation**: Karvonen et al. (2025), "SAEBench," ICML 2025, arXiv:2503.09532. Chanin et al. (2024), "A is for Absorption," arXiv:2409.14507. Paulo & Belrose (2025), "Sparse Autoencoders Trained on the Same Data Learn Different Features," arXiv:2501.16615.

**SAE critique**: Smith (2024), "The 'strong' feature hypothesis could be wrong," Alignment Forum. Smith et al. (2025), "Negative Results for Sparse Autoencoders on Downstream Tasks," GDM Safety Research blog.

**Feature geometry**: Engels et al. (2024), "Not All Language Model Features Are One-Dimensionally Linear," ICLR 2025, arXiv:2405.14860. Li et al. (2025), "The Geometry of Concepts," Entropy 27(4).

**Meta-SAEs**: Leask et al. (2025), "Sparse Autoencoders Do Not Find Canonical Units of Analysis," ICLR 2025, arXiv:2502.04878.

**Circuit methods**: Marks et al. (2025), "Sparse Feature Circuits," ICLR 2025, arXiv:2403.19647. Ameisen et al. (2025), "Circuit Tracing: Revealing Computational Graphs in Language Models," Transformer Circuits Thread. Conmy et al. (2023), "Towards Automated Circuit Discovery," NeurIPS 2023, arXiv:2304.14997. Hanna et al. (2024), "Attribution Patching Outperforms Automated Circuit Discovery," BlackboxNLP at EMNLP 2024.

**Causal methods**: Vig et al. (2020), "Investigating Gender Bias Using Causal Mediation Analysis," NeurIPS 2020, arXiv:2004.12265. Meng et al. (2022), "Locating and Editing Factual Associations in GPT," NeurIPS 2022, arXiv:2202.05262. Geiger et al. (2025), "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability," JMLR 26(83):1–64. Sundararajan et al. (2017), "Axiomatic Attribution for Deep Networks," ICML 2017, arXiv:1703.01365.

**Steering vectors**: Turner et al. (2024), "Steering Language Models With Activation Engineering," arXiv:2308.10248. Panickssery et al. (2024), "Steering Llama 2 via Contrastive Activation Addition," ACL 2024, arXiv:2312.06681. Zou et al. (2023), "Representation Engineering," arXiv:2310.01405.

**Probing**: Belrose et al. (2023), "Eliciting Latent Predictions with the Tuned Lens," arXiv:2303.08112. Geva et al. (2021), "Transformer Feed-Forward Layers Are Key-Value Memories," EMNLP 2021.

**AutoInterp**: Bills et al. (2023), "Language Models Can Explain Neurons in Language Models," OpenAI. Paulo et al. (2025), "Automatically Interpreting Millions of Features," ICML 2025, arXiv:2410.13928.

**Auditing and evaluation**: AuditBench (Anthropic, 2026), arXiv:2602.22755. Bricken et al. (2025), "Building and evaluating alignment auditing agents," Anthropic Alignment Science. Marks et al. (2025), "Auditing language models for hidden objectives," arXiv:2503.10965. Karvonen et al. (2026), "Activation Oracles," arXiv:2512.15674.

**Model diffing**: Minder et al. (2025), "Overcoming Sparsity Artifacts in Crosscoders," arXiv:2504.02922. Lindsey et al. (2024), "Sparse Crosscoders for Cross-Layer Features," Transformer Circuits Thread.

**Emergent misalignment**: Wang et al. (2025), "Persona Features Control Emergent Misalignment," arXiv:2506.19823, OpenAI/Nature.

**Domain-specific SAEs**: O'Neill et al. (2025), "Resurrecting the Salmon: Rethinking Mechanistic Interpretability with Domain-Specific SAEs," arXiv:2508.09363. Gallifant et al. (2025), "Sparse Autoencoder Features for Classifications and Transferability," EMNLP 2025.

**Open problems**: Sharkey et al. (2025), "Open Problems in Mechanistic Interpretability," arXiv:2501.16496, TMLR.

**AGORA**: Arnold et al. (2024), "Introducing the AI Governance and Regulatory Archive," AIES 2024, DOI:10.1609/aies.v7i1.31615.

**Legal NLP**: Chalkidis et al. (2020), "LEGAL-BERT," Findings of EMNLP 2020.

---

## Proposed paper structure for the revision

The revised paper should follow this narrative arc: (1) **Motivation** — policy text analysis needs mechanistic interpretability, not just black-box classification; connect to AuditBench's finding that interpretability tools help with auditing. (2) **Bottom-up discovery** — Phase 1 results showing what Gemma 2 "sees" in policy text, independent of labels. (3) **Causal validation** — Phases 2–3 establishing multi-method convergence and layer-wise mechanisms. (4) **Honest limitations** — Phase 5 width sensitivity and Phase 8 strong feature hypothesis engagement. (5) **Practical value** — Phase 4 comparison showing the interpretability-performance tradeoff is worth it, and Phase 7 connecting to real auditing use cases. (6) **Cross-model robustness** — Phase 6 replication. The key shift from the original is that the paper tells a mechanistic story first, then evaluates it, rather than starting from classification benchmarks.

This plan produces approximately 10 main-text figures and 6 main-text tables, with 8 appendix sections containing detailed supplementary evidence. The total experimental scope requires running Gemma 2 inference on ~1,000 AGORA documents at 2 model sizes × 3 SAE widths × 4 layers, plus 3 re-trained SAEs for stability analysis, plus crosscoder training. This is computationally intensive but feasible with standard academic GPU/TPU access.