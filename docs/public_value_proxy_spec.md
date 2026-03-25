Public Value Proxy Specification

Goal

This project no longer treats AGORA tags as the final conceptual target. Instead, it uses AGORA risk and harm tags as observable proxies for higher level public values from the AGORA descriptive paper. The central claim becomes: some internal SAE features may reflect public value level structure rather than only narrow policy tag wording.

Design Principles

1. Public values are higher level constructs and AGORA tags are noisy operational proxies.
2. Main claims should rely on public value families with at least two nontrivial proxy tags.
3. Main proxy selection should use both annotated coverage and validated coverage.
4. Ambiguous tags that map to several values should be demoted to secondary use where possible.
5. Governance strategy tags remain important, but are treated as modulators of public value feature recruitment rather than as the primary target concept.

Selected Main Public Value Families

1. Sustainability
   Rationale: the descriptive paper links sustainability most directly to safety issues and unreliability, with ecological harm as a related harm level signal.
   Main proxies:
   1. Risk factors: Safety
   2. Risk factors: Reliability
   Secondary proxies:
   1. Harms: Ecological harm
   Notes:
   1. Harms: Harm to infrastructure was discussed in the paper but rejected as too inferential for the main crosswalk.
   2. Discovery should use the two risk proxies first, then evaluate transfer to ecological harm.

2. Protection of individual rights
   Rationale: this family has the strongest multi proxy support in the crosswalk and the best coverage in AGORA.
   Main proxies:
   1. Risk factors: Security
   2. Risk factors: Privacy
   3. Harms: Violation of civil or human rights, including privacy
   Secondary proxies:
   1. Risk factors: Bias
   Notes:
   1. Bias is demoted to secondary because it also supports equality and neutrality.
   2. Main discovery should rely on privacy, security, and rights violation, then test whether the discovered feature bank transfers to bias related segments.

3. Transparency and accountability
   Rationale: the descriptive paper uses lack of transparency and lack of interpretability as the clearest proxy pair for openness and accountability.
   Main proxies:
   1. Risk factors: Transparency
   Secondary proxies:
   1. Risk factors: Interpretability and explainability
   Notes:
   1. This family is intentionally grouped because the paper maps the same AGORA risks to both openness and accountability.
   2. The family framing is safer than trying to fully disentangle openness from accountability with the current crosswalk alone.

4. Equality and neutrality
   Rationale: bias and discrimination provide the cleanest operational route to this family.
   Main proxies:
   1. Risk factors: Bias
   2. Harms: Discrimination
   Secondary proxies:
   1. Harms: Violation of civil or human rights, including privacy
   Notes:
   1. Rights violation is secondary because it is broader than equality and also supports individual rights.
   2. Discovery should focus on bias and discrimination first.

Exploratory Families

1. Common good
   Candidate proxies:
   1. Risk factors: Safety
   2. Harms: Harm to health/safety
   3. Harms: Financial loss
   4. Harms: Harm to property
   5. Harms: Harm to infrastructure
   6. Harms: Ecological harm
   7. Harms: Violation of civil or human rights, including privacy
   8. Harms: Detrimental content
   Risk:
   1. The proxy set is broad and semantically diffuse.

2. Human dignity
   Candidate proxies:
   1. Risk factors: Privacy
   2. Risk factors: Bias
   3. Harms: Violation of civil or human rights, including privacy
   4. Harms: Detrimental content
   Risk:
   1. The family is plausible but still broad enough to invite interpretive disagreement.

3. Effectiveness and reliability
   Candidate proxies:
   1. Risk factors: Reliability
   Risk:
   1. The current crosswalk gives too little proxy diversity for a strong main claim.

4. Innovation
   Candidate proxies:
   1. No direct main proxy from the risk and harm crosswalk.
   Risk:
   1. This should not be a main public value claim under the current design.

Proxy Selection Criteria

1. Main proxies should have at least moderate annotated coverage.
2. Main proxies should also have enough validated positives to support a robustness check.
3. Secondary proxies may be sparser, but they must support transfer or stress tests.
4. A tag should be demoted if it is central to more than one family and could create ambiguous attribution.

Current Coverage Snapshot

The counts below are from the current AGORA segment table after simple inspection, before any length or quality filtering.

1. Sustainability family
   1. Risk factors: Safety = 378 total, 376 annotated, 103 validated
   2. Risk factors: Reliability = 187 total, 185 annotated, 51 validated
   3. Harms: Ecological harm = 46 total, 45 annotated, 5 validated

2. Individual rights family
   1. Risk factors: Security = 397 total, 382 annotated, 99 validated
   2. Risk factors: Privacy = 283 total, 275 annotated, 54 validated
   3. Harms: Violation of civil or human rights, including privacy = 378 total, 364 annotated, 95 validated
   4. Risk factors: Bias = 250 total, 248 annotated, 59 validated

3. Transparency and accountability family
   1. Risk factors: Transparency = 327 total, 317 annotated, 87 validated
   2. Risk factors: Interpretability and explainability = 74 total, 72 annotated, 19 validated

4. Equality and neutrality family
   1. Risk factors: Bias = 250 total, 248 annotated, 59 validated
   2. Harms: Discrimination = 202 total, 199 annotated, 45 validated

Governance Strategy Categories

These categories come from the AGORA descriptive paper and will be used as modulators rather than as the main target labels.

1. Regulation and Enforcement
   1. Strategies: Governance development
   2. Strategies: Performance requirements
   3. Strategies: Licensing, registration, and certification
   4. Strategies: Tiering
   5. Strategies: Evaluation
   6. Strategies: Disclosure

2. Direct Services and Public Projects
   1. Strategies: Government support
   2. Strategies: Pilots and testbeds

3. Information and Capacity Building
   1. Strategies: Government study or report
   2. Strategies: New institution
   3. Strategies: Convening

Research Use

1. Discovery
   1. Use main proxies only.
   2. Discover candidate SAE features on the training split.

2. Transfer
   1. Evaluate whether features discovered from one proxy transfer to held out segments from another proxy in the same family.

3. Family level analysis
   1. Build shared and specific feature banks at the family level.

4. Strategy modulation
   1. Test whether governance strategies recruit certain public value family features more strongly after controlling for proxy tags and text covariates.

5. Causal validation
   1. Intervene on family feature banks and measure held out behavioral changes on policy text.
