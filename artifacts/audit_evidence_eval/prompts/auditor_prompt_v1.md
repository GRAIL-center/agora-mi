# Auditor Prompt Template

Use `policy_interp.audit_evidence_suite.render_auditor_prompt(package)` to render a case specific prompt. The auditor must return a single JSON object with issue findings, policy spans, evidence ids, confounds, unsupported claims, and overall confidence.
