---
applyTo: "**"
---

# Code Review Instructions

When performing code review, use `/CODE_REVIEW.md` as the source of truth for:

- Review checklist (reuse and boundaries, primitives and ξ → 0 numerics, the NumPyro distribution contract, point processes, extraction, public API and docs, tests, modern Python)
- xtremax-specific checks (stable kernels, supports that `biject_to` can handle, padded point-process buffers, float32 accuracy)
- Output format and priority levels
- Suggestion type emojis and review tone

Key principles:
- Sacrifice *cleverness* for *clarity*. Sacrifice *brevity* for *explicitness*.
- Don't worry about formatting — CI (ruff format, pre-commit) handles that automatically.
- Be **constructive** and **specific**. Acknowledge good patterns with 👍.
- Every suggestion must include a concrete alternative, not just criticism.
