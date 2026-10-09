---
name: code-review
description: Review a change or pull request in the pyrox workspace against CODE_REVIEW.md, the contracts in AGENTS.md (sites, Parameterized, numerics), the package AGENTS.md files and reuse of gaussx / kernellib / geonnax primitives.
---

# Code review

Read "Reuse before you write" and "The contracts" in
[`AGENTS.md`](../../../AGENTS.md) and the `AGENTS.md` of each package the
diff touches; for every function, class or module the diff adds, search
[`docs/capabilities.md`](../../../docs/capabilities.md) for an existing
equivalent; then apply [`CODE_REVIEW.md`](../../../CODE_REVIEW.md) and report
in its format.
