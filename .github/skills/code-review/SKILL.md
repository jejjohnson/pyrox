---
name: code-review
description: Review a change or pull request in the pyrox workspace against CODE_REVIEW.md, the contracts in AGENTS.md (sites, Parameterized, numerics), the package AGENTS.md files and reuse of gaussx / kernellib / geonnax primitives.
---

# Code review

Follow the same steps as the repo's review skill,
[`.claude/skills/pyrox-review/SKILL.md`](../../../.claude/skills/pyrox-review/SKILL.md):
read the touched packages' `AGENTS.md`; check the diff for re-implemented
functionality with the procedure in
[`.claude/agents/reuse-reviewer.md`](../../../.claude/agents/reuse-reviewer.md)
(search [`docs/capabilities.md`](../../../docs/capabilities.md) for every
helper the diff adds); check it for model and numerical defects with the
procedure in
[`.claude/agents/model-reviewer.md`](../../../.claude/agents/model-reviewer.md);
then apply [`CODE_REVIEW.md`](../../../CODE_REVIEW.md) and report in its
format.
