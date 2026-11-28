"""The linear-algebra choices of a GP prior, composed into one gaussx strategy.

A prior carries three separate choices (pyrox#550): the ``solver`` that
applies $A^{-1}$, the ``preconditioner`` that speeds it up, and the
``logdet`` that estimates $\\log \\det A$. `compose_solver_fields` turns
them into the one `gaussx.AbstractSolverStrategy` every entry point uses,
through `gaussx.compose_solver`.
"""

from __future__ import annotations

import gaussx as gx


def _field_name(value: object) -> str:
    return "None" if value is None else type(value).__name__


def compose_solver_fields(
    owner: str,
    solver: gx.AbstractSolveStrategy | None,
    preconditioner: gx.AbstractPreconditioner | None,
    logdet: gx.AbstractLogdetStrategy | None,
) -> gx.AbstractSolverStrategy:
    """One strategy from a prior's ``solver``, ``preconditioner`` and ``logdet``.

    All three ``None`` is `gaussx.DenseSolver`, and a ``solver`` alone is
    returned unchanged, so a prior that sets none of the new fields keeps
    its strategy exactly. The checks are on types only, so this runs under
    ``jit``.

    Args:
        owner: The class whose fields these are, named in the error.
        solver: The solve strategy; ``None`` is `gaussx.DenseSolver`.
        preconditioner: The preconditioner of the solve and, for a
            `gaussx.BBMMSolver`, of its log-determinant.
        logdet: A separate log-determinant strategy; ``None`` keeps the
            solver's own.

    Returns:
        The composed `gaussx.AbstractSolverStrategy`.

    Raises:
        ValueError: For a combination `gaussx.compose_solver` rejects (a
            preconditioner with an exact solver, two preconditioners, ...),
            naming the three fields.
    """
    try:
        return gx.compose_solver(solver, preconditioner=preconditioner, logdet=logdet)
    except ValueError as error:
        raise ValueError(
            f"{owner}(solver={_field_name(solver)}, "
            f"preconditioner={_field_name(preconditioner)}, "
            f"logdet={_field_name(logdet)}) cannot be combined: {error}"
        ) from error
