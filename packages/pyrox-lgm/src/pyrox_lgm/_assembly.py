"""Sparse assembly of an LGM's latent precision and projector.

Every component precision is turned into COO triplets whose *pattern* is a
host (numpy) constant and whose *values* are traced, so the assembled
block-diagonal precision and the stacked projector keep one sparsity
pattern for every theta: gaussx caches the symbolic Cholesky per pattern and
reuses it across Newton steps, theta-mode iterations and design points.
"""

from __future__ import annotations

import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array


def full_coo(op: lx.AbstractLinearOperator) -> tuple[np.ndarray, np.ndarray, Array]:
    """``(rows, cols, values)`` of every stored entry, both triangles.

    Duplicates are allowed (they sum). Sparse forms for the operators the
    components produce: `gaussx.SparseOperator`, diagonal, tridiagonal,
    symmetric `gaussx.BlockTriDiag`, `gaussx.Kronecker` / `gaussx.KroneckerSum`
    of those, and scalar multiples. Any other operator (a `Generic` structure
    can be any lineax operator) is assembled densely: its values may be
    traced, so its pattern cannot be read from them.
    """
    if isinstance(op, gx.SparseOperator):
        r = np.asarray(op.pattern.rows)
        c = np.asarray(op.pattern.cols)
        v = op.values
        if op.pattern.symmetric:  # lower triangle stored: mirror it
            off = r != c
            return (
                np.concatenate([r, c[off]]),
                np.concatenate([c, r[off]]),
                jnp.concatenate([v, v[jnp.asarray(np.flatnonzero(off))]]),
            )
        return r, c, v
    if isinstance(op, lx.DiagonalLinearOperator):
        n = op.in_size()
        return np.arange(n), np.arange(n), op.diagonal
    if isinstance(op, lx.TridiagonalLinearOperator):
        n = op.in_size()
        i = np.arange(n - 1)
        return (
            np.concatenate([np.arange(n), i + 1, i]),
            np.concatenate([np.arange(n), i, i + 1]),
            jnp.concatenate([op.diagonal, op.lower_diagonal, op.upper_diagonal]),
        )
    if isinstance(op, gx.BlockTriDiag):
        nb, d = op.diagonal.shape[0], op.diagonal.shape[1]
        start = np.arange(nb) * d
        a, b = np.divmod(np.arange(d * d), d)
        rows = [np.add.outer(start, a).ravel()]
        cols = [np.add.outer(start, b).ravel()]
        vals = [op.diagonal[:, a, b].reshape(-1)]
        if nb > 1:
            rows += [
                np.add.outer(start[1:], a).ravel(),
                np.add.outer(start[:-1], b).ravel(),
            ]
            cols += [
                np.add.outer(start[:-1], b).ravel(),
                np.add.outer(start[1:], a).ravel(),
            ]
            sub = op.sub_diagonal[:, a, b].reshape(-1)
            vals += [sub, sub]
        return np.concatenate(rows), np.concatenate(cols), jnp.concatenate(vals)
    if isinstance(op, gx.Kronecker):
        # A ⊗ B ⊗ C ... folded left to right, one pair at a time.
        r, c, v = full_coo(op.operators[0])
        for factor in op.operators[1:]:
            rb, cb, vb = full_coo(factor)
            nb = factor.in_size()
            r = np.add.outer(r * nb, rb).ravel()
            c = np.add.outer(c * nb, cb).ravel()
            v = (v[:, None] * vb[None, :]).reshape(-1)
        return r, c, v
    if isinstance(op, gx.KroneckerSum):
        ra, ca, va = full_coo(op.A)
        rb, cb, vb = full_coo(op.B)
        na, nb = op.A.in_size(), op.B.in_size()
        ia, ib = np.arange(nb), np.arange(na)
        return (
            np.concatenate(
                [np.add.outer(ra * nb, ia).ravel(), np.add.outer(ib * nb, rb).ravel()]
            ),
            np.concatenate(
                [np.add.outer(ca * nb, ia).ravel(), np.add.outer(ib * nb, cb).ravel()]
            ),
            jnp.concatenate([jnp.repeat(va, nb), jnp.tile(vb, na)]),
        )
    if isinstance(op, lx.MulLinearOperator):
        r, c, v = full_coo(op.operator)
        return r, c, op.scalar * v
    if isinstance(op, lx.TaggedLinearOperator):
        return full_coo(op.operator)
    matrix = op.matrix if isinstance(op, lx.MatrixLinearOperator) else op.as_matrix()
    n, m = matrix.shape
    r, c = np.divmod(np.arange(n * m), m)
    return r, c, matrix.reshape(-1)


def coo_matmul(a, b, n: int):
    """``A @ B`` for COO triplets ``(rows, cols, values)`` of ``n x n`` matrices.

    The output pattern and the pairing of factors are computed on the host
    from the patterns alone; only the values are traced, so the product can
    be differentiated in them and keeps a theta-free pattern.
    """
    ra, ca, va = a
    rb, cb, vb = b
    order = np.argsort(rb, kind="stable")
    counts = np.bincount(rb[order], minlength=n)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    per = counts[ca]  # B entries in row k for each A entry (i, k)
    ia = np.repeat(np.arange(ra.size), per)
    offset = np.arange(ia.size) - np.repeat(np.cumsum(per) - per, per)
    ib = order[np.repeat(starts[ca], per) + offset]
    keys = ra[ia].astype(np.int64) * n + cb[ib]
    uniq, inv = np.unique(keys, return_inverse=True)
    vals = jax.ops.segment_sum(va[ia] * vb[ib], jnp.asarray(inv), uniq.size)
    return (uniq // n).astype(int), (uniq % n).astype(int), vals


def coo_power(a, power: int, n: int):
    """``A^power`` (``power >= 1``) by repeated `coo_matmul`."""
    out = a
    for _ in range(power - 1):
        out = coo_matmul(out, a, n)
    return out


def block_diagonal(
    blocks: list[lx.AbstractLinearOperator],
) -> gx.SparseOperator:
    """Symmetric `gaussx.SparseOperator` with the given diagonal blocks."""
    rows, cols, vals = [], [], []
    start = 0
    for op in blocks:
        r, c, v = full_coo(op)
        rows.append(r + start)
        cols.append(c + start)
        vals.append(v)
        start += op.in_size()
    r, c = np.concatenate(rows), np.concatenate(cols)
    lower = r >= c
    v = jnp.concatenate(vals)[jnp.asarray(np.flatnonzero(lower))]
    return gx.SparseOperator.from_coo(
        r[lower],
        c[lower],
        v,
        (start, start),
        symmetric=True,
        tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )


def hstack(blocks: list[lx.AbstractLinearOperator], n_rows: int) -> gx.SparseOperator:
    """``[B_1 | ... | B_k]`` as a (general) `gaussx.SparseOperator`."""
    rows, cols, vals = [], [], []
    start = 0
    for op in blocks:
        if op.out_size() != n_rows:
            raise ValueError(
                f"projector block has {op.out_size()} rows, expected {n_rows}"
            )
        r, c, v = full_coo(op)
        rows.append(r)
        cols.append(c + start)
        vals.append(v)
        start += op.in_size()
    return gx.SparseOperator.from_coo(
        np.concatenate(rows),
        np.concatenate(cols),
        jnp.concatenate(vals),
        (n_rows, start),
    )
