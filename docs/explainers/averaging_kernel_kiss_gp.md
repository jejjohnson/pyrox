# Is the averaging kernel a KISS-GP interpolation?

*Explainer · how KISS-GP's interpolation weights and the methane column averaging kernel relate, followed through one TROPOMI example.*

[Open the full page](pages/averaging-kernel-kiss-gp.html){ .md-button .md-button--primary }

**Where it sits.** The full page is a self-contained document (operator chart, model boxes, units checks, a code sketch and worked numbers) that opens on its own and links back here. It uses one running example throughout: three TROPOMI XCH₄ pixels over a 25 km model grid with 12 retrieval layers.

## TL;DR: same algebra, different job

- **The shared core is real.** Any linear map of a Gaussian field is Gaussian, with covariance $\mathbf{H} \mathbf{K} \mathbf{H}^\top$. KISS-GP's $\mathbf{W} \mathbf{K}_{UU} \mathbf{W}^\top$ and a column operator's $\tilde{\mathbf{A}} \mathbf{K}_z \tilde{\mathbf{A}}^\top$ are both instances.
- **KISS-GP's $\mathbf{W}$ is horizontal and approximate.** It interpolates a grid to a point: 16 nonzeros per row in 2-D, rows sum to 1, and the error vanishes as the grid gets finer.
- **The column averaging kernel is vertical and exact.** It collapses the layer dimension to one XCH₄ number: dense over all layers, not summing to 1, and set by the instrument.
- **In an inversion they compose.** Row $n$ of the observation operator is $\tilde{\mathbf{a}}_n^\top \otimes \mathbf{w}_n^\top$; with a separable prior $\mathbf{K}_z \otimes \mathbf{K}_s$ the column covariance is $(\tilde{\mathbf{A}} \mathbf{K}_z \tilde{\mathbf{A}}^\top) \circ (\mathbf{W} \mathbf{K}_s \mathbf{W}^\top)$.
- **The averaging kernel is a posterior smoother.** With an identity Jacobian it is the GP smoother matrix $\mathbf{K}(\mathbf{K} + \sigma^2 \mathbf{I})^{-1}$, and its trace (the DOFS) is the GP's effective degrees of freedom. $\mathbf{W}$ lives in the prior and has no such meaning.
- **Notation trap.** In Rodgers' notation $\mathbf{K}$ is the Jacobian; on the page $\mathbf{J}$ is the Jacobian and every $\mathbf{K}$ is a covariance.

## On the page

1. The shared core: looking at a GP through a linear map
2. KISS-GP: $\mathbf{W}$ stands in for point evaluation
3. The column averaging kernel collapses height
4. Where they meet: one row, two factors
5. The averaging kernel as a posterior smoother
6. Side by side
