---
status: draft
version: 0.1.0
---

# pyrox x GP Evaluation Metrics: Gap Analysis

**Subject:** Evaluation metrics sourced from GPyTorch's `metrics/` module
and standard GP literature.

**Date:** 2026-04-02

---

## 1  Summary

Evaluation metrics live in `pyrox.gp`, not gaussx. They operate on GP predictions
(mean, variance) and observed targets -- no linear algebra primitives needed.
gaussx provides the solvers and distributions that produce predictions; pyrox.gp
evaluates how good those predictions are.

---

## 2  Gap Catalog

### Gap 1: NLPD (Negative Log Predictive Density)

**Type:** Probabilistic

**Math:** The negative log predictive density under a Gaussian predictive distribution:

$$\text{NLPD} = -\frac{1}{N} \sum_{i=1}^{N} \log p(y_i \mid \mu_i, \sigma_i^2) = \frac{1}{N} \sum_{i=1}^{N} \left[\frac{1}{2}\log(2\pi\sigma_i^2) + \frac{(y_i - \mu_i)^2}{2\sigma_i^2}\right]$$

Lower is better. Decomposes into a calibration term (variance matching residuals)
and a sharpness term (small variances rewarded). This is the single most important
metric for GP regression evaluation.

```python
def nlpd(
    y_true: Float[Array, " N"],
    mu_pred: Float[Array, " N"],
    var_pred: Float[Array, " N"],
) -> Float[Array, ""]:
    """Negative log predictive density (Gaussian). Lower is better."""
    ...
```

**Ref:** Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning.* MIT Press. Ch. 2.

---

### Gap 2: Calibration Error

**Type:** Probabilistic

**Math:** Expected calibration error (ECE) for regression. For a set of
confidence levels $\{p_1, \ldots, p_L\}$, compare the empirical coverage
to the nominal level:

$$\text{ECE} = \frac{1}{L} \sum_{l=1}^{L} \left| \hat{p}_l - p_l \right|$$

where $\hat{p}_l = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\!\left[y_i \in \text{CI}_{p_l}(x_i)\right]$
is the fraction of observations falling inside the $p_l$-level prediction interval
$\text{CI}_{p_l} = [\mu_i - z_{p_l/2}\,\sigma_i,\; \mu_i + z_{p_l/2}\,\sigma_i]$.

A perfectly calibrated model has $\hat{p}_l = p_l$ for all levels.

```python
def calibration_error(
    y_true: Float[Array, " N"],
    mu_pred: Float[Array, " N"],
    var_pred: Float[Array, " N"],
    confidence_levels: Float[Array, " L"] | None = None,
) -> Float[Array, ""]:
    """Expected calibration error across confidence levels. Lower is better."""
    ...
```

**Ref:** Kuleshov, V., Fenner, N., & Ermon, S. (2018). *Accurate Uncertainties for Deep Learning Using Calibrated Regression.* ICML.

---

### Gap 3: Coverage (Prediction Interval)

**Type:** Probabilistic

**Math:** The empirical coverage at confidence level $p$ is the fraction of
test observations falling inside the $p\%$ prediction interval:

$$\text{Coverage}(p) = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\!\left[\mu_i - z_{p/2}\,\sigma_i \leq y_i \leq \mu_i + z_{p/2}\,\sigma_i\right]$$

where $z_{p/2} = \Phi^{-1}\!\left(\frac{1+p}{2}\right)$ is the standard normal quantile.
For a well-calibrated model, $\text{Coverage}(p) \approx p$.

```python
def coverage(
    y_true: Float[Array, " N"],
    mu_pred: Float[Array, " N"],
    var_pred: Float[Array, " N"],
    confidence_level: float = 0.95,
) -> Float[Array, ""]:
    """Fraction of observations inside the prediction interval."""
    ...
```

**Ref:** Gneiting, T. & Raftery, A. E. (2007). *Strictly Proper Scoring Rules, Prediction, and Estimation.* JASA.

---

### Gap 4: Sharpness (Interval Width)

**Type:** Probabilistic

**Math:** Mean prediction interval width at confidence level $p$:

$$\text{Sharpness}(p) = \frac{1}{N} \sum_{i=1}^{N} 2\,z_{p/2}\,\sigma_i$$

where $z_{p/2} = \Phi^{-1}\!\left(\frac{1+p}{2}\right)$.
Smaller intervals are better, *conditional on maintaining coverage*.
Sharpness without coverage is meaningless -- always report both.

```python
def sharpness(
    var_pred: Float[Array, " N"],
    confidence_level: float = 0.95,
) -> Float[Array, ""]:
    """Mean prediction interval width. Lower is better (given coverage)."""
    ...
```

**Ref:** Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). *Probabilistic Forecasts, Calibration and Sharpness.* JRSS-B.

---

### Gap 5: CRPS (Continuous Ranked Probability Score)

**Type:** Probabilistic

**Math:** The CRPS measures the integrated squared distance between the
predictive CDF $F$ and the empirical CDF of the observation:

$$\text{CRPS}(F, y) = \int_{-\infty}^{\infty} \left(F(t) - \mathbf{1}\{y \leq t\}\right)^2 \, dt$$

For a Gaussian predictive distribution $F = \mathcal{N}(\mu, \sigma^2)$,
the CRPS has a closed-form expression:

$$\text{CRPS}(\mu, \sigma, y) = \sigma \left[\frac{1}{\sqrt{\pi}} - 2\,\varphi(z) - z\,(2\Phi(z) - 1)\right]$$

where $z = \frac{y - \mu}{\sigma}$, $\varphi$ is the standard normal PDF, and
$\Phi$ is the standard normal CDF. Average over all test points:

$$\overline{\text{CRPS}} = \frac{1}{N} \sum_{i=1}^{N} \text{CRPS}(\mu_i, \sigma_i, y_i)$$

Lower is better. The CRPS is a strictly proper scoring rule and has the same
units as the observations.

```python
def crps_gaussian(
    y_true: Float[Array, " N"],
    mu_pred: Float[Array, " N"],
    var_pred: Float[Array, " N"],
) -> Float[Array, ""]:
    """Closed-form CRPS for Gaussian predictive distributions. Lower is better."""
    ...
```

**Ref:** Gneiting, T. & Raftery, A. E. (2007). *Strictly Proper Scoring Rules, Prediction, and Estimation.* JASA.

---

### Gap 6: RMSE / MAE

**Type:** Point

**Math:** Standard point prediction metrics using the predictive mean:

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \mu_i)^2}$$

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \mu_i|$$

RMSE penalizes large errors more heavily; MAE is more robust to outliers.
Both ignore predictive uncertainty -- always pair with a probabilistic metric.

```python
def rmse(
    y_true: Float[Array, " N"],
    mu_pred: Float[Array, " N"],
) -> Float[Array, ""]:
    """Root mean squared error."""
    ...

def mae(
    y_true: Float[Array, " N"],
    mu_pred: Float[Array, " N"],
) -> Float[Array, ""]:
    """Mean absolute error."""
    ...
```

**Ref:** Standard. See Rasmussen & Williams (2006) Ch. 2 for GP context.

---

### Gap 7: Quantile Calibration Plot

**Type:** Diagnostic

**Math:** Plot the empirical coverage $\hat{p}_l$ against the nominal confidence
level $p_l$ for a range of levels $p_l \in \{0.05, 0.10, \ldots, 0.95\}$:

$$\hat{p}_l = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\!\left[F_i^{-1}\!\left(\frac{1-p_l}{2}\right) \leq y_i \leq F_i^{-1}\!\left(\frac{1+p_l}{2}\right)\right]$$

A perfectly calibrated model lies on the diagonal $\hat{p} = p$.
Points above the diagonal indicate over-coverage (conservative uncertainties);
points below indicate under-coverage (overconfident).

```python
def quantile_calibration_data(
    y_true: Float[Array, " N"],
    mu_pred: Float[Array, " N"],
    var_pred: Float[Array, " N"],
    n_levels: int = 19,
) -> tuple[Float[Array, " L"], Float[Array, " L"]]:
    """Return (nominal_levels, empirical_coverages) for calibration plot."""
    ...
```

**Ref:** Kuleshov, V., Fenner, N., & Ermon, S. (2018). *Accurate Uncertainties for Deep Learning Using Calibrated Regression.* ICML.

---

### Gap 8: Reliability Diagram

**Type:** Diagnostic (classification)

**Math:** For GP classification, bin predictions by confidence and compare
predicted probability to observed accuracy. Partition predictions into $B$
bins by predicted confidence:

$$\text{Bin}_b = \{i : p_i \in (l_b, u_b]\}, \qquad \text{acc}(b) = \frac{1}{|\text{Bin}_b|} \sum_{i \in \text{Bin}_b} \mathbf{1}[y_i = \hat{y}_i]$$

$$\text{conf}(b) = \frac{1}{|\text{Bin}_b|} \sum_{i \in \text{Bin}_b} p_i$$

Plot $\text{acc}(b)$ vs $\text{conf}(b)$. A calibrated classifier has
$\text{acc}(b) \approx \text{conf}(b)$ for all bins. The ECE for classification is:

$$\text{ECE} = \sum_{b=1}^{B} \frac{|\text{Bin}_b|}{N} \left|\text{acc}(b) - \text{conf}(b)\right|$$

```python
def reliability_diagram_data(
    y_true: Int[Array, " N"],
    prob_pred: Float[Array, " N"],
    n_bins: int = 10,
) -> tuple[Float[Array, " B"], Float[Array, " B"], Int[Array, " B"]]:
    """Return (mean_confidence, mean_accuracy, bin_counts) for reliability diagram."""
    ...
```

**Ref:** Niculescu-Mizil, A. & Caruana, R. (2005). *Predicting Good Probabilities with Supervised Learning.* ICML.

---

## 3  Notes

- All metrics are pure functions: `metric(y_true, mu_pred, var_pred) -> scalar`
- NLPD is the most important single metric for GP regression evaluation
- Calibration + sharpness together give a more complete picture than NLPD alone
- CRPS is a strictly proper scoring rule that rewards both calibration and sharpness
- Diagnostic plots (Gaps 7-8) return data arrays, not rendered figures -- plotting is the caller's responsibility

---

## 4  References

1. Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning.* MIT Press.
2. Kuleshov, V., Fenner, N., & Ermon, S. (2018). *Accurate Uncertainties for Deep Learning Using Calibrated Regression.* ICML.
3. Gneiting, T. & Raftery, A. E. (2007). *Strictly Proper Scoring Rules, Prediction, and Estimation.* JASA.
4. Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). *Probabilistic Forecasts, Calibration and Sharpness.* JRSS-B.
5. Niculescu-Mizil, A. & Caruana, R. (2005). *Predicting Good Probabilities with Supervised Learning.* ICML.
