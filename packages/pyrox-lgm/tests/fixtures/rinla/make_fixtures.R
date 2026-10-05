# Golden R-INLA fixtures for pyrox-lgm's inla() (jejjohnson/pyrox#254).
#
# Each case simulates (or loads) its data, fits it in R-INLA with the priors
# pyrox-lgm uses by default, and writes the data and the posterior summaries
# to <case>.json next to this script. The Python side refits the same data
# and compares (tests/test_rinla_fixtures.py).
#
# Prior correspondences (pyrox-lgm default -> R-INLA):
#   PCPrecision(1, 0.01)             -> pc.prec, param = c(1, 0.01)
#   PCAR1Rho(0.5, 0.5)               -> pc.cor0, param = c(0.5, 0.5)
#   PCBYM2Phi(0.5, 2/3)              -> pc,      param = c(0.5, 2/3)
#   PCMatern(r0, 0.5, 1, 0.01)       -> inla.spde2.pcmatern(prior.range = c(r0, 0.5),
#                                                           prior.sigma = c(1, 0.01))
#   FixedEffects(prior_precision=1e-3) -> control.fixed(prec.intercept = prec = 0.001)
#   RW1 / RW2 / Besag scale_model=True -> scale.model = TRUE
# and strategy = "gaussian" with the VB mean correction (pyrox-lgm's default
# "vb"); the simplified-Laplace marginals are stored too, under "sla".
#
# Regenerate (needs R and INLA, https://www.r-inla.org):
#   Rscript packages/pyrox-lgm/tests/fixtures/rinla/make_fixtures.R

suppressPackageStartupMessages({
  library(INLA)
  library(fmesher)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))

pc_prec <- list(prec = list(prior = "pc.prec", param = c(1, 0.01)))
ctrl_fixed <- list(prec.intercept = 0.001, prec = 0.001)

fit <- function(formula, data, strategy, ...) {
  inla(
    formula, data = data,
    control.fixed = ctrl_fixed,
    control.inla = list(
      strategy = strategy, int.strategy = "auto",
      control.vb = list(enable = TRUE)
    ),
    num.threads = "1:1", ...
  )
}

summarise <- function(res) {
  fx <- res$summary.fixed
  list(
    fixed = lapply(setNames(seq_len(nrow(fx)), rownames(fx)), function(i) list(
      mean = fx$mean[i], sd = fx$sd[i], q025 = fx[i, "0.025quant"],
      q50 = fx[i, "0.5quant"], q975 = fx[i, "0.975quant"]
    )),
    random = lapply(res$summary.random, function(d) list(mean = d$mean, sd = d$sd)),
    hyperpar = lapply(
      setNames(seq_len(nrow(res$summary.hyperpar)), rownames(res$summary.hyperpar)),
      function(i) list(
        mean = res$summary.hyperpar$mean[i],
        q50 = res$summary.hyperpar[i, "0.5quant"]
      )
    ),
    mlik_integration = res$mlik[1, 1],
    mlik_gaussian = res$mlik[2, 1],
    # internal scale: log precisions, logit(phi), log((1+rho)/(1-rho)),
    # log range, log sd; pyrox-lgm's unconstrained u uses the same maps.
    theta_mode = as.list(setNames(res$mode$theta, names(res$mode$theta))),
    theta_cov = res$misc$cov.intern
  )
}

run_case <- function(name, formula, data, extra, ...) {
  out <- list(
    inla_version = as.character(packageVersion("INLA")),
    data = extra,
    vb = summarise(fit(formula, data, "gaussian", ...)),
    sla = summarise(fit(formula, data, "simplified.laplace", ...))
  )
  write_json(out, file.path(here, paste0(name, ".json")),
             digits = NA, auto_unbox = TRUE, pretty = TRUE)
  cat("wrote", name, "\n")
}

# 1. RW2 + intercept, Gaussian, on a noisy sine.
set.seed(1)
n <- 50
t <- 1:n
y <- sin(t / 6) + 0.5 + rnorm(n, sd = 0.3)
run_case(
  "rw2_gaussian",
  y ~ 1 + f(t, model = "rw2", scale.model = TRUE, hyper = pc_prec),
  data.frame(y = y, t = t),
  list(y = y, t = t - 1),
  control.family = list(hyper = pc_prec)
)

# 2. AR(1) + intercept, Gaussian.
set.seed(2)
n <- 80
x <- as.numeric(arima.sim(list(ar = 0.7), n = n, sd = sqrt(1 - 0.7^2)))
y <- 1 + x + rnorm(n, sd = 0.5)
t <- 1:n
ar1_hyper <- list(
  prec = list(prior = "pc.prec", param = c(1, 0.01)),
  rho = list(prior = "pc.cor0", param = c(0.5, 0.5))
)
run_case(
  "ar1_gaussian",
  y ~ 1 + f(t, model = "ar1", hyper = ar1_hyper),
  data.frame(y = y, t = t),
  list(y = y, t = t - 1),
  control.family = list(hyper = pc_prec)
)

# 3. Scotland lip cancer: BYM2 + AFF covariate, Poisson with expected counts.
data(Scotland)
g <- inla.read.graph(system.file("demodata/scotland.graph", package = "INLA"))
edges <- do.call(rbind, lapply(seq_len(g$n), function(i) {
  nb <- g$nbs[[i]]
  nb <- nb[nb > i]
  if (length(nb)) cbind(i - 1, nb - 1) else NULL
}))
sc <- data.frame(
  y = Scotland$Counts, E = Scotland$E, x = Scotland$X / 10, region = Scotland$Region
)
bym2_hyper <- list(
  prec = list(prior = "pc.prec", param = c(1, 0.01)),
  phi = list(prior = "pc", param = c(0.5, 2 / 3))
)
run_case(
  "scotland_bym2",
  y ~ 1 + x + f(region, model = "bym2", graph = g, scale.model = TRUE, hyper = bym2_hyper),
  sc,
  list(
    y = sc$y, offset = log(sc$E), x = sc$x, region = sc$region - 1,
    n_nodes = g$n, edges = edges
  ),
  family = "poisson", E = sc$E
)

# 4. SPDE (alpha = 2) on a small mesh, Poisson, a smooth surface on [0, 1]^2.
set.seed(4)
n <- 150
loc <- cbind(runif(n), runif(n))
mesh <- fm_mesh_2d(loc = loc, max.edge = c(0.15, 0.4), cutoff = 0.05, offset = c(0.1, 0.3))
y <- rpois(n, exp(0.5 + sin(3 * loc[, 1]) * cos(3 * loc[, 2])))
spde <- inla.spde2.pcmatern(mesh, alpha = 2, prior.range = c(0.3, 0.5), prior.sigma = c(1, 0.01))
A <- inla.spde.make.A(mesh, loc)
stk <- inla.stack(
  data = list(y = y), A = list(A, 1),
  effects = list(s = seq_len(spde$n.spde), intercept = rep(1, n))
)
run_case(
  "spde_poisson",
  y ~ -1 + intercept + f(s, model = spde),
  inla.stack.data(stk),
  list(
    y = y, loc = loc, vertices = mesh$loc[, 1:2], triangles = mesh$graph$tv - 1,
    range0 = 0.3
  ),
  family = "poisson", control.predictor = list(A = inla.stack.A(stk))
)

# 5. Bernoulli probability-of-detection toy: RW2 over 50 size bins + wind.
set.seed(5)
n_obs <- 500
n_bins <- 50
size <- sample.int(n_bins, n_obs, replace = TRUE)
wind <- rnorm(n_obs)
eta <- -1 + 3 * sin(seq(-1.5, 1.5, length.out = n_bins))[size] - 0.5 * wind
y <- as.numeric(runif(n_obs) < plogis(eta))
run_case(
  "pod_bernoulli",
  y ~ 1 + wind + f(size, model = "rw2", scale.model = TRUE, hyper = pc_prec, values = 1:n_bins),
  data.frame(y = y, size = size, wind = wind),
  list(y = y, size = size - 1, wind = wind, n_bins = n_bins),
  family = "binomial", Ntrials = rep(1, n_obs)
)
