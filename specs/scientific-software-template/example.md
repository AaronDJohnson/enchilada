# Scientific Software Specification: Enchilada residual updates

**Specification ID / revision**: enchilada-residual-example / 1
**Feature Directory**: `specs/scientific-software-template`
**Created / updated**: 2026-09-08 / 2026-09-08
**Status**: Draft — worked example; acceptance cases are proposed
**Scope type**: retrospective contract with proposed scientific validation
**Source baseline**: `dd412c3d47d3a209b632f43ee3220215936a31cc`
**Scientific reviewer**: package maintainer; review pending

This is a filled example of the [scientific template](../../.specify/templates/overrides/spec-template.md),
focused on one capability. It illustrates how to specify a mathematical contract
and expose an evidence gap. It does not certify Enchilada or prescribe a new
production sampler. The [package baseline](../001-package-baseline/spec.md) retains
the broader current-behavior inventory.

## 1. Scientific purpose and scope

Enchilada coordinates additive signal blocks sharing an observation. Correct
residuals let a block condition its update on the other blocks' current signal
contributions. The computed quantity is the observation minus selected stored
templates, in the observation's original units and representation.

This example covers sequential updates of finite, real, time-domain arrays on a
shared grid with fixed noise. The coordinator owns residual accounting and copies
of published signal arrays. The caller keeps the observation unchanged; each block
owns its inference algorithm and private state.

The proposed statistical case concerns two synthetic linear Gaussian blocks.
Arbitrary contributor samplers, adaptive algorithms, noise inference, frequency
conventions, ephemerides, and astrophysical model validity are outside this example.
The smallest deterministic case has observation 10 and two changing contributions.

## 2. Mathematical contract

### Definitions and conventions

| Symbol | Meaning and units | Shape and convention |
| --- | --- | --- |
| `y` | observed samples, dimensionless in these fixtures | vector on one shared time grid |
| `h_j` | block `j`'s most recently adopted signal | same shape, units, and sample order as `y` |
| `r_j` | residual handed to block `j` | excludes that block's stored contribution from the subtraction |
| `r` | full residual | observation minus every adopted signal |

The finite observation and declared grid remain fixed during a run. Signals are
additive in the chosen observable. Blocks must not modify shared observation or
noise objects. These are contributor obligations; the coordinator does not prove
additivity or prevent every mutation of a shared object.

### Governing equations and assumptions

For a cycle updating blocks in registration order:

```text
r_j = y - sum(h_k_new for k < j) - sum(h_k_old for k > j)
h_j_new = published signal from block j's update(r_j)
r = y - sum(h_j_current for every registered j)
```

Indices are positions in registration order. Earlier blocks' adopted updates are
visible to later blocks in the same cycle. An initialized block's contribution is
already present when subsequent blocks initialize.

For the proposed statistical fixture, let `a` and `b` have independent standard
normal priors and let the additive observation noise have covariance `I`:

```text
y = X theta + noise,   theta = [a, b]^T
X = [[1, 1],          y = [1, -1]^T
     [0, 1]]

p(theta | y) is Gaussian with
precision Q = I + X^T X = [[2, 1], [1, 3]]
covariance C = Q^(-1) = (1/5) [[3, -1], [-1, 2]]
mean mu = C X^T y = [3/5, -1/5]^T

a | b, y ~ Normal((1 - b)/2, variance=1/2)
b | a, y ~ Normal(-a/3,      variance=1/3)
```

Completing the square in `theta^T theta + (y-X theta)^T (y-X theta)` gives
the stated precision and linear term; `Q C = I` and `Q mu = X^T y` provide
independent algebraic checks of the reference values.

### Exact quantity and approximation

Residual accounting is a finite-array operation. Floating-point error is the
relevant approximation. The proposed inference fixture targets the stated joint
posterior; a finite chain estimates its moments with Monte Carlo error and possible
initialization bias. Publishing a conditional mean changes the shared state and
does not implement these specified conditional draws.

## 3. Algorithm and state contract

For each block, construct its conditional residual, call its update, validate the
returned template, copy its signal into the ledger, and then visit the next block.
Later modification of a block's published array must not change its ledger entry.
The [implementation](../../src/enchilada/wheel.py) makes these stages explicit.

For the proposed Gaussian case, a block publishes the template of its current
scalar draw. Record each `(a, b)` pair after a complete cycle. Each conditional uses
the appropriate current value of the other parameter; a collection of private
sampler histories is not automatically a set of joint draws.

Under the fixture's correct full conditional draws, their sequential composition
preserves the joint target. Convergence and finite-run accuracy require additional
analysis and evidence; the [Gibbs sampling discussion in Gelman's computation notes](https://sites.stat.columbia.edu/gelman/bayescomputation/bdachapters10and11.pdf)
provides the general conditional-update framework.

Invalid templates are rejected before normal adoption. Earlier successful block
updates remain after a later failure; there is no whole-cycle rollback. This
fixed-noise example excludes warning-induced failures during noise adoption and
does not make a general transactional guarantee.

## 4. Numerical contract and error budget

VAL-001 uses float64 arrays containing small integers. Every specified subtraction
is exactly representable, so elementwise equality is required (`atol=0`, `rtol=0`).
This bound applies to that fixture; arbitrary residual scales require their own
roundoff and cancellation analysis.

VAL-002 compares five posterior moments against exact rational references. Its
acceptance procedure separates statistical agreement from adequate precision.
Posterior standard deviations set meaningful scales; they are not the standard
errors of the estimated moments. The chosen precision target is a proposed
integration-test requirement, not an astrophysical error budget.

Singular systems and unknown noise variance are excluded by the fixture. There is
no iterative linear solve in the specified block updates. Maximum chain length and
warm-up are fixed before observing validation results.

## 5. Scientific requirements

### SCI-001 — Current residuals preserve additive accounting

- **Requirement**: On the stated fixed grid, initialization and successful ordered
  updates MUST obey the residual equations, using copied adopted signal arrays.
- **Scientific consequence**: Subtracting a block's own signal, using stale values,
  or mutating stored contributions changes the data the next inference step fits.
- **Assumptions and dependencies**: Additivity, stable registration order, finite
  arrays, and contributor compliance with the ownership rules.
- **Acceptance**: VAL-001.
- **Evidence status**: unverified.
- **Evidence references and limits**: EV-001 identifies related existing tests and
  source evidence. The complete changing-signal and buffer-reuse fixture specified
  here is not yet a recorded acceptance run.

### SCI-002 — The specified Gaussian integration recovers joint moments

- **Requirement**: With the two conditional-sampling blocks defined above, the
  recorded full-cycle states MUST meet VAL-002's joint-moment acceptance rule.
- **Scientific consequence**: Plausible parameter means can coexist with incorrect
  variances or dependence, making reported uncertainties unreliable.
- **Assumptions and dependencies**: Correct Gaussian draws, independent RNG streams
  across chains, current scalar states, fixed priors/noise, and SCI-001.
- **Acceptance**: VAL-002.
- **Evidence status**: unverified.
- **Evidence references and limits**: EV-002; the fixture and its test harness are
  proposed. No current claim of arbitrary-block posterior correctness follows.

## 6. Validation and acceptance

### VAL-001 — Changing contributions and published-buffer isolation

- **Claims and assessment type**: SCI-001; verification.
- **Fixture**: One dimensionless channel, two samples `[10, 10]`, sample rate 1 Hz,
  epoch 0, float64. Block A initializes to `[2, 2]`, B to `[3, 3]`. In one cycle A
  publishes `[4, 4]`, then B publishes `[1, 1]`.
- **Independent reference**: Scalar arithmetic. A initializes with `[10, 10]`, B
  with `[8, 8]`; A updates with `[7, 7]`, B with `[6, 6]`; final residual `[5, 5]`.
- **Metric and acceptance rule**: Exact elementwise equality for every stated
  value. After the cycle, overwrite A's published buffer with `[99, 99]`; its
  ledger entry must remain `[4, 4]`, the residual `[5, 5]`, and the observation
  `[10, 10]`. Use explicit zero absolute and relative tolerances.
- **Execution**: Planned automated test; no executable node is claimed here.
- **Failure sensitivity**: An implementation using the old A value gives B a
  residual of 8; subtracting the block's own signal or retaining its mutable buffer
  also violates the expected results.
- **Coverage and limits**: Tests update ordering, self-exclusion, copying, and
  initialization in a small exact case. It does not establish general floating-point
  accuracy, arbitrary failure behavior, or statistical validity.

### VAL-002 — Correlated Gaussian posterior with independent reference

- **Claims and assessment type**: SCI-002; verification of the specified integration.
- **Fixture**: The exact `X`, `y`, priors, and conditional distributions above.
  Use 32 chains, float64, PCG64 generators seeded by the 32 children of
  `SeedSequence(20260908)`. Repeat the initial pairs `(-4,-4)`, `(-4,4)`, `(4,-4)`,
  `(4,4)` eight times in that order, assigning an independent stream to each chain.
  Discard 1,024 complete cycles and retain the next 8,192 without thinning.
- **Independent reference**: Closed-form `mu` and `C` above, derived without the
  residual implementation or sampler. Examine the five probes `a`, `b`,
  `(a-3/5)^2`, `(b+1/5)^2`, and `(a-3/5)(b+1/5)` with expectations
  `[3/5, -1/5, 3/5, 2/5, -1/5]`. Center second moments on the analytical mean.
- **Metric and acceptance rule**: For each probe, compute its mean within each
  chain, then the grand mean and the sample standard deviation of those 32 means
  divided by `sqrt(32)`. This standard error uses independent chain replicates;
  it does not treat correlated within-chain draws as independent. Let `h` be that
  standard error times the Student-t quantile with 31 degrees of freedom at
  `1 - 0.001/(2*5)`. Require both absolute error at most `h` and
  `h <= 0.02*s`, where probe scales `s` are
  `[sqrt(3/5), sqrt(2/5), 3/5, 2/5, sqrt((3/5)*(2/5))]`.
  The first condition checks agreement; the second bounds interval half-width to
  2% of its scientific scale. This proposed precision target is intended to resolve
  biases of 5% of those scales; calibration must assess its detection power.
  Excess width is inconclusive, never a pass.
- **Statistical qualifications**: The nominal family false-rejection rate is
  0.001 using Bonferroni adjustment and approximately Gaussian chain means. That
  approximation and sensitivity to relevant bias require calibration before this
  procedure is accepted. The update equations give
  `E[b_next - mu_b | b_current] = (b_current - mu_b)/6`;
  a burn-in justification must also account for distributional initialization error.
- **Execution**: Planned integration test; the exact harness, locked environment,
  runtime budget, and calibration results are open. Fix them before acceptance
  runs; retain every run, seed, diagnostic, and failure.
- **Failure sensitivity**: Check deliberately stale residual updates and publication
  of conditional means instead of draws. The variance and cross-moment probes must
  be sensitive to these errors; mean recovery alone is insufficient.
- **Coverage and limits**: A linear two-parameter case with known noise. Passing
  does not establish convergence for nonlinear, multimodal, or adaptive samplers,
  or validity of an astrophysical likelihood.

Both cases are required to support both claims. A missing or inconclusive case
leaves its claim unverified. Existing tests elsewhere do not waive these criteria.

## 7. Reproducibility and evidence

Required reproducibility is exact for VAL-001 and statistical under VAL-002's
declared procedure. Numerical streams are compared only in an explicitly recorded
NumPy/backend environment. Any future acceptance run must retain code and spec
revisions, dirty patch identity, command, environment, RNG setup, and raw outputs.

| Evidence ID | Claims / cases | Baseline and procedure | Observed result / artifact | Limits |
| --- | --- | --- | --- | --- |
| EV-001 | SCI-001 / VAL-001 | Baseline commit above; inspect [Wheel](../../src/enchilada/wheel.py) and [existing tests](../../tests/test_wheel.py) | `TestNoAddBack`, `TestLedger`, and `TestDocumentedContracts` cover related accounting and copying behavior | Source/test inventory; not a recorded execution of the full VAL-001 fixture |
| EV-002 | SCI-002 / VAL-002 | Proposed procedure in this example | Not run; fixture not implemented | Joint mean/covariance acceptance and calibration remain open |

The existing [toy regression](../../tests/test_examples.py) checks parameter means
against generating truth within `max(5 * posterior_std, 0.05)` and positive reported
spread. That is useful example evidence, but it does not compare joint moments to
the analytical posterior required here. The [galactic-binary example](../../examples/gb_model.py)
publishes a model at the final walker mean; it must not be assumed to implement the
conditional-draw fixture specified in this document.

## 8. Open decisions and scientific change control

| Open decision or limitation | Scientific consequence / affected claims | Resolution criterion and responsible role |
| --- | --- | --- |
| OPEN: implement and record VAL-001 | SCI-001 lacks its complete specified acceptance evidence | Maintainer runs the exact changing-signal/buffer case and records artifacts |
| OPEN: review and calibrate VAL-002 | SCI-002's statistical acceptance procedure is provisional | Scientific reviewer assesses burn-in, nominal error rate, precision, and sensitivity using correct and deliberately wrong kernels |
| OPEN: implement the Gaussian fixture and freeze its execution environment | SCI-002 has no acceptance run | Contributor provides reproducible blocks, harness, and retained run records after procedure review |

Changes to the model, conditional state, oracle, tolerances, or sampling procedure
must explain the scientific reason and identify affected claims. The maintainer
reviews the change; a scientific reviewer assesses statistical criteria. Preserve
old evidence and mark it stale when its assumptions or procedure change.

**Readiness decision**: Draft example, scientific review pending. It is ready to
illustrate specification writing; its scientific claims remain unverified under
the acceptance rules above. No checklist completion or test success is asserted.
