# Noise exchange and weighting

## Purpose

Share one current noise model with consistent one-sided spectral and variance
conventions. This draft describes commit `dd412c3d47d3a209b632f43ee3220215936a31cc`.
Models are shared references treated as immutable. Likelihood construction and
general cross-channel covariance remain contributor responsibilities.
See [evidence](../../evidence.md) for numerical fixtures and limitations.

## Requirements

### Requirement: Preserve noise until a new publication

A campaign SHALL support absent, initially fixed, or block-published noise;
`Template.noise=None` publishes nothing and retains the current model.

**Spec Kit mapping**: FR-028.

#### Scenario: Keep fixed noise across signal updates

- **GIVEN** an observation carrying noise model M
- **WHEN** signal blocks publish templates with no noise model
- **THEN** subsequent residuals still carry M by identity

### Requirement: Support a noise-only contributor

A noise-only block SHALL use a zero signal template with its noise publication,
making the new model available to subsequently formed residuals.

**Spec Kit mapping**: FR-029.

#### Scenario: Refresh weighting without subtracting a signal

- **GIVEN** a noise block registered before a signal block
- **WHEN** the noise block publishes model M2 with zero signal
- **THEN** the signal block receives M2 and its residual samples are unchanged by that zero contribution

### Requirement: Keep one noise owner and identify replacement

The Wheel SHALL keep one current noise model and emit `NoiseOverwrittenWarning`
when a different block replaces the current block owner; repeated publications by
one owner and the first takeover of initial observation noise are silent. Under
normal warning handling the new model becomes current; models are not combined.

**Spec Kit mapping**: FR-030.

#### Scenario: Identify competing noise contributors

- **GIVEN** block instrument has published a noise model
- **WHEN** block foreground publishes another under normal warning handling
- **THEN** a warning names both owners and the foreground model becomes current for subsequent residuals

### Requirement: Evaluate the observation's positive-frequency grid

Noise-spectrum access SHALL evaluate the model on non-DC `rfftfreq(N,d=dt)` bins,
dispatch an explicit requested channel when supplied, and expose DC as positive
infinity for zero statistical weight.

**Spec Kit mapping**: FR-031.

#### Scenario: Evaluate a channel-specific spectrum without querying DC

- **GIVEN** `N=4`, `fs=2`, and a channel-specific model
- **WHEN** the spectrum for A is requested
- **THEN** the model receives frequencies `[0.5,1.0]` for A and the returned full grid starts with infinite DC power

### Requirement: Validate one-sided spectral power on consumption

A consumed noise spectrum SHALL be finite and strictly positive at every non-DC
bin and follow the convention below; a missing callable `psd` or invalid values
fail at consumption. With `X=dt*rfft(x)`, an interior bin has
`E[abs(X_k)^2] = (Tobs/2)*S_k`. White time noise of variance `sigma^2` uses
`S_k=2*sigma^2/fs`. Even-length Nyquist has one real degree of freedom and cannot
be weighted as an ordinary interior complex bin by a likelihood.

**Spec Kit mapping**: FR-032.

#### Scenario: Reject zero spectral power

- **GIVEN** a noise model that returns zero at one positive-frequency bin
- **WHEN** its spectrum is consumed
- **THEN** consumption raises a noise-spectrum validation error

#### Scenario: Check the independent white-noise normalization fixture

- **GIVEN** seed 1, 40 white-noise realizations, `N=8192`, `fs=0.2`, and `sigma=0.7`
- **WHEN** the existing fixture averages the power ratios to `(Tobs/2)*S` over bins selected by `[10:-10]`
- **THEN** the mean ratio differs from 1 by at most 0.05, under this fixed regression criterion

### Requirement: Integrate the zero-mean time variance consistently

Noise-variance access SHALL compute `df * sum(w_k*S_k)` over non-DC bins, with
weight 1 except weight 1/2 for an even-length Nyquist bin. With no noise model,
spectrum and variance access both return `None`.

**Spec Kit mapping**: FR-033.

#### Scenario: Check white-noise variance for both parities

- **GIVEN** the white-noise fixtures with `sigma=0.7`, `N=2048` and `N=2049`
- **WHEN** variance is integrated from the one-sided spectrum
- **THEN** it agrees with `expected=sigma^2*(1-1/N)` within absolute error `max(1e-12,1e-12*abs(expected))`, matching the existing pytest comparison's absolute floor and relative tolerance

#### Scenario: Handle a DC-only observation

- **GIVEN** a one-sample observation with a valid noise model
- **WHEN** its zero-mean noise variance is requested
- **THEN** the variance is zero because no positive-frequency bins remain
