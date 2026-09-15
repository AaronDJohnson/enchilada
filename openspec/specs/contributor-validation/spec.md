# Contributor checks and examples

## Purpose

Help scientists integrate a block and interpret what the available checks actually
establish. This draft describes commit `dd412c3d47d3a209b632f43ee3220215936a31cc`.
Conformance establishes exercised interface behavior, while inference correctness
requires model-specific evidence. See [the evidence record](../../evidence.md).

## Requirements

### Requirement: Exercise a real block through normal campaign validation

`check_block(block, observed, n_cycles=2)` SHALL register the supplied instance on
a scratch Wheel and run the requested cycles with normal return validation.
The helper mutates the actual block's state and does not clone or restore it.

**Spec Kit mapping**: FR-039.

#### Scenario: Check a disposable contributor instance

- **GIVEN** a conforming block with an update counter and valid observed data
- **WHEN** `check_block` runs with its default count
- **THEN** initialization and two updates occur on that instance and the helper returns normally

### Requirement: Limit the helper's noise and scientific claims accurately

The conformance helper SHALL evaluate the default spectrum interface only when
the final noise object is present and differs by identity from the observation's
initial model. The result does not establish every intermediate/channel spectrum,
unchanged initial noise, convergence, or posterior correctness.

**Spec Kit mapping**: FR-040.

#### Scenario: Check the last changed model only

- **GIVEN** a block publishing M1 during initialization and M2 on its final update
- **WHEN** conformance checking reaches its noise check
- **THEN** M2's default spectrum is exercised without asserting that M1 or every channel has been validated

### Requirement: Provide a diagnostic zero-signal block

`EchoBlock` SHALL report initialization context and the first channel's residual
RMS magnitude, count its own updates, and publish zero signal templates.

**Spec Kit mapping**: FR-041.

#### Scenario: Diagnose a campaign without fitting sources

- **GIVEN** an EchoBlock on a valid campaign
- **WHEN** three cycles run
- **THEN** its counter is three, its contribution remains zero, and its diagnostics describe the input context and residual magnitude

### Requirement: Provide executable minimal integration examples

The package SHALL provide the minimal script and annotated notebook demonstrating
campaign assembly, independently owned blocks, and repeated cycles.

**Spec Kit mapping**: FR-042.

#### Scenario: Execute the minimal script

- **GIVEN** the core development environment
- **WHEN** `examples/demo.py` runs
- **THEN** its two no-op blocks finish three cycles and report residual RMS

#### Scenario: Execute the annotated demonstration

- **GIVEN** the notebook's numerical-orbit dependencies are available
- **WHEN** the code cells of `examples/demo.ipynb` execute in order in a fresh process
- **THEN** they complete without error and demonstrate the documented integration points

### Requirement: Retain a seeded synthetic inference regression

The project SHALL provide the two-signal, sampled-white-noise toy fit with a
deterministic-seed truth-recovery check. The existing criterion is a regression
for that example, not a calibrated test of arbitrary samplers or joint covariance.

**Spec Kit mapping**: FR-043.

#### Scenario: Reproduce the recorded toy acceptance fixture

- **GIVEN** slow amplitude 3, fast amplitude 2, noise standard deviation 0.5, and seed 0
- **WHEN** the toy fit runs 200 cycles and discards 80 burn-in cycles
- **THEN** every reported spread is positive and each mean is within `max(5*posterior_std,0.05)` of its injected truth

### Requirement: Keep the external galactic-binary example's scope explicit

The galactic-binary example SHALL identify its separately supplied model code,
waveform/sampler dependencies, and recorded environment limits without adding that
stack to core requirements. External scientific recovery is not certified by the
package test suite; the example publishes a model at a final walker-mean estimate.

**Spec Kit mapping**: FR-044.

#### Scenario: Interpret the example publication correctly

- **GIVEN** the implementation in `examples/gb_model.py` and the external notebook's setup instructions
- **WHEN** a contributor reviews the returned signal's meaning
- **THEN** it is identified as a model rendered from a point estimate, with no automatic claim that those publications are exact conditional draws

### Requirement: Identify documented contract violations clearly

For documented guarded inputs, diagnostics SHALL identify the relevant setting,
block operation, channel, grid, noise model, or orbit span; common data-attribute
misspellings receive discoverability hints. This does not standardize every error
from arbitrary malformed external objects.

**Spec Kit mapping**: FR-045.

#### Scenario: Diagnose a block's incompatible return

- **GIVEN** a block named source returning the wrong channel grid from its update
- **WHEN** campaign return validation fails
- **THEN** the diagnostic identifies source's update and the relevant grid mismatch

#### Scenario: Help identify a misspelled grid attribute

- **GIVEN** an observation and a common misspelling of one of its documented public data attributes
- **WHEN** that attribute is accessed
- **THEN** `AttributeError` includes a discoverability hint instead of creating a dynamic attribute
