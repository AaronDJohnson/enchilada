# Observation data

## Purpose

Define a common sample grid and scientific representation for observations,
residuals, and block inputs. This is a draft retrospective contract at commit
`dd412c3d47d3a209b632f43ee3220215936a31cc`. Physical channel meaning remains a campaign
agreement. Scenarios specify checks; their evidence status is in [evidence](../../evidence.md).

## Requirements

### Requirement: Declare observation conventions

`L1Data` SHALL carry channel arrays, sample rate in Hz, ordered channel names, TDI
generation, observable, original time-sample count, epoch in seconds, and domain.
Omitted epoch and domain default to zero and time; noise and orbit default to absent.

**Spec Kit mapping**: FR-001.

#### Scenario: Construct a synthetic time observation

- **GIVEN** one real floating channel A with four samples, sample rate 2 Hz, and explicit generation and observable
- **WHEN** an observation is constructed with default epoch and domain
- **THEN** it describes time samples at 0, 0.5, 1, and 1.5 seconds and retains the supplied conventions

### Requirement: Validate the declared grid and representation

Observation construction SHALL reject invalid metadata or arrays that do not match
the declared grid: non-positive/non-finite rate, non-finite epoch, empty convention
strings, invalid domain, empty/duplicate channels, mismatched keys, wrong rank or
length, and unsupported numeric kinds. Time arrays are real floating; frequency
arrays are complex. Construction does not establish physical validity or reject
every non-finite sample; campaign finiteness is a separate boundary.

**Spec Kit mapping**: FR-002.

#### Scenario: Reject inconsistent channel lengths

- **GIVEN** channels A and E with lengths four and five on a declared four-sample time grid
- **WHEN** the observation is constructed
- **THEN** construction raises a validation error identifying the inconsistency

#### Scenario: Reject a real array declared to be a spectrum

- **GIVEN** three float64 bins declared as frequency-domain data with original sample count four
- **WHEN** the observation is constructed
- **THEN** construction rejects the real numeric kind

### Requirement: Derive omitted time-sample counts

A time-domain observation SHALL resolve an omitted sample count, represented by
zero, from its channel arrays and require all channels to match the resolved count.

**Spec Kit mapping**: FR-003.

#### Scenario: Derive the time grid length

- **GIVEN** two real channels with 1,025 samples each and no explicit count
- **WHEN** the observation is constructed
- **THEN** its original time-sample count is 1,025

### Requirement: Preserve frequency-grid parity explicitly

A frequency-domain observation SHALL require the original positive integer
time-sample count `N` and require `N // 2 + 1` complex bins in every channel.

**Spec Kit mapping**: FR-004.

#### Scenario: Refuse to guess parity from bin count

- **GIVEN** 513 complex bins, which could represent 1,024 or 1,025 original samples
- **WHEN** construction omits `N`
- **THEN** construction rejects the omission and explains the parity ambiguity

### Requirement: Preserve scientific context through data operations

Residual formation and actual representation conversion SHALL preserve the sample
grid, channel order, observable, generation, and orbit reference; each residual
carries the Wheel's current noise reference. Representation conversion retains
the noise reference of its input.

**Spec Kit mapping**: FR-005.

#### Scenario: Convert a residual after a noise publication

- **GIVEN** a campaign with an attached orbit and a newly published noise object
- **WHEN** a residual is formed and converted to the other domain
- **THEN** both objects retain the campaign conventions and the same orbit and current noise references

### Requirement: Expose consistent grid quantities

Observation grid accessors SHALL implement `Tobs=N/fs`, `dt=1/fs`, `df=1/Tobs`,
`fny=fs/2`, and first-sample epoch `t0`, with the documented long/short aliases.
`aliases()` returns an independent copy of the name mapping.

**Spec Kit mapping**: FR-006.

#### Scenario: Distinguish duration from the final sample time

- **GIVEN** `N=4`, `fs=2`, and `epoch=100`
- **WHEN** timing quantities are inspected
- **THEN** duration is 2 seconds, spacing is 0.5 seconds, and the final sample is at 101.5 seconds

### Requirement: Apply a consistent Fourier convention

Domain conversion SHALL implement `X=dt*rfft(x)` and
`x=irfft(X/dt, n=N)` for each channel, preserving `N`; conversion to the current
domain returns the same object. The frequency grid is `rfftfreq(N, d=dt)`.
No epoch phase factor, window, or resampling is introduced. Round-trip recovery
applies to spectra originating from valid real time data; arbitrary imaginary DC
or even-length Nyquist coefficients are not preserved by an inverse real transform.

**Spec Kit mapping**: FR-007.

#### Scenario: Round-trip even and odd records

- **GIVEN** the existing real-data fixtures with `N=1024` and `N=1025`
- **WHEN** each is converted to frequency and back to time
- **THEN** all samples agree with the original using explicit `rtol=1e-7`, `atol=1e-12`, with the original count retained

#### Scenario: Check the absolute transform scale independently

- **GIVEN** four samples `[1,0,0,0]` at `fs=2`
- **WHEN** the channel is converted to frequency
- **THEN** its three coefficients are `[0.5,0.5,0.5]`, agreeing exactly in this representable fixture

### Requirement: Check available ephemeris coverage

Observation construction SHALL require an orbit's advertised `t_range`, when
present, to cover `epoch` through `epoch+(N-1)*dt`, including endpoints. Without
an advertised range, the early coverage check is skipped. Delayed response queries
may need additional margins supplied by the caller.

**Spec Kit mapping**: FR-008.

#### Scenario: Accept a table ending at the last sample

- **GIVEN** the four-sample grid at epoch 100 and rate 2 Hz, with orbit coverage `[100,101.5]`
- **WHEN** the observation is constructed
- **THEN** coverage is accepted without requiring the unsampled time 102

### Requirement: Reject non-finite campaign observations

Wheel construction SHALL reject non-finite observed values in any channel before
initializing any block; gap masks and automatic repair are outside this baseline.

**Spec Kit mapping**: FR-009.

#### Scenario: Reject a non-finite sample before model initialization

- **GIVEN** an otherwise valid observation with a NaN in channel E
- **WHEN** a Wheel is constructed
- **THEN** construction fails with a non-finite-data diagnostic before any block can initialize
