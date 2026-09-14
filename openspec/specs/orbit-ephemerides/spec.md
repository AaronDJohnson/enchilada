# Orbit ephemerides

## Purpose

Give all contributing response models the same constellation positions, clock,
coordinate frame, and nominal arm context. This draft describes commit
`dd412c3d47d3a209b632f43ee3220215936a31cc`; it does not establish an astrophysical
response error budget. See [evidence](../../evidence.md) for fixture accuracy and
the limits of interpolation and frame tests.

## Requirements

### Requirement: Share an optional immutable ephemeris

An attached orbit SHALL remain a shared reference on the observation and derived
residuals, with participants treating it as immutable and using query times on the
observation's absolute clock. Orbit-free campaigns remain valid.

**Spec Kit mapping**: FR-034.

#### Scenario: Use a common constellation in two blocks

- **GIVEN** an observation with an attached orbit and two registered blocks
- **WHEN** both blocks receive residuals
- **THEN** both receive the same orbit object and interpret query times on the observation's clock

### Requirement: Define spacecraft coordinate and arm conventions

The orbit interface SHALL expose nominal arm length `L` in metres, transfer
frequency `fstar` in Hz, and `positions(t)` as ecliptic Cartesian `(x,y,z)` arrays,
each shaped `(3,m)` for `m` query times and ordered by spacecraft then time.
Numerical orbits also support scalar queries with coordinate shape `(3,1)`.

**Spec Kit mapping**: FR-035.

#### Scenario: Preserve coordinate axes for a vector query

- **GIVEN** two valid absolute query times
- **WHEN** spacecraft positions are requested
- **THEN** each coordinate has shape `(3,2)`, with each row referring to the same spacecraft across both times

### Requirement: Interpolate inside the table and refuse extrapolation

A numerical orbit SHALL use cubic-spline interpolation within its tabulated range,
accept exact endpoints, and reject queries outside it. Valid construction requires
finite positions and at least two finite strictly increasing times; all malformed
rank errors are not normalized into one package-specific diagnostic.

**Spec Kit mapping**: FR-036.

#### Scenario: Respect both temporal boundaries

- **GIVEN** a numerical table spanning absolute times 100 through 200
- **WHEN** positions at 100, 200, 99, and 201 are requested separately
- **THEN** the endpoints are accepted and the two outside requests raise errors

#### Scenario: Check the existing interpolation regression

- **GIVEN** the existing 200-node, 30-day circular-orbit fixture
- **WHEN** positions are evaluated at its three between-node query times
- **THEN** X/Y coordinates agree with the analytic fixture using `rtol=1e-7`, `atol=1` metre

### Requirement: Preserve frame and axes through supported loaders

Numerical orbit loading SHALL support in-memory arrays, the documented HDF5 layout,
and external objects with `compute_position(times)`, preserving time epochs and
spacecraft/time/coordinate axes. Array input defaults to ecliptic; HDF5 and external
input default to equatorial. Equatorial conversion uses `epsilon=0.40909280422232897`
radians and `(x, cos(epsilon)*y+sin(epsilon)*z, -sin(epsilon)*y+cos(epsilon)*z)`.

**Spec Kit mapping**: FR-037.

#### Scenario: Convert an equatorial axis without exchanging spacecraft

- **GIVEN** equatorial input containing a spacecraft position `(0,1,0)` metres
- **WHEN** it is loaded with the equatorial frame declared
- **THEN** that spacecraft's ecliptic position is `(0,cos(epsilon),-sin(epsilon))`, compared with `rtol=1e-7`, `atol=1e-3` metre

#### Scenario: Preserve an absolute HDF5 epoch

- **GIVEN** a supported HDF5 table whose sampling attributes are `t0`, `dt`, and `size`
- **WHEN** it is loaded
- **THEN** tabulated times are `t0 + arange(size)*dt` without subtracting the absolute epoch

#### Scenario: Normalize the supported external position layout

- **GIVEN** external positions shaped `(m,9)` or `(m,3,3)` with time first
- **WHEN** the external-orbit loader consumes them
- **THEN** it preserves spacecraft and coordinate identities in `(3,m,3)` storage and rejects other documented unsupported layouts

### Requirement: Provide documented nominal arm defaults and overrides

Numerical orbit construction SHALL honor supplied arm/transfer overrides where
offered; otherwise it derives `L` from the mean three pairwise separations and
`fstar=299792458/(2*pi*L)`. A non-positive or non-finite derived mean falls back to
`2.5e9` metres. Explicit overrides receive no separate positivity/consistency guard.

**Spec Kit mapping**: FR-038.

#### Scenario: Supply a deterministic fallback for a degenerate table

- **GIVEN** valid times but all three spacecraft positions coincide and no overrides are supplied
- **WHEN** the numerical orbit is constructed
- **THEN** nominal arm length is `2.5e9` metres and transfer frequency is derived from that value
