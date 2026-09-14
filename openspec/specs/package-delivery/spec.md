# Package delivery

## Purpose

Preserve the installation and verification paths needed to use the scientific
core independently of optional integrations. This draft records repository
configuration at `dd412c3d47d3a209b632f43ee3220215936a31cc`, whose metadata version is
0.1.0. It is not a fresh claim about external package availability or a record of
publication. See [evidence](../../evidence.md) for execution boundaries.

## Requirements

### Requirement: Keep the numerical core independently installable

Core installation and import SHALL require only the declared runtime NumPy
dependency on supported Python versions; numerical orbits and notebooks remain
optional extras. The baseline declares Python `>=3.12` and NumPy `>=1.23`.

**Spec Kit mapping**: FR-046.

#### Scenario: Run without optional orbit and notebook dependencies

- **GIVEN** a supported core-only installation
- **WHEN** Enchilada is imported and an orbit-free campaign is run
- **THEN** SciPy, HDF5, lisaorbits, and notebook dependencies are unnecessary

### Requirement: Expose the documented public and typed surface

The package SHALL expose `Block`, `NoiseBlock`, `NoiseOverwrittenWarning`,
`NumericOrbit`, `Orbit`, `L1Data`, `Template`, `Wheel`, `__version__`, and
`dataclasses.replace` as `replace`, with a distributed `py.typed` marker and valid
data aliases. Version uses installed metadata, falling back to `0+unknown` when
metadata is unavailable. Typing does not make unknown attributes valid members.

**Spec Kit mapping**: FR-047.

#### Scenario: Inspect a built distribution

- **GIVEN** an installed built wheel
- **WHEN** its public names, version, and typing marker are inspected
- **THEN** the documented names and metadata version are available and `py.typed` is present

### Requirement: Ship a source distribution that can run its tests

The source distribution SHALL include the test suite and supporting example files,
with a configured verification job that runs them from the unpacked distribution
independently of the checkout.

**Spec Kit mapping**: FR-048.

#### Scenario: Verify the packaged source in isolation

- **GIVEN** a source distribution built from the baseline configuration
- **WHEN** the configured sdist job unpacks it outside the checkout and installs its dependencies
- **THEN** the packaged tests and supporting examples are available for the packaged-source test run

### Requirement: Maintain the configured quality and compatibility checks

The main CI configuration SHALL check lint, formatting, types, tests with at least
95% branch-inclusive coverage, builds, and installed-wheel imports; separate jobs
exercise core-only interpreters, declared dependency floors, and the unpacked sdist.
Passing these checks does not establish scientific completeness.

**Spec Kit mapping**: FR-049.

#### Scenario: Refuse a change that misses the coverage floor

- **GIVEN** the main CI environment with its coverage gate enabled
- **WHEN** the test run measures branch-inclusive coverage below 95%
- **THEN** that quality gate fails

#### Scenario: Check the distinct installation configurations

- **GIVEN** the checked-in CI workflow
- **WHEN** its jobs are inspected
- **THEN** core-only Python 3.12/3.13/3.14, dependency-floor resolution, and unpacked-sdist checks are distinct from the full-extra matrix

### Requirement: Gate publication on a matching version and release checks

Tagged publication SHALL require the configured release gates, a tag matching the
project version under package-version normalization, and a successful wheel smoke
check; manual workflow dispatch stops before publication. The release workflow
does not include main CI's separate unpacked-sdist job.

**Spec Kit mapping**: FR-050.

#### Scenario: Reject a mismatched tag

- **GIVEN** metadata version 0.1.0 and a release tag `v0.2.0`
- **WHEN** the release build checks the tag
- **THEN** the version gate fails and publication cannot proceed

#### Scenario: Rehearse the release without publishing

- **GIVEN** a manually dispatched release workflow
- **WHEN** its validation and build jobs complete
- **THEN** the tag-only publication condition keeps the publication job from running
