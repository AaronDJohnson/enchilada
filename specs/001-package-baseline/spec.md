# Feature Specification: enchilada Package Baseline

**Feature Branch**: `template-return`

**Feature Directory**: `specs/001-package-baseline`

**Created**: 2026-09-08

**Status**: Draft — current behavior documented for maintainer review

**Baseline Commit**: `dd412c3d47d3a209b632f43ee3220215936a31cc`

**Package Version**: `0.1.0` in project metadata, including the unreleased template-return changes on this branch

**Input**: User description: "I want to make a spec kit specification for this entire package. Can you build that for me?"

## Purpose and Scope

enchilada lets independently developed source-population and noise models participate in one LISA global fit. A campaign coordinator supplies the observation and shared scientific conventions; each contributing group supplies a block containing its own model and sampler. The package coordinates their updates and keeps the residual accounting consistent.

This is a retrospective specification of the entire package at the baseline commit. Requirements describe supported behavior and contributor responsibilities. They do not establish a roadmap or claim that every behavior already has an automated acceptance test. Exact public interfaces, numerical conventions, and operational qualifications are in the [package contract](contracts/package-contract.md); [traceability](traceability.md) records evidence and gaps.

The scope includes observation and template containers, ordered block orchestration, noise exchange, shared ephemerides and numerical orbit loaders, conformance helpers, examples, and package delivery. Waveform generation, likelihood construction, sampling algorithms, and scientific convergence decisions belong to block authors.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Coordinate independently owned source models (Priority: P1)

As a campaign coordinator, I want several source-class teams to fit the same observation, with each block receiving the data its own sources must explain, so their contributions form one joint model.

**Why this priority**: Correct conditional data and residual accounting are the package's central purpose.

**Independent Test**: Use a synthetic observation and two deterministic blocks with known initial and updated contributions; inspect the data handed to each block and the resulting full residual without any waveform library.

**Acceptance Scenarios**:

1. **Given** two blocks registered in order, **When** the second initializes, **Then** it receives the observation minus the first block's initial contribution.
2. **Given** observations equal to 10 and initial contributions of 2 and 3, **When** the first block updates its contribution to 4, **Then** it receives 7, the second subsequently receives 6, and the full residual is 3 if the second keeps its contribution at 3.
3. **Given** a registered block, **When** several cycles are requested, **Then** it updates once per cycle in registration order and retains its own sampler state across calls.
4. **Given** a block with no remaining sources, **When** it returns an explicit zero contribution, **Then** its previous contribution is replaced by zero without a withdrawal warning.
5. **Given** a block returning an observation/residual object instead of a template, **When** it initializes or updates, **Then** the package rejects the result and explains the template-return contract.

### User Story 2 - Establish consistent scientific data conventions (Priority: P1)

As a scientist preparing a campaign, I want every block to receive the same sampling grid, channel labels, physical interpretation, and epoch, so teams cannot silently use different conventions.

**Why this priority**: Inconsistent conventions can produce plausible but scientifically incorrect fits.

**Independent Test**: Construct valid and invalid observations, derive their timing quantities, and transform both even- and odd-length time series to frequency space and back.

**Acceptance Scenarios**:

1. **Given** real time series with a common length, **When** an observation is constructed without a sample count, **Then** the count is derived and all channels must match it.
2. **Given** a frequency-only observation without its original time-sample count, **When** construction is attempted, **Then** the package explains that the frequency-bin count cannot determine the original parity.
3. **Given** valid data with either parity, **When** it is transformed and inverted, **Then** the original sample count and metadata are retained and the original samples are recovered within numerical tolerance.
4. **Given** mismatched channels, representation, lengths, or invalid run settings, **When** data is constructed, **Then** construction fails with a relevant diagnostic; each block additionally checks whether it supports the declared physical conventions.
5. **Given** non-finite observed samples, **When** a campaign is created, **Then** it fails before calling a block and explains the absence of gap-mask support.

### User Story 3 - Share a current noise estimate (Priority: P1)

As a noise-model contributor, I want to publish a common noise estimate that signal blocks can consume with consistent normalization, so every subsequent update uses the current estimate.

**Why this priority**: Source and noise estimates condition on one another in a global fit.

**Independent Test**: Combine a zero-signal noise block with a signal block that records the model it receives; evaluate a known white-noise model on even and odd sample counts.

**Acceptance Scenarios**:

1. **Given** a fixed initial noise model, **When** signal blocks publish only signal templates, **Then** the same noise model remains available.
2. **Given** a noise block publishing a new model, **When** a later block initializes or updates, **Then** it receives the new model without a signal contribution from the noise block.
3. **Given** two blocks publishing noise models, **When** ownership changes between them, **Then** the later model becomes current and a warning identifies both owners; the models are not automatically combined.
4. **Given** an available noise model, **When** its spectrum or time-domain variance is requested, **Then** the result uses the observation's grid, supports channel-specific models, excludes DC from weighting, and accounts for the even-length Nyquist bin.
5. **Given** no noise model, **When** either noise quantity is requested, **Then** it is absent; **Given** an invalid model, **When** its spectrum is consumed, **Then** consumption fails with a model-related diagnostic.

### User Story 4 - Share and load constellation ephemerides (Priority: P2)

As a response-model author, I want all blocks to use the campaign's constellation positions on the same clock and coordinate frame, so independently computed responses describe the same observation.

**Why this priority**: Real-data response consistency needs a shared ephemeris; synthetic campaigns can operate without one.

**Independent Test**: Load a synthetic spacecraft-position table, check interpolation and frame conversion, attach it to an observation, and query its boundaries.

**Acceptance Scenarios**:

1. **Given** an attached orbit reporting its valid time span, **When** the observation is constructed, **Then** that span must cover the first through last sample, including exact endpoints.
2. **Given** a valid orbit, **When** blocks receive residuals, **Then** they receive the same orbit reference and query positions in ecliptic metres on the observation's clock.
3. **Given** supported array, ephemeris-file, or external-orbit inputs, **When** a numerical orbit is loaded, **Then** spacecraft and coordinate axes retain their meaning and equatorial inputs are converted to the shared frame.
4. **Given** a numerical orbit, **When** positions outside its tabulated time span are requested, **Then** it refuses extrapolation; valid endpoints remain accessible.

### User Story 5 - Inspect progress and contain invalid updates (Priority: P2)

As a campaign coordinator, I want to inspect residuals and individual contributions between cycles and receive useful failures, so I can diagnose and checkpoint my own campaign.

**Why this priority**: Long-running fits need observability and clear boundaries around failures.

**Independent Test**: Run deterministic blocks with a recording callback, alter returned diagnostic arrays, and introduce a block with an invalid return.

**Acceptance Scenarios**:

1. **Given** a running campaign, **When** full residuals, conditional residuals, or individual contributions are requested, **Then** the values reflect the current ledger and their arrays can be edited without altering the ledger or observed arrays.
2. **Given** a callback and a request for three cycles, **When** all cycles complete, **Then** the callback runs after each full cycle with indices 0, 1, and 2 and access to the coordinator.
3. **Given** a block failing return validation during a cycle, **When** the error propagates, **Then** its invalid contribution is not adopted, later blocks are not updated, and the incomplete cycle's callback is not called; earlier successful updates remain.
4. **Given** a block failing initialization or initial return validation, **When** registration is attempted, **Then** campaign membership and ledger entries remain as before the attempt, subject to the shared-object responsibilities in the contract.

### User Story 6 - Check a contributed block before integration (Priority: P2)

As a block author, I want a small conformance check and a no-op example block, so I can verify my integration without assembling a production campaign.

**Why this priority**: Teams need fast feedback at the protocol boundary.

**Independent Test**: Exercise conforming, off-grid, legacy-return, and invalid-noise blocks through the helper on synthetic data.

**Acceptance Scenarios**:

1. **Given** a conforming block, **When** the conformance helper initializes it and performs the requested updates, **Then** the helper completes without error in either supported representation.
2. **Given** a block returning the wrong type or grid, **When** the helper exercises that return, **Then** it raises the underlying contract error.
3. **Given** a newly published final noise model, **When** conformance checking finishes, **Then** its default spectrum interface is exercised; channel-specific correctness and earlier noise publications require the author's own checks.
4. **Given** the no-op diagnostic block, **When** it runs, **Then** it reports context and residual magnitude, counts its updates, and leaves signal contributions at zero.

### User Story 7 - Learn through executable examples (Priority: P2)

As a new contributor, I want examples that progress from orchestration to an actual fit, so I can see which responsibilities belong to the package and which belong to my block.

**Why this priority**: Executable examples make the integration contract usable.

**Independent Test**: Execute the minimal script, its notebook with orbit dependencies available, and the seeded toy fit; inspect the external galactic-binary example's setup instructions separately.

**Acceptance Scenarios**:

1. **Given** the core development environment, **When** the minimal example runs, **Then** two no-op blocks complete three cycles and report the residual.
2. **Given** the seeded toy campaign, **When** two signal blocks and one sampled-noise block run, **Then** their posterior summaries satisfy the documented truth-recovery tolerance.
3. **Given** a scientist using the galactic-binary notebook, **When** they prepare the environment, **Then** its external waveform, sampler, plotting dependencies, and recorded platform limits are explicit; those model components remain example code.

### User Story 8 - Install and maintain the package (Priority: P2)

As a user or downstream packager, I want a small typed core and verifiable release artifacts, so I can integrate the orchestration layer without installing an entire scientific modeling stack.

**Why this priority**: Independent teams and packaging systems need predictable installation and compatibility claims.

**Independent Test**: Inspect declared dependencies and public exports, exercise the supported CI configurations, install a built wheel, and run the suite from an unpacked source distribution.

**Acceptance Scenarios**:

1. **Given** a core-only installation, **When** the package is imported and an orbit-free campaign runs, **Then** optional orbit and notebook dependencies are unnecessary.
2. **Given** an installed wheel, **When** consumers inspect it, **Then** the version, documented public exports, and typing marker are available.
3. **Given** a source distribution, **When** a downstream packager unpacks it, **Then** the tests and supporting examples needed to run its suite are present.
4. **Given** a tagged release, **When** publication proceeds, **Then** its tag matches the package version and the configured release quality gates and artifact smoke check have passed.

### Edge Cases

| Condition | Required outcome or scope boundary |
| --- | --- |
| No registered blocks | Residual data equals the observation; requested cycle callbacks still run. |
| Zero requested cycles | No updates or callbacks; initialization already performed during registration remains in effect. |
| Negative, Boolean, or non-integral cycle count | Reject the cycle request. |
| Empty, duplicate, or non-string block name; missing operation | Reject before invoking initialization. |
| Unknown contribution or excluded-block name | Reject and identify available block names. |
| A zero contribution in only one channel | Preserve independent per-channel contributions. |
| Wider-precision template than observation | Preserve the template and promote residual arithmetic as needed. |
| A reused template buffer or edited diagnostic arrays | Previously adopted ledger entries remain unchanged. |
| Non-finite template values in any channel | Reject at the campaign return boundary before adoption. |
| A one-sample time series | Permit the positive sample count; its spectrum has only DC and its zero-mean noise variance is zero. |
| Even and odd counts sharing a frequency-bin count | Preserve or require the original time-sample count; never infer parity from bins. |
| Noise omitted by a signal block | Keep the existing model; omission is not a clear operation. |
| The same noise block refreshes its model | Update without an ownership warning. |
| A block takes over the observation's initial noise model | Update without an ownership warning. |
| Orbit without a reported validity span | Skip the early span check; response authors remain responsible for valid query times. |
| Delayed response queries outside the observation span | Require sufficient ephemeris margin; observation construction alone cannot guarantee it. |
| Exceptions, including a callback exception | Propagate to the caller; no automatic retry or whole-cycle rollback. |

## Requirements *(mandatory)*

### Functional Requirements

#### Observation and scientific conventions

- **FR-001**: A campaign observation MUST declare its samples, sample rate, channels, TDI generation, and physical observable; the representation MUST default to time and the epoch to zero when omitted.
- **FR-002**: Observation construction MUST validate a positive finite sample rate, finite epoch, non-empty convention strings, supported representation, unique non-empty channel collection, matching channel keys, and one-dimensional arrays with the representation's required lengths and numeric kind.
- **FR-003**: Time-domain observations MUST derive an omitted sample count from their arrays and validate all channels against the resolved positive integer count.
- **FR-004**: Observations constructed directly in frequency space MUST require the original positive time-sample count and use the corresponding one-sided grid.
- **FR-005**: Residual formation and representation conversion MUST preserve campaign conventions and the shared orbit; residuals MUST carry the current noise model.
- **FR-006**: Users MUST be able to obtain observation duration, sample interval, frequency spacing, Nyquist frequency, sample count, and epoch through the documented long and short names, and discover their equivalences.
- **FR-007**: Representation conversion MUST use the specified Fourier normalization, preserve sample count, invert valid time-origin data within numerical tolerance for both parities, and leave an already matching representation unchanged.
- **FR-008**: An attached ephemeris reporting a validity span MUST cover the observation's first and last sample at construction.
- **FR-009**: Campaign construction MUST reject non-finite observed samples before any block is initialized.

#### Block participation and templates

- **FR-010**: A contributing block MUST expose a name, an initialization operation, and an update operation; its name MUST remain stable while registered.
- **FR-011**: Registration MUST reject invalid or duplicate names and missing callable operations before invoking initialization.
- **FR-012**: Successful registration MUST initialize the block once using the observation minus previously registered contributions, with the current noise model attached.
- **FR-013**: Initialization exceptions and failed initial return validation MUST leave campaign membership, ledger entries, and the current noise slot unchanged; this guarantee does not restore block-owned state or undo prohibited mutation of shared objects.
- **FR-014**: Each successful initialization or update MUST return a distinct template representing only that block's current summed signal, covering the campaign's channels and grid with finite values of the correct numeric kind.
- **FR-015**: A block MUST be able to publish an explicit zero template, including after previously publishing a nonzero contribution, without a withdrawal warning.
- **FR-016**: Templates MUST carry signal samples and an optional noise publication only; they MUST NOT carry or replace campaign settings or the orbit.
- **FR-017**: Block authors MUST validate conventions their models cannot support and retain responsibility for parameters, randomness, chains, tuning, checkpoints, and external resources.

#### Campaign orchestration and inspection

- **FR-018**: Each full cycle MUST update every registered block once in registration order.
- **FR-019**: Each block update MUST receive the observation minus every other block's latest contribution, with its own contribution excluded from subtraction.
- **FR-020**: A valid returned template MUST replace that block's previous ledger entry directly, and subsequent block updates MUST use the replacement.
- **FR-021**: A run MUST accept non-negative integer cycle counts, including supported integer scalar equivalents, reject invalid counts, and perform no updates or callbacks for zero cycles.
- **FR-022**: Users MUST be able to retrieve the full residual, the residual conditional on one named block, and a named block's current contribution; unknown block names MUST be rejected.
- **FR-023**: Residual and contribution access MUST provide independent arrays, and adopted template arrays MUST be copied so later buffer reuse cannot change the ledger.
- **FR-024**: Residual arithmetic MUST accommodate wider-precision templates without narrowing stored contributions to the observation's precision.
- **FR-025**: An optional progress callback MUST run after each completed cycle with the coordinator and a zero-based index local to that run invocation.
- **FR-026**: A block exception or return-validation failure MUST stop the run and propagate; an invalid return MUST NOT replace its ledger entry, and no later block or incomplete-cycle callback may run.
- **FR-027**: A failure MUST NOT imply automatic rollback of earlier successful updates or restoration of any block's internal state; restarting and checkpoint recovery remain caller responsibilities.

#### Noise exchange and weighting

- **FR-028**: A campaign MUST support no noise model, an initially supplied fixed model, or a model published by a block; omitting a publication MUST retain the existing model.
- **FR-029**: A noise-only block MUST return a zero signal template and publish its new noise model for all subsequently formed residuals.
- **FR-030**: The campaign MUST maintain one current noise model, warn when ownership changes between blocks, and avoid ownership warnings for repeated publications by one owner or takeover of the observation's initial model.
- **FR-031**: Noise-spectrum access MUST evaluate the model on the positive-frequency observation grid, optionally dispatch by channel, and expose DC as infinite noise power for zero statistical weight.
- **FR-032**: Noise spectra MUST use the one-sided normalization specified in the package contract and be finite and strictly positive at every non-DC bin; missing spectrum operations or invalid values MUST fail when consumed.
- **FR-033**: Noise-variance access MUST integrate the same spectrum with DC excluded and the even-length Nyquist bin half-weighted, producing the zero-mean time-series variance; both noise accessors MUST return no value when the model is absent.

#### Shared constellation ephemerides

- **FR-034**: The observation MAY carry a fixed shared ephemeris, which blocks MUST treat as immutable and use on the same time reference as the observation.
- **FR-035**: The orbit interface MUST expose nominal arm length, transfer frequency, and the three spacecraft's Cartesian coordinates in ecliptic metres at requested times.
- **FR-036**: Numerical ephemerides MUST interpolate their tabulated positions, permit endpoint queries, and reject requests outside the table's time span.
- **FR-037**: Numerical ephemerides MUST support in-memory arrays, the documented ephemeris-file layout, and supported external orbit objects, with explicit frame handling and preservation of spacecraft/time/coordinate axes.
- **FR-038**: Numerical ephemerides MUST accept the documented arm-length and transfer-frequency overrides where offered, otherwise derive the nominal arm length and transfer frequency, with the documented fallback for a degenerate position table.

#### Contributor tools and examples

- **FR-039**: A conformance helper MUST initialize a supplied block on a scratch campaign and perform a configurable number of updates, defaulting to two, using the campaign's normal return validation.
- **FR-040**: After those updates, the helper MUST exercise the default spectrum interface if the final noise model is present and differs by identity from the observation's initial model; conformance MUST NOT be presented as proof of sampler correctness or exhaustive noise validation.
- **FR-041**: A diagnostic no-op block MUST expose initialization context, report residual magnitude, retain its own update count, and publish zero signal templates.
- **FR-042**: The project MUST provide an executable minimal demonstration and an annotated notebook showing the campaign integration points.
- **FR-043**: The project MUST provide a deterministic-seed toy fit with two signal components and sampled noise whose posterior summaries can be checked against known truth.
- **FR-044**: The galactic-binary integration example MUST identify its separately supplied model code, external dependencies, and recorded environment limits without adding that modeling stack to core package requirements.
- **FR-045**: Contract failures MUST identify the relevant setting, block operation, channel, grid, model, or orbit span sufficiently to locate the violation; common data-attribute misspellings MUST provide discoverability hints.

#### Package delivery and maintenance

- **FR-046**: Core installation and import MUST require only the declared core numerical dependency; numerical-orbit and notebook capabilities MUST be optional installation choices.
- **FR-047**: The distributed package MUST expose its documented public names, installed version, replacement helper, and typing information; valid aliases MUST remain usable and misspelled data attributes MUST remain detectable by supported type checking.
- **FR-048**: Source distributions MUST include the test suite and supporting examples, and the project MUST verify that the packaged suite runs from the unpacked source tree.
- **FR-049**: The configured main CI gate MUST check lint, formatting, types, tests with at least 95% branch-inclusive coverage, artifact builds, and installed-wheel imports; additional jobs MUST exercise the declared core-only and dependency-floor configurations.
- **FR-050**: Tagged publication MUST require a matching project version, the configured release gates, and a successful wheel smoke check; a manually requested release workflow MUST stop before publication.

### Key Entities *(include if feature involves data)*

| Entity | Meaning and relationships |
| --- | --- |
| Observation | The campaign's input channel samples and shared scientific context; supplies the baseline for every residual. |
| Residual | An observation-shaped view of data after selected current contributions are subtracted; includes the current noise model. |
| Template | One block's current summed signal across all campaign channels, optionally accompanied by a noise publication. |
| Block | An independently owned contributor with a stable name, initialization, update behavior, and private sampler state. |
| Wheel | The campaign coordinator holding block order, the observation reference, current contributions, and noise ownership. |
| Ledger entry | The copied signal samples last accepted from one named block. |
| Noise model | The single current source of one-sided noise power, optionally channel-dependent. |
| Orbit | The shared constellation ephemeris and its arm-length/transfer-frequency context. |
| Cycle | One ordered pass updating all blocks; a block update may contain many internal sampler steps. |

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In the two-block accounting scenario, every handed residual and final contribution agrees with direct subtraction of the known current signals, within the declared numeric comparison tolerance, over multiple cycles.
- **SC-002**: A campaign with three blocks requested to run four cycles produces exactly four updates per block and four completed-cycle notifications in the documented order.
- **SC-003**: Editing inspected residuals, inspected contributions, or a previously returned template buffer changes zero stored observation or ledger samples.
- **SC-004**: For the reference even/odd fixtures of 1,024 and 1,025 samples, a representation round trip retains every channel and convention and agrees with the original samples within the baseline comparison tolerance (`rtol=1e-7`, `atol=1e-12`).
- **SC-005**: For the reference white-noise fixtures, integrated variance agrees with `sigma² × (1 − 1/N)` within relative tolerance `1e-12` for both parities; measured interior spectral power agrees with its predicted normalization within 5% in the seeded ensemble fixture.
- **SC-006**: Every invalid-input category in the tested validation matrix is rejected at its documented boundary, before that invalid signal contribution can be consumed by another block.
- **SC-007**: The reference orbit interpolation and frame-conversion fixtures meet their recorded numerical tolerances, accept both table endpoints, and reject all tested out-of-span requests.
- **SC-008**: With seed 0, 200 cycles, and 80 burn-in cycles, all three toy-fit posterior means lie within the larger of five posterior standard deviations or 0.05 of their injected truths; all posterior standard deviations are positive.
- **SC-009**: A contributor can complete the minimal example and block conformance workflow without a waveform or sampler dependency, and the annotated demonstration executes when its orbit dependencies are available.
- **SC-010**: Each declared delivery configuration has a repeatable validation path, and both an installed wheel and an unpacked source distribution support their documented consumer workflows.

The [traceability record](traceability.md) distinguishes automated evidence, source-reviewed behavior, manual checks, and validation still outside this baseline run. Coverage percentage alone is not proof that every scientific or operational edge case has been tested.

## Assumptions

- The requested deliverable is a specification of the existing branch. Future capability changes will be specified separately and explicitly supersede relevant baseline requirements.
- Users are scientific-software contributors and campaign coordinators. The package is an in-process library; it has no end-user account system, hosted service, command-line campaign runner, or graphical campaign interface.
- Observations are finite, uniformly sampled, gap-free, and fit in memory. Channel labels and physical observable strings express campaign agreements; the package does not prove those physical agreements are correct or convert channel bases.
- The caller preserves the original observed arrays for the lifetime of the campaign. Frozen containers do not make their arrays or attached objects deeply immutable. Blocks and callbacks follow the ownership rules in the contract.
- Block names remain stable, execution is sequential, and callbacks inspect the campaign without mutating its structure. Thread safety, parallel scheduling, automatic retries, persistent checkpoints, and block removal are outside scope.
- A noise model is a shared immutable object replaced by publication. The package has one noise slot, no clear-noise operation, no automatic component composition, and no general cross-channel covariance consumer.
- Orbit-free campaigns are valid. Response authors provide any additional ephemeris margin needed for delayed evaluations; the early sample-span check is only a coarse compatibility check.
- Waveforms, priors, likelihoods, sampler tuning, convergence assessment, external-process cleanup, and scientific correctness remain block-author responsibilities. The two-method protocol has no finalization callback.
- The [package contract](contracts/package-contract.md) records the baseline dependency and platform matrix. Its external-example matrix is repository documentation at this commit, not a fresh claim about current upstream wheel availability.
- Historical performance figures are illustrative measurements, not a throughput or latency guarantee. No new performance target is imposed by this specification.
