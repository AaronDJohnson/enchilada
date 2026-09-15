# Block orchestration

## Purpose

Maintain additive signal accounting while independently owned blocks update in a
fixed order. This draft describes commit `dd412c3d47d3a209b632f43ee3220215936a31cc`.
For each channel, with observation `d` and adopted signals `T_j`:

```text
full residual:        r = d - sum_j(T_j)
input to block i:   r_i = d - sum_(j != i)(T_j)
accepted update:    T_i = copy(returned signal)
```

Earlier accepted updates in a cycle are immediately visible to later blocks.
Correct accounting alone does not establish a joint posterior sampler. The caller
preserves observed arrays; blocks and callbacks respect shared-object ownership.
See [evidence and scientific limits](../../evidence.md).

## Requirements

### Requirement: Define a stable block participation contract

A participating block SHALL expose a stable, non-empty string `name` and callable
`start(residual)` and `update(residual)` operations returning `Template` objects.
The contributor preserves its name while registered.

**Spec Kit mapping**: FR-010.

#### Scenario: Participate with independently owned state

- **GIVEN** a block with a valid name, both operations, and a private update counter
- **WHEN** it initializes and completes two updates
- **THEN** the same block instance retains its counter and returns its current signals through the declared operations

### Requirement: Validate registration before invoking a block

Registration SHALL reject empty, non-string, or duplicate names and missing or
non-callable operations before invoking initialization.

**Spec Kit mapping**: FR-011.

#### Scenario: Reject duplicate membership without side effects from initialization

- **GIVEN** a registered block named A and another block with the same name
- **WHEN** the second block is added
- **THEN** registration fails before calling its `start` operation

### Requirement: Initialize against existing contributions

Successful registration SHALL call the new block's initialization once with the
observation minus already registered signals and with the current noise model.

**Spec Kit mapping**: FR-012.

#### Scenario: Initialize two contributors sequentially

- **GIVEN** observation 10 in every sample and block A initialized with signal 2
- **WHEN** block B initializes
- **THEN** B receives 8 in every sample, while A's initial input was 10

### Requirement: Contain initialization and initial-validation failures

A block initialization exception or failure to validate its initial return SHALL
leave Wheel membership, existing ledger entries, and the noise slot unchanged.
This guarantee excludes block-internal side effects, prohibited shared-object
mutation, and exceptions during subsequent state adoption.

**Spec Kit mapping**: FR-013.

#### Scenario: Reject an invalid initial template

- **GIVEN** an existing campaign and a joining block that returns a template with a missing channel
- **WHEN** the initial return is validated
- **THEN** the joining block is not registered and existing accounting and noise remain unchanged

### Requirement: Validate each published signal at the campaign boundary

Every successful block return SHALL be a `Template` containing its own summed
signal, with all campaign channels, matching lengths and real/complex kind, and
finite values. Reusing a valid container or buffer is allowed; adoption copies it.

**Spec Kit mapping**: FR-014.

#### Scenario: Reject the observation object as a signal publication

- **GIVEN** an initialized block
- **WHEN** its update returns `L1Data` instead of `Template`
- **THEN** the return is rejected with the block operation and expected return contract identified

#### Scenario: Reject a non-finite contribution in any channel

- **GIVEN** a valid previously adopted signal
- **WHEN** the block publishes an infinity in channel E
- **THEN** return validation fails before that signal is adopted

### Requirement: Accept explicit zero contributions

A block SHALL be able to replace its prior contribution with an explicit zero
template, including zeros in selected channels, without a signal-withdrawal warning.

**Spec Kit mapping**: FR-015.

#### Scenario: Remove a previously modeled source

- **GIVEN** a block previously contributing 3 in every sample
- **WHEN** it returns a zero template
- **THEN** its adopted contribution becomes zero without a withdrawal warning

### Requirement: Limit template publications to signal and optional noise

A template SHALL carry signal arrays and an optional noise publication only;
campaign grid settings and orbit are supplied by the observation/residual contract.

**Spec Kit mapping**: FR-016.

#### Scenario: Publish noise alongside an existing signal buffer

- **GIVEN** a valid signal template
- **WHEN** `with_noise(model)` is used
- **THEN** the new container shares its signal arrays and carries the supplied noise reference without replacing the campaign's grid or orbit

### Requirement: Make contributor scientific responsibilities explicit

A block author SHALL reject conventions its model cannot support and own its
parameters, RNG, chains, adaptation, checkpoints, and external-resource lifecycle.
Protocol conformance does not establish a correct likelihood or conditional kernel.

**Spec Kit mapping**: FR-017.

#### Scenario: Refuse an unsupported physical observable

- **GIVEN** a block whose model supports fractional frequency only
- **WHEN** it initializes against an observation labeled phase
- **THEN** that block rejects the convention instead of silently interpreting phase samples as fractional frequency

### Requirement: Execute complete cycles in registration order

A full cycle SHALL call every registered block's update exactly once in
registration order, continuing its existing state across successive run calls.

**Spec Kit mapping**: FR-018.

#### Scenario: Count complete ordered passes

- **GIVEN** blocks A, B, and C registered in that order
- **WHEN** four cycles are requested
- **THEN** the update sequence is A, B, C repeated four times, with four updates per block

### Requirement: Supply the current conditional residual

Each block update SHALL receive `d - sum_(j != i)(T_j)` using the latest adopted
signals, leaving its own signal out of the subtraction.

**Spec Kit mapping**: FR-019.

#### Scenario: Update against the other block only

- **GIVEN** observation 10 and current contributions A=2 and B=3
- **WHEN** A is called to update
- **THEN** A receives 7, requiring no addition of its previous contribution

### Requirement: Adopt the published signal before subsequent updates

A valid publication SHALL directly replace the block's previous ledger signal
before the next block updates, without reconstructing it by subtracting residuals.

**Spec Kit mapping**: FR-020.

#### Scenario: Observe a changed signal within the same cycle

- **GIVEN** observation 10 and initialized contributions A=2 and B=3
- **WHEN** A publishes 4 and B subsequently publishes 1
- **THEN** B receives 6 and the full residual after the cycle is 5, exactly for these small float64 integer fixtures

### Requirement: Validate requested cycle counts

Run requests SHALL accept non-negative Python or NumPy integer counts, exclude
Booleans, and reject negative or non-integral values; zero requests perform no
updates or callbacks.

**Spec Kit mapping**: FR-021.

#### Scenario: Leave initialized state unchanged for zero cycles

- **GIVEN** a campaign whose blocks have already initialized
- **WHEN** zero cycles are requested with a callback
- **THEN** no block update or callback occurs and the initialized ledger remains available

#### Scenario: Reject a Boolean count

- **GIVEN** a valid campaign
- **WHEN** `True` is supplied as the cycle count
- **THEN** the run request is rejected rather than interpreted as one cycle

### Requirement: Expose full and conditional accounting

The Wheel SHALL expose full residuals, residuals excluding subtraction of a named
block, and named adopted contributions, rejecting unknown names.

**Spec Kit mapping**: FR-022.

#### Scenario: Inspect a two-block campaign

- **GIVEN** observation 10 and adopted contributions A=4 and B=1
- **WHEN** full residual, A's conditional residual, and A's contribution are inspected
- **THEN** they contain 5, 9, and 4 respectively; requesting unknown block C fails

### Requirement: Isolate adopted and inspected signal arrays

Ledger adoption and diagnostic access SHALL supply independent signal arrays so
later edits to published buffers, inspected contributions, or residuals cannot
change stored observation or ledger samples. Original observed arrays and attached
objects remain shared references under the caller's preservation obligation.

**Spec Kit mapping**: FR-023.

#### Scenario: Reuse a publication buffer safely

- **GIVEN** A has published buffer `[4,4]`, B contributes `[1,1]`, and the observation is `[10,10]`
- **WHEN** A's published buffer is later overwritten with `[99,99]`
- **THEN** A's stored contribution remains `[4,4]` and the full residual remains `[5,5]`

#### Scenario: Edit diagnostics without changing the campaign

- **GIVEN** a returned residual and an inspected contribution
- **WHEN** the caller overwrites their arrays
- **THEN** a fresh inspection yields the original campaign values

### Requirement: Preserve sufficient arithmetic precision

Residual arithmetic SHALL promote to accommodate included wider-precision signals
without narrowing the stored templates or altering the observation's stored dtype.

**Spec Kit mapping**: FR-024.

#### Scenario: Combine different floating-point precisions

- **GIVEN** float32 observations and a float64 adopted signal
- **WHEN** a residual including that signal is computed
- **THEN** the residual uses float64 while the observation stays float32 and the signal stays float64

### Requirement: Notify only completed cycles

An optional callback SHALL run after each complete cycle with `(cycle, wheel)` and
a zero-based index local to that run invocation, including campaigns without blocks.

**Spec Kit mapping**: FR-025.

#### Scenario: Inspect every completed pass

- **GIVEN** a callback recording the campaign state
- **WHEN** four cycles finish successfully
- **THEN** it receives indices 0, 1, 2, and 3 after the respective updates have been adopted

### Requirement: Stop at a block exception or invalid return

A block exception or return-validation failure SHALL propagate, preserve that
block's prior adopted signal, and prevent later block updates and the incomplete
cycle's callback. Exceptions during adoption have the narrower limits stated below.

**Spec Kit mapping**: FR-026.

#### Scenario: Reject an invalid update in the middle of a cycle

- **GIVEN** A has successfully updated and B returns a template of the wrong length before C's turn
- **WHEN** B's return is validated
- **THEN** B's old contribution remains, C is not called, the incomplete-cycle callback is skipped, and the error propagates

### Requirement: Expose partial progress after failure

A failed run SHALL retain earlier successful updates and leave block-owned state
under contributor control; it provides no automatic whole-cycle rollback, retry,
checkpoint recovery, or cleanup. Escalating a noise-ownership warning to an exception
can interrupt adoption after recording a signal; adoption is not transactional.

**Spec Kit mapping**: FR-027.

#### Scenario: Recover a campaign with partial progress explicitly

- **GIVEN** A adopts a new signal and B then raises during its update
- **WHEN** the caller catches the exception
- **THEN** A's new signal remains adopted, B's internal side effects are not undone, and any recovery is the caller's responsibility
