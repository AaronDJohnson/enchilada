"""Orchestrator-owned data, covariance, sampler state, and signal accounting."""

import warnings
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace

import numpy as np

from enchilada._signal_sum import Signal, SignalSum
from enchilada.block import Block
from enchilada.block_result import BlockResult, check_on_grid
from enchilada.covariance import DataCovariance
from enchilada.data import L1Data
from enchilada.domains import WDMGrid
from enchilada.ledger import Ledger
from enchilada.translated_covariance import TranslatedCovariance, copy_covariance
from enchilada.translation import _target_grid, transform

DomainSelection = tuple[str, WDMGrid | None] | None


class NoiseOverwrittenWarning(RuntimeWarning):
    """A second block replaces the current noise owner's covariance.

    Wheel keeps one covariance, not a sum of separate noise publications.
    Publish a combined covariance from a single noise block when modelling
    several components.
    """


class Wheel:
    """Run stateless blocks with an explicit, complete state owned by the Wheel.

    The pristine observation is copied at construction. A second L1Data stores
    the full working residual, rebuilt after every accepted update. For block i,
    the input is d - sum(j != i, T_j); the block's own signal remains in its input.
    An absent TDI signal contributes zero, including when a block previously
    returned a signal. Ledger holds complete BlockResults. Public data, covariance,
    and ledger access returns copies. Orbit resources must be treated as immutable.

    Registration requires and adopts an explicit initial_block_result. It never
    calls model initialization or consumes random draws.
    Each cycle calls sample with those inputs and the current_block_result from
    the ledger. Sampling receives the supplied RNG and independent input snapshots.
    Failed calls/validation/copying leave that block's state and the RNG unchanged.
    Earlier successful updates in a failed cycle remain committed. Exceptions in
    on_cycle_complete occur after that entire cycle has committed.

    `initial_noise_covariance` supplies the starting DataCovariance. If omitted,
    covariance stays unset until a block supplies it. Blocks receive covariance
    separately from L1Data. The original data domain stays canonical. Register
    a block with data_domain to translate its inputs and returned estimates.
    The block that owns the current covariance must return it on every call;
    other blocks can omit covariance without changing it.
    """

    def __init__(
        self,
        observed_data: L1Data,
        initial_noise_covariance: DataCovariance | TranslatedCovariance | None = None,
        *,
        random_seed: int | None = None,
    ) -> None:
        """Initialize a campaign from observations and an optional covariance.

        Args:
            observed_data: L1 observations and shared run settings. The Wheel
                makes independent pristine and working copies.
            initial_noise_covariance: Starting covariance. None leaves covariance
                unset until a block supplies it.
            random_seed: Seed for the campaign's NumPy generator. None requests
                nondeterministic initialization.
        """
        if not isinstance(observed_data, L1Data):
            raise TypeError("observed_data must be L1Data")
        for ch in observed_data.channel_names:
            if not np.isfinite(observed_data.channel_data[ch]).all():
                raise ValueError(
                    f"observed_data.channel_data[{ch!r}] has non-finite samples; "
                    "enchilada has no data-quality mask for gaps; "
                    "fill or trim them first"
                )
        if initial_noise_covariance is not None and not isinstance(
            initial_noise_covariance, (DataCovariance, TranslatedCovariance)
        ):
            raise TypeError(
                "initial_noise_covariance must be a DataCovariance, "
                "TranslatedCovariance or None"
            )
        if initial_noise_covariance is not None:
            initial_noise_covariance = copy_covariance(initial_noise_covariance)
            initial_noise_covariance.check_compatible(observed_data)
        self._observed_data = self._copy_data(observed_data)
        self._working_residual = self._copy_data(observed_data)
        self._noise_covariance = initial_noise_covariance
        self._noise_owner_block_name: str | None = None
        self._ledger = Ledger()
        self._blocks: list[tuple[str, Block]] = []
        self._block_indices: dict[str, int] = {}
        self._block_domains: dict[str, DomainSelection] = {}
        self._signal_sum = SignalSum()
        self._rng = np.random.default_rng(random_seed)

    @staticmethod
    def _copy_data(data_to_copy: L1Data) -> L1Data:
        """Copy channel arrays, preserving immutable orbit data."""
        return replace(
            data_to_copy,
            channel_data={
                ch: arr.copy() for ch, arr in data_to_copy.channel_data.items()
            },
        )

    @property
    def observed_data(self) -> L1Data:
        """A defensive copy of the pristine observation."""
        return self._copy_data(self._observed_data)

    @property
    def working_residual(self) -> L1Data:
        """A defensive copy of the current full residual."""
        return self._copy_data(self._working_residual)

    @property
    def noise_covariance(self) -> DataCovariance | TranslatedCovariance | None:
        """The current covariance, as an independent copy."""
        return (
            None
            if self._noise_covariance is None
            else copy_covariance(self._noise_covariance, validated=True)
        )

    @property
    def ledger(self) -> Ledger:
        """Store of complete results; use ledger.snapshot() for a copied dictionary."""
        return self._ledger

    def add(
        self,
        block_to_register: Block,
        *,
        initial_block_result: BlockResult,
        data_domain: str | None = None,
        num_frequency_divisions: int | None = None,
        num_time_divisions: int | None = None,
    ) -> None:
        """Register a block from a complete, explicitly prepared starting result.

        block_to_register must have a unique name and implement sample.
        Failed registration leaves the campaign unchanged.

        initial_block_result supplies a complete starting estimate, parameters,
        and sampler continuation state, for example at an injected signal.
        Wheel copies and validates it without calling an initialization method or
        consuming random draws. The result is required and cannot be None.
        Its signal uses the selected block domain/grid, or the observation grid when no
        data_domain is selected. Covariance carries its own source grid metadata.
        The caller must keep estimates, parameters, and sampler state consistent.
        This sets the starting state only; sample still runs normally.

        data_domain selects time, frequency or wdm for this block's residual,
        covariance and current result. None preserves their current domains.
        For WDM, specify either division count; the other follows from the
        observation length. Both counts must be even. An existing observation
        WDM grid is reused if neither count is supplied.
        """
        if not isinstance(initial_block_result, BlockResult):
            raise TypeError("initial_block_result must be a BlockResult")
        domain_selection: DomainSelection = None
        if data_domain is None:
            if num_frequency_divisions is not None or num_time_divisions is not None:
                raise ValueError("WDM division counts require data_domain='wdm'")
        else:
            domain_selection = (
                data_domain,
                _target_grid(
                    data_domain,
                    self._observed_data.num_time_samples,
                    self._observed_data.wdm_grid,
                    num_frequency_divisions,
                    num_time_divisions,
                ),
            )
        block_name = getattr(block_to_register, "name", None)
        if not isinstance(block_name, str) or not block_name:
            raise ValueError(
                f"block name must be a non-empty string, got {block_name!r}"
            )
        if block_name in self._ledger:
            raise ValueError(f"block name {block_name!r} already registered")
        if not callable(getattr(block_to_register, "sample", None)):
            raise TypeError(
                f"block {block_name!r} does not implement sample; "
                "a Block needs name and sample(conditional_residual, "
                "noise_covariance, current_block_result, *, rng)."
            )
        new_block_result: object = initial_block_result
        source_reference = self._observed_data
        if (
            domain_selection is not None
            and initial_block_result.tdi_signal_contribution is not None
        ):
            domain, grid = domain_selection
            # Only the seed's grid is needed. Transforming observation
            # values here wastes work and can overflow independently of it.
            source_reference = replace(
                self._observed_data,
                channel_data=initial_block_result.tdi_signal_contribution,
                data_domain=domain,
                wdm_grid=grid,
            )
        if domain_selection is not None:
            new_block_result = self._canonical_result(
                new_block_result, source_reference
            )
        self._adopt(
            block_name,
            new_block_result,
            "initial_block_result",
            self._rng,
            block_to_register=block_to_register,
            domain_selection=domain_selection,
        )

    def _block_inputs(
        self,
        domain_selection: DomainSelection,
        exclude_block_name: str | None = None,
    ) -> tuple[L1Data, DataCovariance | TranslatedCovariance | None]:
        conditional_residual = self.residual(exclude_block_name=exclude_block_name)
        noise_covariance = self.noise_covariance
        if domain_selection is not None:
            domain, grid = domain_selection
            conditional_residual = transform(
                conditional_residual,
                domain,
                num_frequency_divisions=None
                if grid is None
                else grid.num_frequency_divisions,
            )
            if noise_covariance is not None:
                noise_covariance = transform(
                    noise_covariance,
                    domain,
                    num_frequency_divisions=None
                    if grid is None
                    else grid.num_frequency_divisions,
                )
        return conditional_residual, noise_covariance

    def _canonical_result(self, result: object, source_reference: L1Data) -> object:
        if not isinstance(result, BlockResult):
            return result  # _adopt supplies the block/method-specific error.
        return transform(
            result,
            self._observed_data.data_domain,
            reference_data=source_reference,
            num_frequency_divisions=(
                None
                if self._observed_data.wdm_grid is None
                else self._observed_data.wdm_grid.num_frequency_divisions
            ),
        )

    def run(
        self,
        num_cycles: int,
        on_cycle_complete: Callable[[int, "Wheel"], None] | None = None,
    ) -> None:
        """Visit blocks in registration order; notify after each complete cycle.

        Cycle indices are local to this invocation. Multiple run calls continue
        the accepted ledger and RNG; run(0) does nothing.

        Args:
            num_cycles: Non-negative number of complete Gibbs cycles to attempt.
            on_cycle_complete: Optional callback receiving (cycle_index, wheel)
                after every block in a cycle has returned an accepted result.
                Indices start at zero for each run call. Partial cycles do not
                invoke the callback.
        """
        if (
            not isinstance(num_cycles, (int, np.integer))
            or isinstance(num_cycles, bool)
            or num_cycles < 0
        ):
            raise ValueError(
                f"num_cycles must be a non-negative integer, got {num_cycles!r}"
            )
        for cycle_index in range(num_cycles):
            for block_name, block in self._blocks:
                candidate_rng = deepcopy(self._rng)
                domain_selection = self._block_domains[block_name]
                conditional_residual, noise_covariance = self._block_inputs(
                    domain_selection, block_name
                )
                current_block_result = self._ledger[block_name]
                if domain_selection is not None:
                    domain, grid = domain_selection
                    current_block_result = transform(
                        current_block_result,
                        domain,
                        reference_data=self._observed_data,
                        num_frequency_divisions=None
                        if grid is None
                        else grid.num_frequency_divisions,
                    )
                new_block_result: object = block.sample(
                    conditional_residual=conditional_residual,
                    noise_covariance=noise_covariance,
                    current_block_result=current_block_result,
                    rng=candidate_rng,
                )
                if domain_selection is not None:
                    new_block_result = self._canonical_result(
                        new_block_result, conditional_residual
                    )
                self._adopt(block_name, new_block_result, "sample", candidate_rng)
            if on_cycle_complete is not None:
                on_cycle_complete(cycle_index, self)

    def _form_residual(self, signal_sum: Signal) -> L1Data:
        """Subtract an aggregate signal from pristine observations."""
        channel_data = {}
        for ch, observed_channel_values in self._observed_data.channel_data.items():
            if signal_sum is None:
                residual_channel_values = observed_channel_values.copy()
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    residual_channel_values = observed_channel_values - signal_sum[ch]
            if not np.isfinite(residual_channel_values).all():
                raise ValueError(f"residual in channel {ch!r} became non-finite")
            channel_data[ch] = residual_channel_values
        return replace(self._observed_data, channel_data=channel_data)

    def residual(self, exclude_block_name: str | None = None) -> L1Data:
        """Return a copied full residual or a block's conditional residual.

        exclude_block_name identifies a registered block whose signal is left
        in the observations. None subtracts every block's signal.
        """
        if exclude_block_name is None:
            return self.working_residual
        self._require_block_name(exclude_block_name)
        return self._form_residual(
            self._signal_sum.excluding(self._block_indices[exclude_block_name])
        )

    def contribution(self, block_name: str) -> dict[str, np.ndarray]:
        """Copy the registered block's summed TDI signal.

        block_name must identify a registered block. An absent contribution
        returns fresh zero arrays on the observation grid.
        """
        self._require_block_name(block_name)
        signal = self._ledger._results_by_block_name[block_name].tdi_signal_contribution
        if signal is None:
            return {
                ch: np.zeros_like(arr)
                for ch, arr in self._observed_data.channel_data.items()
            }
        return {ch: arr.copy() for ch, arr in signal.items()}

    def _require_block_name(self, block_name: str) -> None:
        if block_name not in self._ledger:
            raise ValueError(
                f"unknown block {block_name!r}; registered: {sorted(self._ledger)}"
            )

    def _adopt(
        self,
        block_name: str,
        new_block_result: object,
        source_method_name: str,
        candidate_rng: np.random.Generator,
        *,
        block_to_register: Block | None = None,
        domain_selection: DomainSelection = None,
    ) -> None:
        """Validate a candidate result and RNG before committing campaign state.

        source_method_name identifies the block call or initial result in diagnostics.
        block_to_register is supplied only when adopting a new block's initial result.
        """
        if not isinstance(new_block_result, BlockResult):
            raise TypeError(
                f"{block_name}.{source_method_name} must return a BlockResult, got "
                f"{type(new_block_result).__name__}"
            )
        # Snapshot BEFORE validation so subsequent use only touches owned state.
        candidate_block_result = deepcopy(new_block_result)
        candidate_block_result.__post_init__()
        signal = candidate_block_result.tdi_signal_contribution
        if signal is not None:
            check_on_grid(
                signal,
                channel_names=self._observed_data.channel_names,
                num_time_samples=self._observed_data.num_time_samples,
                data_domain=self._observed_data.data_domain,
                wdm_grid=self._observed_data.wdm_grid,
                context_label=(
                    f"{block_name}.{source_method_name} tdi_signal_contribution"
                ),
            )
            for ch, arr in signal.items():
                if not np.isfinite(arr).all():
                    raise ValueError(
                        f"{block_name}.{source_method_name} returned non-finite "
                        f"samples in channel {ch!r}"
                    )
        candidate_noise_covariance = self._noise_covariance
        candidate_noise_owner_block_name = self._noise_owner_block_name
        if (
            block_name == candidate_noise_owner_block_name
            and candidate_block_result.noise_covariance is None
        ):
            raise ValueError(
                f"{block_name}.{source_method_name} must return its complete "
                "noise_covariance "
                "because it owns the current covariance"
            )
        if candidate_block_result.noise_covariance is not None:
            # Revalidate: callers can edit arrays inside a frozen dataclass.
            candidate_noise_covariance = copy_covariance(
                candidate_block_result.noise_covariance
            )
            candidate_noise_covariance.check_compatible(self._observed_data)
            candidate_block_result = replace(
                candidate_block_result,
                noise_covariance=candidate_noise_covariance,
            )
            candidate_noise_owner_block_name = block_name
        candidate_block_results = {
            **self._ledger._results_by_block_name,
            block_name: candidate_block_result,
        }
        block_index = (
            len(self._blocks)
            if block_to_register is not None
            else self._block_indices[block_name]
        )
        candidate_signal_sum = self._signal_sum.with_signal(block_index, signal)
        candidate_residual = self._form_residual(candidate_signal_sum.total)
        candidate_block_indices = (
            {**self._block_indices, block_name: block_index}
            if block_to_register is not None
            else self._block_indices
        )
        next_blocks = (
            self._blocks
            if block_to_register is None
            else [*self._blocks, (block_name, block_to_register)]
        )
        candidate_block_domains = (
            self._block_domains
            if block_to_register is None
            else {**self._block_domains, block_name: domain_selection}
        )
        next_rng = deepcopy(candidate_rng)
        if (
            candidate_noise_owner_block_name != self._noise_owner_block_name
            and self._noise_owner_block_name is not None
        ):
            warnings.warn(
                f"{block_name}.{source_method_name} replaced the noise model that "
                f"{self._noise_owner_block_name!r} owns; "
                "Wheel does not combine noise models. "
                "Publish one combined model from a single noise block.",
                NoiseOverwrittenWarning,
                stacklevel=3,
            )
        # All potentially failing work, including warnings-as-errors, is done.
        self._ledger._results_by_block_name = candidate_block_results
        self._working_residual = candidate_residual
        self._signal_sum = candidate_signal_sum
        self._block_indices = candidate_block_indices
        self._block_domains = candidate_block_domains
        self._noise_covariance, self._noise_owner_block_name = (
            candidate_noise_covariance,
            candidate_noise_owner_block_name,
        )
        self._rng = next_rng
        self._blocks = next_blocks
