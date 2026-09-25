"""Galactic-binary waveform, noise, and sampler code for the enchilada example.

These model components are supplied by the application and used by the companion
notebook through enchilada's ``L1Data`` and ``Wheel`` interfaces.

Requires the LISA stack in ``requirements-gb.txt``: ``gbgpu``, ``eryn``,
``lisaanalysistools``. The companion notebook additionally plots with
``matplotlib`` and ``corner``.
"""

from dataclasses import dataclass, field

import numpy as np

from enchilada import BlockState, DataCovariance, L1Data, TranslatedCovariance


# ---------------------------------------------------------------- noise model
class FixedLISANoise:
    """One-sided LISA PSD from lisatools ``get_sensitivity`` (A == E).

    The only contract a enchilada noise object needs: ``psd(freqs[, channel])``.
    """

    def psd(self, f, channel=None):
        from lisatools.sensitivity import A1TDISens, get_sensitivity

        return get_sensitivity(np.asarray(f), sens_fn=A1TDISens)


# ---------------------------------------------------------------- waveform helpers
def _new_gb():
    from gbgpu.gbgpu import GBGPU

    return GBGPU()


def gb_template(gb, params, angles, Tobs, dt, NB):
    """GB (amp, f0, fdot, phi0) + fixed angles -> (start_ind, hA, hE)."""
    amp, f0, fdot, phi0 = params
    iota, psi, lam, beta = angles
    gb.run_wave(amp, f0, fdot, 0.0, phi0, iota, psi, lam, beta, N=NB, T=Tobs, dt=dt)
    return int(gb.start_inds[0]), np.asarray(gb.A[0]), np.asarray(gb.E[0])


def waveform(params, angles, Tobs, dt, NB=128):
    """Convenience one-off template (own GBGPU instance) -> (start_ind, hA, hE).

    For hot loops (a likelihood) pass a reused ``gb`` to :func:`gb_template`
    instead; this is for diagnostics/plots.
    """
    return gb_template(_new_gb(), params, angles, Tobs, dt, NB)


def scatter(start_ind, cols, n_rfft, chans):
    """Narrowband template columns -> full one-sided rfft-grid dict."""
    out = {ch: np.zeros(n_rfft, dtype=complex) for ch in chans}
    for ch in chans:
        out[ch][start_ind : start_ind + len(cols[ch])] = cols[ch]
    return out


def inner(d, h, S, band, df):
    """Noise-weighted squared residual norm ``4 df sum |d-h|^2 / S`` over a band."""
    r = d[band] - h
    return 4.0 * df * np.sum((np.abs(r) ** 2) / S[band]).real


# ---------------------------------------------------------------- data generation
def inject_gb(truth, angles, Tobs, dt, n_samples, channels, noise, NB=128, seed=42):
    """Synthesize one GB in stationary noise on the one-sided rfft grid.

    Note this dataset never exists as a time series -- GBGPU emits narrowband
    frequency-domain templates directly -- which is why the caller must state
    ``num_time_samples`` when wrapping it in an ``L1Data``. Data loaded as a time
    series instead gets it derived, and ``L1Data.to_frequency()`` carries it.

    Returns ``(tdi, info)``: ``tdi`` is the channel->array dict to wrap in a
    ``L1Data``; ``info`` carries ``band``, the noiseless ``signal``, the
    optimal ``snr``, and the template's ``start_ind`` for diagnostics.
    """
    df = 1.0 / Tobs
    freqs = np.fft.rfftfreq(n_samples, dt)
    n_rfft = freqs.size

    gb = _new_gb()
    si, hA, hE = gb_template(gb, truth, angles, Tobs, dt, NB)
    signal = scatter(si, {"A": hA, "E": hE}, n_rfft, channels)

    S = np.empty(n_rfft)
    S[0] = np.inf
    S[1:] = noise.psd(freqs[1:])

    rng = np.random.default_rng(seed)
    tdi = {}
    for ch in channels:
        draw = np.sqrt(S / (4.0 * df)) * (
            rng.standard_normal(n_rfft) + 1j * rng.standard_normal(n_rfft)
        )
        draw[0] = 0.0
        if n_samples % 2 == 0:
            # The real Nyquist coefficient carries the full S/(2 df) variance.
            draw[-1] = np.sqrt(2.0) * draw[-1].real
        tdi[ch] = signal[ch] + draw

    band = slice(si, si + NB)
    snr = np.sqrt(sum(inner(np.zeros(n_rfft), h, S, band, df) for h in (hA, hE)))
    return tdi, {"band": band, "signal": signal, "snr": snr, "start_ind": si}


# ---------------------------------------------------------------- the block
class GBBlock:
    """One galactic binary with a uniform-box prior and Eryn stretch sampling.

    The block owns its sampler continuation in ``self.state``. The caller-side
    ``draw_gb_prior`` helper initializes all walkers before registration. Each call
    rebuilds GBGPU and Eryn, restores numerical walker/RNG state, and evaluates
    the conditional residual.
    ``BlockState.parameters`` and its signal describe the first current walker,
    a posterior draw after mixing. ``state.samples`` holds only this cycle's
    samples; collect a chain with the Wheel callback. ``state.updates``
    counts calls to ``sample``, each of which runs ``steps_per_cycle`` steps.

    This CPU example uses one temperature and fixed ensemble splits. Adaptive
    moves or tempering would require carrying their additional evolving state.

    Register with ``wheel.add(block, initial_block_state=initial,
    data_domain="frequency")``. This narrowband likelihood requires native
    frequency covariance with independent channels;
    generic translated covariance operators need a different likelihood. Wheel
    can restore a native frequency covariance that was stored in time or WDM.
    """

    SUPPORTED_CHANNELS = ("A", "E")
    PARAMETER_NAMES = ("amp", "f0", "fdot", "phi0")

    @dataclass
    class State:
        coords: dict[str, np.ndarray] = field(default_factory=dict)
        inds: dict[str, np.ndarray] = field(default_factory=dict)
        betas: np.ndarray | None = None
        random_state: tuple | None = None
        samples: np.ndarray = field(default_factory=lambda: np.empty((0, 4)))
        updates: int = 0
        sampler: str = "Eryn"
        representative: str = "first walker"

    def __init__(
        self,
        bounds,
        angles,
        name="gb",
        n_walkers=24,
        steps_per_cycle=40,
        band=128,
    ):
        b = np.asarray(bounds, float)
        if b.shape != (4, 2) or not np.isfinite(b).all() or np.any(b[:, 0] >= b[:, 1]):
            raise ValueError("bounds must contain four finite increasing intervals")
        if len(angles) != 4 or not np.isfinite(angles).all():
            raise ValueError("angles must contain four finite fixed angles")
        if n_walkers < 8 or steps_per_cycle < 1 or band < 1:
            raise ValueError(
                "need at least 8 walkers, 1 step per cycle, and 1 band bin"
            )
        self.name = name
        self.bounds = tuple(tuple(row) for row in b)
        self.angles = tuple(angles)
        self.nw, self.k, self.NB = n_walkers, steps_per_cycle, band
        self.state = self.State()

    def _context(
        self,
        conditional_residual: L1Data,
        noise_covariance: DataCovariance | TranslatedCovariance | None,
    ):
        if conditional_residual.data_domain != "frequency":
            raise ValueError(f"{self.name}: GBBlock requires frequency-domain data")
        if conditional_residual.physical_observable != "fractional_frequency":
            raise ValueError(f"{self.name}: GBGPU emits fractional-frequency TDI")
        unsupported = set(conditional_residual.channel_names) - set(
            self.SUPPORTED_CHANNELS
        )
        if unsupported:
            raise ValueError(f"{self.name}: unsupported channels {sorted(unsupported)}")
        if noise_covariance is None:
            raise ValueError(f"{self.name}: GBBlock requires a noise covariance")
        if (
            not isinstance(noise_covariance, DataCovariance)
            or noise_covariance.data_domain != "frequency"
        ):
            raise ValueError(
                f"{self.name}: this narrowband likelihood requires a native "
                "frequency-domain covariance"
            )
        noise_covariance.check_compatible(conditional_residual)
        if np.any(
            noise_covariance.covariance_matrix[
                :, ~np.eye(len(conditional_residual.channel_names), dtype=bool)
            ]
        ):
            raise ValueError(
                f"{self.name}: this likelihood requires uncorrelated channels"
            )
        return _new_gb(), {
            ch: noise_covariance.noise_psd(channel_name=ch)
            for ch in conditional_residual.channel_names
        }

    def _waveform(self, gb, params, conditional_residual):
        si, hA, hE = gb_template(
            gb,
            params,
            self.angles,
            conditional_residual.Tobs,
            conditional_residual.dt,
            self.NB,
        )
        # The narrowband likelihood below uses the interior-bin weight. Reject
        # a signal overlapping DC or an even-length time series' Nyquist bin.
        end = conditional_residual.num_time_samples // 2 + (
            conditional_residual.num_time_samples % 2
        )
        if si < 1 or si + self.NB > end:
            raise ValueError(
                f"{self.name}: GB band must lie strictly inside the rfft grid"
            )
        return si, {"A": hA, "E": hE}

    def _render(self, gb, params, conditional_residual):
        si, cols = self._waveform(gb, params, conditional_residual)
        return conditional_residual.block_state(
            scatter(
                si,
                cols,
                conditional_residual.num_time_samples // 2 + 1,
                conditional_residual.channel_names,
            ),
            parameters=dict(zip(self.PARAMETER_NAMES, map(float, params), strict=True)),
        )

    def sample(
        self,
        conditional_residual: L1Data,
        noise_covariance: DataCovariance | TranslatedCovariance | None,
        current_block_state: BlockState,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        from eryn.ensemble import EnsembleSampler
        from eryn.moves import StretchMove
        from eryn.prior import ProbDistContainer, UniformDistribution
        from eryn.state import State

        gb, psds = self._context(conditional_residual, noise_covariance)

        def loglike(x):
            si, cols = self._waveform(gb, np.asarray(x).ravel(), conditional_residual)
            band = slice(si, si + self.NB)
            # Drop the parameter-independent full-data norm. Subtracting only
            # a moving band's data norm would change the target as f0 moves.
            return sum(
                4.0
                * conditional_residual.df
                * np.sum(
                    (
                        np.real(
                            conditional_residual.channel_data[ch][band].conj()
                            * cols[ch]
                        )
                        - 0.5 * np.abs(cols[ch]) ** 2
                    )
                    / psds[ch][band]
                )
                for ch in conditional_residual.channel_names
            )

        sampler = EnsembleSampler(
            self.nw,
            4,
            loglike,
            priors={
                "model_0": ProbDistContainer(
                    {
                        i: UniformDistribution(lo, hi)
                        for i, (lo, hi) in enumerate(self.bounds)
                    }
                )
            },
            # Eryn randomizes splits with NumPy's global RNG by default. Fixed
            # splits keep draws in the saved Eryn RNG state while preserving
            # the affine-invariant stretch proposal.
            moves=StretchMove(randomize_split=False, use_gpu=False),
        )
        current_state = self.state
        if current_state.random_state is None or not current_state.coords:
            raise ValueError(
                "Initialize GBBlock.state with draw_gb_prior or a checkpoint"
            )
        sampler.random_state = current_state.random_state
        # Deliberately omit cached log_like/log_prior: another block may have
        # changed the residual or covariance since these walkers were last used.
        initial = State(
            current_state.coords,
            inds=current_state.inds,
            betas=current_state.betas,
            random_state=current_state.random_state,
            copy=True,
        )
        final = sampler.run_mcmc(initial, self.k, progress=False)
        result = self._render(
            gb, final.branches_coords["model_0"][0, 0, 0], conditional_residual
        )
        self.state = self.State(
            coords=final.branches_coords,
            inds=final.branches_inds,
            betas=final.betas,
            random_state=sampler.random_state,
            samples=sampler.get_chain()["model_0"].reshape(-1, 4),
            updates=current_state.updates + 1,
        )
        return result


def draw_gb_prior(
    block: GBBlock,
    reference_data: L1Data,
    noise_covariance: DataCovariance | TranslatedCovariance | None,
    *,
    rng: np.random.Generator,
) -> BlockState:
    """Initialize block-owned walkers and return the initial physical publication.

    Call this example helper with frequency-domain data and native frequency
    covariance, using an initialization RNG separate from Wheel's sampling RNG.
    This resets ``block.state``. Wheel never calls the helper or draws initial
    parameters itself. A fresh block instance belongs to each independent campaign.
    """
    gb, _ = block._context(reference_data, noise_covariance)
    bounds = np.asarray(block.bounds)
    coords = rng.uniform(bounds[:, 0], bounds[:, 1], size=(1, block.nw, 1, 4))
    # Eryn consumes RandomState tuples, not Generator.bit_generator.state.
    random_state = np.random.RandomState(int(rng.integers(2**32))).get_state()
    result = block._render(gb, coords[0, 0, 0], reference_data)
    block.state = block.State(
        coords={"model_0": coords},
        inds={"model_0": np.ones((1, block.nw, 1), dtype=bool)},
        random_state=random_state,
    )
    return result
