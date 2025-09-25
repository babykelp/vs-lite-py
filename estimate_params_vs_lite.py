from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Sequence, List, Optional

import numpy as np


# -----------------------------
# Configuration data-classes
# -----------------------------

@dataclass
class Priors:
    """Prior configuration for growth parameters.


    kind
    One of {"fourbet", "uniform"} to mirror MATLAB options.
    - "fourbet": 4-parameter Beta with (a, b, lo, hi) per parameter.
    - "uniform": use uniform bounds (lo, hi) per parameter.


    The defaults match Tolwinski-Ward et al. (2013).
    """

    kind: str = "fourbet" # MATLAB default

    # For "uniform": (lo, hi)
    T1: Optional[Tuple[float, float]] = (0.0, 8.5)
    T2: Optional[Tuple[float, float]] = (9.0, 20.0)
    M1: Optional[Tuple[float, float]] = (0.01, 0.03)
    M2: Optional[Tuple[float, float]] = (0.1, 0.5)

    # For "fourbet": rows correspond to T1, T2, M1, M2; columns (a, b, lo, hi)
    fourbetparams: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [9.0, 5.0, 0.0, 9.0],
                [3.5, 3.5, 10.0, 24.0],
                [1.5, 2.8, 0.0, 0.1],
                [1.5, 2.5, 0.1, 0.5],
            ],
            dtype=float,
        )
    )

@dataclass
class ErrorModel:
    """Error model configuration for RW ~ Model + error.

    kind
        'iid' (errormod=0), 'ar1' (1), or 'ar2' (2).
    estimate
        If False and errorpars are supplied, keep them fixed.
    errorpars
        For iid: (sigma2w,)
        For ar1: (phi1, tau2)
        For ar2: (phi1, phi2, tau2)
    """

    kind: str = "iid"
    estimate: bool = True
    errorpars: Optional[Tuple[float, ...]] = None


@dataclass
class Sampler:
    """MCMC/estimation settings (mirrors MATLAB names)."""

    nsamp: int = 1000  # total iterations
    nbi: int = 200  # burn-in
    thin: int = 1
    nchain: int = 3  # for Rhat diagnostic
    random_seed: Optional[int] = None
    pt_ests: str = "mle"  # or "med"
    convthresh: float = 0.1
    verbose: bool = True

@dataclass
class Runtime:
    """Non-statistical runtime/model switches."""

    hydroclim: str = "P"  # 'P' or 'M'
    substep: bool = False  # for leaky bucket when hydroclim == 'P'
    intwindow: Tuple[int, int] = (0, 12)  # [I0, If]

@dataclass
class CalibrationWindows:
    gparscalint: Optional[np.ndarray] = None  # indices (0-based)
    eparscalint: Optional[np.ndarray] = None  # indices (0-based)

@dataclass
class Settings:
    priors: Priors = field(default_factory=Priors)
    error_model: ErrorModel = field(default_factory=ErrorModel)
    sampler: Sampler = field(default_factory=Sampler)
    runtime: Runtime = field(default_factory=Runtime)
    windows: CalibrationWindows = field(default_factory=CalibrationWindows)

@dataclass
class ChainRecord:
    """Holds per-chain raw draws produced by the sampler."""
    # length = nsamp + nbi for each
    Tt: np.ndarray
    To: np.ndarray
    Mt: np.ndarray
    Mo: np.ndarray
    # 3D array: (12, nyears, nsamp+nbi)
    Gterms_dist: np.ndarray
    # error-model specific arrays (any of these may be None depending on errormod)
    sig2rw: Optional[np.ndarray] = None          # iid only
    phi1: Optional[np.ndarray] = None            # ar1/ar2
    phi2: Optional[np.ndarray] = None            # ar2 only
    tau2: Optional[np.ndarray] = None            # ar1/ar2
    # log-likelihood (length = nsamp + nbi)
    logL: Optional[np.ndarray] = None
# -----------------------------
# Main function (translated signature & I/O contract)
# -----------------------------

def estimate_vslite_params_v2_3(
    T: np.ndarray,
    P: np.ndarray,
    phi: float | np.ndarray,
    RW: np.ndarray,
    **kwargs: Any,
):
    """
    Estimate VSLite growth response parameters (T1, T2, M1, M2) from
    calibration-interval monthly temperature and precipitation, latitude,
    and annual ring-width indices.

    Parameters
    ----------
    T : np.ndarray
        Monthly temperature, shape (12, nyears).
    P : np.ndarray
        Monthly precipitation, shape (12, nyears).
    phi : float or np.ndarray
        Site latitude in degrees North. Either a scalar or an array of length nyears.
    RW : np.ndarray
        Standardized annual ring-width index, shape (nyears,).

    Other Parameters (kwargs)
    -------------------------
    MATLAB-style property/value equivalents are supported, e.g.::

        errormod=0|1|2, gparscalint=..., eparscalint=..., errorpars=...,\
        pt_ests='mle'|'med', hydroclim='P'|'M', substep=0|1, intwindow=(0,12),\
        nsamp=1000, nbi=200, nchain=3, gparpriors='fourbet'|'uniform',\
        T1priorsupp=(lo,hi), T2priorsupp=..., M1priorsupp=..., M2priorsupp=...,\
        fourbetparams=(4x4 array), convthresh=0.1, verbose=1|0

    You can also pass nested dicts: priors=..., error_model=..., sampler=...

    Returns
    -------
    T1, T2, M1, M2, extras : tuple[float, float, float, float, dict]
        Point estimates (medians or MLEs in the final implementation),
        plus an ``extras`` dict for additional outputs (posterior samples,
        diagnostics, settings, etc.).
    """

    # ---- Convert inputs to numpy arrays and validate shapes ----
    T = np.asarray(T, dtype=float)
    P = np.asarray(P, dtype=float)
    RW = np.asarray(RW, dtype=float)

    if T.ndim != 2 or P.ndim != 2:
        raise ValueError("T and P must be 2D arrays with shape (12, nyears).")
    if T.shape[0] != 12 or P.shape[0] != 12:
        raise ValueError("T and P must have 12 rows (months).")
    if T.shape != P.shape:
        raise ValueError("T and P must have the same shape.")

    nyears = T.shape[1]

    if RW.ndim != 1 or RW.shape[0] != nyears:
        raise ValueError(
            f"RW must be 1D with length nyears={nyears}, got {RW.shape}."
        )

    # Latitude can be scalar or length nyears
    phi_arr = np.asarray(phi, dtype=float)
    if phi_arr.size == 1:
        phi_arr = np.full(nyears, float(phi_arr))
    elif phi_arr.ndim == 1 and phi_arr.shape[0] == nyears:
        pass
    else:
        raise ValueError(
            "phi must be a scalar or a 1D array of length nyears."
        )

    # ---- Build settings from kwargs ----
    settings = Settings()
    settings = _apply_kwargs(settings, kwargs, nyears)

    # ---- Preprocess: z-score RW to match noise assumptions ----
    RWz = _zscore(RW)

    # ---- Soil moisture handling (hydroclim 'P' or 'M') ----
    # Constants per MATLAB code
    Mmax = 0.76
    Mmin = 0.01
    muth = 5.8
    mth = 4.886
    alpha = 0.093
    Minit = 200.0
    dr = 1000.0

    Nyrs = nyears

    if settings.runtime.hydroclim == "P":
        if settings.runtime.substep:
            M = leakybucket_submonthly(1, Nyrs, phi_arr, T, P, Mmax, Mmin, alpha, mth, muth, dr, Minit / dr)
        else:
            M = leakybucket_monthly(1, Nyrs, phi_arr, T, P, Mmax, Mmin, alpha, mth, muth, dr, Minit / dr)
    else:  # 'M' supplied in P slot
        M = P

    # ---- Insolation growth term gE (12 x 1) ----
    gE = Compute_gE(phi_arr)

    # ---- Placeholder for the full Bayesian estimation ----
    # TODO in later steps:
    #   * Construct priors for T1, T2, M1, M2 according to settings.priors
    #   * Define the forward VSLite growth model and hydroclimate handling
    #   * Specify error model (iid / AR(1) / AR(2)) per settings.error_model
    #   * Implement MCMC and compute point estimates per settings.sampler.pt_ests

    T1 = np.nan
    T2 = np.nan
    M1 = np.nan
    M2 = np.nan

    # --- Placeholder: list of ChainRecord produced by the sampler ---
    # When you port the MCMC section, push one ChainRecord per chain into this list.
    chains: List[ChainRecord] = []

    # Postprocess (MATLAB POSTPROCESS SAMPLES analogue)
    post = _postprocess_chains(settings, chains, nyears)

    # --- Advanced optional outputs placeholder structure (MATLAB varargout analogue) ---
    # Always present keys:
    extras: Dict[str, Any] = {
        "settings": settings,
        "posterior": None,      # will hold raw draws when implemented
        "diagnostics": {},
        # postprocess outputs (None if sampler not yet implemented)
        "chains": post.get("chains"),
        "ensembles": post.get("ensembles"),
        # growth parameter posteriors (Nsamp arrays) when implemented
        "T1dist": None,
        "T2dist": None,
        "M1dist": None,
        "M2dist": None,
        # Gelman-Rubin stats and convergence flag (populated after MCMC)
        "Rhats": None,
        "convwarning": None,
    }

    # Error-model-specific placeholders (match MATLAB ordered varargout semantics)
    if settings.error_model.kind == "iid":
        extras.update({
            "sig2rw": None,         # scalar point estimate
            "sigma2rwdist": None,   # Nsamp vector
            "Gdist": None,          # shape (12, nyears, Nsamp)
        })
    elif settings.error_model.kind == "ar1":
        extras.update({
            "phi1": None,       # scalar point estimate
            "phi1dist": None,   # Nsamp vector
            "tau2": None,       # scalar point estimate
            "tau2dist": None,   # Nsamp vector
            "Gdist": None,      # shape (12, nyears, Nsamp)
        })
    elif settings.error_model.kind == "ar2":
        extras.update({
            "phi1": None,
            "phi2": None,
            "phi1dist": None,
            "phi2dist": None,
            "tau2": None,
            "tau2dist": None,
            "Gdist": None,
        })


    return T1, T2, M1, M2, extras


# -----------------------------
# Helpers
# -----------------------------

# Allowed top-level kwargs (MATLAB-style + nested dicts)
_DEF_ALLOWED_TOP_KWARGS = {
    # nested
    "priors",
    "error_model",
    "sampler",
    # MATLAB-style
    "errormod",
    "gparscalint",
    "eparscalint",
    "errorpars",
    "pt_ests",
    "hydroclim",
    "substep",
    "intwindow",
    "nsamp",
    "nbi",
    "nchain",
    "gparpriors",
    "T1priorsupp",
    "T2priorsupp",
    "M1priorsupp",
    "M2priorsupp",
    "fourbetparams",
    "convthresh",
    "verbose",
}


def _apply_kwargs(settings: Settings, kwargs: Dict[str, Any], nyears: int) -> Settings:
    """Update Settings dataclasses from kwargs (MATLAB varargin analogue).

    Supports both nested dicts (priors=..., error_model=..., sampler=...)
    and MATLAB-style property/value pairs listed above.
    """
    # First pass: guard unknown keys
    for k in kwargs:
        if k not in _DEF_ALLOWED_TOP_KWARGS:
            raise TypeError(
                f"Unknown keyword argument '{k}'. Allowed: {_DEF_ALLOWED_TOP_KWARGS}."
            )

    # Nested dict updates
    if "priors" in kwargs:
        _update_dataclass(settings.priors, kwargs["priors"])  # type: ignore[arg-type]
    if "error_model" in kwargs:
        _update_dataclass(settings.error_model, kwargs["error_model"])  # type: ignore[arg-type]
    if "sampler" in kwargs:
        _update_dataclass(settings.sampler, kwargs["sampler"])  # type: ignore[arg-type]

    # MATLAB-style mappings
    if "errormod" in kwargs:
        emod = int(kwargs["errormod"])  # 0,1,2
        if emod == 0:
            settings.error_model.kind = "iid"
        elif emod == 1:
            settings.error_model.kind = "ar1"
        elif emod == 2:
            settings.error_model.kind = "ar2"
        else:
            raise ValueError("errormod must be 0, 1, or 2")

    if "errorpars" in kwargs and kwargs["errorpars"] is not None:
        pars = tuple(np.asarray(kwargs["errorpars"], dtype=float).tolist())
        settings.error_model.errorpars = pars
        settings.error_model.estimate = False

    if "gparscalint" in kwargs and kwargs["gparscalint"] is not None:
        settings.windows.gparscalint = _norm_indices(kwargs["gparscalint"], nyears)

    if "eparscalint" in kwargs and kwargs["eparscalint"] is not None:
        eidx = _norm_indices(kwargs["eparscalint"], nyears)
        # MATLAB note: Must be contiguous if using AR(1). Apply also to AR(2).
        if settings.error_model.kind in {"ar1", "ar2"} and not _is_contiguous(eidx):
            raise ValueError("eparscalint must be contiguous for AR(1)/AR(2) error model.")
        settings.windows.eparscalint = eidx

    if "pt_ests" in kwargs:
        val = str(kwargs["pt_ests"]).lower()
        if val not in {"mle", "med"}:
            raise ValueError("pt_ests must be 'mle' or 'med'")
        settings.sampler.pt_ests = val

    if "hydroclim" in kwargs:
        hc = str(kwargs["hydroclim"]).upper()
        if hc not in {"P", "M"}:
            raise ValueError("hydroclim must be 'P' or 'M'")
        settings.runtime.hydroclim = hc

    if "substep" in kwargs:
        settings.runtime.substep = bool(kwargs["substep"])  # 0/1 accepted

    if "intwindow" in kwargs:
        iw = _to_pair(kwargs["intwindow"])  # (I0, If)
        settings.runtime.intwindow = (int(iw[0]), int(iw[1]))

    if "nsamp" in kwargs:
        ns = int(kwargs["nsamp"])
        if not (200 <= ns <= 10000):
            raise ValueError("nsamp must be in [200, 10000]")
        settings.sampler.nsamp = ns

    if "nbi" in kwargs:
        settings.sampler.nbi = int(kwargs["nbi"])  # MATLAB default 200

    if "nchain" in kwargs:
        settings.sampler.nchain = int(kwargs["nchain"])  # default 3

    if "gparpriors" in kwargs:
        kind = str(kwargs["gparpriors"]).lower()
        if kind == "fourbet":
            settings.priors.kind = "fourbet"
        elif kind == "uniform":
            settings.priors.kind = "uniform"
        else:
            raise ValueError("gparpriors must be 'fourbet' or 'uniform'")

    # Uniform prior supports
    for key, attr in [
        ("T1priorsupp", "T1"),
        ("T2priorsupp", "T2"),
        ("M1priorsupp", "M1"),
        ("M2priorsupp", "M2"),
    ]:
        if key in kwargs and kwargs[key] is not None:
            lo, hi = _to_pair(kwargs[key])
            if not (float(lo) < float(hi)):
                raise ValueError(f"{key} lower bound must be < upper bound")
            setattr(settings.priors, attr, (float(lo), float(hi)))

    if "fourbetparams" in kwargs and kwargs["fourbetparams"] is not None:
        fb = np.asarray(kwargs["fourbetparams"], dtype=float)
        if fb.shape != (4, 4):
            raise ValueError("fourbetparams must be a 4x4 array")
        settings.priors.fourbetparams = fb

    if "convthresh" in kwargs:
        cv = float(kwargs["convthresh"])
        if cv <= 0:
            raise ValueError("convthresh must be > 0")
        settings.sampler.convthresh = cv

    if "verbose" in kwargs:
        settings.sampler.verbose = bool(kwargs["verbose"])  # 0/1 accepted

    # Final sanity checks
    _validate_settings(settings)

    return settings


def _update_dataclass(obj: Any, updates: Dict[str, Any]) -> None:
    if not isinstance(updates, dict):
        raise TypeError(
            f"Expected a dict of updates for {obj.__class__.__name__}, got {type(updates)}."
        )
    for k, v in updates.items():
        if not hasattr(obj, k):
            raise AttributeError(
                f"{obj.__class__.__name__} has no field '{k}'."
            )
        setattr(obj, k, v)


def _to_pair(x: Any) -> Tuple[float, float]:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size != 2:
        raise ValueError("Expected a pair of numbers")
    return float(arr[0]), float(arr[1])


def _norm_indices(x: Any, n: int) -> np.ndarray:
    """Normalize year indices to a sorted, unique 0-based integer array.

    Accepts 0-based or 1-based indices; if any index == n, we assume 1-based
    and subtract 1. Raises on out-of-range values.
    """
    idx = np.asarray(x, dtype=int).ravel()
    if idx.size == 0:
        raise ValueError("Index vector must be non-empty")

    # Heuristic: treat as 1-based if any index equals n or min(idx) == 1
    if (idx == n).any() or idx.min() == 1:
        idx = idx - 1

    if idx.min() < 0 or idx.max() >= n:
        raise IndexError(f"Index out of bounds for nyears={n}")

    return np.unique(np.sort(idx))


def _is_contiguous(idx: np.ndarray) -> bool:
    return np.all(np.diff(idx) == 1)


def _validate_settings(s: Settings) -> None:
    # intwindow sanity
    i0, i1 = s.runtime.intwindow
    if not (0 <= i0 <= 12 and 0 <= i1 <= 12 and i0 <= i1):
        raise ValueError("intwindow must be within [0,12] and i0 <= i1")

    # priors sanity
    if s.priors.kind == "uniform":
        for name in ("T1", "T2", "M1", "M2"):
            lo, hi = getattr(s.priors, name)  # type: ignore[misc]
            if not (lo < hi):
                raise ValueError(f"Uniform prior for {name} must have lo < hi")
    elif s.priors.kind == "fourbet":
        fb = s.priors.fourbetparams
        if fb.shape != (4, 4):
            raise ValueError("fourbetparams must be 4x4")
        # basic checks: bounds monotonic and shapes > 0
        for row in range(4):
            a, b, lo, hi = fb[row]
            if not (a > 0 and b > 0 and lo < hi):
                raise ValueError("Invalid fourbetparams row: shape>0 and lo<hi required")

# -----------------------------
# Data prep & model stubs (to be filled)
# -----------------------------

def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return np.zeros_like(x)
    return (x - mu) / sd


def leakybucket_monthly(
    start_month: int,
    Nyrs: int,
    phi_arr: np.ndarray,
    T: np.ndarray,
    P: np.ndarray,
    Mmax: float,
    Mmin: float,
    alpha: float,
    mth: float,
    muth: float,
    dr: float,
    Minit_over_dr: float,
) -> np.ndarray:
    """Monthly Leaky Bucket placeholder.
    Returns an array M shaped like T (12, Nyrs). TODO: implement physics.
    """
    # For now: simple bounded smoothing of precipitation as a placeholder.
    M = np.clip(P / (dr if dr else 1.0), Mmin, Mmax)
    return M


def leakybucket_submonthly(
    start_month: int,
    Nyrs: int,
    phi_arr: np.ndarray,
    T: np.ndarray,
    P: np.ndarray,
    Mmax: float,
    Mmin: float,
    alpha: float,
    mth: float,
    muth: float,
    dr: float,
    Minit_over_dr: float,
) -> np.ndarray:
    """Submonthly Leaky Bucket placeholder (calls monthly version for now)."""
    return leakybucket_monthly(
        start_month, Nyrs, phi_arr, T, P, Mmax, Mmin, alpha, mth, muth, dr, Minit_over_dr
    )


def Compute_gE(phi_arr: np.ndarray) -> np.ndarray:
    """Insolation growth term gE placeholder.
    Returns a (12, 1) column vector normalized to [0,1]. TODO: replace with VS-Lite’s formulation.
    """
    months = np.arange(12)
    # crude seasonality by latitude sign: more growth in warm half of year
    season = 0.5 * (1 + np.sin(2 * np.pi * (months - 2) / 12.0))
    gE = season.reshape(12, 1)
    # ensure [0,1]
    gE = (gE - gE.min()) / (gE.max() - gE.min() + 1e-12)
    return gE

def _postprocess_chains(
    settings: Settings,
    chains: List[ChainRecord],
    nyears: int,
) -> Dict[str, Any]:
    """
    Python equivalent of the MATLAB post-processing block:
    - concatenates per-chain series into 'chains' (with chain dimension)
    - builds 'ensembles' by removing burn-in and concatenating across chains
    - returns a dictionary of outputs to merge into `extras`
    """
    nsamp = settings.sampler.nsamp
    nbi   = settings.sampler.nbi
    nchain = settings.sampler.nchain
    err_kind = settings.error_model.kind  # 'iid' | 'ar1' | 'ar2'

    # Safety if sampling not yet implemented
    if not chains:
        return {
            "chains": None,
            "ensembles": None,
        }

    # --- stacks with chain axis (= last axis) ---
    def _stack(name: str) -> np.ndarray:
        return np.stack([getattr(c, name) for c in chains], axis=-1)

    # Core growth parameters
    Ttchains = _stack("Tt")  # shape (nsamp+nbi, nchain)
    Tochains = _stack("To")
    Mtchains = _stack("Mt")
    Mochains = _stack("Mo")

    # Gterms: shape (12, nyears, nsamp+nbi, nchain)
    Gterms_chains = np.stack([c.Gterms_dist for c in chains], axis=-1)

    # Error-model parts
    sig2rwchains = phi1chains = phi2chains = tau2chains = None
    if err_kind == "iid":
        sig2rwchains = _stack("sig2rw")
    elif err_kind in {"ar1", "ar2"}:
        phi1chains = _stack("phi1")
        tau2chains = _stack("tau2")
        if err_kind == "ar2":
            phi2chains = _stack("phi2")

    # logL
    logLchains = _stack("logL")

    # --- ensembles: strip burn-in (nbi) then concat across chains (column-wise) ---
    sl = slice(nbi, None)  # nbi..end
    Ttensemb = np.concatenate([c.Tt[sl] for c in chains], axis=0)
    Toensemb = np.concatenate([c.To[sl] for c in chains], axis=0)
    Mtensemb = np.concatenate([c.Mt[sl] for c in chains], axis=0)
    Moensemb = np.concatenate([c.Mo[sl] for c in chains], axis=0)

    Gterms_distensemb = np.concatenate([c.Gterms_dist[:, :, sl] for c in chains], axis=2)
    # Gterms_distensemb shape: (12, nyears, nsamp_total_post_burn), where the 3rd axis is all chains concatenated

    sig2rwensemb = phi1ensemb = phi2ensemb = tau2ensemb = None
    if err_kind == "iid":
        sig2rwensemb = np.concatenate([c.sig2rw[sl] for c in chains], axis=0)  # type: ignore[index]
    elif err_kind in {"ar1", "ar2"}:
        phi1ensemb = np.concatenate([c.phi1[sl] for c in chains], axis=0)      # type: ignore[index]
        tau2ensemb = np.concatenate([c.tau2[sl] for c in chains], axis=0)      # type: ignore[index]
        if err_kind == "ar2":
            phi2ensemb = np.concatenate([c.phi2[sl] for c in chains], axis=0)  # type: ignore[index]

    logLensemb = np.concatenate([c.logL[sl] for c in chains], axis=0)          # type: ignore[index]

    # Bundle results like MATLAB variable names (chains & ensemb)
    out: Dict[str, Any] = {
        "chains": {
            "Ttchains": Ttchains,
            "Tochains": Tochains,
            "Mtchains": Mtchains,
            "Mochains": Mochains,
            "Gterms_chains": Gterms_chains,
            "logLchains": logLchains,
        },
        "ensembles": {
            "Ttensemb": Ttensemb,
            "Toensemb": Toensemb,
            "Mtensemb": Mtensemb,
            "Moensemb": Moensemb,
            "Gterms_distensemb": Gterms_distensemb,
            "logLensemb": logLensemb,
        },
    }

    if err_kind == "iid":
        out["chains"]["sig2rwchains"] = sig2rwchains
        out["ensembles"]["sig2rwensemb"] = sig2rwensemb
    elif err_kind == "ar1":
        out["chains"]["phi1chains"] = phi1chains
        out["chains"]["tau2chains"] = tau2chains
        out["ensembles"]["phi1ensemb"] = phi1ensemb
        out["ensembles"]["tau2ensemb"] = tau2ensemb
    elif err_kind == "ar2":
        out["chains"]["phi1chains"] = phi1chains
        out["chains"]["phi2chains"] = phi2chains
        out["chains"]["tau2chains"] = tau2chains
        out["ensembles"]["phi1ensemb"] = phi1ensemb
        out["ensembles"]["phi2ensemb"] = phi2ensemb
        out["ensembles"]["tau2ensemb"] = tau2ensemb

    return out
# Optional convenience alias with a more Pythonic name
estimate_vslite_params = estimate_vslite_params_v2_3
