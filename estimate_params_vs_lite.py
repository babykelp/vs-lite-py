from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Sequence, List, Optional

import numpy as np
from typing import Iterable
from numpy.random import default_rng, Generator


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

    ensembles = post.get("ensembles") if post else None
    chains_dict = post.get("chains") if post else None

    # --- point estimates ('med' or 'mle') ---
    T1, T2, M1, M2, err_pts = _select_point_estimates(settings, ensembles, settings.error_model.kind)

    # --- R-hats and convergence flag ---
    Rhats = None
    convwarning = None
    if chains_dict:
        Rhats = _compute_rhats(settings, chains_dict)
        # convwarning: any |Rhat - 1| > convthresh
        diffs = [abs(v - 1.0) for v in Rhats.values() if np.isfinite(v)]
        convwarning = bool(any(d > settings.sampler.convthresh for d in diffs))
    
        # --- Build ordered Rhats vector (MATLAB order) & verbose printout ---
    Rhats_vec = None
    if Rhats:
        order = ["T1", "T2", "M1", "M2"]
        if settings.error_model.kind == "iid":
            order += ["sigma2rw"]
            heading = "    Rhat for T1, T2, M1, M2, sigma2rw:"
        elif settings.error_model.kind == "ar1":
            order += ["phi1", "tau2"]
            heading = "    Rhat for T1, T2, M1, M2, phi1, tau2:"
        elif settings.error_model.kind == "ar2":
            order += ["phi1", "phi2", "tau2"]
            heading = "    Rhat for T1, T2, M1, M2, phi1, phi2, tau2:"
        else:
            heading = "    Rhat values:"

        Rhats_vec = np.array([Rhats.get(k, np.nan) for k in order], dtype=float)

        if settings.sampler.verbose:
            print(heading)
            print(Rhats_vec)

        # convwarning: any |Rhat - 1| > convthresh
        convwarning = bool(np.any(np.abs(Rhats_vec - 1.0) > settings.sampler.convthresh))


    # --- Advanced optional outputs placeholder structure (MATLAB varargout analogue) ---
    # Always present keys:
    extras: Dict[str, Any] = {
        "settings": settings,
        "posterior": None,
        "diagnostics": {},
        "chains": chains_dict,
        "ensembles": ensembles,
        "T1dist": None,
        "T2dist": None,
        "M1dist": None,
        "M2dist": None,
        "Rhats": Rhats,                 # dict per-parameter
        "Rhats_vec": Rhats_vec,         # ordered numpy array (MATLAB-compatible order)
        "convwarning": convwarning,     # bool
    }

    if ensembles:
        extras.update({
            "Ttensemb": ensembles.get("Ttensemb"),
            "Toensemb": ensembles.get("Toensemb"),
            "Mtensemb": ensembles.get("Mtensemb"),
            "Moensemb": ensembles.get("Moensemb"),
            "logLensemb": ensembles.get("logLensemb"),
            # error-model specific ensembles
            "sig2rwensemb": ensembles.get("sig2rwensemb") if settings.error_model.kind == "iid" else None,
            "phi1ensemb": ensembles.get("phi1ensemb") if settings.error_model.kind in {"ar1", "ar2"} else None,
            "phi2ensemb": ensembles.get("phi2ensemb") if settings.error_model.kind == "ar2" else None,
            "tau2ensemb": ensembles.get("tau2ensemb") if settings.error_model.kind in {"ar1", "ar2"} else None,
            "Gterms_distensemb": ensembles.get("Gterms_distensemb"),
    })


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

def _gelman_rubin92(chain_mat: np.ndarray, nsamp: int, nbi: int) -> float:
    """
    Gelman & Rubin (1992) potential scale reduction factor, R-hat.

    Parameters
    ----------
    chain_mat : ndarray, shape (nsamp+nbi, nchain)
        Stacked draws for one parameter across chains (rows = iterations).
    nsamp : int
        Total iterations per chain (post-burn + burn).
    nbi : int
        Burn-in iterations per chain.

    Returns
    -------
    float
        R-hat (>= 1). Values near 1 indicate convergence.
    """
    if chain_mat.ndim != 2:
        raise ValueError("chain_mat must be 2D (iterations x chains)")

    # use only post-burn-in samples
    x = chain_mat[nbi:, :]
    n, m = x.shape  # n = iterations per chain, m = number of chains
    if n < 2 or m < 2:
        return np.nan  # not enough data for R-hat

    chain_means = np.mean(x, axis=0)
    chain_vars  = np.var(x, axis=0, ddof=1)

    W = np.mean(chain_vars)                         # within-chain variance
    B = n * np.var(chain_means, ddof=1)             # between-chain variance
    var_hat = ((n - 1) / n) * W + (B / n)           # marginal posterior variance estimate

    if W <= 0:
        return np.inf

    Rhat = np.sqrt(var_hat / W)
    return float(Rhat)


def _compute_rhats(settings: Settings, chains_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute R-hats for all relevant series using _gelman_rubin92.
    Expects entries like 'Ttchains', 'Tochains', 'Mtchains', 'Mochains', and
    (depending on error model) 'sig2rwchains' or 'phi1chains'/'tau2chains'.
    """
    nsamp = settings.sampler.nsamp
    nbi   = settings.sampler.nbi
    errk  = settings.error_model.kind

    rhats: Dict[str, float] = {}

    for name in ["Ttchains", "Tochains", "Mtchains", "Mochains"]:
        arr = chains_dict.get(name)
        if arr is not None:
            rhats[name.replace("chains", "")] = _gelman_rubin92(arr, nsamp, nbi)

    if errk == "iid":
        arr = chains_dict.get("sig2rwchains")
        if arr is not None:
            rhats["sigma2rw"] = _gelman_rubin92(arr, nsamp, nbi)
    elif errk in {"ar1", "ar2"}:
        arr = chains_dict.get("phi1chains")
        if arr is not None:
            rhats["phi1"] = _gelman_rubin92(arr, nsamp, nbi)
        arr = chains_dict.get("tau2chains")
        if arr is not None:
            rhats["tau2"] = _gelman_rubin92(arr, nsamp, nbi)
        if errk == "ar2":
            arr = chains_dict.get("phi2chains")
            if arr is not None:
                rhats["phi2"] = _gelman_rubin92(arr, nsamp, nbi)

    return rhats

def _select_point_estimates(
    settings: Settings,
    ensembles: Optional[Dict[str, np.ndarray]],
    err_kind: str,
) -> Tuple[float, float, float, float, Dict[str, float]]:
    """
    Implements MATLAB's pt_ests block:
    - If 'med': returns medians of post-burn-in ensembles
    - If 'mle': returns values at index of max logLensemb
    Also returns error-model point estimates in a small dict.
    """
    # Defaults if nothing available yet
    T1 = T2 = M1 = M2 = np.nan
    err_pts: Dict[str, float] = {}

    if not ensembles:
        return T1, T2, M1, M2, err_pts

    Ttensemb = ensembles.get("Ttensemb")
    Toensemb = ensembles.get("Toensemb")
    Mtensemb = ensembles.get("Mtensemb")
    Moensemb = ensembles.get("Moensemb")
    logLensemb = ensembles.get("logLensemb")

    if settings.sampler.pt_ests == "med":
        if Ttensemb is not None: T1 = float(np.median(Ttensemb))
        if Toensemb is not None: T2 = float(np.median(Toensemb))
        if Mtensemb is not None: M1 = float(np.median(Mtensemb))
        if Moensemb is not None: M2 = float(np.median(Moensemb))

        if err_kind == "iid":
            sig2rw = ensembles.get("sig2rwensemb")
            if sig2rw is not None:
                err_pts["sig2rw"] = float(np.median(sig2rw))
        elif err_kind == "ar1":
            phi1 = ensembles.get("phi1ensemb")
            tau2 = ensembles.get("tau2ensemb")
            if phi1 is not None: err_pts["phi1hat"] = float(np.median(phi1))
            if tau2 is not None: err_pts["tau2hat"] = float(np.median(tau2))
        elif err_kind == "ar2":
            phi1 = ensembles.get("phi1ensemb")
            phi2 = ensembles.get("phi2ensemb")
            tau2 = ensembles.get("tau2ensemb")
            if phi1 is not None: err_pts["phi1hat"] = float(np.median(phi1))
            if phi2 is not None: err_pts["phi2hat"] = float(np.median(phi2))
            if tau2 is not None: err_pts["tau2hat"] = float(np.median(tau2))

    else:  # 'mle'
        if logLensemb is None or logLensemb.size == 0:
            return T1, T2, M1, M2, err_pts

        mle_ind = int(np.argmax(logLensemb))

        if Ttensemb is not None and mle_ind < Ttensemb.size:
            T1 = float(Ttensemb[mle_ind])
        if Toensemb is not None and mle_ind < Toensemb.size:
            T2 = float(Toensemb[mle_ind])
        if Mtensemb is not None and mle_ind < Mtensemb.size:
            M1 = float(Mtensemb[mle_ind])
        if Moensemb is not None and mle_ind < Moensemb.size:
            M2 = float(Moensemb[mle_ind])

        if err_kind == "iid":
            sig2rw = ensembles.get("sig2rwensemb")
            if sig2rw is not None and mle_ind < sig2rw.size:
                err_pts["sig2rw"] = float(sig2rw[mle_ind])
        elif err_kind == "ar1":
            phi1 = ensembles.get("phi1ensemb")
            tau2 = ensembles.get("tau2ensemb")
            if phi1 is not None and mle_ind < phi1.size:
                err_pts["phi1hat"] = float(phi1[mle_ind])
            if tau2 is not None and mle_ind < tau2.size:
                err_pts["tau2hat"] = float(tau2[mle_ind])
        elif err_kind == "ar2":
            phi1 = ensembles.get("phi1ensemb")
            phi2 = ensembles.get("phi2ensemb")
            tau2 = ensembles.get("tau2ensemb")
            if phi1 is not None and mle_ind < phi1.size:
                err_pts["phi1hat"] = float(phi1[mle_ind])
            if phi2 is not None and mle_ind < phi2.size:
                err_pts["phi2hat"] = float(phi2[mle_ind])
            if tau2 is not None and mle_ind < tau2.size:
                err_pts["tau2hat"] = float(tau2[mle_ind])

    return T1, T2, M1, M2, err_pts

def _month_lengths_nonleap() -> np.ndarray:
    """Days per month for a non-leap year (shape: (12,))."""
    return np.array([31,28,31,30,31,30,31,31,30,31,30,31], dtype=int)


def _monthly_daylight_factor(phi_deg: np.ndarray | float) -> np.ndarray:
    """
    Approximate monthly mean daylight factor L/12 used by Thornthwaite:
    factor = daylength_hours / 12.0  (shape: (12,))

    Uses mid-month solar declination and the standard daylength formula.
    """
    phi = np.deg2rad(np.asarray(phi_deg, dtype=float))
    # mid-month day-of-year (non-leap)
    mid_doy = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349], dtype=float)
    # Cooper (1969) declination in radians
    delta = 0.409 * np.sin(2 * np.pi * (mid_doy - 80) / 365.0)

    # hour angle
    cos_omega = -np.tan(phi) * np.tan(delta)
    cos_omega = np.clip(cos_omega, -1.0, 1.0)
    omega = np.arccos(cos_omega)
    # daylength in hours: 2*omega * 24/(2*pi)
    daylength = (2.0 * omega) * (24.0 / (2.0 * np.pi))
    return daylength / 12.0  # L/12

def _thornthwaite_pet_mm(T_monthly: np.ndarray, ndl: np.ndarray, cdays: np.ndarray) -> np.ndarray:
    """
    Thornthwaite PET (mm) by month, vectorized over years.

    Parameters
    ----------
    T_monthly : (12, Nyrs) monthly mean air temperature in °C
    ndl       : (12,) daylight factor L/12
    cdays     : (12,) days per month

    Returns
    -------
    PET : (12, Nyrs) in mm
    """
    T = np.array(T_monthly, dtype=float)
    T[T < 0] = 0.0  # Thornthwaite: negative T → PET=0

    # Heat index I computed per year
    # I = sum( (T_i/5)^1.514 ), across months with T>0
    with np.errstate(invalid="ignore"):
        I = np.sum((T / 5.0) ** 1.514, axis=0)

    # Empirical exponent a(I)
    a = 6.75e-7 * I**3 - 7.71e-5 * I**2 + 1.792e-2 * I + 0.49239

    # PET in cm for a 30-day 12-hour month: 16 * (10*T/I)^a
    # Then scale by daylight factor and month length per Thornthwaite (1948).
    # Handle I == 0 → PET = 0
    PET = np.zeros_like(T)
    for y in range(T.shape[1]):
        if I[y] <= 0 or not np.isfinite(I[y]):
            continue
        # base PET in cm
        base = 16.0 * (10.0 * (T[:, y] / I[y])) ** a[y]
        base[T[:, y] <= 0] = 0.0
        # scale to actual month length and daylight
        PET[:, y] = base * ndl * (cdays / 30.0) * 10.0  # cm → mm

    return PET

def _ndl_and_monthly_daylight(phi_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Replicates the MATLAB daylight block:
    - daily normalized daylength 'ndl' over 365 days
    - monthly daylight factor (nhrs/12) and month-day counts

    Returns
    -------
    ndl_daily : (365,)   normalized day length (0..1), MATLAB 'ndl'
    daylight_factor : (12,)  nhrs/12 for the middle of each month
    cdays : (13,) cumulative day counts for month boundaries (MATLAB 'cdays')
    ndays : (13,) month-day vector with a leading 0 (MATLAB 'ndays')
    """
    # MATLAB: ndays = [0 31 28 31 30 31 30 31 31 30 31 30 31];
    ndays = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=int)
    cdays = np.cumsum(ndays)

    # Convert latitude to radians
    latr = np.deg2rad(phi_deg)

    # MATLAB: sd = asin(sin(pi*23.5/180) * sin(pi * (((1:365) - 80)/180)))'
    days = np.arange(1, 366, dtype=float)
    sd = np.arcsin(np.sin(np.pi * 23.5 / 180.0) * np.sin(np.pi * ((days - 80.0) / 180.0)))

    # MATLAB:
    # y = -tan(lat)*tan(sd); clamp to [-1,1]
    y = -np.tan(latr) * np.tan(sd)
    y = np.clip(y, -1.0, 1.0)

    # hdl = acos(y)
    hdl = np.arccos(y)

    # dtsi = hdl*sin(lat)*sin(sd) + cos(lat)*cos(sd)*sin(hdl)
    dtsi = hdl * (np.sin(latr) * np.sin(sd)) + (np.cos(latr) * np.cos(sd) * np.sin(hdl))

    # ndl = dtsi / max(dtsi)
    ndl_daily = dtsi / np.max(dtsi)

    # ---- monthly mean daylength for Thornthwaite (middle-of-month) ----
    # jday = cdays(1:12) + .5*ndays(2:13)
    jday = cdays[:12].astype(float) + 0.5 * ndays[1:13].astype(float)

    # m_star = 1 - tand(phi)*tand(23.439*cos(jday*pi/182.625))
    # clamp: if <0 -> 0; if 0<m_star<2 keep; if >2 -> 2
    phi_tan = np.tan(np.deg2rad(phi_deg))
    m_star = 1.0 - phi_tan * np.tan(np.deg2rad(23.439 * np.cos(jday * np.pi / 182.625)))
    mmm = np.where(m_star < 0, 0.0, np.where(m_star > 2.0, 2.0, m_star))

    # nhrs = 24*acosd(1-mmm)/180
    nhrs = 24.0 * (np.degrees(np.arccos(1.0 - mmm)) / 180.0)

    # daylight factor used by Thornthwaite: nhrs/12
    daylight_factor = nhrs / 12.0

    return ndl_daily, daylight_factor, cdays.astype(float), ndays.astype(float)

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
    syear: int,
    eyear: int,
    phi: np.ndarray | float,
    T: np.ndarray,
    P: np.ndarray,
    Mmax: float,
    Mmin: float,
    alph: float,
    m_th: float,
    mu_th: float,
    rootd: float,
    M0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Monthly CPC-style leaky-bucket soil-moisture integrator (coarse monthly step).

    Parameters
    ----------
    syear, eyear : int
        Start/end indices (MATLAB passes 1, Nyrs). We derive Nyrs from T/P shape.
    phi : float or (Nyrs,) array
        Latitude in degrees N (scalar or per-year; function uses the scalar).
    T : (12, Nyrs)
        Monthly mean air temperature (°C).
    P : (12, Nyrs)
        Monthly precipitation (mm).
    Mmax, Mmin : float
        Soil moisture bounds (v/v).
    alph, m_th, mu_th : float
        Runoff/leakage parameters (as in MATLAB).
    rootd : float
        Root/bucket depth (mm).
    M0 : float
        Initial soil moisture (v/v) at the start (previous month).

    Returns
    -------
    M : (12, Nyrs)
        Volumetric soil moisture (v/v).
    potEv : (12, Nyrs)
        Potential evapotranspiration (mm) via Thornthwaite scaling.
    ndl_daily : (365,)
        Normalized daily daylength (0..1) for the latitude.
    cdays : (13,)
        Cumulative day counts for month boundaries, with a leading 0 (MATLAB-like).
    """
    # --- checks & shapes ---
    T = np.asarray(T, dtype=float)
    P = np.asarray(P, dtype=float)
    assert T.shape == P.shape and T.shape[0] == 12, "T and P must be (12, Nyrs)"
    nyears = T.shape[1]

    # MATLAB: if (M0 < 0) M0 = 200/rootd
    if M0 < 0.0:
        M0 = 200.0 / float(rootd)

        # Latitude (use scalar)
    phi_scalar = float(phi[0]) if np.ndim(phi) == 1 else float(phi)

    # Exact MATLAB-equivalent daylight pieces
    ndl_daily, daylight_factor, cdays, ndays = _ndl_and_monthly_daylight(phi_scalar)

    # MATLAB: L = (ndays(2:13)/30) .* (nhrs/12)
    L = (ndays[1:13] / 30.0) * daylight_factor  # shape (12,)

    # --- Initialize outputs (12 x Nyrs) ---
    M = np.full_like(T, np.nan, dtype=float)
    potEv = np.full_like(T, np.nan, dtype=float)

    # --- Year cycle ---
    for cyear in range(nyears):  # 0-based; MATLAB had 1:nyrs
        # --- Month cycle ---
        for t in range(12):      # 0..11; MATLAB had 1:12
            # ----- Thornthwaite PET (Ep) for current month -----
            Tt = float(T[t, cyear])
            if Tt < 0.0:
                Ep = 0.0
            elif 0.0 <= Tt < 26.5:
                istar = (T[:, cyear] / 5.0).copy()
                istar[istar < 0.0] = 0.0
                I = float(np.sum(istar ** 1.514))
                if I <= 0.0 or not np.isfinite(I):
                    Ep = 0.0
                else:
                    a = (6.75e-7) * I**3 - (7.71e-5) * I**2 + (1.79e-2) * I + 0.49
                    Ep = 16.0 * L[t] * (10.0 * Tt / I) ** a
            else:  # T >= 26.5
                Ep = -415.85 + 32.25 * Tt - 0.43 * (Tt ** 2)

            potEv[t, cyear] = Ep

            # ----- CPC Leaky Bucket monthly (no substeps) -----
            if t > 0:
                # use previous month in same year
                prev_sm = float(M[t - 1, cyear])
                Etrans = Ep * prev_sm * rootd / (Mmax * rootd)
                G = mu_th * alph / (1.0 + mu_th) * prev_sm * rootd
                R = P[t, cyear] * (prev_sm * rootd / (Mmax * rootd)) ** m_th + \
                    (alph / (1.0 + mu_th)) * prev_sm * rootd
                dWdt = float(P[t, cyear]) - Etrans - R - G
                M[t, cyear] = prev_sm + dWdt / rootd

            elif t == 0 and cyear > 0:
                # use previous December
                prev_sm = float(M[11, cyear - 1])
                Etrans = Ep * prev_sm * rootd / (Mmax * rootd)
                G = mu_th * alph / (1.0 + mu_th) * prev_sm * rootd
                R = P[t, cyear] * (prev_sm * rootd / (Mmax * rootd)) ** m_th + \
                    (alph / (1.0 + mu_th)) * prev_sm * rootd
                dWdt = float(P[t, cyear]) - Etrans - R - G
                M[t, cyear] = prev_sm + dWdt / rootd

            else:  # t == 0 and cyear == 0
                # initial month of simulation
                M0_eff = float(M0)
                if M0_eff < 0.0:
                    M0_eff = 0.20  # 200 mm at rootd=1000 mm; matches MATLAB's .20
                Etrans = Ep * M0_eff * rootd / (Mmax * rootd)
                G = mu_th * alph / (1.0 + mu_th) * (M0_eff * rootd)
                R = P[t, cyear] * (M0_eff * rootd / (Mmax * rootd)) ** m_th + \
                    (alph / (1.0 + mu_th)) * M0_eff * rootd
                dWdt = float(P[t, cyear]) - Etrans - R - G
                M[t, cyear] = M0_eff + dWdt / rootd

            # error-catching (same as MATLAB)
            if M[t, cyear] <= Mmin:
                M[t, cyear] = Mmin
            if M[t, cyear] >= Mmax:
                M[t, cyear] = Mmax
            if not np.isfinite(M[t, cyear]):
                M[t, cyear] = Mmin


    return M, potEv, ndl_daily, cdays



def leakybucket_submonthly(
    syear: int,
    eyear: int,
    phi: np.ndarray | float,
    T: np.ndarray,
    P: np.ndarray,
    Mmax: float,
    Mmin: float,
    alph: float,
    m_th: float,
    mu_th: float,
    rootd: float,
    M0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sub-monthly (daily) CPC-style leaky-bucket soil-moisture integrator (Python translation scaffold).

    Parameters
    ----------
    syear, eyear : int
        Start/end indices (MATLAB passes 1, Nyrs). We ignore their absolute values and
        derive Nyrs from T/P shape.
    phi : float or (Nyrs,) array
        Latitude in degrees N (scalar or per-year).
    T : (12, Nyrs)
        Monthly mean air temperature (°C).
    P : (12, Nyrs)
        Monthly precipitation (mm).
    Mmax, Mmin : float
        Soil moisture bounds (v/v).
    alph, m_th, mu_th : float
        Runoff/leakage parameters (as in MATLAB).
    rootd : float
        Root/bucket depth (mm).
    M0 : float
        Initial soil moisture (v/v) at the start (previous month).

    Returns
    -------
    M : (12, Nyrs)
        Volumetric soil moisture (v/v).
    potEv : (12, Nyrs)
        Potential evapotranspiration (mm) via Thornthwaite (1948) scaling.
    ndl : (12,)
        Monthly daylight factor L/12.
    cdays : (12,)
        Days per month (non-leap).
    """
    T = np.asarray(T, dtype=float)
    P = np.asarray(P, dtype=float)
    nyears = T.shape[1]
    assert T.shape == P.shape and T.shape[0] == 12, "T and P must be (12, Nyrs)"

        # --- MATLAB: if (M0 < 0) M0 = 200/rootd; ---
    if M0 < 0.0:
        M0 = 200.0 / float(rootd)

    # --- Daylength math per MATLAB block ---
    # Note: MATLAB takes phi as scalar here. If you passed an array, take its first element.
    phi_scalar = float(phi[0]) if np.ndim(phi) == 1 else float(phi)
    ndl_daily, daylight_factor, cdays, ndays = _ndl_and_monthly_daylight(phi_scalar)

    # --- Thornthwaite PET (mm) using the MATLAB daylight factor (nhrs/12) and month length ---
    # Our helper expects `ndl`=L/12 (i.e., nhrs/12) and `cdays` as monthly day counts.
    potEv = _thornthwaite_pet_mm(T, daylight_factor, ndays[1:13])  # pass 12 monthly day counts

     # --- Initialize outputs (12 x Nyrs) ---
    M = np.full_like(T, np.nan, dtype=float)
    potEv = np.full_like(T, np.nan, dtype=float)

    # MATLAB: monthly daylight factor L = (days_in_month/30) * (nhrs/12)
    # day lengths for the 12 months (no leap year)
    month_days = ndays[1:13]  # shape (12,)
    L_month = (month_days / 30.0) * daylight_factor  # shape (12,)

    # ADDED BY NICK: if M0 < 0 → set to 200/rootd
    if M0 < 0.0:
        M0 = 200.0 / float(rootd)

    # --- Year cycle ---
    for cyear in range(nyears):
        # --- Month cycle ---
        for t in range(12):
            # ----- Thornthwaite PET (Ep) for current month -----
            Tt = float(T[t, cyear])

            if Tt < 0.0:
                Ep = 0.0
            elif 0.0 <= Tt < 26.5:
                istar = (T[:, cyear] / 5.0).copy()
                istar[istar < 0.0] = 0.0
                I = float(np.sum(istar ** 1.514))
                if I <= 0.0 or not np.isfinite(I):
                    Ep = 0.0
                else:
                    a = (6.75e-7) * I**3 - (7.71e-5) * I**2 + (1.79e-2) * I + 0.49
                    Ep = 16.0 * L_month[t] * (10.0 * Tt / I) ** a
            else:  # T >= 26.5
                Ep = -415.85 + 32.25 * Tt - 0.43 * (Tt ** 2)

            potEv[t, cyear] = Ep

            # ----- CPC Leaky Bucket with substeps in 2 mm increments -----
            dp = 2.0  # mm per increment
            Pval = float(P[t, cyear])
            nstep = int(np.floor(Pval / dp)) + 1  # at least 1 substep
            Pinc = Pval / nstep
            alphinc = float(alph) / nstep
            Epinc = Ep / nstep

            # handling for sm_init (previous month's soil moisture)
            if t > 0:
                M0_eff = M[t - 1, cyear]
            elif t == 0 and cyear > 0:
                M0_eff = M[11, cyear - 1]
            else:
                M0_eff = float(M0)

            sm0 = float(M0_eff)

            for _ in range(nstep):
                # evapotranspiration (linear in soil moisture fraction)
                # Etrans = Epinc * sm0*rootd / (Mmax*rootd) == Epinc * (sm0/Mmax)
                Etrans = Epinc * (sm0 / float(Mmax))

                # groundwater loss via percolation:
                # G = mu_th * alphinc/(1+mu_th) * sm0*rootd
                G = float(mu_th) * (alphinc / (1.0 + float(mu_th))) * sm0 * float(rootd)

                # runoff (surface + subsurface):
                # R = Pinc*(sm0*rootd/(Mmax*rootd))^m_th + (alphinc/(1+mu_th))*sm0*rootd
                frac = sm0 / float(Mmax)
                frac = np.clip(frac, 0.0, 1.0)
                R = Pinc * (frac ** float(m_th)) + (alphinc / (1.0 + float(mu_th))) * sm0 * float(rootd)

                # water balance (units mm), convert back to volumetric via /rootd
                dWdt = Pinc - Etrans - R - G
                sm1 = sm0 + dWdt / float(rootd)

                # enforce bounds
                sm0 = max(float(Mmin), min(float(Mmax), sm1))

            M[t, cyear] = sm0

            # error-catching (exactly as MATLAB)
            if M[t, cyear] <= float(Mmin):
                M[t, cyear] = float(Mmin)
            if M[t, cyear] >= float(Mmax):
                M[t, cyear] = float(Mmax)
            if not np.isfinite(M[t, cyear]):
                M[t, cyear] = float(Mmin)


    return M, potEv, ndl_daily, cdays


def Compute_gE(phi: float | np.ndarray) -> np.ndarray:
    """
    VS-Lite scaled daylength term gE from latitude (deg N).
    Returns a (12, 1) column vector, matching MATLAB.

    Notes
    -----
    The original MATLAB expects a scalar phi. If an array is passed,
    we use its first element.
    """
    # ensure scalar latitude in radians
    phi_scalar = float(phi[0]) if np.ndim(phi) == 1 else float(phi)
    latr = np.deg2rad(phi_scalar)

    # month-day vectors with leading 0 (MATLAB-compatible)
    ndays = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=int)
    cdays = np.cumsum(ndays)  # length 13, month boundaries

    # daily solar declination (365 days, no leap-year adjustment)
    days = np.arange(1, 366, dtype=float)
    sd = np.arcsin(np.sin(np.pi * 23.5 / 180.0) * np.sin(np.pi * ((days - 80.0) / 180.0)))

    # helper terms
    y = -np.tan(latr) * np.tan(sd)
    y = np.clip(y, -1.0, 1.0)
    hdl = np.arccos(y)

    # normalized daylength over the year
    dtsi = hdl * (np.sin(latr) * np.sin(sd)) + (np.cos(latr) * np.cos(sd) * np.sin(hdl))
    ndl = dtsi / np.max(dtsi)

    # monthly means of normalized daylength
    gE = np.empty((12, 1), dtype=float)
    for t in range(12):
        start = cdays[t]
        end = cdays[t + 1]  # exclusive
        gE[t, 0] = float(np.mean(ndl[start:end]))

    return gE

def _ar1_precision_matrix(phi: float, tau2: float, n: int) -> np.ndarray:
    """
    Precision (inverse covariance) matrix for a stationary AR(1) process of length n,
    with innovations variance tau2 and coefficient phi.

    Q = (1/tau2) * tri-diagonal with:
        diag: [1, 1+phi^2, ..., 1+phi^2, 1]
        off-diag: -phi
    """
    Q = np.zeros((n, n), dtype=float)
    if n <= 0:
        return Q
    inv_tau2 = 1.0 / float(tau2)
    # main diagonal
    Q[np.arange(n), np.arange(n)] = 1.0
    if n >= 2:
        Q[1:-1, 1:-1] += phi ** 2
        Q[np.arange(n-1), np.arange(1, n)] = -phi
        Q[np.arange(1, n), np.arange(n-1)] = -phi
    Q *= inv_tau2
    return Q

# -------- Conditional parameter sampling subroutines --------

def Tt_U_aux(
    Ttcurr: float,
    T: np.ndarray,            # shape (12, Ny)
    To: float,                # scalar current To
    gM: np.ndarray,           # shape (12, Ny) from previous step
    RW: np.ndarray,           # shape (Ny,)
    errorpars: Iterable[float],
    gE: np.ndarray,           # shape (12, 1)
    Gterms: np.ndarray,       # current g matrix (12, Ny)
    att: float, btt: float,   # uniform prior support for Tt
    intwindow: tuple[int, int],
    cyrs: np.ndarray,         # 0-based year indices used for likelihood
    rng: Generator | None = None,
) -> float:
    """
    Uniform-prior proposal for Tt with Metropolis-Hastings acceptance.
    Mirrors MATLAB Tt_U_aux, including integration-window logic and
    white/AR(1) error models.

    Notes
    -----
    - `cyrs` must be a 1D array of 0-based indices into columns of T/Gterms.
    - `gE` is (12,1); we'll broadcast it row-wise.
    """
    rng = default_rng() if rng is None else rng
    Ny = T.shape[1]
    I_0, I_f = int(intwindow[0]), int(intwindow[1])

    # --- propose from the prior ---
    Ttprop = float(rng.uniform(att, btt))

    # --- build gT for the proposal ---
    gTprop = np.full_like(T, np.nan, dtype=float)
    gTprop[T < Ttprop] = 0.0
    gTprop[T > To]     = 1.0
    mask_mid = (T > Ttprop) & (T < To)
    # (T - Ttprop) / (To - Ttprop) where mask_mid true
    gTprop[mask_mid] = (T[mask_mid] - Ttprop) / max(To - Ttprop, 1e-12)

    # g matrices (12 x Ny)
    gprop = gE.reshape(12, 1) * np.minimum(gM, gTprop)
    gcurr = np.array(Gterms, dtype=float, copy=True)

    # --- apply integration window ---
    # Months are 0..11 in Python.
    if I_0 < 0:
        # include previous year's tail months
        startmo = 12 + I_0      # inclusive row index [startmo..11]
        endmo   = I_f - 1       # inclusive row index for current-year segment [0..endmo]

        # rows startmo..11 for all years
        tail_prop = gprop[startmo:12, :]   # shape (k, Ny)
        tail_curr = gcurr[startmo:12, :]

        # first "previous-season" column is mean across years (to fill year 0)
        tail_mean_prop = np.mean(tail_prop, axis=1, keepdims=True)  # (k,1)
        tail_mean_curr = np.mean(tail_curr, axis=1, keepdims=True)  # (k,1)

        # shift one year to the right for cols 1..Ny-1
        tail_prop_shift = tail_prop[:, :-1] if Ny > 1 else np.empty((tail_prop.shape[0], 0))
        tail_curr_shift = tail_curr[:, :-1] if Ny > 1 else np.empty((tail_curr.shape[0], 0))

        prevseas_prop = np.concatenate([tail_mean_prop, tail_prop_shift], axis=1)  # (k, Ny)
        prevseas_curr = np.concatenate([tail_mean_curr, tail_curr_shift], axis=1)

        # current-year window rows: 0..endmo
        main_prop = gprop[: (endmo + 1), :]  # (endmo+1, Ny)
        main_curr = gcurr[: (endmo + 1), :]

        gprop_win = np.concatenate([prevseas_prop, main_prop], axis=0)
        gcurr_win = np.concatenate([prevseas_curr, main_curr], axis=0)

    else:
        # no previous-year carryover
        startmo = I_0           # inclusive
        endmo   = I_f - 1       # inclusive
        gprop_win = gprop[startmo : (endmo + 1), :]
        gcurr_win = gcurr[startmo : (endmo + 1), :]

    # --- likelihood ratio under chosen error model ---
    err = np.asarray(tuple(errorpars), dtype=float)
    if err.size == 1:
        # White noise error model
        sigma2rw = float(err[0])

        s_gcurr = np.sum(gcurr_win, axis=0)  # length Ny
        s_gprop = np.sum(gprop_win, axis=0)

        # normalize using mean/std across ALL years (not just cyrs)
        z_curr = (s_gcurr - np.mean(s_gcurr)) / (np.std(s_gcurr) + 1e-12)
        z_prop = (s_gprop - np.mean(s_gprop)) / (np.std(s_gprop) + 1e-12)

        resid_curr = RW[cyrs] - np.sqrt(max(1.0 - sigma2rw, 0.0)) * z_curr[cyrs]
        resid_prop = RW[cyrs] - np.sqrt(max(1.0 - sigma2rw, 0.0)) * z_prop[cyrs]

        expcurr = np.sum(resid_curr ** 2)
        expprop = np.sum(resid_prop ** 2)
        HR = np.exp(-0.5 * (expprop - expcurr) / max(sigma2rw, 1e-12))

    elif err.size == 2:
        # AR(1) error model: error = [phi1, tau2]
        phi1, tau2 = float(err[0]), float(err[1])
        sigma2rw = tau2 / max(1.0 - phi1**2, 1e-12)

        Q = _ar1_precision_matrix(phi1, tau2, len(cyrs))

        s_gcurr = np.sum(gcurr_win, axis=0)
        s_gprop = np.sum(gprop_win, axis=0)

        z_curr = (s_gcurr - np.mean(s_gcurr)) / (np.std(s_gcurr) + 1e-12)
        z_prop = (s_gprop - np.mean(s_gprop)) / (np.std(s_gprop) + 1e-12)

        Wcurr = z_curr[cyrs] * np.sqrt(max(1.0 - sigma2rw, 0.0))
        Wprop = z_prop[cyrs] * np.sqrt(max(1.0 - sigma2rw, 0.0))

        rc = (RW[cyrs] - Wcurr).reshape(-1, 1)
        rp = (RW[cyrs] - Wprop).reshape(-1, 1)

        logLprop = float(-0.5 * (rp.T @ Q @ rp))
        logLcurr = float(-0.5 * (rc.T @ Q @ rc))
        HR = np.exp(logLprop - logLcurr)

    else:
        raise ValueError("errorpars must have length 1 (iid) or 2 (AR1).")

    # --- accept/reject ---
    accept = rng.random() < min(1.0, float(HR))
    return Ttprop if accept else float(Ttcurr)

def Tt_lit_aux(
    Ttcurr: float,
    T: np.ndarray,            # (12, Ny)
    To: float,                # current To (scalar)
    gM: np.ndarray,           # (12, Ny) from previous step
    RW: np.ndarray,           # (Ny,)  (use z-scored RW if you did earlier)
    errorpars: Iterable[float],
    gE: np.ndarray,           # (12, 1)
    Gterms: np.ndarray,       # current g matrix (12, Ny)
    aT1: float, bT1: float,   # beta shapes (att, btt in MATLAB)
    slpT1: float, intT1: float,   # 4-parameter beta transform: slp*Beta(a,b) + int
    intwindow: tuple[int, int],
    cyrs: np.ndarray,         # 0-based year indices used for likelihood
    rng: Generator | None = None,
) -> float:
    """
    4-parameter Beta prior proposal for Tt with Metropolis–Hastings accept/reject.
    Direct port of MATLAB Tt_lit_aux.

    Notes
    -----
    - `cyrs` must be 0-based indices into columns of T/Gterms.
    - `gE` should be shape (12,1); it will be broadcast row-wise.
    """
    rng = default_rng() if rng is None else rng
    Ny = T.shape[1]
    I_0, I_f = int(intwindow[0]), int(intwindow[1])

    # --- propose from the 4-parameter Beta prior: Ttprop = slp*Beta(a,b)+int ---
    Ttprop = float(slpT1 * rng.beta(aT1, bT1) + intT1)

    # --- build gT for the proposal ---
    gTprop = np.full_like(T, np.nan, dtype=float)
    gTprop[T < Ttprop] = 0.0
    gTprop[T > To]     = 1.0
    mask_mid = (T > Ttprop) & (T < To)
    denom = max(To - Ttprop, 1e-12)
    gTprop[mask_mid] = (T[mask_mid] - Ttprop) / denom

    # g matrices (12 x Ny)
    gprop = gE.reshape(12, 1) * np.minimum(gM, gTprop)
    gcurr = np.array(Gterms, dtype=float, copy=True)

    # --- apply integration window ---
    if I_0 < 0:
        # include part of previous year
        startmo = 12 + I_0          # inclusive index in [0..11]
        endmo   = I_f - 1           # inclusive
        tail_prop = gprop[startmo:12, :]
        tail_curr = gcurr[startmo:12, :]

        tail_mean_prop = np.mean(tail_prop, axis=1, keepdims=True)  # (k,1)
        tail_mean_curr = np.mean(tail_curr, axis=1, keepdims=True)

        tail_prop_shift = tail_prop[:, :-1] if Ny > 1 else np.empty((tail_prop.shape[0], 0))
        tail_curr_shift = tail_curr[:, :-1] if Ny > 1 else np.empty((tail_curr.shape[0], 0))

        prevseas_prop = np.concatenate([tail_mean_prop, tail_prop_shift], axis=1)
        prevseas_curr = np.concatenate([tail_mean_curr, tail_curr_shift], axis=1)

        main_prop = gprop[: (endmo + 1), :]
        main_curr = gcurr[: (endmo + 1), :]

        gprop_win = np.concatenate([prevseas_prop, main_prop], axis=0)
        gcurr_win = np.concatenate([prevseas_curr, main_curr], axis=0)
    else:
        startmo = I_0
        endmo   = I_f - 1
        gprop_win = gprop[startmo : (endmo + 1), :]
        gcurr_win = gcurr[startmo : (endmo + 1), :]

    # --- likelihood ratio under chosen error model ---
    err = np.asarray(tuple(errorpars), dtype=float)
    if err.size == 1:
        # White noise error model
        sigma2rw = float(err[0])

        s_gcurr = np.sum(gcurr_win, axis=0)
        s_gprop = np.sum(gprop_win, axis=0)

        z_curr = (s_gcurr - np.mean(s_gcurr)) / (np.std(s_gcurr) + 1e-12)
        z_prop = (s_gprop - np.mean(s_gprop)) / (np.std(s_gprop) + 1e-12)

        resid_curr = RW[cyrs] - np.sqrt(max(1.0 - sigma2rw, 0.0)) * z_curr[cyrs]
        resid_prop = RW[cyrs] - np.sqrt(max(1.0 - sigma2rw, 0.0)) * z_prop[cyrs]

        expcurr = np.sum(resid_curr ** 2)
        expprop = np.sum(resid_prop ** 2)
        HR = np.exp(-0.5 * (expprop - expcurr) / max(sigma2rw, 1e-12))

    elif err.size == 2:
        # AR(1) error model: error = [phi1, tau2]
        phi1, tau2 = float(err[0]), float(err[1])
        sigma2rw = tau2 / max(1.0 - phi1**2, 1e-12)

        Q = _ar1_precision_matrix(phi1, tau2, len(cyrs))

        s_gcurr = np.sum(gcurr_win, axis=0)
        s_gprop = np.sum(gprop_win, axis=0)

        z_curr = (s_gcurr - np.mean(s_gcurr)) / (np.std(s_gcurr) + 1e-12)
        z_prop = (s_gprop - np.mean(s_gprop)) / (np.std(s_gprop) + 1e-12)

        Wcurr = z_curr[cyrs] * np.sqrt(max(1.0 - sigma2rw, 0.0))
        Wprop = z_prop[cyrs] * np.sqrt(max(1.0 - sigma2rw, 0.0))

        rc = (RW[cyrs] - Wcurr).reshape(-1, 1)
        rp = (RW[cyrs] - Wprop).reshape(-1, 1)

        logLprop = float(-0.5 * (rp.T @ Q @ rp))
        logLcurr = float(-0.5 * (rc.T @ Q @ rc))
        HR = np.exp(logLprop - logLcurr)
    else:
        raise ValueError("errorpars must have length 1 (iid) or 2 (AR1).")

    # --- accept or reject the proposal ---
    accept = (rng.random() < min(1.0, float(HR)))
    return Ttprop if accept else float(Ttcurr)

def To_U_aux(
    Tocurr: float,
    T: np.ndarray,            # (12, Ny)
    Tt: float,                # scalar current Tt
    gM: np.ndarray,           # (12, Ny)
    RW: np.ndarray,           # (Ny,)
    errorpars: Iterable[float],
    gE: np.ndarray,           # (12, 1)
    Gterms: np.ndarray,       # (12, Ny)
    ato: float, bto: float,   # uniform prior bounds for To
    intwindow: tuple[int, int],
    cyrs: np.ndarray,         # 0-based indices
    rng: Generator | None = None,
) -> float:
    """
    Uniform-prior proposal for To with Metropolis-Hastings acceptance.
    Mirrors MATLAB To_U_aux, including integration-window handling and
    white/AR(1) error models.
    """
    rng = default_rng() if rng is None else rng
    Ny = T.shape[1]
    I_0, I_f = int(intwindow[0]), int(intwindow[1])

    # --- propose from the prior ---
    Toprop = float(rng.uniform(ato, bto))

    # --- build gT for the proposal ---
    gTprop = np.full_like(T, np.nan, dtype=float)
    gTprop[T < Tt]     = 0.0
    gTprop[T > Toprop] = 1.0
    mask_mid = (T > Tt) & (T < Toprop)
    denom = max(Toprop - Tt, 1e-12)
    gTprop[mask_mid] = (T[mask_mid] - Tt) / denom

    # g matrices
    gprop = gE.reshape(12, 1) * np.minimum(gM, gTprop)
    gcurr = np.array(Gterms, dtype=float, copy=True)

    # --- apply integration window ---
    if I_0 < 0:
        startmo = 12 + I_0
        endmo   = I_f - 1

        tail_prop = gprop[startmo:12, :]
        tail_curr = gcurr[startmo:12, :]

        tail_mean_prop = np.mean(tail_prop, axis=1, keepdims=True)
        tail_mean_curr = np.mean(tail_curr, axis=1, keepdims=True)

        tail_prop_shift = tail_prop[:, :-1] if Ny > 1 else np.empty((tail_prop.shape[0], 0))
        tail_curr_shift = tail_curr[:, :-1] if Ny > 1 else np.empty((tail_curr.shape[0], 0))

        prevseas_prop = np.concatenate([tail_mean_prop, tail_prop_shift], axis=1)
        prevseas_curr = np.concatenate([tail_mean_curr, tail_curr_shift], axis=1)

        main_prop = gprop[: (endmo + 1), :]
        main_curr = gcurr[: (endmo + 1), :]

        gprop_win = np.concatenate([prevseas_prop, main_prop], axis=0)
        gcurr_win = np.concatenate([prevseas_curr, main_curr], axis=0)
    else:
        startmo = I_0
        endmo   = I_f - 1
        gprop_win = gprop[startmo : (endmo + 1), :]
        gcurr_win = gcurr[startmo : (endmo + 1), :]

    # --- likelihood ratio ---
    err = np.asarray(tuple(errorpars), dtype=float)
    if err.size == 1:
        sigma2rw = float(err[0])

        s_gcurr = np.sum(gcurr_win, axis=0)
        s_gprop = np.sum(gprop_win, axis=0)

        z_curr = (s_gcurr - np.mean(s_gcurr)) / (np.std(s_gcurr) + 1e-12)
        z_prop = (s_gprop - np.mean(s_gprop)) / (np.std(s_gprop) + 1e-12)

        resid_curr = RW[cyrs] - np.sqrt(max(1.0 - sigma2rw, 0.0)) * z_curr[cyrs]
        resid_prop = RW[cyrs] - np.sqrt(max(1.0 - sigma2rw, 0.0)) * z_prop[cyrs]

        expcurr = np.sum(resid_curr ** 2)
        expprop = np.sum(resid_prop ** 2)
        HR = np.exp(-0.5 * (expprop - expcurr) / max(sigma2rw, 1e-12))

    elif err.size == 2:
        phi1, tau2 = float(err[0]), float(err[1])
        sigma2rw = tau2 / max(1.0 - phi1**2, 1e-12)

        Q = _ar1_precision_matrix(phi1, tau2, len(cyrs))

        s_gcurr = np.sum(gcurr_win, axis=0)
        s_gprop = np.sum(gprop_win, axis=0)

        z_curr = (s_gcurr - np.mean(s_gcurr)) / (np.std(s_gcurr) + 1e-12)
        z_prop = (s_gprop - np.mean(s_gprop)) / (np.std(s_gprop) + 1e-12)

        Wcurr = z_curr[cyrs] * np.sqrt(max(1.0 - sigma2rw, 0.0))
        Wprop = z_prop[cyrs] * np.sqrt(max(1.0 - sigma2rw, 0.0))

        rc = (RW[cyrs] - Wcurr).reshape(-1, 1)
        rp = (RW[cyrs] - Wprop).reshape(-1, 1)

        logLprop = float(-0.5 * (rp.T @ Q @ rp))
        logLcurr = float(-0.5 * (rc.T @ Q @ rc))
        HR = np.exp(logLprop - logLcurr)
    else:
        raise ValueError("errorpars must have length 1 (iid) or 2 (AR1).")

    accept = (rng.random() < min(1.0, float(HR)))
    return Toprop if accept else float(Tocurr)

from numpy.random import default_rng, Generator
from typing import Iterable

def To_lit_aux(
    Tocurr: float,
    T: np.ndarray,            # (12, Ny)
    Tt: float,                # scalar current Tt
    gM: np.ndarray,           # (12, Ny)
    RW: np.ndarray,           # (Ny,)
    errorpars: Iterable[float],
    gE: np.ndarray,           # (12, 1)
    Gterms: np.ndarray,       # (12, Ny)
    ato: float, bto: float,   # Beta(a,b) for proposal
    slp: float,               # hi - lo (scale)
    intr: float,              # lo (offset)
    intwindow: tuple[int, int],
    cyrs: np.ndarray,         # 0-based indices of years used in likelihood
    rng: Generator | None = None,
) -> float:
    """
    'Literature' prior sampler for To: proposal Toprop = slp * Beta(a,b) + intr.
    Mirrors MATLAB To_lit_aux, including integration-window handling and
    white/AR(1) error models for the MH ratio.
    """
    rng = default_rng() if rng is None else rng
    Ny = T.shape[1]
    I_0, I_f = int(intwindow[0]), int(intwindow[1])

    # --- propose from scaled Beta prior ---
    Toprop = float(slp * rng.beta(ato, bto) + intr)

    # --- build gT for the proposal ---
    gTprop = np.full_like(T, np.nan, dtype=float)
    gTprop[T < Tt]       = 0.0
    gTprop[T > Toprop]   = 1.0
    mask_mid = (T > Tt) & (T < Toprop)
    denom = max(Toprop - Tt, 1e-12)
    gTprop[mask_mid] = (T[mask_mid] - Tt) / denom

    # g matrices
    gprop = gE.reshape(12, 1) * np.minimum(gM, gTprop)
    gcurr = np.array(Gterms, dtype=float, copy=True)

    # --- apply integration window ---
    if I_0 < 0:
        startmo = 12 + I_0
        endmo   = I_f - 1

        tail_prop = gprop[startmo:12, :]
        tail_curr = gcurr[startmo:12, :]

        tail_mean_prop = np.mean(tail_prop, axis=1, keepdims=True)
        tail_mean_curr = np.mean(tail_curr, axis=1, keepdims=True)

        tail_prop_shift = tail_prop[:, :-1] if Ny > 1 else np.empty((tail_prop.shape[0], 0))
        tail_curr_shift = tail_curr[:, :-1] if Ny > 1 else np.empty((tail_curr.shape[0], 0))

        prevseas_prop = np.concatenate([tail_mean_prop, tail_prop_shift], axis=1)
        prevseas_curr = np.concatenate([tail_mean_curr, tail_curr_shift], axis=1)

        main_prop = gprop[: (endmo + 1), :]
        main_curr = gcurr[: (endmo + 1), :]

        gprop_win = np.concatenate([prevseas_prop, main_prop], axis=0)
        gcurr_win = np.concatenate([prevseas_curr, main_curr], axis=0)
    else:
        startmo = I_0
        endmo   = I_f - 1
        gprop_win = gprop[startmo : (endmo + 1), :]
        gcurr_win = gcurr[startmo : (endmo + 1), :]

    # --- likelihood ratio ---
    err = np.asarray(tuple(errorpars), dtype=float)
    if err.size == 1:
        sigma2rw = float(err[0])

        s_gcurr = np.sum(gcurr_win, axis=0)
        s_gprop = np.sum(gprop_win, axis=0)

        z_curr = (s_gcurr - np.mean(s_gcurr)) / (np.std(s_gcurr) + 1e-12)
        z_prop = (s_gprop - np.mean(s_gprop)) / (np.std(s_gprop) + 1e-12)

        resid_curr = RW[cyrs] - np.sqrt(max(1.0 - sigma2rw, 0.0)) * z_curr[cyrs]
        resid_prop = RW[cyrs] - np.sqrt(max(1.0 - sigma2rw, 0.0)) * z_prop[cyrs]

        expcurr = np.sum(resid_curr ** 2)
        expprop = np.sum(resid_prop ** 2)
        HR = np.exp(-0.5 * (expprop - expcurr) / max(sigma2rw, 1e-12))

    elif err.size == 2:
        phi1, tau2 = float(err[0]), float(err[1])
        sigma2rw = tau2 / max(1.0 - phi1**2, 1e-12)

        Q = _ar1_precision_matrix(phi1, tau2, len(cyrs))

        s_gcurr = np.sum(gcurr_win, axis=0)
        s_gprop = np.sum(gprop_win, axis=0)

        z_curr = (s_gcurr - np.mean(s_gcurr)) / (np.std(s_gcurr) + 1e-12)
        z_prop = (s_gprop - np.mean(s_gprop)) / (np.std(s_gprop) + 1e-12)

        Wcurr = z_curr[cyrs] * np.sqrt(max(1.0 - sigma2rw, 0.0))
        Wprop = z_prop[cyrs] * np.sqrt(max(1.0 - sigma2rw, 0.0))

        rc = (RW[cyrs] - Wcurr).reshape(-1, 1)
        rp = (RW[cyrs] - Wprop).reshape(-1, 1)

        logLprop = float(-0.5 * (rp.T @ Q @ rp))
        logLcurr = float(-0.5 * (rc.T @ Q @ rc))
        HR = np.exp(logLprop - logLcurr)
    else:
        raise ValueError("errorpars must have length 1 (iid) or 2 (AR1).")

    # accept / reject
    accept = (rng.random() < min(1.0, float(HR)))
    return Toprop if accept else float(Tocurr)

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
