"""Canonical uncertainty distributions, nominal trajectories and transport inputs."""
from __future__ import annotations

from math import exp, isfinite
from statistics import NormalDist
from typing import Mapping
import warnings

import numpy as np

ALL = "ALL"
SHAPES = ("linear", "early", "late", "logistic", "random_walk", "almost_flat")
SUPPORTED_DISTRIBUTIONS = ("normal", "uniform", "triangular", "lognormal", "beta")
# Updated in place by parameters.py so imported references survive its reload.
TRANSPORT_SURROGATE_INPUTS = {}
# Public parameter names stay unchanged through scenarios and surrogate inputs.
TRANSPORT_BINDINGS = {
    "PASSENGER_DEMAND_GROWTH": "passenger_demand_multiplier",   # used as uncertainty
    "BIKE_ASC_SHIFT": "bike_asc_shift",                         # used as uncertainty
    # Not used as uncertainties, but kept active: these are inputs of the transport
    # surrogate (they take their nominal values). Do not comment them out.
    "PT_ASC_SHIFT": "pt_asc_shift",
    "EBIKE_SHARE": "ebike_share",
    "ROAD_FREIGHT_GROWTH": "road_freight_multiplier",
}
POST_MODELLING_TARGETS = {
    # ---- used as uncertainties ----
    "DISCOUNT_RATE", "CAPEX_MULTIPLIER",
    "C_TT_BIKE",
    "BENEFIT_HEALTH_BIKE_PER_KM",
    "CAR_CO2_KG_PER_VEHICLE_KM", "CO2_VALUE_CHF_PER_TONNE",
    "BIKE_PEAK_SHARE",
    "OPEX_MULTIPLIER",

    # ---- not used as uncertainties (values still come from NOMINAL_PARAMS) ----
    # "OPEX_RATE", "OPEX_MULTIPLIER", "C_FLEX",
    # "C_TT_PT", "C_TT_CAR", "C_TT_WALK",
    # "C_TT_PT_WAITING", "C_TT_PT_ACCESS", "C_TT_PT_TRANSFER",
    # "BENEFIT_HEALTH_WALK_PER_KM", "BENEFIT_SOCIOECONOMIC_PT_PER_TRIP",
    # "CO2_VALUE_ANNUAL_GROWTH",
    # "RAIL_ENERGY_WH_PER_GROSS_TONNE_KM", "RAIL_ELECTRICITY_CO2_G_PER_KWH", "NUMBER_OF_PEAK_HOURS",
    # "ACCIDENT_COST_MULTIPLIER", "LOCAL_AIR_COST_MULTIPLIER", "CROWDING_SLOPE", "CROWDING_MAX",
    # "C_NOISE_CAR", "C_AIR_CAR", "C_ACCIDENT_CAR", "C_NOISE_PT", "C_AIR_PT", "C_ACCIDENT_PT",
    # "TRAIN_GROSS_TONNES", "PEAK_TO_ANNUAL_TRAIN", "C_INV_STAGE1", "C_INV_STAGE2",
    # "EQUIVALENT_DAYS_PER_YEAR", "PT_CAPACITY_BASELINE", "CAPACITY_INCREASE_STAGE1",
    # "CAPACITY_INCREASE_STAGE2", "CAPACITY_INCREASE_COMBINED",
    # *(f"{mode}_PEAK_SHARE" for mode in ("PT", "CAR", "WALK")),
}
# Physical/economic domains are validation rules, not scenario clipping rules.
PARAMETER_DOMAINS = {
    key: {"minimum": 0.0} for key in POST_MODELLING_TARGETS
    if key not in {"DISCOUNT_RATE", "CO2_VALUE_ANNUAL_GROWTH"}
}
PARAMETER_DOMAINS.update({
    "PASSENGER_DEMAND_GROWTH": {"minimum": -1.0},
    "ROAD_FREIGHT_GROWTH": {"minimum": -1.0},
    "EBIKE_SHARE": {"minimum": 0.0, "maximum": 1.0},
    "DISCOUNT_RATE": {"minimum": -1.0, "strict_minimum": True},
    "CO2_VALUE_ANNUAL_GROWTH": {"minimum": -1.0, "strict_minimum": True},
    "CROWDING_MAX": {"minimum": 1.0},
    "NUMBER_OF_PEAK_HOURS": {"minimum": 1.0, "maximum": 24.0, "integer": True},
    **{key: {"minimum": 0.0, "strict_minimum": True} for key in
       ("PT_CAPACITY_BASELINE", "TRAIN_GROSS_TONNES", "EQUIVALENT_DAYS_PER_YEAR")},
    **{f"{mode}_PEAK_SHARE": {"minimum": 0.0, "maximum": 1.0, "strict_minimum": True}
       for mode in ("PT", "CAR", "BIKE", "WALK")},
})


def _validate_domain(key, values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{key} produces a nonfinite trajectory.")
    domain = PARAMETER_DOMAINS.get(key, {})
    low, high = domain.get("minimum", -np.inf), domain.get("maximum", np.inf)
    below = values <= low if domain.get("strict_minimum") else values < low
    if np.any(below | (values > high)):
        bracket = "(" if domain.get("strict_minimum") else "["
        raise ValueError(f"{key} has values outside its domain {bracket}{low}, {high}]. "
                         "Review its distribution or nominal endpoint; values are not clipped.")
    if domain.get("integer") and not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{key} must use whole numbers.")


def _configuration(registry=None, nominal=None, n_years=None, tail_probability=None):
    if any(value is None for value in (registry, nominal, n_years, tail_probability)):
        import parameters as p
        registry = p.STRUCTURAL_UNCERTAINTIES if registry is None else registry
        nominal = p.NOMINAL_PARAMS if nominal is None else nominal
        n_years = p.N_YEARS if n_years is None else n_years
        tail_probability = (p.TRANSPORT_SURROGATE_TAIL_PROBABILITY
                            if tail_probability is None else tail_probability)
    if not isinstance(n_years, int) or not 10 <= n_years <= 100:
        raise ValueError("N_YEARS must be an integer between 10 and 100.")
    return registry, nominal, n_years, float(tail_probability)


def supported_parameters(nominal=None) -> list[str]:
    if nominal is None:
        import parameters as p
        nominal = p.NOMINAL_PARAMS
    return [key for key in nominal if key in TRANSPORT_BINDINGS or key in POST_MODELLING_TARGETS]


def parameter_catalogue() -> dict:
    return {key: {"category": "transport_modelling" if key in TRANSPORT_BINDINGS else "post_modelling",
                  **PARAMETER_DOMAINS.get(key, {})}
            for key in supported_parameters()}


def select_uncertainties(selection=ALL, use="scenarios", category=None) -> list[str]:
    import parameters as p
    registry = p.STRUCTURAL_UNCERTAINTIES
    names = list(registry) if selection == ALL else ([selection] if isinstance(selection, str) else list(selection))
    if set(names) - registry.keys():
        raise KeyError(f"Unknown uncertainty parameters: {sorted(set(names) - registry.keys())}")
    if use not in ("scenarios", "sensitivity", ALL, None):
        raise ValueError("use must be scenarios, sensitivity or ALL.")
    if category not in (None, "transport_modelling", "post_modelling"):
        raise ValueError(f"Unknown category: {category!r}")
    return [name for name in names if (use in (None, ALL) or registry[name]["use"] == use)
            and (category is None or registry[name]["category"] == category)]


def _fraction(t, shape, u=None):
    if shape == "linear":
        return t
    if shape == "early":
        return t ** 0.4
    if shape == "late":
        return t ** 2.5
    if shape == "almost_flat":
        return t ** 10  # Most of the change occurs near the end of the horizon.
    if shape == "logistic":
        raw = 1.0 / (1.0 + np.exp(-10.0 * (t - 0.5)))
        lo, hi = 1 / (1 + exp(5.0)), 1 / (1 + exp(-5.0))
        return (raw - lo) / (hi - lo)
    if shape == "random_walk":
        if u is None:
            return t
        rng = np.random.default_rng(int(float(u) * (2**31 - 1)))
        walk = np.r_[0.0, np.cumsum(rng.normal(size=len(t) - 1))]
        bridge = (walk - t * walk[-1]) / np.sqrt(max(len(t) - 1, 1))
        return np.clip(t + 0.20 * bridge, 0.0, 1.0)
    raise ValueError(f"Unknown trajectory shape: {shape!r}")


def _clip(value, spec):
    return np.clip(value, float(spec.get("hard_minimum", -np.inf)), float(spec.get("hard_maximum", np.inf)))


def _distribution_value(u, spec, reference):
    distribution = spec.get("distribution", "normal")
    if distribution not in SUPPORTED_DISTRIBUTIONS:
        raise ValueError(f"Unsupported distribution: {distribution!r}")
    integer = spec.get("integer", False)
    if not isinstance(integer, bool):
        raise ValueError("integer must be True or False.")
    if integer:
        if distribution != "uniform":
            raise ValueError("integer=True is supported for uniform distributions.")
        bounds = [float(spec["minimum"]), float(spec["maximum"]), float(reference)]
        if any(not isfinite(value) or not value.is_integer() for value in bounds):
            raise ValueError("Integer uniform bounds and nominal value must be whole numbers.")
    if u is None:
        value = float(_clip(reference, spec))
        return int(value) if integer else value
    if not isfinite(float(u)) or not 0 <= float(u) <= 1:
        raise ValueError("An uncertainty quantile must lie in [0, 1].")
    q = float(np.clip(u, 1e-9, 1 - 1e-9))
    if distribution in ("normal", "lognormal"):
        sigma = float(spec.get("sigma", 0.0))
        if not isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be finite and nonnegative.")
        z = NormalDist().inv_cdf(q)
        if distribution == "normal":
            value = reference + z * sigma * (abs(reference) if spec.get("relative_sigma", False) else 1.0)
        else:
            if reference <= 0:
                raise ValueError("Lognormal nominal value must be positive.")
            statistic = spec.get("nominal_statistic", "mode")
            coefficient = {"mode": 1.0, "median": 0.0, "mean": -0.5}
            if statistic not in coefficient:
                raise ValueError("Lognormal nominal_statistic must be mode, median or mean.")
            value = reference * exp(coefficient[statistic] * sigma ** 2 + z * sigma)
    else:
        low, high = float(spec["minimum"]), float(spec["maximum"])
        if not isfinite(low) or not isfinite(high) or not low <= reference <= high:
            raise ValueError("The nominal endpoint must lie within finite distribution bounds.")
        if high == low:
            value = float(_clip(low, spec))
            return int(value) if integer else value
        if distribution == "uniform":
            value = (min(high, low + np.floor(float(u) * (high - low + 1))) if integer
                     else low + float(u) * (high - low))
        elif distribution == "triangular":
            from scipy.stats import triang
            value = low + (high - low) * triang.ppf(float(u), (reference - low) / (high - low))
        else:
            from scipy.stats import beta
            alpha, beta_shape = float(spec["alpha"]), float(spec["beta"])
            if min(alpha, beta_shape) <= 0 or not isfinite(alpha + beta_shape):
                raise ValueError("Beta shape parameters must be finite and positive.")
            value = low + (high - low) * beta.ppf(float(u), alpha, beta_shape)
    value = float(_clip(value, spec))
    return int(value) if integer else value


def _path(parameter, u, registry, nominal, n_years, shape="linear"):
    if parameter not in supported_parameters(nominal):
        raise KeyError(f"Unsupported parameter: {parameter}")
    spec = registry.get(parameter)
    base = float(nominal[parameter])
    end = float(nominal.get(parameter + "_Y40", base))
    if spec is None:
        if u is not None:
            raise ValueError(f"{parameter} has no uncertainty distribution; use u=None.")
        return base + _fraction(np.linspace(0, 1, n_years), shape) * (end - base)
    if not spec["transient"]:
        return np.full(n_years, _distribution_value(u, spec, base))
    initial = spec.get("initial_distribution")
    start_value = _distribution_value(u, initial, base) if initial else base
    end_value = _distribution_value(u, spec, end)
    return start_value + _fraction(np.linspace(0, 1, n_years), shape, u) * (end_value - start_value)


def get_trajectory(name, u=None, shape="linear", *, nominal=None):
    import parameters as p
    reference = p.NOMINAL_PARAMS if nominal is None else {**p.NOMINAL_PARAMS, **nominal}
    return _path(name, u, p.STRUCTURAL_UNCERTAINTIES, reference, p.N_YEARS, shape)


def build_paths(draws=None, shapes=None, *, nominal=None):
    import parameters as p
    draws, shapes = dict(draws or {}), dict(shapes or {})
    if draws.keys() - p.STRUCTURAL_UNCERTAINTIES.keys():
        raise KeyError(f"No uncertainty distribution for {sorted(draws.keys() - p.STRUCTURAL_UNCERTAINTIES.keys())}")
    known = supported_parameters()
    if shapes.keys() - set(known):
        raise KeyError(f"Unknown trajectory parameters: {sorted(shapes.keys() - set(known))}")
    return clip_transport_paths({key: get_trajectory(key, draws.get(key), shapes.get(key, "linear"), nominal=nominal)
                                 for key in known})


def nominal_paths(*, nominal=None):
    return build_paths(nominal=nominal)


def is_deterministic(name_or_spec):
    import parameters as p
    if isinstance(name_or_spec, str) and name_or_spec not in supported_parameters():
        raise KeyError(f"Unsupported parameter: {name_or_spec}")
    spec = p.STRUCTURAL_UNCERTAINTIES.get(name_or_spec) if isinstance(name_or_spec, str) else name_or_spec
    if spec is None:
        return True
    def zero(distribution):
        if distribution.get("distribution", "normal") in ("normal", "lognormal"):
            return float(distribution.get("sigma", 0)) == 0
        return float(distribution["minimum"]) == float(distribution["maximum"])
    return zero(spec) and (not spec.get("initial_distribution") or zero(spec["initial_distribution"]))


def target_path(paths, parameter_name, *, nominal=None):
    return np.asarray(paths[parameter_name]) if parameter_name in paths else get_trajectory(parameter_name, nominal=nominal)


def year_parameters(paths, t, params=None):
    import parameters as p
    values = dict(p.NOMINAL_PARAMS if params is None else params)
    for key in supported_parameters():
        values[key] = float(target_path(paths, key, nominal=params)[t])
    return values


def _distribution_bounds(spec, reference, tail):
    if spec.get("distribution", "normal") in ("uniform", "triangular", "beta"):
        lo, hi = float(spec["minimum"]), float(spec["maximum"])
    else:
        lo, hi = (_distribution_value(tail / 2, spec, reference),
                  _distribution_value(1 - tail / 2, spec, reference))
    return float(_clip(lo, spec)), float(_clip(hi, spec))


def transport_surrogate_inputs(*, registry=None, nominal=None, n_years=None, tail_probability=None):
    registry, nominal, _, tail = _configuration(registry, nominal, n_years, tail_probability)
    if not 0 < tail < 1:
        raise ValueError("The training tail probability must lie between 0 and 1.")
    inputs = {}
    for key in TRANSPORT_BINDINGS:
        if key not in nominal:
            raise ValueError(f"Keep a fixed general value for supported transport parameter {key}.")
        base, end = float(nominal[key]), float(nominal.get(key + "_Y40", nominal[key]))
        spec = registry.get(key)
        limits = [base, end]
        if spec:
            limits = [base, end] if spec["transient"] else [base]
            limits.extend(_distribution_bounds(spec, end if spec["transient"] else base, tail))
            if spec.get("initial_distribution"):
                limits.extend(_distribution_bounds(spec["initial_distribution"], base, tail))
        if not all(isfinite(value) for value in limits):
            raise ValueError(f"{key} requires finite training bounds.")
        low, high = min(limits), max(limits)
        if key == "EBIKE_SHARE" and not 0 <= low <= high <= 1:
            raise ValueError("EBIKE_SHARE training bounds must lie in [0, 1]; use a bounded distribution or physical hard limits.")
        if key in ("PASSENGER_DEMAND_GROWTH", "ROAD_FREIGHT_GROWTH") and low < -1:
            raise ValueError(f"{key} training range cannot imply negative demand.")
        if high - low > 1e-12:
            inputs[key] = {"minimum": low, "maximum": high, "nominal": base}
    return inputs


def transport_surrogate_training_bounds(tail_probability=None):
    return {key: (spec["minimum"], spec["maximum"])
            for key, spec in transport_surrogate_inputs(tail_probability=tail_probability).items()}


def nominal_transport_surrogate_inputs():
    import parameters  # Load project configuration before reading its derived metadata.
    return {key: spec["nominal"] for key, spec in TRANSPORT_SURROGATE_INPUTS.items()}


def complete_transport_inputs(values=None):
    import parameters as p
    values = dict(values or {})
    if values.keys() - TRANSPORT_BINDINGS.keys():
        raise ValueError(f"Unsupported transport inputs: {sorted(values.keys() - TRANSPORT_BINDINGS.keys())}")
    return {key: float(values.get(key, p.NOMINAL_PARAMS[key])) for key in TRANSPORT_BINDINGS}


def native_transport_inputs(values=None):
    """Convert canonical growth inputs to the native FSM's multiplier arguments."""
    canonical = complete_transport_inputs(values)
    native = {TRANSPORT_BINDINGS[key]: value for key, value in canonical.items()}
    native["passenger_demand_multiplier"] += 1.0
    native["road_freight_multiplier"] += 1.0
    return native


def transport_surrogate_inputs_from_uncertainty_values(values: Mapping):
    import parameters  # Load project configuration before reading its derived metadata.
    result = {}
    for key, spec in TRANSPORT_SURROGATE_INPUTS.items():
        value = float(values.get(key, spec["nominal"]))
        if not isfinite(value):
            raise ValueError(f"{key} must be finite.")
        if not spec["minimum"] - 1e-12 <= value <= spec["maximum"] + 1e-12:
            warnings.warn(f"{key} exceeds its surrogate training range; clipping to "
                          f"[{spec['minimum']}, {spec['maximum']}]. Review uncertainty tails or rebuild with wider coverage.",
                          UserWarning, stacklevel=2)
            value = float(np.clip(value, spec["minimum"], spec["maximum"]))
        result[key] = value
    return result


def clip_transport_paths(paths):
    """Use the same bounded transport paths for assignment and annual appraisal."""
    import parameters  # Load project configuration before reading its derived metadata.
    clipped = dict(paths)
    for key, spec in TRANSPORT_SURROGATE_INPUTS.items():
        if key not in clipped:
            continue
        values = np.asarray(clipped[key], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{key} must be finite.")
        if np.any((values < spec["minimum"] - 1e-12) | (values > spec["maximum"] + 1e-12)):
            warnings.warn(f"{key} exceeds its surrogate training range; clipping the trajectory to "
                          f"[{spec['minimum']}, {spec['maximum']}]. Review uncertainty tails or rebuild with wider coverage.",
                          UserWarning, stacklevel=2)
            clipped[key] = np.clip(values, spec["minimum"], spec["maximum"])
    return clipped


def transport_surrogate_inputs_from_paths(paths, t):
    import parameters  # Load project configuration before reading its derived metadata.
    return transport_surrogate_inputs_from_uncertainty_values({key: float(target_path(paths, key)[t])
                                                               for key in TRANSPORT_SURROGATE_INPUTS})


def validate_params(*, registry=None, nominal=None, n_years=None, tail_probability=None):
    registry, nominal, n_years, tail = _configuration(registry, nominal, n_years, tail_probability)
    if not 0 <= nominal["DISCOUNT_RATE"] <= 0.2:
        raise ValueError("DISCOUNT_RATE must lie between 0 and 0.20.")
    if not isfinite(nominal["EBIKE_SPEED_MULTIPLIER"]) or nominal["EBIKE_SPEED_MULTIPLIER"] <= 0:
        raise ValueError("EBIKE_SPEED_MULTIPLIER must be finite and positive.")
    days, hours = nominal["EQUIVALENT_DAYS_PER_YEAR"], nominal["NUMBER_OF_PEAK_HOURS"]
    if not isfinite(days) or days <= 0 or not isfinite(hours) or not 0 < hours <= 24:
        raise ValueError("Use positive equivalent days and 1–24 peak hours.")
    if not float(hours).is_integer():
        raise ValueError("NUMBER_OF_PEAK_HOURS must be a whole number.")
    for mode in ("PT", "CAR", "BIKE", "WALK"):
        share = nominal[f"{mode}_PEAK_SHARE"]
        if not isfinite(share) or not 0 < share <= 1.0 / hours:
            raise ValueError(f"{mode} peak share must be positive and peak hours times its share cannot exceed one.")
    for key in ("PT_HEADWAY_BASELINE", "PT_CAPACITY_BASELINE", "TRAIN_GROSS_TONNES"):
        if not isfinite(nominal[key]) or nominal[key] <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    for key in ("CAPACITY_INCREASE_STAGE1", "CAPACITY_INCREASE_STAGE2", "CAPACITY_INCREASE_COMBINED"):
        increase = nominal[key]
        if not isfinite(increase) or increase < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
    if not isfinite(nominal["OPEX_RATE"]) or nominal["OPEX_RATE"] < 0:
        raise ValueError("OPEX_RATE must be finite and nonnegative.")
    for key in ("C_TT_PT_WAITING", "C_TT_PT_ACCESS", "C_TT_PT_TRANSFER",
                "CAR_CO2_KG_PER_VEHICLE_KM", "CO2_VALUE_CHF_PER_TONNE",
                "RAIL_ENERGY_WH_PER_GROSS_TONNE_KM", "RAIL_ELECTRICITY_CO2_G_PER_KWH"):
        if not isfinite(nominal[key]) or nominal[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
    for key in ("APPRAISAL_START_YEAR", "CO2_REFERENCE_YEAR"):
        if not isfinite(nominal[key]) or not float(nominal[key]).is_integer():
            raise ValueError(f"{key} must be a calendar year.")
    if not isfinite(nominal["CO2_VALUE_ANNUAL_GROWTH"]) or nominal["CO2_VALUE_ANNUAL_GROWTH"] <= -1:
        raise ValueError("CO2_VALUE_ANNUAL_GROWTH must be finite and greater than -1.")
    for key, spec in registry.items():
        if key not in supported_parameters(nominal):
            raise ValueError(f"{key} has no supported model binding or general nominal value.")
        expected = "transport_modelling" if key in TRANSPORT_BINDINGS else "post_modelling"
        if spec.get("category") != expected or spec.get("use") not in ("scenarios", "sensitivity"):
            raise ValueError(f"{key} requires category={expected!r} and a valid use.")
        if not isinstance(spec.get("transient"), bool):
            raise ValueError(f"{key}: transient must be True or False.")
        if spec.get("integer", False) and spec["transient"]:
            raise ValueError(f"{key}: integer uncertainties use a constant path; set transient=False.")
        if key == "NUMBER_OF_PEAK_HOURS" and not spec.get("integer", False):
            raise ValueError("NUMBER_OF_PEAK_HOURS requires integer=True.")
        if key == "DISCOUNT_RATE" and spec["transient"]:
            raise ValueError("DISCOUNT_RATE uses one constant rate; set transient=False.")
        if spec.get("initial_distribution") and not spec["transient"]:
            raise ValueError(f"{key}: initial_distribution requires transient=True.")
        forbidden = {"name", "parameter_name", "surrogate_input", "trajectory_offset", "trajectory_scale",
                     "nominal", "nominal_start", "nominal_end", "sigma_start", "sigma_end"} & spec.keys()
        if forbidden:
            raise ValueError(f"{key}: remove duplicate mapping/trajectory fields {sorted(forbidden)}; use general nominal values and endpoint distribution.")
        for distribution in (spec, spec.get("initial_distribution")):
            if not distribution:
                continue
            if distribution.get("distribution", "normal") == "normal" and ("minimum" in distribution or "maximum" in distribution):
                raise ValueError(f"{key}: normal uses sigma, not minimum/maximum.")
            if distribution.get("hard_minimum", -np.inf) > distribution.get("hard_maximum", np.inf):
                raise ValueError(f"{key} has reversed hard limits.")
        # Check full support for bounded distributions and the configured
        # training-tail quantiles for unbounded distributions.
        bounded = spec.get("distribution", "normal") in ("uniform", "triangular", "beta")
        quantiles = (None, 0.0, 1.0) if bounded else (None, tail / 2, 1 - tail / 2)
        for quantile in quantiles:
            _validate_domain(key, _path(key, quantile, registry, nominal, n_years))
    for key in supported_parameters(nominal):
        _validate_domain(key, _path(key, None, {}, nominal, n_years))
    for key in TRANSPORT_BINDINGS:
        base = float(nominal[key])
        end = float(nominal.get(key + "_Y40", base))
        if not all(isfinite(value) for value in (base, end)):
            raise ValueError(f"{key} nominal values must be finite.")
        if key in ("PASSENGER_DEMAND_GROWTH", "ROAD_FREIGHT_GROWTH") and min(base, end) < -1:
            raise ValueError(f"{key} cannot imply negative demand.")
        if key == "EBIKE_SHARE" and not 0 <= min(base, end) <= max(base, end) <= 1:
            raise ValueError("EBIKE_SHARE must lie in [0, 1].")
    transport_surrogate_inputs(registry=registry, nominal=nominal, n_years=n_years, tail_probability=tail)


def configure(*, registry, nominal, n_years, tail_probability):
    """Validate project settings and refresh shared surrogate metadata once."""
    validate_params(registry=registry, nominal=nominal, n_years=n_years,
                    tail_probability=tail_probability)
    inputs = transport_surrogate_inputs(registry=registry, nominal=nominal,
                                       n_years=n_years, tail_probability=tail_probability)
    TRANSPORT_SURROGATE_INPUTS.clear()
    TRANSPORT_SURROGATE_INPUTS.update(inputs)
