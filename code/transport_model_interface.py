"""
transport_model_interface.py
============================
Notebook-facing bridge to the Canton Zürich transport model (FSM).

This module serves as a project-neutral, modular interface between Jupyter
notebooks and the packaged transport_core model, with prepared spatial zones,
multimodal skims, and logit mode choice.

WHAT THIS MODULE PROVIDES
-------------------------
1. Preflight checks for prepared transport data inputs (load_transport_context).
2. Spatial lookup utilities by municipality / region (get_zone_ids_for_municipalities).
3. Stage-aware multimodal mode choice simulation (run_transport_mode_choice).
4. Automated extraction of corridor metrics for CBA evaluation (extract_corridor_metrics).
5. Interactive dashboard widgets for exploratory analysis (mode_choice_dashboard).
6. Optional WebGL / Lonboard network map visualization (full_network_explorer).

Students do NOT need to modify this file; all project-specific interventions
and economic assumptions are configured in stages.py and parameters.py.
"""
from __future__ import annotations


from copy import deepcopy
from dataclasses import dataclass, fields, replace
import importlib
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default Constants
# ---------------------------------------------------------------------------

DEFAULT_CORRIDOR_BUFFER_M = 500.0
DEFAULT_MAX_GATES = 100
DEFAULT_GATE_SEPARATION_M = 500.0
DEFAULT_PASSTHROUGH_FRACTION = 0.05

# ---------------------------------------------------------------------------
# Global Settings & Toggles
# ---------------------------------------------------------------------------

ASSIGNMENT_SETTINGS = {
    # MSA is the teaching-model default: link flows update BPR travel times
    # explicitly. Planning uses the separately prepared OD surrogate.
    "method": "MSA",
    "force_regenerate_corridor_network": False,  # Reuse prepared roads unless a rebuild is explicitly requested.
    "max_iterations": 70,
    "min_iterations": 8,
    "stopping_rule": "road_relative_gap",
    "road_gap_threshold": 0.02,
    "check_every": 4,
    "relative_gap_threshold": 0.10,  # Optional successive-change diagnostic only.
    "od_threshold": 0.0,
    "drive_occupancy": 1.14,
    "modal_feedback": True,
}

def resolve_assignment_settings(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve the current notebook defaults and validate explicit solver options."""
    settings = {
        "stopping_rule": "successive_change", "road_gap_threshold": None,
        "check_every": 4, "flow_gap_threshold": None, "demand_gap_threshold": None,
        **ASSIGNMENT_SETTINGS,
    }
    unknown = set(overrides or {}).difference(settings)
    if unknown:
        raise ValueError(f"Unknown assignment settings: {sorted(unknown)}")
    settings.update(overrides or {})
    if str(settings["method"]).upper() != "MSA":
        raise ValueError("Native transport evaluation requires method='MSA'.")
    settings["method"] = "MSA"
    for name in ("max_iterations", "min_iterations", "check_every"):
        value = settings[name]
        if int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
        settings[name] = int(value)
    if settings["max_iterations"] < settings["min_iterations"]:
        raise ValueError("max_iterations must be at least min_iterations.")
    for name in ("relative_gap_threshold", "od_threshold", "road_gap_threshold",
                 "flow_gap_threshold", "demand_gap_threshold"):
        value = settings[name]
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative.")
    if not np.isfinite(settings["drive_occupancy"]) or settings["drive_occupancy"] <= 0:
        raise ValueError("drive_occupancy must be positive and finite.")
    if settings["stopping_rule"] not in {"successive_change", "road_relative_gap"}:
        raise ValueError("Unknown coupled-assignment stopping_rule.")
    return settings


def assignment_solver_kwargs(settings: Mapping[str, Any] | None = None, *, coupled: bool = True) -> dict[str, Any]:
    """Select the arguments supported by the fixed-demand or coupled solver."""
    resolved = resolve_assignment_settings(settings)
    names = ["drive_occupancy", "max_iterations", "min_iterations", "relative_gap_threshold", "od_threshold",
             "stopping_rule", "road_gap_threshold", "check_every"]
    if coupled:
        names += ["flow_gap_threshold", "demand_gap_threshold"]
    return {name: resolved[name] for name in names}


def assignment_settings_match(
    diagnostics: Mapping[str, Any],
    settings: Mapping[str, Any] | None = None,
    *,
    modal_feedback: bool | None = None,
    passthrough_fraction: float | None = None,
) -> bool:
    """Check solver settings before reusing a result; callers also check its physical inputs."""
    requested = resolve_assignment_settings(settings)
    expected = {key: requested[key] for key in ("max_iterations", "min_iterations", "drive_occupancy")}
    expected["od_threshold_veh_h"] = requested["od_threshold"]
    if diagnostics.get("stopping_rule") != requested["stopping_rule"]:
        return False
    if requested["stopping_rule"] == "road_relative_gap":
        expected["road_gap_threshold"] = (
            requested["road_gap_threshold"] if requested["road_gap_threshold"] is not None
            else requested["relative_gap_threshold"]
        )
        expected["check_every"] = requested["check_every"]
    else:
        expected["relative_l1_threshold"] = requested["relative_gap_threshold"]
        if modal_feedback:
            expected.update({key: requested[key] if requested[key] is not None
                             else requested["relative_gap_threshold"]
                             for key in ("flow_gap_threshold", "demand_gap_threshold")})
    if modal_feedback is not None and diagnostics.get("modal_feedback") != modal_feedback:
        return False
    if passthrough_fraction is not None:
        expected["passthrough_fraction"] = passthrough_fraction
    return all(
        key in diagnostics and diagnostics[key] is not None
        and np.isclose(float(diagnostics[key]), float(value), rtol=1e-9, atol=1e-12)
        for key, value in expected.items()
    )


# ---------------------------------------------------------------------------
# NumPy unpickling compatibility shim (handles NumPy 1.x vs 2.x pickles)
# ---------------------------------------------------------------------------

from transport_core import _alias_numpy_core_for_pickle_compat


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TransportContext:
    """Prepared inputs and modules shared across notebook transport runs."""
    project_root: Path
    model_dir: Path
    zones: Any
    baseline_od: pd.DataFrame
    road_background_od: pd.DataFrame
    travel_times: dict[str, pd.DataFrame]
    lengths: dict[str, pd.DataFrame]
    assignment_network: dict[str, Any]
    modules: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in TransportContext. Available keys: {list(self.keys())}")

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def __iter__(self):
        return iter(self.keys())

    def keys(self) -> list[str]:
        return [f.name for f in fields(self)]

    def values(self) -> list[Any]:
        return [getattr(self, f.name) for f in fields(self)]

    def items(self) -> list[tuple[str, Any]]:
        return [(f.name, getattr(self, f.name)) for f in fields(self)]

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class ModeChoiceResult:
    """Memory-conscious output of a multimodal mode-choice run."""
    drive_od: pd.DataFrame
    od_by_mode: dict[str, pd.DataFrame]
    summary: pd.DataFrame
    scenario: dict[str, Any]
    travel_times: dict[str, pd.DataFrame]
    lengths: dict[str, pd.DataFrame]
    cache_key: tuple[Any, ...]
    assigned_edges: Any = None
    assigned_metadata: dict = None
    # Preserve the stage-adjusted, uncongested skim when a coupled assignment
    # returns an updated congested skim. This makes later-year reruns stable.
    uncongested_travel_times: dict[str, pd.DataFrame] | None = None
    # Three stage-adjusted components before the e-bike time adjustment.
    technology_base_times: dict[str, pd.DataFrame] | None = None

    @property
    def stage(self) -> int:
        if isinstance(self.scenario, dict):
            return int(self.scenario.get("stage", 0))
        return 0

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in ModeChoiceResult. Available keys: {list(self.keys())}")

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def __iter__(self):
        return iter(self.keys())

    def keys(self) -> list[str]:
        return [f.name for f in fields(self)] + ["stage"]

    def values(self) -> list[Any]:
        return [getattr(self, k) for k in self.keys()]

    def items(self) -> list[tuple[str, Any]]:
        return [(k, getattr(self, k)) for k in self.keys()]

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.keys()}


@dataclass
class CorridorContext:
    """Topology, cordon gates, and OD mappings for one model corridor.

    Keeping these related objects together prevents notebook cells from
    rebuilding slightly different corridor definitions for maps and audit
    tables.  The class is project-neutral: MehrSpur is selected through
    ``parameters.CORRIDOR_MUNICIPALITIES`` rather than hard-coded here.
    """

    polygon: Any
    polygon_gdf: Any
    zones: Any
    zone_ids: list[str]
    edges: Any
    nodes: Any
    gates: Any
    zone_node_map: dict[str, int]
    external_zone_to_entry_gate: dict[str, str]
    external_zone_to_exit_gate: dict[str, str]
    metadata: dict[str, Any]


@dataclass
class AssignmentResult:
    """One corridor assignment plus the data needed to audit it."""

    links: Any
    history: pd.DataFrame
    diagnostics: dict[str, Any]
    demand_matrix: pd.DataFrame
    demand_breakdown: pd.DataFrame
    gate_totals: pd.DataFrame
    # Populated only by the coupled congestion--mode-choice calculation.
    mode_result: ModeChoiceResult | None = None
    # Directional routing-zone/gate delay target; unavailable for old caches.
    local_od_delay_min: pd.DataFrame | None = None


# =============================================================================
# 1. PREFLIGHT AND CONTEXT LOADING
# =============================================================================

def configured_input_paths(project_root: str | Path) -> dict[str, Path]:
    """Read prepared input locations from the transport core configuration."""
    import importlib.util

    config_file = Path(project_root).resolve() / "code" / "transport_core" / "config.py"
    spec = importlib.util.spec_from_file_location("_transport_input_configuration", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return {key: Path(getattr(config, name)).resolve() for key, name in {
        "zones": "ZONES_FILE", "demand": "DEMAND_PACKAGE_FILE", "skims": "SKIM_PACKAGE_FILE",
        "network": "ASSIGNMENT_NETWORK_FILE", "mode_choice": "MODE_CHOICE_FILE",
        "lookups": "LOOKUP_PACKAGE_FILE",
    }.items()}


def context_cache_status(project_root: str | Path) -> dict[str, Any]:
    """Check context-cache freshness using file metadata, without loading matrices."""
    import json

    root = Path(project_root).resolve()
    cache_file = root / "cache" / "tmi_context.pkl"
    metadata_file = cache_file.with_suffix(".json")
    paths = configured_input_paths(root)
    input_files = [paths[name] for name in ("zones", "demand", "skims", "network")]
    identity = {"schema": 1, "inputs": {
        str(path): {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        if path.exists() else None for path in input_files
    }}
    ready, reason = True, "Prepared transport inputs match the cached context."
    if not cache_file.exists():
        ready, reason = False, "Transport context cache is missing."
    elif metadata_file.exists():
        try:
            recorded = json.loads(metadata_file.read_text(encoding="utf-8"))
            if recorded != identity:
                ready, reason = False, "Prepared transport inputs changed since the context was cached."
        except (OSError, ValueError) as error:
            ready, reason = False, f"Transport context cache metadata is unreadable: {error}"
    elif any(path.exists() and path.stat().st_mtime_ns > cache_file.stat().st_mtime_ns
             for path in input_files):
        ready, reason = False, "Prepared transport inputs are newer than the existing context cache."
    else:
        reason = "Existing context cache predates metadata tracking; prepared inputs are not newer."
    return {"ready": ready, "reason": reason, "identity": identity,
            "metadata_present": metadata_file.exists()}


def transport_input_status(project_root: str | Path) -> pd.DataFrame:
    """Check availability of prepared FSM transport model input files."""
    root = Path(project_root).resolve()
    paths = configured_input_paths(root)

    rows = [
        ("Zones", paths["zones"], True, "Spatial zones & attributes"),
        ("Passenger & Background Demand", paths["demand"], True, "OD matrices"),
        ("Multimodal Skims", paths["skims"], True, "Travel times & distances"),
        ("Intervention Lookups", paths["lookups"], True, "Municipality and stop selectors"),
        ("Road Network", paths["network"], True, "Road links and nodes"),
        ("Mode-choice Parameters", paths["mode_choice"], True, "Logit coefficients"),
    ]

    status = pd.DataFrame(rows, columns=["item", "path", "required", "purpose"])
    status["present"] = status["path"].map(lambda p: Path(p).exists())
    status["relative_path"] = status["path"].map(
        lambda p: str(Path(p).relative_to(root)) if Path(p).is_relative_to(root) else str(p)
    )
    return status[["item", "required", "present", "relative_path", "purpose"]]


def readiness_flags(status: pd.DataFrame) -> dict[str, bool]:
    """Summarise data readiness."""
    by_item = status.set_index("item")["present"].to_dict()
    ready = all(by_item.values())
    return {
        "network": bool(by_item.get("Zones", False) and by_item.get("Road Network", False)),
        "mode_choice": ready,
    }


def _import_transport_modules() -> dict[str, Any]:
    """Load namespaced transport modules without changing the import path."""
    module_names = (
        "config",
        "input_packages",
        "zoning",
        "travel_times",
        "interventions",
        "mode_choice_zurich"
    )
    modules: dict[str, Any] = {}
    for name in module_names:
        modules[name] = importlib.import_module(f"transport_core.{name}")
    return modules

def _load_assignment_network(prepared_dir: Path, network_file: Path | None = None) -> dict[str, Any]:
    """Load prepared road network without importing pandana/network.py."""
    network_file = network_file or prepared_dir / "assignment_network.pkl"
    if not network_file.exists():
        return {}
    import pickle
    with open(network_file, "rb") as f:
        return pickle.load(f)


def configured_detailed_network_path(
    project_root: str | Path, *, must_exist: bool = True,
) -> Path | None:
    """Return the detailed-network cache selected by the case-study config.

    ``parameters.DETAILED_NETWORK_FILE`` is the authoritative project-specific
    selector. The historical filenames remain fallbacks for projects that have
    not opted in. An explicit but missing path fails loudly so a case study
    cannot silently train on another corridor or on the macro network. Notebook
    preparation uses ``must_exist=False`` to obtain the target for a new cache.
    """
    import parameters as p

    root = Path(project_root).resolve()
    configured = getattr(p, "DETAILED_NETWORK_FILE", None)
    if configured:
        candidate = Path(configured)
        target = candidate if candidate.is_absolute() else root / candidate
        if must_exist and not target.is_file():
            raise FileNotFoundError(
                f"Configured detailed network does not exist: {target}"
            )
        return target.resolve()

    for relative in (
        "data/processed/detailed_network.pkl",
        "data/processed/mehrspur_detailed_network.pkl",
    ):
        candidate = root / relative
        if candidate.is_file():
            return candidate.resolve()
    return None


def _detailed_network_scope(
    corridor_municipalities: Sequence[str] | None = None,
    buffer_m: float | None = None,
) -> dict[str, Any]:
    """Describe the configured footprint of a detailed road network."""
    import parameters as p

    mode = getattr(p, "CORRIDOR_DEFINITION_MODE", "municipalities")
    if mode == "zones":
        selected = getattr(p, "CORRIDOR_ZONE_IDS", [])
        scope = {"mode": mode, "zone_ids": sorted(set(map(str, selected)))}
    elif mode == "municipalities":
        selected = p.CORRIDOR_MUNICIPALITIES if corridor_municipalities is None else corridor_municipalities
        scope = {"mode": mode, "municipalities": sorted(set(map(str, selected)))}
    else:
        raise ValueError("CORRIDOR_DEFINITION_MODE must be 'zones' or 'municipalities'.")
    if not selected:
        raise ValueError(f"The corridor {mode} selector is empty.")
    if buffer_m is not None:
        scope["buffer_m"] = float(buffer_m)
    return scope


def _validate_detailed_network(
    network: Mapping[str, Any],
    *,
    corridor_municipalities: Sequence[str] | None = None,
    buffer_m: float | None = None,
) -> None:
    """Reject caches from another corridor, allowing matching legacy municipality caches."""
    if not isinstance(network, Mapping) or not {"edges", "nodes", "zone_node_map"}.issubset(network):
        raise ValueError("Detailed network must contain edges, nodes and zone_node_map.")
    requested = _detailed_network_scope(corridor_municipalities, buffer_m)
    metadata = network.get("metadata", {})
    recorded = metadata.get("corridor_scope")
    if recorded is None:
        legacy_municipalities = metadata.get("corridor_municipalities")
        if requested["mode"] != "municipalities" or legacy_municipalities is None:
            raise ValueError("Detailed network has no matching corridor scope. Rebuild it in Notebook 02.")
        recorded = {"mode": "municipalities", "municipalities": sorted(set(map(str, legacy_municipalities)))}
    selector = "zone_ids" if requested["mode"] == "zones" else "municipalities"
    if recorded.get("mode") != requested["mode"] or sorted(set(map(str, recorded.get(selector, [])))) != requested[selector]:
        raise ValueError("Detailed network belongs to a different corridor. Select or rebuild the correct cache in Notebook 02.")
    if "buffer_m" in requested and "buffer_m" in recorded and requested["buffer_m"] != float(recorded["buffer_m"]):
        raise ValueError("Detailed network uses a different corridor buffer. Rebuild it in Notebook 02.")


def _detailed_network_file_identity(path: Path) -> dict[str, Any]:
    """Identify an already loaded cache without rereading its network arrays."""
    stat = path.stat()
    return {"path": path.resolve().as_posix(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _read_detailed_network(
    path: Path,
    *,
    corridor_municipalities: Sequence[str] | None = None,
    buffer_m: float | None = None,
) -> dict[str, Any]:
    import pickle

    try:
        with path.open("rb") as handle:
            network = pickle.load(handle)
        _validate_detailed_network(network, corridor_municipalities=corridor_municipalities, buffer_m=buffer_m)
    except Exception as error:
        raise ValueError(f"Cannot use detailed network {path}: {error}") from error
    # In-memory provenance only; loading a cache does not rewrite it.
    network.setdefault("metadata", {})["_cache_source"] = _detailed_network_file_identity(path)
    return network


def load_transport_context(
    project_root: str | Path,
    *,
    require_mode_choice: bool = True,
    read_only: bool = False,
    allow_missing_detailed_network: bool = False,
) -> TransportContext:
    """Load prepared inputs, refreshing their cache only when inputs change.

    ``read_only=True`` permits loading but never creates directories or writes caches.
    Notebook 02 may set ``allow_missing_detailed_network=True`` while preparing a
    configured cache for the first time. Existing invalid caches still raise.
    """
    import pickle
    import json

    root = Path(project_root).resolve()
    import parameters as p
    target_file = configured_detailed_network_path(root, must_exist=not allow_missing_detailed_network)
    detailed_network = None
    if target_file is not None and target_file.exists():
        try:
            detailed_network = _read_detailed_network(target_file)
            print(f"[OK] Automatically loaded detailed corridor network from {target_file.name}")
        except ValueError as error:
            if getattr(p, "DETAILED_NETWORK_FILE", None):
                raise
            print(f"Warning: {error} Using the prepared macroscopic network.")
    elif target_file is not None:
        print(f"Detailed network will be prepared in Notebook 02: {target_file}")
    status = transport_input_status(root)
    flags = readiness_flags(status)

    if require_mode_choice and not flags["mode_choice"]:
        missing = status.loc[status["required"] & ~status["present"], "relative_path"]
        raise FileNotFoundError("Missing prepared transport inputs: " + ", ".join(missing.astype(str)))

    model_dir = root / "code" / "transport_core"
    prepared_dir = root / "data" / "transport" / "prepared"
    modules = _import_transport_modules()

    cache_dir = root / "cache"
    cache_file = cache_dir / "tmi_context.pkl"
    cache_metadata = cache_dir / "tmi_context.json"
    input_paths = configured_input_paths(root)
    cache_status = context_cache_status(root)
    cache_identity = cache_status["identity"]

    def write_cache_metadata():
        if not read_only:
            cache_dir.mkdir(exist_ok=True)
            temporary = cache_metadata.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(cache_identity, indent=2) + "\n", encoding="utf-8")
            temporary.replace(cache_metadata)

    def _get_active_network(macro_net, loaded_zones):
        nonlocal detailed_network
        if detailed_network is None:
            return macro_net
        if detailed_network.get("metadata", {}).get("merge_policy") != _DETAILED_NETWORK_MERGE_POLICY:
            metadata = {key: value for key, value in detailed_network.get("metadata", {}).items() if key != "_cache_source"}
            buffer_m = float(metadata.get("corridor_scope", {}).get("buffer_m", 800.0))
            core_zones = _get_core_zones(loaded_zones, getattr(p, "CORRIDOR_MUNICIPALITIES", []))
            detailed_network = _merge_detailed_road_network(
                macro_net, detailed_network, loaded_zones, core_zones,
                core_zones.geometry.union_all().buffer(buffer_m), metadata=metadata,
            )
            if not read_only:
                temporary = target_file.with_suffix(".tmp.pkl")
                with temporary.open("wb") as handle:
                    pickle.dump(detailed_network, handle, protocol=pickle.HIGHEST_PROTOCOL)
                temporary.replace(target_file)
                detailed_network["metadata"]["_cache_source"] = _detailed_network_file_identity(target_file)
            print("[OK] Updated detailed-network zone attachments from existing inputs.")
        return detailed_network

    if cache_file.exists():
        try:
            if not cache_status["ready"]:
                raise ValueError(cache_status["reason"])
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
            required = {"zones", "baseline_od", "road_background", "travel_times", "lengths", "assignment_network"}
            if not required.issubset(cached_data):
                raise ValueError("transport context cache has an unsupported schema")
            # Adopt an existing compatible pickle without changing its contents.
            if not cache_metadata.exists():
                write_cache_metadata()
            return TransportContext(
                project_root=root,
                model_dir=model_dir,
                zones=cached_data["zones"],
                baseline_od=cached_data["baseline_od"],
                road_background_od=cached_data["road_background"],
                travel_times=cached_data["travel_times"],
                lengths=cached_data["lengths"],
                assignment_network=_get_active_network(cached_data["assignment_network"], cached_data["zones"]),
                modules=modules,
            )
        except Exception as e:
            print(f"Warning: Failed to load cached context ({e}). Regenerating...")

    assignment_network = _load_assignment_network(prepared_dir, input_paths["network"])
    zones, baseline_od, road_background = modules["zoning"].load_model_inputs()
    travel_times, lengths = modules["travel_times"].load_travel_times(zones)

    cached_data = {
        "zones": zones,
        "baseline_od": baseline_od,
        "road_background": road_background,
        "travel_times": travel_times,
        "lengths": lengths,
        "assignment_network": assignment_network,
    }
    try:
        if not read_only:
            cache_dir.mkdir(exist_ok=True)
            temporary = cache_file.with_suffix(".pkl.tmp")
            with open(temporary, "wb") as f:
                pickle.dump(cached_data, f)
            temporary.replace(cache_file)
            write_cache_metadata()
    except Exception as e:
        print(f"Warning: Failed to write context cache ({e}).")

    return TransportContext(
        project_root=root,
        model_dir=model_dir,
        zones=zones,
        baseline_od=baseline_od,
        road_background_od=road_background,
        travel_times=travel_times,
        lengths=lengths,
        assignment_network=_get_active_network(assignment_network, zones),
        modules=modules,
    )


# =============================================================================
# 2. SPATIAL HELPERS & CORRIDOR SELECTION
# =============================================================================

def resolve_corridor_zone_ids(
    zones: pd.DataFrame,
    municipality_names: Sequence[str] | set[str] | None = None,
) -> list[str]:
    """Resolve the configured zone/municipality selector and reject unknown entries."""
    import parameters as p

    mode = getattr(p, "CORRIDOR_DEFINITION_MODE", "municipalities")
    if mode == "zones":
        selected = list(map(str, getattr(p, "CORRIDOR_ZONE_IDS", [])))
        column = zones["grid_id"].astype(str)
    elif mode == "municipalities":
        selected = list(municipality_names if municipality_names is not None else p.CORRIDOR_MUNICIPALITIES)
        column = zones["municipality_name"].astype(str)
    else:
        raise ValueError("CORRIDOR_DEFINITION_MODE must be 'zones' or 'municipalities'.")
    if not selected:
        raise ValueError(f"The corridor {mode} selector is empty.")
    unknown = set(selected).difference(column)
    if unknown:
        raise ValueError(f"Unknown corridor {mode}: {sorted(unknown)}")
    return zones.loc[column.isin(selected), "grid_id"].astype(str).tolist()


def get_zone_ids_for_municipalities(
    context: TransportContext | Any,
    municipality_names: list[str] | set[str] | None = None,
) -> list[str]:
    """Return a list of grid_id strings matching the given municipality names or explicit corridor zones."""
    zones = context.zones if hasattr(context, "zones") else context
    return resolve_corridor_zone_ids(zones, municipality_names)


def resolve_area_zone_ids(
    zones: pd.DataFrame,
    area_names: Sequence[str],
    *,
    allowed_zone_ids: Sequence[str] | set[str] | None = None,
) -> dict[str, list[str]]:
    """Resolve municipality or quartier labels to disjoint FSM zone lists."""
    requested = list(dict.fromkeys(map(str, area_names)))
    if not requested:
        raise ValueError("At least one municipality or quartier must be supplied.")

    required = {"grid_id", "municipality_name"}
    missing = required.difference(zones.columns)
    if missing:
        raise ValueError(f"Zone data are missing columns: {sorted(missing)}")

    zone_table = zones[["grid_id", "municipality_name"]].copy()
    zone_table["grid_id"] = zone_table["grid_id"].astype(str)
    zone_table["municipality_name"] = zone_table["municipality_name"].astype(str)
    if "city_quartier" in zones.columns:
        zone_table["city_quartier"] = zones["city_quartier"].fillna("").astype(str)
    else:
        zone_table["city_quartier"] = ""

    allowed = None if allowed_zone_ids is None else set(map(str, allowed_zone_ids))
    result: dict[str, list[str]] = {}
    assigned: dict[str, str] = {}
    for name in requested:
        matching = zone_table.loc[
            zone_table["municipality_name"].eq(name)
            | zone_table["city_quartier"].eq(name),
            "grid_id",
        ]
        if allowed is not None:
            matching = matching.loc[matching.isin(allowed)]
        zone_ids = list(dict.fromkeys(matching))
        if not zone_ids:
            raise ValueError(f"No corridor zones matched area {name!r}.")
        for zone_id in zone_ids:
            previous = assigned.get(zone_id)
            if previous is not None:
                raise ValueError(
                    f"Corridor areas {previous!r} and {name!r} overlap at zone {zone_id!r}."
                )
            assigned[zone_id] = name
        result[name] = zone_ids
    return result


def _get_core_zones(zones: pd.DataFrame, corridor_municipalities: list[str]) -> pd.DataFrame:
    """Get core network zones based on the student's chosen definition mode in parameters.py."""
    zone_ids = resolve_corridor_zone_ids(zones, corridor_municipalities)
    return zones.loc[zones["grid_id"].astype(str).isin(zone_ids)].copy()


# =============================================================================
# 3. NATIVE FSM SCENARIO GENERATION & MODE CHOICE
# =============================================================================


def _corridor_mode_summary(
    od_by_mode: dict[str, pd.DataFrame],
    corridor_zone_ids: list[str] | set[str] | None = None,
) -> pd.DataFrame:
    """Summarise modal split across all OD pairs touching the corridor."""
    reference = next(iter(od_by_mode.values()))
    labels = reference.index.astype(str)

    if corridor_zone_ids is not None and len(corridor_zone_ids) > 0:
        internal = np.isin(labels, np.asarray(list(corridor_zone_ids), dtype=str))
        involved = internal[:, None] | internal[None, :]
    else:
        involved = np.ones((len(labels), len(labels)), dtype=bool)

    mode_matrices = {
        "Car (Driver)": od_by_mode["drive"],
        "Public Transport": od_by_mode["pt_walk"] + od_by_mode["pt_bike"],
        "Bicycle": od_by_mode["bike"],
        "Walking": od_by_mode["walk"],
    }

    rows = []
    for mode, frame in mode_matrices.items():
        val = frame.to_numpy(dtype=float)
        rows.append({
            "mode": mode,
            "trips": float(val[involved].sum()),
        })

    summary = pd.DataFrame(rows)
    total_trips = float(summary["trips"].sum())
    summary["share"] = summary["trips"] / max(total_trips, 1e-9)
    return summary


def run_transport_mode_choice(
    context: TransportContext,
    *,
    stage: int,
    stage_specs: Mapping[int, Mapping[str, Any]],
    corridor_zone_ids: list[str] | set[str] | None = None,
    corridor_municipalities: list[str] | None = None,
    trip_rate_multiplier: float = 1.0,
    demand_multiplier: float | None = None,
    pt_asc_shift: float = 0.0,
    bike_asc_shift: float = 0.0,
    ebike_share: float | None = None,
) -> ModeChoiceResult:
    """
    Run 5-alternative mode choice across Canton Zürich using fixed skims and stage specs.

    Parameters
    ----------
    context : TransportContext
        Loaded transport model context.
    stage : int
        Active stage ID (0, 1, 2, ...).
    stage_specs : Mapping
        Stage specifications dict from stages.py.
    corridor_zone_ids : list[str] | None
        Optional subset of zone IDs for localized corridor summary.
    demand_multiplier : float | None
        Uniform multiplier of every baseline OD pair. If omitted, the
        equivalent trip_rate_multiplier argument is used.
    """
    stage = int(stage)

    if not isinstance(stage_specs, dict):
        raise TypeError(
            f"Expected 'stage_specs' to be a dictionary of stage definitions, "
            f"but got {type(stage_specs).__name__}. "
            f"\nDid you pass the 'stages' module instead of calling stages.get_stages(params)? "
            f"\nExample fix: stage_specs = stages.get_stages(p.NOMINAL_PARAMS)"
        )

    if stage not in stage_specs:
        raise KeyError(f"Unknown stage: {stage}. Available stages: {list(stage_specs.keys())}")

    mult = demand_multiplier if demand_multiplier is not None else trip_rate_multiplier
    trip_multiplier = float(mult)
    if not np.isfinite(trip_multiplier) or trip_multiplier < 0:
        raise ValueError("demand_multiplier must be finite and nonnegative.")
    modules = context.modules

    # Pass down the native lists directly from stages.py
    stage_spec = dict(stage_specs[int(stage)])
    scenario = {
        "name": f"stage_{stage}",
        "stage": stage,
    }

    # 1. Base values from stage spec
    for key, value in stage_spec.items():
        if key not in ("name",):
            scenario[key] = value
    scenario["demand_multiplier"] = trip_multiplier

    # 2. Overrides from explicit function arguments (e.g. from the dashboard)
    if ebike_share is not None:
        scenario["ebike_share"] = float(ebike_share)

    total_od = context.baseline_od * trip_multiplier

    travel_times = modules["travel_times"].apply_perceived_time_policies(
        context.travel_times,
        context.zones,
    )
    lengths = {key: value.copy() for key, value in context.lengths.items()}

    # Apply interventions directly using the native FSM schema
    intervention_keys = ["bike_highways", "railway_expansions", "mobility_hubs", "road_capacity"]
    has_interventions = any(scenario.get(k) for k in intervention_keys)
    if has_interventions:
        travel_times, lengths, _ = modules["interventions"].apply_interventions(
            travel_times,
            lengths,
            scenario,
            zones=context.zones,
        )

    technology_base_times = {key: travel_times[key] for key in
                             ("bike", "access_pt_bike", "egress_pt_bike")
                             if key in travel_times}
    travel_times, _ = modules["travel_times"].apply_ebike_share(
        travel_times,
        float(scenario.get("ebike_share", 0.0)),
        speed_multiplier=float(stage_spec.get("EBIKE_SPEED_MULTIPLIER", 1.5)),
    )
    # Apply fixed minutes after technology adjustments so a bike saving is not
    # rescaled by the e-bike share. Exact MSA and surrogate use this same path.
    # Existing external passengers are appraised later, outside mode choice.
    from additional.section_flows import apply_section_time_saving
    travel_times, section_metadata = apply_section_time_saving(
        context, travel_times, stage_spec
    )
    scenario.update(section_metadata)

    parameters = modules["mode_choice_zurich"].load_mode_choice_parameters()
    # Override native parameters if provided in stage_specs
    for k in parameters:
        if k in stage_spec:
            parameters[k] = stage_spec[k]
    for asc_name in ("ASC_PT_WALK", "ASC_PT_BIKE"):
        parameters[asc_name] = float(parameters[asc_name]) + float(pt_asc_shift)
        scenario[asc_name] = parameters[asc_name]
    # ASC_CAR remains the utility reference.  A positive shift therefore
    # represents greater bicycle acceptance relative to the car alternative.
    parameters["ASC_BIKE"] = float(parameters["ASC_BIKE"]) + float(bike_asc_shift)
    scenario["ASC_BIKE"] = parameters["ASC_BIKE"]
    od_by_mode, _ = modules["mode_choice_zurich"].mode_split_aggregated(
        travel_times,
        lengths,
        total_od,
        walk_allowed_mask=modules["travel_times"].standalone_walk_mask(
            lengths,
            context.zones,
        ),
        parameters=parameters,
    )

    summary = _corridor_mode_summary(od_by_mode, corridor_zone_ids)

    cache_key = (
        stage,
        round(trip_multiplier, 6),
        round(float(pt_asc_shift), 6),
        round(float(bike_asc_shift), 6),
        round(float(scenario.get("ebike_share", 0.0)), 6),
        scenario.get("section_signature"),
    )

    assigned_edges = None
    assigned_metadata = None
    uncongested_times = dict(travel_times)

    return ModeChoiceResult(
        drive_od=od_by_mode["drive"],
        od_by_mode=od_by_mode,
        summary=summary,
        scenario=scenario,
        travel_times=travel_times,
        lengths=lengths,
        cache_key=cache_key,
        assigned_edges=assigned_edges,
        assigned_metadata=assigned_metadata,
        uncongested_travel_times=uncongested_times,
        technology_base_times=technology_base_times,
    )


def _technology_base_components(context, mode_result):
    """Recover stage-adjusted cycling components before any e-bike scaling."""
    stored = getattr(mode_result, "technology_base_times", None)
    if stored is not None:
        return stored
    modules = context.modules
    times = modules["travel_times"].apply_perceived_time_policies(context.travel_times, context.zones)
    if any(mode_result.scenario.get(key) for key in
           ("bike_highways", "railway_expansions", "mobility_hubs", "road_capacity")):
        times, _, _ = modules["interventions"].apply_interventions(
            times, {key: value.copy() for key, value in context.lengths.items()},
            mode_result.scenario, zones=context.zones,
        )
    return {key: times[key] for key in ("bike", "access_pt_bike", "egress_pt_bike") if key in times}


def mode_choice_with_ebike_share(context, mode_result, ebike_share, corridor_zone_ids=None):
    """Recalculate mode choice after applying the current e-bike share once."""
    from dataclasses import replace
    from additional.section_flows import apply_section_time_saving, section_config

    share = float(ebike_share)
    if not np.isfinite(share) or not 0.0 <= share <= 1.0:
        raise ValueError("E-bike share must be finite and between zero and one.")
    if share == float(mode_result.scenario.get("ebike_share", 0.0)):
        return mode_result
    technology_base = _technology_base_components(context, mode_result)
    times = dict(mode_result.uncongested_travel_times or mode_result.travel_times)
    times.update(technology_base)
    times, _ = context.modules["travel_times"].apply_ebike_share(
        times, share, speed_multiplier=float(mode_result.scenario.get("EBIKE_SPEED_MULTIPLIER", 1.5)),
    )
    scenario = {**mode_result.scenario, "ebike_share": share}
    if section_config(scenario.get("section_config"))["mode"] == "BIKE":
        times, _ = apply_section_time_saving(context, times, scenario)
    total_od = sum(mode_result.od_by_mode.values())
    native = context.modules["mode_choice_zurich"]
    parameters = native.load_mode_choice_parameters()
    parameters.update({key: scenario[key] for key in parameters if key in scenario})
    od_by_mode, _ = native.mode_split_aggregated(
        times, mode_result.lengths, total_od,
        walk_allowed_mask=context.modules["travel_times"].standalone_walk_mask(mode_result.lengths, context.zones),
        parameters=parameters,
    )
    return replace(mode_result, drive_od=od_by_mode["drive"], od_by_mode=od_by_mode,
                   summary=_corridor_mode_summary(od_by_mode, corridor_zone_ids), scenario=scenario,
                   travel_times=times, uncongested_travel_times=times,
                   technology_base_times=technology_base,
                   assigned_edges=None, assigned_metadata=None,
                   cache_key=tuple(mode_result.cache_key) + ("ebike_share", share))


# =============================================================================
# 4. CORRIDOR METRICS EXTRACTION (FOR CBA CALCULATOR)
# =============================================================================

def _welfare_od_sample(
    context: TransportContext,
    mode_result: ModeChoiceResult,
    corridor_zone_ids: Sequence[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Aligned passenger OD quantities and per-trip hours for time welfare.

    Keep all five modes, including both PT access submodes, separate. These
    temporary arrays are reduced to matched moments when preparing the response table.
    The appraisal includes trips with either endpoint in the corridor and
    covered through journeys of the selected section mode. Background
    vehicles affect congestion, not passenger
    welfare quantities. PT components include access, egress and walking at
    transfers, in addition to in-vehicle time and waiting.
    """
    import hashlib
    import json

    labels = pd.Index(sorted(mode_result.od_by_mode["drive"].index.astype(str)))
    internal = np.isin(labels, list(map(str, corridor_zone_ids))) if (
        corridor_zone_ids is not None and len(corridor_zone_ids) > 0
    ) else None
    cordon_mask = (internal[:, None] | internal[None, :]) if internal is not None else np.ones(
        (len(labels), len(labels)), dtype=bool
    )
    from additional.section_flows import intervention_masks, physical_car_times, section_welfare_times
    section_masks = intervention_masks(labels, mode_result.scenario.get("section_config"))
    mask = cordon_mask.copy()
    for covered in section_masks.values():
        mask |= covered

    def values(frame: pd.DataFrame | None, *, optional: bool = False) -> np.ndarray:
        if frame is None:
            if optional:
                return np.zeros(int(mask.sum()), dtype=float)
            raise ValueError("A required OD quantity or time skim is missing for matched welfare.")
        aligned = frame.rename(index=str, columns=str).reindex(index=labels, columns=labels)
        result = aligned.to_numpy(dtype=float)[mask]
        if not np.isfinite(result).all():
            raise ValueError("Matched OD welfare requires finite, aligned quantities and time skims.")
        return result

    quantities = {}
    for key, mode in (("car", "drive"), ("pt_walk", "pt_walk"),
                      ("pt_bike", "pt_bike"), ("bike", "bike"), ("walk", "walk")):
        covered = section_masks.get(mode, section_masks.get(key, False))
        # A PT counting section does not expand unrelated car/bike/walk welfare.
        quantities[key] = values(mode_result.od_by_mode[mode]) * (cordon_mask | covered)[mask]
    if any((quantity < 0.0).any() for quantity in quantities.values()):
        raise ValueError("Matched OD welfare requires nonnegative passenger quantities.")
    times = mode_result.travel_times
    uncongested = mode_result.uncongested_travel_times or times
    components = {
        # Connector/parking penalties in the mode-choice skim are perceived
        # costs, not observed journey minutes. Keep the physical skim for CBA.
        "car_freeflow": values(physical_car_times(context, mode_result)) / 60.0,
        "car_delay": np.maximum(
            values(times.get("drive", times.get("car")))
            - values(uncongested.get("drive", uncongested.get("car"))), 0.0
        ) / 60.0,
    }
    for mode in ("pt_walk", "pt_bike"):
        components[f"{mode}_ivt"] = values(times.get(f"ivt_{mode}", times.get(mode))) / 60.0
        components[f"{mode}_wait"] = (
            values(times.get(f"initial_wait_{mode}")) + values(times.get(f"transfer_wait_{mode}"), optional=True)
        ) / 60.0
        components[f"{mode}_access"] = values(times.get(f"access_{mode}")) / 60.0
        components[f"{mode}_egress"] = values(times.get(f"egress_{mode}")) / 60.0
        components[f"{mode}_transfer_walk"] = values(times.get(f"transfer_physical_{mode}")) / 60.0
    for mode in ("bike", "walk"):
        components[f"{mode}_time"] = values(times.get(mode)) / 60.0
    components.update({key: np.asarray(hours, dtype=float)[mask]
                       for key, hours in section_welfare_times(context, mode_result, labels).items()})
    digest = hashlib.sha256(json.dumps(list(labels)).encode("utf-8"))
    digest.update(np.flatnonzero(mask).astype("<i8").tobytes())
    return {"od_keys": digest.hexdigest(), "quantities": quantities, "times": components}


_WELFARE_TIME_BUFFER_CACHE: dict[bytes, bytes] = {}


def _compact_welfare_od_sample(
    context: TransportContext,
    mode_result: ModeChoiceResult,
    corridor_zone_ids: Sequence[str] | set[str],
) -> dict[str, Any]:
    """Losslessly compressed zone-OD state for annual trajectory appraisal.

    Apply the Rule of Half before aggregating: demand-weighted municipal mean
    times manufacture benefits when the mix of underlying OD journeys changes.
    Compression keeps exact float64 quantities and times, and identical fixed
    skims share immutable buffers across years/stages. The bounded buffer cache
    contains no demand arrays and does not change the welfare calculation.
    """
    import hashlib
    import zlib

    sample = _welfare_od_sample(context, mode_result, corridor_zone_ids)

    def pack(values: np.ndarray, *, share: bool = False) -> bytes:
        raw = np.asarray(values, dtype="<f8").tobytes()
        if not share:
            return zlib.compress(raw, level=1)
        key = hashlib.sha256(raw).digest()
        if key not in _WELFARE_TIME_BUFFER_CACHE:
            if len(_WELFARE_TIME_BUFFER_CACHE) >= 128:
                del _WELFARE_TIME_BUFFER_CACHE[next(iter(_WELFARE_TIME_BUFFER_CACHE))]
            _WELFARE_TIME_BUFFER_CACHE[key] = zlib.compress(raw, level=1)
        return _WELFARE_TIME_BUFFER_CACHE[key]

    return {
        "od_keys": sample["od_keys"],
        "encoding": "zlib-float64-v1",
        "od_count": len(sample["quantities"]["car"]),
        "quantities": {mode: pack(q) for mode, q in sample["quantities"].items()},
        "times": {component: pack(t, share=component != "car_delay")
                  for component, t in sample["times"].items()},
        "aggregation": "exact directional zone OD, lossless compression",
    }


def extract_corridor_metrics(
    context: TransportContext,
    mode_result: ModeChoiceResult,
    corridor_zone_ids: list[str] | set[str] | None = None,
    *,
    p_co2_kg_per_km: float = 0.139,
    min_distance_km: float | None = None,
) -> dict[str, float]:
    """
    Extract aggregate physical and demand indicators from a mode choice run.

    The resulting dictionary is a direct drop-in input for simulation_engine.py!

    Returns:
        total_trips     : Total peak-hour passenger trips
        car_trips       : Peak-hour car passenger trips (persons, including drivers)
        pt_trips        : Peak-hour public transport trips
        bike_trips      : Peak-hour cycling trips
        walk_trips      : Peak-hour walking trips
        car_share_trips : Car modal split share (raw trips, unfiltered)
        pt_share_trips  : PT modal split share (raw trips, unfiltered)
        bike_share_trips: Bike modal split share (raw trips, unfiltered)
        walk_share_trips: Walk modal split share (raw trips, unfiltered)
        car_share       : Car modal split share (PKM-based, strategic trips >= 5km)
        pt_share        : PT modal split share (PKM-based, strategic trips >= 5km)
        bike_share      : Bike modal split share (PKM-based, strategic trips >= 5km)
        walk_share      : Walk modal split share (PKM-based, strategic trips >= 5km)
        car_dist_km     : Peak-hour vehicle-km travelled by car (VKT)
        car_person_km   : Peak-hour car passenger-km before occupancy conversion
        car_tt_hours    : Peak-hour total car in-vehicle travel time (hours)
        pt_tt_hours     : Peak-hour total PT in-vehicle travel time (hours)
        car_co2_tonnes_peak : Peak-hour car emissions using p_co2_kg_per_km
        co2_tonnes      : Compatibility alias for car_co2_tonnes_peak
        avg_tt_min      : Car/PT mean in-vehicle minutes, excluding road delay

    Quantities describe one modeled peak hour. Annual appraisal recomputes
    emissions from vehicle/train activity and that year's emission factors.
    """
    od_by_mode = mode_result.od_by_mode
    reference = next(iter(od_by_mode.values()))
    labels = reference.index.astype(str)

    if corridor_zone_ids is not None and len(corridor_zone_ids) > 0:
        internal = np.isin(labels, np.asarray(list(corridor_zone_ids), dtype=str))
        involved = internal[:, None] | internal[None, :]
    else:
        involved = np.ones((len(labels), len(labels)), dtype=bool)

    # Trips
    car_trips = float(od_by_mode["drive"].to_numpy(dtype=float)[involved].sum())
    pt_trips = float((od_by_mode["pt_walk"] + od_by_mode["pt_bike"]).to_numpy(dtype=float)[involved].sum())
    bike_trips = float(od_by_mode["bike"].to_numpy(dtype=float)[involved].sum())
    walk_trips = float(od_by_mode["walk"].to_numpy(dtype=float)[involved].sum())
    total_trips = car_trips + pt_trips + bike_trips + walk_trips

    # Quantities in the modal OD matrices are people. Convert only vehicle-based
    # distance/emission costs by occupancy; all time costs retain person-hours.
    car_lengths = context.lengths.get("drive", context.lengths.get("car", mode_result.lengths.get("drive")))
    times = mode_result.travel_times
    uncongested = mode_result.uncongested_travel_times or times

    def aligned_values(frame: pd.DataFrame | None) -> np.ndarray:
        if frame is None:
            return np.zeros(involved.shape, dtype=float)
        values = frame.rename(index=str, columns=str).reindex(index=labels, columns=labels).to_numpy(dtype=float)
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    quantities = {mode: aligned_values(frame) for mode, frame in od_by_mode.items()}

    from additional.section_flows import intervention_masks, physical_car_times, section_metrics
    section_masks = intervention_masks(labels, mode_result.scenario.get("section_config"))
    appraisal_masks = {mode: involved | section_masks.get(mode, False)
                       for mode in quantities}

    def person_hours(mode: str, frame: pd.DataFrame | None, *, appraisal: bool = False) -> float:
        scope = appraisal_masks[mode] if appraisal else involved
        return float((quantities[mode] * aligned_values(frame))[scope].sum() / 60.0)

    car_lengths_km = aligned_values(car_lengths) / 1000.0
    car_person_km = float((quantities["drive"] * car_lengths_km)[involved].sum())
    occupancy = float((mode_result.assigned_metadata or {}).get(
        "drive_occupancy", ASSIGNMENT_SETTINGS.get("drive_occupancy", 1.14)
    ))
    if not np.isfinite(occupancy) or occupancy <= 0.0:
        raise ValueError("Car vehicle metrics require a positive, finite drive occupancy.")
    car_vkt = car_person_km / occupancy

    # === IDEA 1 & 2: Distance-weighted PKM Modal Split with Spatial Filtering ===
    # Filter out local trips (< min_distance_km) to avoid dilution of the core PT/Car metrics.
    # We use car_lengths_km as a proxy for physical OD distance for all modes.
    import parameters as p
    actual_min_dist = min_distance_km if min_distance_km is not None else getattr(p, "MIN_STRATEGIC_PKM_DISTANCE", 5.0)

    strategic_mask = involved & (car_lengths_km >= actual_min_dist)

    car_pkm = float((od_by_mode["drive"].to_numpy(dtype=float) * car_lengths_km)[strategic_mask].sum())
    pt_pkm = float(((od_by_mode["pt_walk"] + od_by_mode["pt_bike"]).to_numpy(dtype=float) * car_lengths_km)[strategic_mask].sum())
    bike_pkm = float((od_by_mode["bike"].to_numpy(dtype=float) * car_lengths_km)[strategic_mask].sum())
    walk_pkm = float((od_by_mode["walk"].to_numpy(dtype=float) * car_lengths_km)[strategic_mask].sum())

    total_pkm = car_pkm + pt_pkm + bike_pkm + walk_pkm

    car_physical_times = physical_car_times(context, mode_result)
    car_tt_total_hours = person_hours("drive", car_physical_times)
    car_delay = np.maximum(
        aligned_values(times.get("drive", times.get("car")))
        - aligned_values(uncongested.get("drive", uncongested.get("car"))), 0.0
    )
    car_delay_person_hours = float((quantities["drive"] * car_delay)[involved].sum() / 60.0)
    pt_hours = dict.fromkeys(("ivt", "wait", "access", "egress", "transfer"), 0.0)
    for mode in ("pt_walk", "pt_bike"):
        pt_hours["ivt"] += person_hours(mode, times.get(f"ivt_{mode}", times.get(mode)))
        pt_hours["wait"] += person_hours(mode, times.get(f"initial_wait_{mode}"))
        pt_hours["wait"] += person_hours(mode, times.get(f"transfer_wait_{mode}"))
        pt_hours["access"] += person_hours(mode, times.get(f"access_{mode}"))
        pt_hours["egress"] += person_hours(mode, times.get(f"egress_{mode}"))
        pt_hours["transfer"] += person_hours(mode, times.get(f"transfer_physical_{mode}"))
    pt_tt_total_hours = pt_hours["ivt"]
    pt_wait_total_hours = pt_hours["wait"]
    bike_tt_total_hours = person_hours("bike", times.get("bike"))
    walk_tt_total_hours = person_hours("walk", times.get("walk"))

    co2_kg = car_vkt * float(p_co2_kg_per_km)

    total_tt_hours = car_tt_total_hours + pt_tt_total_hours
    motorized_trips = car_trips + pt_trips
    avg_tt_min = (total_tt_hours / max(motorized_trips, 1.0)) * 60.0

    metrics = {
        "total_trips": total_trips,
        "car_trips": car_trips,
        "pt_trips": pt_trips,
        "bike_trips": bike_trips,
        "walk_trips": walk_trips,

        # Original trip-based modal split (kept for reference/legacy checks)
        "car_share_trips": car_trips / max(total_trips, 1e-9),
        "pt_share_trips": pt_trips / max(total_trips, 1e-9),
        "bike_share_trips": bike_trips / max(total_trips, 1e-9),
        "walk_share_trips": walk_trips / max(total_trips, 1e-9),

        # Primary modal split: PKM-based for strategic trips (>= 5km)
        "car_share": car_pkm / max(total_pkm, 1e-9),
        "pt_share": pt_pkm / max(total_pkm, 1e-9),
        "bike_share": bike_pkm / max(total_pkm, 1e-9),
        "walk_share": walk_pkm / max(total_pkm, 1e-9),

        "car_dist_km": car_vkt,
        "car_vehicle_km": car_vkt,
        "car_person_km": car_person_km,
        "car_vehicle_trips": car_trips / occupancy,
        "drive_occupancy": occupancy,
        "pt_dist_km": pt_pkm,
        "car_tt_hours": car_tt_total_hours,
        "car_delay_person_hours": car_delay_person_hours,
        "congestion_delay_hours": car_delay_person_hours,
        "pt_tt_hours": pt_tt_total_hours,
        "pt_wait_hours": pt_wait_total_hours,
        "pt_access_hours": pt_hours["access"],
        "pt_egress_hours": pt_hours["egress"],
        "pt_transfer_walk_hours": pt_hours["transfer"],
        "bike_tt_hours": bike_tt_total_hours,
        "walk_tt_hours": walk_tt_total_hours,
        "bike_dist_km": float((quantities["bike"] * aligned_values(mode_result.lengths.get("bike")))[involved].sum() / 1000.0),
        "walk_dist_km": float((quantities["walk"] * aligned_values(mode_result.lengths.get("walk")))[involved].sum() / 1000.0),
        "co2_tonnes": co2_kg / 1000.0,
        "car_co2_tonnes_peak": co2_kg / 1000.0,
        "avg_tt_min": avg_tt_min,
    }
    # Preserve cordon indicators above for teaching plots and adaptive triggers.
    # These separate time-cost anchors include the extra covered through users,
    # matching the per-mode OD scope used by _welfare_od_sample.
    metrics.update({
        "appraisal_car_trips": float(quantities["drive"][appraisal_masks["drive"]].sum()),
        "appraisal_pt_trips": sum(float(quantities[mode][appraisal_masks[mode]].sum())
                                   for mode in ("pt_walk", "pt_bike")),
        "appraisal_bike_trips": float(quantities["bike"][appraisal_masks["bike"]].sum()),
        "appraisal_walk_trips": float(quantities["walk"][appraisal_masks["walk"]].sum()),
        "appraisal_car_tt_hours": person_hours("drive", car_physical_times, appraisal=True),
        "appraisal_car_delay_person_hours": float(
            (quantities["drive"] * car_delay)[appraisal_masks["drive"]].sum() / 60.0
        ),
        "appraisal_bike_tt_hours": person_hours("bike", times.get("bike"), appraisal=True),
        "appraisal_walk_tt_hours": person_hours("walk", times.get("walk"), appraisal=True),
    })
    for component, suffixes in {
        "tt": ("ivt",), "wait": ("initial_wait", "transfer_wait"),
        "access": ("access",), "egress": ("egress",),
        "transfer_walk": ("transfer_physical",),
    }.items():
        metrics[f"appraisal_pt_{component}_hours"] = sum(
            person_hours(mode, times.get(f"{suffix}_{mode}"), appraisal=True)
            for mode in ("pt_walk", "pt_bike") for suffix in suffixes
        )
    metrics.update(section_metrics(context, mode_result))
    return metrics


def run_simulation(
    context: TransportContext,
    *,
    stage: int,
    stage_specs: Mapping[int, Mapping[str, Any]],
    corridor_zone_ids: list[str] | set[str] | None = None,
    corridor_municipalities: list[str] | None = None,
    trip_rate_multiplier: float = 1.0,
    demand_multiplier: float | None = None,
    pt_asc_shift: float = 0.0,
    bike_asc_shift: float = 0.0,
    road_freight_multiplier: float = 1.0,
    ebike_share: float | None = None,
    corridor_context: CorridorContext | None = None,
    assignment_settings: Mapping[str, Any] | None = None,
    verbose: bool = False,
) -> tuple[ModeChoiceResult, dict[str, float]]:
    """
    Unified function that executes mode choice and route assignment based on ASSIGNMENT_SETTINGS.

    Returns:
        mode_result (ModeChoiceResult): The raw FSM result.
        metrics (dict): The aggregated physical/demand metrics, augmented with delay if MSA is used.
    """
    # 1. Run mode choice
    mode_result = run_transport_mode_choice(
        context,
        stage=stage,
        stage_specs=stage_specs,
        corridor_zone_ids=corridor_zone_ids,
        corridor_municipalities=corridor_municipalities,
        trip_rate_multiplier=trip_rate_multiplier,
        demand_multiplier=demand_multiplier,
        pt_asc_shift=pt_asc_shift,
        bike_asc_shift=bike_asc_shift,
        ebike_share=ebike_share,
    )

    # 2. Select the assignment and extract its final physical metrics once.
    min_dist_km = stage_specs[stage].get("_min_distance_km") if stage_specs and stage in stage_specs else None
    # This function is called inside stage loops, sensitivity sweeps, and
    # parallel workers. A per-call message therefore floods notebook output;
    # callers can opt in while debugging.
    if verbose:
        print("  [Simulation] Extracting corridor metrics...")
    settings = resolve_assignment_settings(assignment_settings)
    method = settings["method"]

    import parameters as p
    has_corridor = bool(corridor_municipalities) or corridor_context is not None or getattr(p, "CORRIDOR_DEFINITION_MODE", "municipalities") == "zones"
    if method == "MSA" and has_corridor:
        # A notebook editor may supply a stage-specific road network. When it
        # does not, construct the normal corridor and apply any documented
        # link-level ``road_capacity`` entries from the selected stage.
        if corridor_context is None:
            corridor = build_corridor_context(
                context,
                corridor_municipalities=list(corridor_municipalities or []),
                name="configured project corridor",
            )
        else:
            corridor = corridor_context
        corridor = ensure_stage_road_context(corridor, stage_specs.get(stage, {}))
        if bool(settings["modal_feedback"]):
            assignment = run_coupled_corridor_assignment(
                context,
                corridor,
                mode_result,
                # ``mode_result`` already contains the requested demand scale.
                demand_multiplier=1.0,
                background_multiplier=road_freight_multiplier,
                **assignment_solver_kwargs(settings),
            )
            mode_result = assignment.mode_result or mode_result
            corridor_zone_ids = corridor_zone_ids or corridor.zone_ids
        else:
            assignment = run_corridor_assignment(
                context,
                corridor,
                mode_result,
                background_multiplier=road_freight_multiplier,
                **assignment_solver_kwargs(settings, coupled=False),
            )
        mode_result.assigned_edges = assignment.links
        mode_result.assigned_metadata = assignment.diagnostics
    metrics = extract_corridor_metrics(context, mode_result, corridor_zone_ids, min_distance_km=min_dist_km)
    if method == "MSA" and has_corridor:
        metrics["congestion_vehicle_delay_hours"] = assignment.diagnostics.get(
            "total_delay_hours", 0.0
        )

    return mode_result, metrics


def transport_state_from_mode(
    context: TransportContext,
    mode_result: ModeChoiceResult,
    corridor_zone_ids: Sequence[str] | None = None,
    *,
    include_welfare: bool = False,
) -> dict[str, Any]:
    """Package an already solved native state for the shared appraisal interface."""
    state = {
        "schema_version": 1, "source": "coupled_msa", "stage": int(mode_result.stage),
        "metrics": extract_corridor_metrics(context, mode_result, corridor_zone_ids),
        "mode_result": mode_result,
        "assignment_diagnostics": dict(mode_result.assigned_metadata or {}),
    }
    if include_welfare:
        state["welfare_od"] = _compact_welfare_od_sample(context, mode_result, corridor_zone_ids)
    return state


def acquire_native_transport_state(
    context: TransportContext,
    mode_result: ModeChoiceResult,
    *,
    stage: int,
    corridor_municipalities: Sequence[str] | None,
    corridor_context: CorridorContext | None = None,
    passenger_demand_multiplier: float = 1.0,
    pt_asc_shift: float = 0.0,
    bike_asc_shift: float = 0.0,
    ebike_share: float | None = None,
    road_freight_multiplier: float = 1.0,
    assignment_settings: Mapping[str, Any] | None = None,
    assignment_result: AssignmentResult | None = None,
    include_welfare: bool = False,
) -> dict[str, Any]:
    """Solve one native transport state, independently of annual cost valuation."""
    from copy import copy

    settings = resolve_assignment_settings(assignment_settings)
    if not settings["modal_feedback"]:
        raise ValueError("Passenger time appraisal requires coupled MSA OD skims.")
    if int(stage) != int(mode_result.stage):
        raise ValueError("The requested stage differs from the supplied native mode state.")
    effective_mode = mode_result
    if pt_asc_shift != 0.0 or bike_asc_shift != 0.0:
        effective_mode = copy(mode_result)
        effective_mode.scenario = dict(mode_result.scenario)
        base_parameters = context.modules["mode_choice_zurich"].load_mode_choice_parameters()
        for name, shift in (("ASC_PT_WALK", pt_asc_shift), ("ASC_PT_BIKE", pt_asc_shift), ("ASC_BIKE", bike_asc_shift)):
            effective_mode.scenario[name] = float(effective_mode.scenario.get(name, base_parameters[name])) + float(shift)
    if ebike_share is not None and abs(float(ebike_share) - float(mode_result.scenario.get("ebike_share", 0.0))) > 1e-12:
        effective_mode = mode_choice_with_ebike_share(
            context, effective_mode, float(ebike_share),
            corridor_zone_ids=getattr(corridor_context, "zone_ids", None),
        )
    corridor = corridor_context or build_corridor_context(
        context, corridor_municipalities=list(corridor_municipalities or []), name="configured project corridor"
    )
    corridor = ensure_stage_road_context(corridor, mode_result.scenario)
    assignment = assignment_result
    if assignment is None:
        assignment = run_coupled_corridor_assignment(
            context, corridor, effective_mode,
            demand_multiplier=passenger_demand_multiplier,
            background_multiplier=road_freight_multiplier,
            **assignment_solver_kwargs(settings),
        )
    if assignment.mode_result is None:
        raise ValueError("The supplied assignment has no coupled passenger mode state.")
    state = transport_state_from_mode(context, assignment.mode_result, corridor.zone_ids,
                                      include_welfare=include_welfare)
    state["assignment_result"] = assignment
    state["assignment_diagnostics"] = dict(assignment.diagnostics)
    return state


# =============================================================================
# 5. INTERACTIVE DASHBOARD WIDGETS
# =============================================================================


# =============================================================================
# 6. CORRIDOR CONTEXT AND CLICKABLE MAP EXPLORERS
# =============================================================================

# Shared palettes keep all network views visually consistent.  Continuous
# metrics use a blue-to-red scale; road classes use stable categorical colours.


def _retain_internal_road_connections(
    local_edges: Any,
    network_edges: Any,
    internal_zone_nodes: Mapping[str, int],
    *,
    name: str,
) -> tuple[Any, dict[str, Any]]:
    """Retain existing road excursions needed to connect the selected zone anchors."""
    import networkx as nx

    if not internal_zone_nodes:
        raise ValueError(f"The {name} cordon contains no internally routed zones.")
    local_graph = nx.DiGraph()
    local_graph.add_edges_from(zip(local_edges["source"], local_edges["target"]))
    internal_nodes = set(internal_zone_nodes.values())
    main_core = max(
        nx.strongly_connected_components(local_graph),
        key=lambda component: (len(component & internal_nodes), len(component)),
    )
    disconnected = internal_nodes.difference(main_core)
    diagnostics = {"routing_extension_links": 0, "routing_extension_nodes": 0,
                   "routing_extension_zone_ids": []}
    if not disconnected:
        return local_edges, diagnostics

    # One shortest existing route in each direction repairs boundary excursions
    # without inventing road links or moving zone attachments to another road.
    full_graph = nx.DiGraph()
    lengths = network_edges.groupby(["source", "target"])["length_m"].min()
    full_graph.add_weighted_edges_from(
        (int(source), int(target), max(float(length), 0.001))
        for (source, target), length in lengths.items()
    )
    _, outward_paths = nx.multi_source_dijkstra(full_graph, main_core, weight="weight")
    _, inward_paths = nx.multi_source_dijkstra(full_graph.reverse(copy=False), main_core, weight="weight")
    unreachable = {
        zone: node for zone, node in internal_zone_nodes.items()
        if node not in outward_paths or node not in inward_paths
    }
    if unreachable:
        examples = ", ".join(f"{zone} (node {node})" for zone, node in list(unreachable.items())[:8])
        raise ValueError(
            f"The {name} zones cannot reach one another even in the full road network: {examples}. "
            "Inspect their road attachments and source-network connectivity; no demand was dropped."
        )
    required_pairs = set()
    for node in disconnected:
        for path in (outward_paths[node], list(reversed(inward_paths[node]))):
            required_pairs.update(zip(path, path[1:]))
    original_pairs = set(zip(local_edges["source"], local_edges["target"]))
    required_pairs.difference_update(original_pairs)
    additions = network_edges.loc[
        [(source, target) in required_pairs for source, target in zip(network_edges["source"], network_edges["target"])]
    ]
    extended = pd.concat([local_edges, additions], ignore_index=True)
    original_nodes = set(local_graph.nodes)
    all_nodes = set(extended["source"]) | set(extended["target"])
    diagnostics.update(
        routing_extension_links=len(additions),
        routing_extension_nodes=len(all_nodes.difference(original_nodes)),
        routing_extension_zone_ids=[zone for zone, node in internal_zone_nodes.items() if node in disconnected],
    )
    return extended, diagnostics


def _reachable_gate_candidates(
    candidates: Any,
    edges: Any,
    internal_zone_nodes: Mapping[str, int],
    *,
    name: str,
) -> tuple[Any, dict[str, Any]]:
    """Keep entry/exit directions that connect to every internal routing node."""
    import networkx as nx

    if not internal_zone_nodes:
        raise ValueError(f"The {name} cordon contains no internally routed zones.")
    graph = nx.DiGraph()
    graph.add_edges_from(zip(edges["source"].astype(int), edges["target"].astype(int)))
    anchor = next(iter(internal_zone_nodes.values()))
    reachable_from_internal = {anchor} | nx.descendants(graph, anchor)
    can_reach_internal = {anchor} | nx.ancestors(graph, anchor)
    internal_core = reachable_from_internal & can_reach_internal
    disconnected = {
        zone: node for zone, node in internal_zone_nodes.items() if node not in internal_core
    }
    if disconnected:
        examples = ", ".join(f"{zone} (node {node})" for zone, node in list(disconnected.items())[:8])
        raise ValueError(
            f"The {name} road cordon separates {len(disconnected)} of "
            f"{len(internal_zone_nodes)} internal zones from the other internal zones: {examples}. "
            "Increase buffer_m or inspect road connectivity before allocating demand; "
            "internal zones have not been dropped or moved to other roads."
        )

    eligible = candidates.copy()
    directions_before = int(eligible[["can_enter", "can_exit"]].to_numpy(dtype=bool).sum())
    eligible["can_enter"] &= eligible["node_id"].isin(can_reach_internal)
    eligible["can_exit"] &= eligible["node_id"].isin(reachable_from_internal)
    usable = eligible["can_enter"] | eligible["can_exit"]
    rejected = eligible.loc[~usable, "node_id"].astype(int).tolist()
    eligible = eligible.loc[usable].copy()
    eligible["both_directions"] = eligible["can_enter"] & eligible["can_exit"]
    eligible["direction"] = np.where(
        eligible["both_directions"], "entry + exit",
        np.where(eligible["can_enter"], "entry", "exit"),
    )
    diagnostics = {
        "eligible_boundary_nodes": len(eligible),
        "excluded_unreachable_boundary_nodes": rejected,
        "excluded_unreachable_gate_directions": directions_before - int(
            eligible[["can_enter", "can_exit"]].to_numpy(dtype=bool).sum()
        ),
        "internal_zone_reachability": "mutually reachable",
    }
    return eligible, diagnostics


def _select_spaced_gates(
    candidates: Any,
    *,
    max_gates: int,
    minimum_separation_m: float,
) -> Any:
    """Prefer high-capacity bidirectional gates while keeping them spatially distinct."""
    if candidates.empty or int(max_gates) < 1:
        return candidates.iloc[0:0].copy()
    ordered = candidates.sort_values(
        ["capacity_vph", "both_directions"], ascending=[False, False]
    )
    selected = []
    both = ordered.loc[ordered["both_directions"].astype(bool)]
    if len(both):
        selected.append(next(both.itertuples(index=False)))
    else:
        entries = ordered.loc[ordered["can_enter"].astype(bool)]
        exits = ordered.loc[ordered["can_exit"].astype(bool)]
        if entries.empty or exits.empty or int(max_gates) < 2:
            raise ValueError("The cordon needs at least one valid entry gate and one exit gate.")
        selected.extend([next(entries.itertuples(index=False)), next(exits.itertuples(index=False))])

    for row in ordered.itertuples(index=False):
        if len(selected) >= int(max_gates):
            break
        if any(row.node_id == existing.node_id for existing in selected):
            continue
        if selected and min(row.geometry.distance(existing.geometry) for existing in selected) < float(minimum_separation_m):
            continue
        selected.append(row)
    selected_ids = [row.node_id for row in selected]
    return candidates.loc[candidates["node_id"].isin(selected_ids)].copy()


def build_corridor_context(
    context: TransportContext,
    corridor_municipalities: list[str] | None = None,
    *,
    buffer_m: float = DEFAULT_CORRIDOR_BUFFER_M,
    max_gates: int = DEFAULT_MAX_GATES,
    gate_separation_m: float = DEFAULT_GATE_SEPARATION_M,
    name: str = "MehrSpur",
) -> CorridorContext:
    """Build one auditable corridor topology shared by maps, OD tables, and assignment.

    Complete inside-to-inside links are retained; links are never geometrically
    cut. Existing road excursions reconnect internal zone anchors where needed.
    Each gate is the retained endpoint of a directed routing-frontier link.
    External zones are mapped separately to eligible entry and exit gates.
    Gate directions must connect to every internal zone; disconnected internal
    zones raise an error so demand cannot silently disappear during assignment.
    """
    import geopandas as gpd

    if not context.assignment_network or "edges" not in context.assignment_network:
        raise ValueError("The transport context does not contain an assignment network.")
    if not corridor_municipalities:
        import parameters as p
        corridor_municipalities = list(getattr(p, "CORRIDOR_MUNICIPALITIES", []))
    zones = context.zones.copy()
    core_zones = _get_core_zones(zones, list(corridor_municipalities))
    if core_zones.empty:
        raise ValueError(f"No zones matched the configured {name} corridor.")
    polygon = core_zones.geometry.union_all().buffer(float(buffer_m))
    polygon_gdf = gpd.GeoDataFrame(
        {"name": [f"{name} modelling cordon"]}, geometry=[polygon], crs=zones.crs
    )

    network_edges = context.assignment_network["edges"].copy()
    network_nodes = context.assignment_network["nodes"].copy()
    node_inside = network_nodes.geometry.intersects(polygon)
    inside_by_node = dict(zip(network_nodes["node_id"].astype(int), node_inside.astype(bool)))
    source_inside = network_edges["source"].map(inside_by_node).fillna(False).astype(bool)
    target_inside = network_edges["target"].map(inside_by_node).fillna(False).astype(bool)

    local_edges = network_edges.loc[source_inside & target_inside].copy()
    if local_edges.empty:
        raise ValueError(f"The {name} cordon contains no complete road links.")
    original_zone_node_map = {
        str(zone): int(node) for zone, node in context.assignment_network["zone_node_map"].items()
    }
    centroids = gpd.GeoSeries(zones["centroid"], crs=zones.crs)
    inside_zone_ids = zones.loc[centroids.intersects(polygon), "grid_id"].astype(str).tolist()
    missing_attachments = sorted(set(inside_zone_ids).difference(original_zone_node_map))
    if missing_attachments:
        raise ValueError(f"The {name} internal zones have no road attachment: {missing_attachments[:8]}.")
    internal_ids = inside_zone_ids
    zone_node_map = {zone_id: original_zone_node_map[zone_id] for zone_id in internal_ids}
    local_edges, routing_extension = _retain_internal_road_connections(
        local_edges, network_edges, zone_node_map, name=name
    )
    used_nodes = set(local_edges["source"].astype(int)) | set(local_edges["target"].astype(int))
    local_nodes = network_nodes.loc[network_nodes["node_id"].astype(int).isin(used_nodes)].copy()
    source_inside = network_edges["source"].isin(used_nodes)
    target_inside = network_edges["target"].isin(used_nodes)

    frontier = source_inside ^ target_inside
    zone_access = network_edges["highway"].eq("manual_connector")
    # Zone access links are artificial attachments, not boundary road crossings.
    crossing = network_edges.loc[frontier & ~zone_access].copy()
    crossing["inside_node"] = np.where(
        source_inside.loc[crossing.index], crossing["source"], crossing["target"]
    ).astype(int)
    crossing["can_exit"] = source_inside.loc[crossing.index].to_numpy(dtype=bool)
    crossing["can_enter"] = target_inside.loc[crossing.index].to_numpy(dtype=bool)
    crossing = crossing.loc[crossing["inside_node"].isin(used_nodes)].copy()

    node_geometry = local_nodes.set_index("node_id").geometry
    candidate_rows = []
    for node_id, group in crossing.groupby("inside_node", sort=False):
        capacity_values = group.get(
            "capacity_vph", pd.Series(0.0, index=group.index)
        )
        capacity = pd.to_numeric(capacity_values, errors="coerce").max()
        highway = group.get("highway", pd.Series("unknown", index=group.index)).astype(str).mode()
        can_enter, can_exit = bool(group["can_enter"].any()), bool(group["can_exit"].any())
        candidate_rows.append({
            "node_id": int(node_id),
            "capacity_vph": float(capacity) if np.isfinite(capacity) else 0.0,
            "highway": highway.iloc[0] if len(highway) else "unknown",
            "can_enter": can_enter,
            "can_exit": can_exit,
            "both_directions": can_enter and can_exit,
            "direction": "entry + exit" if can_enter and can_exit else "entry" if can_enter else "exit",
            "geometry": node_geometry.loc[int(node_id)],
        })
    candidates = gpd.GeoDataFrame(
        candidate_rows,
        columns=["node_id", "capacity_vph", "highway", "can_enter", "can_exit", "both_directions", "direction", "geometry"],
        geometry="geometry",
        crs=zones.crs,
    )
    candidate_count = len(candidates)
    candidates, connectivity = _reachable_gate_candidates(
        candidates, local_edges, zone_node_map, name=name
    )
    if connectivity["excluded_unreachable_gate_directions"] or routing_extension["routing_extension_links"]:
        import warnings
        warnings.warn(
            f"{name} cordon: excluded {len(connectivity['excluded_unreachable_boundary_nodes'])} "
            f"disconnected boundary candidates and {connectivity['excluded_unreachable_gate_directions']} "
            f"unreachable entry/exit directions. Retained {routing_extension['routing_extension_links']} "
            "additional existing links to connect internal zone anchors. "
            "See corridor.metadata for the affected node and zone IDs.",
            UserWarning, stacklevel=2,
        )
    gates = _select_spaced_gates(
        candidates,
        max_gates=int(max_gates),
        minimum_separation_m=float(gate_separation_m),
    ).reset_index(drop=True)
    if gates.empty:
        raise ValueError(f"The {name} cordon produced no usable boundary gates.")
    gates["gate_id"] = [f"G{position:02d}" for position in range(1, len(gates) + 1)]

    corridor_zones = zones.loc[zones["grid_id"].astype(str).isin(internal_ids)].copy()
    zone_node_map.update(dict(zip(gates["gate_id"].astype(str), gates["node_id"].astype(int))))

    external = zones.loc[~zones["grid_id"].astype(str).isin(internal_ids)].copy()
    external_centroids = gpd.GeoSeries(external["centroid"], crs=zones.crs)
    external_xy = np.column_stack((external_centroids.x, external_centroids.y))

    def nearest_gate_mapping(eligible_gates: Any) -> dict[str, str]:
        if eligible_gates.empty:
            raise ValueError("Direction-aware external demand needs an eligible gate.")
        if external.empty:
            return {}
        gate_xy = np.column_stack((eligible_gates.geometry.x, eligible_gates.geometry.y))
        squared_distance = (
            (external_xy[:, None, 0] - gate_xy[None, :, 0]) ** 2
            + (external_xy[:, None, 1] - gate_xy[None, :, 1]) ** 2
        )
        nearest = np.argmin(squared_distance, axis=1)
        return dict(zip(external["grid_id"].astype(str), eligible_gates.iloc[nearest]["gate_id"].astype(str)))

    entry_map = nearest_gate_mapping(gates.loc[gates["can_enter"].astype(bool)])
    exit_map = nearest_gate_mapping(gates.loc[gates["can_exit"].astype(bool)])
    metadata = {
        "name": name,
        "buffer_m": float(buffer_m),
        "max_gates": int(max_gates),
        "gate_separation_m": float(gate_separation_m),
        "internal_zones": len(internal_ids),
        "local_links": len(local_edges),
        "local_nodes": len(local_nodes),
        "candidate_boundary_nodes": candidate_count,
        "excluded_zone_access_frontier_links": int((frontier & zone_access).sum()),
        **routing_extension,
        "routing_nodes_outside_polygon": int((~local_nodes.geometry.intersects(polygon)).sum()),
        **connectivity,
        "selected_gates": len(gates),
        "entry_gates": int(gates["can_enter"].sum()),
        "exit_gates": int(gates["can_exit"].sum()),
        "gate_definition": "retained endpoint of a directed routing-frontier link",
    }
    return CorridorContext(
        polygon=polygon,
        polygon_gdf=polygon_gdf,
        zones=corridor_zones,
        zone_ids=internal_ids,
        edges=local_edges,
        nodes=local_nodes,
        gates=gates,
        zone_node_map=zone_node_map,
        external_zone_to_entry_gate=entry_map,
        external_zone_to_exit_gate=exit_map,
        metadata=metadata,
    )


def build_corridor(context: TransportContext, project_name, **kwargs: Any) -> CorridorContext:
    """Convenience wrapper using the corridor configured in ``parameters.py``."""
    return build_corridor_context(context, name=project_name, **kwargs)

def collapse_od_to_gates(
    matrix: pd.DataFrame,
    context: TransportContext,
    corridor: CorridorContext,
    *,
    passthrough_fraction: float = DEFAULT_PASSTHROUGH_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse canton-wide demand into internal zones and cordon gates.

    The reduced matrix preserves internal-to-internal, inbound, and outbound
    demand exactly. External-to-external demand is grouped by its nearest
    direction-compatible gates and multiplied by ``passthrough_fraction``:
    proximity alone cannot establish which of those trips crosses the cordon.
    """
    fraction = float(np.clip(passthrough_fraction, 0.0, 1.0))
    zone_ids = context.zones["grid_id"].astype(str).tolist()
    values = (
        matrix.copy()
        .set_axis(matrix.index.astype(str), axis=0)
        .set_axis(matrix.columns.astype(str), axis=1)
        .reindex(index=zone_ids, columns=zone_ids, fill_value=0.0)
    )

    internal = [zone for zone in corridor.zone_ids if zone in values.index]
    internal_set = set(internal)
    external = [zone for zone in zone_ids if zone not in internal_set]
    gates = corridor.gates["gate_id"].astype(str).tolist()
    labels = internal + gates
    reduced = pd.DataFrame(0.0, index=labels, columns=labels)
    if internal:
        reduced.loc[internal, internal] = values.loc[internal, internal].to_numpy(dtype=float)

    entry_groups: dict[str, list[str]] = {gate: [] for gate in gates}
    exit_groups: dict[str, list[str]] = {gate: [] for gate in gates}
    for zone in external:
        entry_gate = corridor.external_zone_to_entry_gate.get(zone)
        exit_gate = corridor.external_zone_to_exit_gate.get(zone)
        if entry_gate in entry_groups:
            entry_groups[entry_gate].append(zone)
        if exit_gate in exit_groups:
            exit_groups[exit_gate].append(zone)

    # Inbound trips enter at a gate; outbound trips leave at a gate.
    for gate, external_zones in entry_groups.items():
        if external_zones and internal:
            reduced.loc[gate, internal] += values.loc[external_zones, internal].sum(axis=0)
    for gate, external_zones in exit_groups.items():
        if external_zones and internal:
            reduced.loc[internal, gate] += values.loc[internal, external_zones].sum(axis=1)

    # Retain only the calibrated share of external trips likely to cross the
    # corridor, excluding same-gate movements that do not traverse it.
    for origin_gate, origin_zones in entry_groups.items():
        if not origin_zones:
            continue
        for destination_gate, destination_zones in exit_groups.items():
            if origin_gate == destination_gate or not destination_zones:
                continue
            reduced.loc[origin_gate, destination_gate] += (
                values.loc[origin_zones, destination_zones].to_numpy(dtype=float).sum()
                * fraction
            )

    category_values = [
        float(reduced.loc[internal, internal].to_numpy(dtype=float).sum()) if internal else 0.0,
        float(reduced.loc[gates, internal].to_numpy(dtype=float).sum()) if internal and gates else 0.0,
        float(reduced.loc[internal, gates].to_numpy(dtype=float).sum()) if internal and gates else 0.0,
        float(reduced.loc[gates, gates].to_numpy(dtype=float).sum()) if gates else 0.0,
    ]
    breakdown = pd.DataFrame(
        {
            "category": [
                "internal -> internal", "gate -> internal",
                "internal -> gate", "gate -> gate (pass-through)",
            ],
            "trips": category_values,
        }
    )
    breakdown["share_of_retained"] = breakdown["trips"] / max(
        float(breakdown["trips"].sum()), 1e-9
    )

    if not np.isclose(
        float(breakdown["trips"].sum()),
        float(reduced.to_numpy(dtype=float).sum()),
        rtol=1e-9,
        atol=1e-6,
    ):
        raise AssertionError("Cordon OD categories do not reconcile with the reduced matrix.")
    missing = sorted(set(labels).difference(corridor.zone_node_map))
    if missing:
        raise AssertionError(f"Reduced OD labels have no routing node: {missing}")
    return reduced, breakdown


def gate_totals(demand: pd.DataFrame, corridor: CorridorContext) -> pd.DataFrame:
    """Summarise inbound, outbound, and pass-through demand at every gate."""
    gates = corridor.gates["gate_id"].astype(str).tolist()
    internal = corridor.zone_ids
    rows = []
    for gate in gates:
        inbound = float(demand.loc[gate, internal].sum()) if internal else 0.0
        outbound = float(demand.loc[internal, gate].sum()) if internal else 0.0
        pass_in = float(demand.loc[gates, gate].sum() - demand.loc[gate, gate])
        pass_out = float(demand.loc[gate, gates].sum() - demand.loc[gate, gate])
        rows.append(
            {
                "gate_id": gate,
                "to_internal": inbound,
                "from_internal": outbound,
                "passthrough_in": pass_in,
                "passthrough_out": pass_out,
                "cordon_total": inbound + outbound + 0.5 * (pass_in + pass_out),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# 7. ROUTE ASSIGNMENT
# =============================================================================


# =============================================================================
# 8. LOCAL COARSE MSA / BPR ROAD ASSIGNMENT
# =============================================================================

def coarse_msa_assignment(
    edges: Any,
    zone_node_map: Mapping[str, int],
    demand: pd.DataFrame,
    *,
    nodes: Any | None = None,
    max_iterations: int | None = None,
    min_iterations: int | None = None,
    relative_gap_threshold: float | None = None,
    od_threshold: float | None = None,
    stopping_rule: str | None = None,
    road_gap_threshold: float | None = None,
    check_every: int | None = None,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    """Run a transparent local MSA/BPR assignment for notebook use.

    Road relative gap compares loaded vehicle-minutes with the shortest-path
    vehicle-minutes for the same fixed OD. The optional ``successive_change``
    rule uses ``relative_gap_threshold`` for L1 flow changes. ``nodes`` remains
    for compatibility; endpoints are derived from the directed link table.
    """

    settings = resolve_assignment_settings({name: value for name, value in {
        "max_iterations": max_iterations, "min_iterations": min_iterations,
        "relative_gap_threshold": relative_gap_threshold, "od_threshold": od_threshold,
        "stopping_rule": stopping_rule, "road_gap_threshold": road_gap_threshold,
        "check_every": check_every,
    }.items() if value is not None})
    max_iterations, min_iterations = settings["max_iterations"], settings["min_iterations"]
    relative_gap_threshold, od_threshold = settings["relative_gap_threshold"], settings["od_threshold"]
    road_stopping = settings["stopping_rule"] == "road_relative_gap"
    road_limit = float(settings["road_gap_threshold"] if settings["road_gap_threshold"] is not None else relative_gap_threshold)
    check_every = settings["check_every"]
    started = time.perf_counter()
    local_edges = edges.copy().reset_index(drop=True)
    required = {
        "source", "target", "free_flow_time_min", "capacity_vph", "length_m", "geometry"
    }
    missing = required.difference(local_edges.columns)
    if missing:
        raise ValueError(f"Local assignment links are missing: {sorted(missing)}")
    if int(max_iterations) < int(min_iterations):
        raise ValueError("max_iterations must be at least min_iterations")

    local_demand = demand.copy()
    local_demand.index = local_demand.index.astype(str)
    local_demand.columns = local_demand.columns.astype(str)
    if local_demand.shape[0] != local_demand.shape[1] or list(local_demand.index) != list(local_demand.columns):
        raise ValueError("Assignment demand must be a square, equally labelled OD matrix.")
    demand_values = local_demand.to_numpy(dtype=float)
    if not np.isfinite(demand_values).all() or (demand_values < 0).any():
        raise ValueError("Assignment demand must be finite and non-negative.")

    intrazonal_demand = float(np.trace(demand_values))
    interzonal_demand = max(float(demand_values.sum()) - intrazonal_demand, 0.0)
    retained_mask = demand_values > float(od_threshold)
    np.fill_diagonal(retained_mask, False)
    retained_demand = float(demand_values[retained_mask].sum())
    retained_pairs = int(retained_mask.sum())
    omitted_below_cutoff = max(interzonal_demand - retained_demand, 0.0)

    flows = np.zeros(len(local_edges), dtype=float)
    history: list[dict[str, float]] = []
    served = unserved = 0.0
    unserved_pairs = 0
    road_checks = []
    for iteration in range(1, int(max_iterations) + 1):
        current_travel_time = _bpr_link_times(local_edges, flows)
        aon_flows, served, unserved, unserved_pairs = _all_or_nothing_sparse(
            local_edges,
            zone_node_map,
            local_demand,
            current_travel_time,
            od_threshold=float(od_threshold),
        )
        step = 1.0 / float(iteration)
        updated = flows + step * (aon_flows - flows)
        rel_gap = float(np.abs(updated - flows).sum()) / max(
            float(np.abs(flows).sum()), 1e-9
        )
        flows = updated
        history.append(
            {
                "iteration": iteration,
                "msa_step": step,
                "relative_l1_flow_change": rel_gap,
                "served_vehicles": float(served),
            }
        )
        if road_stopping:
            if iteration == int(max_iterations) or (
                iteration >= int(min_iterations)
                and (iteration - int(min_iterations)) % int(check_every) == 0
            ):
                check_times = _bpr_link_times(local_edges, flows)
                shortest = _corridor_shortest_time_table(
                    local_edges, zone_node_map, check_times
                ).reindex(index=local_demand.index, columns=local_demand.columns).to_numpy(dtype=float)
                feasible = retained_mask & np.isfinite(shortest)
                minimum_time = float(np.sum(demand_values[feasible] * shortest[feasible]))
                actual_time = float(np.dot(flows, check_times))
                gap = (actual_time - minimum_time) / max(actual_time, 1e-12)
                if not np.isfinite(gap) or gap < -1e-8:
                    raise ValueError("Road gap is inconsistent with the assigned fixed OD.")
                gap = max(gap, 0.0)
                road_checks.append({"iteration": iteration,
                    "road_relative_gap_feasible_averaged_od": gap,
                    "averaged_assigned_feasible_vehicles": float(demand_values[feasible].sum())})
                history[-1]["road_relative_gap_feasible_averaged_od"] = gap
                if gap <= road_limit:
                    break
        elif iteration >= int(min_iterations) and rel_gap <= float(relative_gap_threshold):
            break

    capacity = pd.to_numeric(local_edges["capacity_vph"], errors="coerce").fillna(1.0).clip(lower=1.0).to_numpy()
    free_flow_time = pd.to_numeric(local_edges["free_flow_time_min"], errors="coerce").fillna(1.0).to_numpy()
    length_m = pd.to_numeric(local_edges["length_m"], errors="coerce").fillna(0.0).to_numpy()
    final_vc = np.maximum(flows / capacity, 0.0)
    final_travel_time = _bpr_link_times(local_edges, flows)
    final_speed = np.divide(
        length_m / 1000.0,
        final_travel_time / 60.0,
        out=np.zeros(len(local_edges), dtype=float),
        where=final_travel_time > 0,
    )
    assigned_edges = local_edges.copy()
    assigned_edges["flow_vehicles"] = flows
    assigned_edges["volume_capacity_ratio"] = final_vc
    assigned_edges["time_min"] = final_travel_time
    assigned_edges["assigned_speed_kph"] = np.clip(final_speed, 0.0, 130.0)
    assigned_edges["delay_min"] = np.maximum(final_travel_time - free_flow_time, 0.0)

    total_delay_hours = float((assigned_edges["flow_vehicles"] * (assigned_edges["delay_min"] / 60.0)).sum())
    total_vkt = float((assigned_edges["flow_vehicles"] * (length_m / 1000.0)).sum())
    total_vht = float((assigned_edges["flow_vehicles"] * (final_travel_time / 60.0)).sum())
    avg_speed_kph = total_vkt / total_vht if total_vht > 0 else 0.0

    history_df = pd.DataFrame(history)
    metadata = {
        "algorithm": "method of successive averages (MSA) with BPR link times",
        "runtime_s": time.perf_counter() - started,
        "iterations": len(history_df),
        "max_iterations": int(max_iterations),
        "min_iterations": int(min_iterations),
        "total_delay_hours": total_delay_hours,
        "total_vkt": total_vkt,
        "total_vht": total_vht,
        "avg_speed_kph": avg_speed_kph,
        "relative_l1_threshold": float(relative_gap_threshold),
        "final_relative_l1_change": float(history_df.iloc[-1]["relative_l1_flow_change"]) if len(history_df) else 0.0,
        "stopping_rule_met": bool(
            len(history_df) >= int(min_iterations)
            and len(history_df)
            and history_df.iloc[-1]["relative_l1_flow_change"] <= float(relative_gap_threshold)
        ),
        "od_threshold_veh_h": float(od_threshold),
        "requested_total_vehicles": float(demand_values.sum()),
        "intrazonal_not_assigned_vehicles": intrazonal_demand,
        "requested_interzonal_vehicles": interzonal_demand,
        "retained_above_cutoff_vehicles": retained_demand,
        "retained_od_pairs": retained_pairs,
        "omitted_below_cutoff_vehicles": omitted_below_cutoff,
        "served_vehicles": float(served),
        "unserved_vehicles": float(unserved),
        "unserved_od_pairs": int(unserved_pairs),
        "served_share_of_retained": float(served) / max(retained_demand, 1e-9),
        "retained_trips": retained_demand,
        "max_vc": float(final_vc.max()) if len(final_vc) > 0 else 0.0,
        "mean_vc": float(final_vc.mean()) if len(final_vc) > 0 else 0.0,
    }
    metadata["stopping_rule"] = settings["stopping_rule"]
    if road_stopping:
        final_gap = float(road_checks[-1]["road_relative_gap_feasible_averaged_od"])
        metadata.update(
            road_gap_threshold=road_limit, check_every=int(check_every),
            final_road_relative_gap_feasible_averaged_od=final_gap,
            convergence_check_history=road_checks,
            stopping_rule_met=bool(final_gap <= road_limit),
        )
    metadata["termination_reason"] = "converged" if metadata["stopping_rule_met"] else "max_iterations"
    _report_unserved_road_demand(metadata)
    assigned_edges.attrs["metadata"] = metadata
    return assigned_edges, history_df, metadata


def _report_unserved_road_demand(diagnostics: dict[str, Any]) -> None:
    """Expose missing routes separately from assignment convergence."""
    import warnings

    unserved = float(diagnostics.get("unserved_vehicles", 0.0))
    routed = float(diagnostics.get("served_vehicles", 0.0)) + unserved
    diagnostics["unserved_share_of_routing_demand"] = unserved / max(routed, 1e-9)
    if unserved > 1e-6:
        warnings.warn(
            f"Road assignment could not route {unserved:,.1f} vehicles/hour "
            f"({diagnostics['unserved_share_of_routing_demand']:.1%} of routing demand). "
            "Inspect corridor connectivity and zone/gate attachments; a small road gap does not resolve missing routes.",
            UserWarning, stacklevel=2,
        )


def _synthetic_assignment_background(
    context: TransportContext,
    drive_passenger_od: pd.DataFrame,
    road_background_od: pd.DataFrame,
    drive_occupancy: float,
) -> pd.DataFrame:
    """Reproduce the legacy endpoint uplift before original zones become gates."""
    passenger = drive_passenger_od.copy()
    passenger.index = passenger.index.astype(str)
    passenger.columns = passenger.columns.astype(str)
    background = road_background_od.copy()
    background.index = background.index.astype(str)
    background.columns = background.columns.astype(str)
    background = background.reindex(
        index=passenger.index, columns=passenger.columns, fill_value=0.0
    )
    config = context.modules["config"]
    levels = context.zones.set_index(context.zones["grid_id"].astype(str))["Level"]

    def endpoint_factors(labels: pd.Index) -> np.ndarray:
        return np.where(
            levels.reindex(labels).astype(str).eq("Quartier"),
            config.EXTERNAL_BACKGROUND_CITY_FACTOR,
            config.EXTERNAL_BACKGROUND_CANTON_FACTOR,
        )

    uplift = (
        endpoint_factors(passenger.index)[:, None]
        + endpoint_factors(passenger.columns)[None, :]
    )
    internal = passenger / max(float(drive_occupancy), 1e-9) + background
    return internal * uplift


def run_corridor_assignment(
    context: TransportContext,
    corridor: CorridorContext,
    mode_result: ModeChoiceResult,
    *,
    passthrough_fraction: float = DEFAULT_PASSTHROUGH_FRACTION,
    drive_occupancy: float | None = None,
    max_iterations: int | None = None,
    min_iterations: int | None = None,
    relative_gap_threshold: float | None = None,
    od_threshold: float | None = None,
    background_multiplier: float = 1.0,
    stopping_rule: str | None = None,
    road_gap_threshold: float | None = None,
    check_every: int | None = None,
) -> AssignmentResult:
    """Collapse full-canton car demand and assign it to one local corridor.

    Passenger car trips are converted to vehicles using ``drive_occupancy``;
    prepared background road demand is already expressed as vehicles. The
    legacy synthetic endpoint uplift is computed before gate aggregation;
    all three components use the same internal-zone/gate representation.
    """

    settings = resolve_assignment_settings({name: value for name, value in {
        "drive_occupancy": drive_occupancy, "max_iterations": max_iterations,
        "min_iterations": min_iterations, "relative_gap_threshold": relative_gap_threshold,
        "od_threshold": od_threshold,
        "stopping_rule": stopping_rule, "road_gap_threshold": road_gap_threshold,
        "check_every": check_every,
    }.items() if value is not None})
    drive_occupancy = settings["drive_occupancy"]
    max_iterations, min_iterations = settings["max_iterations"], settings["min_iterations"]
    relative_gap_threshold, od_threshold = settings["relative_gap_threshold"], settings["od_threshold"]
    scaled_background = context.road_background_od * max(float(background_multiplier), 0.0)
    passenger, passenger_breakdown = collapse_od_to_gates(
        mode_result.drive_od,
        context,
        corridor,
        passthrough_fraction=passthrough_fraction,
    )
    background, background_breakdown = collapse_od_to_gates(
        scaled_background,
        context,
        corridor,
        passthrough_fraction=passthrough_fraction,
    )
    occupancy = max(float(drive_occupancy), 1e-9)
    synthetic, synthetic_breakdown = collapse_od_to_gates(
        _synthetic_assignment_background(
            context, mode_result.drive_od, scaled_background, occupancy
        ),
        context,
        corridor,
        passthrough_fraction=passthrough_fraction,
    )
    vehicle_demand = passenger / occupancy + background + synthetic

    breakdown = passenger_breakdown.rename(
        columns={"trips": "passenger_car_trips"}
    ).copy()
    breakdown["passenger_vehicles"] = breakdown["passenger_car_trips"] / occupancy
    breakdown["background_vehicles"] = background_breakdown["trips"].to_numpy()
    breakdown["synthetic_background_vehicles"] = synthetic_breakdown["trips"].to_numpy()
    breakdown["assignment_vehicles"] = (
        breakdown["passenger_vehicles"] + breakdown["background_vehicles"]
        + breakdown["synthetic_background_vehicles"]
    )

    links, history, diagnostics = coarse_msa_assignment(
        edges=corridor.edges,
        nodes=corridor.nodes,
        zone_node_map=corridor.zone_node_map,
        demand=vehicle_demand,
        max_iterations=int(max_iterations),
        min_iterations=int(min_iterations),
        relative_gap_threshold=float(relative_gap_threshold),
        od_threshold=float(od_threshold),
        stopping_rule=settings["stopping_rule"],
        road_gap_threshold=settings["road_gap_threshold"],
        check_every=settings["check_every"],
    )
    diagnostics.update(
        {
            "stage": int(mode_result.stage),
            "modal_feedback": False,
            "drive_occupancy": float(drive_occupancy),
            "passthrough_fraction": float(passthrough_fraction),
            "od_threshold_veh_h": float(od_threshold),
            "requested_total_vehicles": float(vehicle_demand.to_numpy().sum()),
            "drive_vehicle_trips": float(passenger.to_numpy().sum() / occupancy),
            "npvm_background_trips": float(background.to_numpy().sum()),
            "external_endpoint_background_trips": float(synthetic.to_numpy().sum()),
            "assignment_vehicle_trips": float(vehicle_demand.to_numpy().sum()),
            "demand_component_scope": "retained corridor OD before assignment cutoff",
            "mode_choice_cache_key": list(mode_result.cache_key),
        }
    )
    links.attrs["metadata"] = diagnostics
    return AssignmentResult(
        links=links,
        history=history,
        diagnostics=diagnostics,
        demand_matrix=vehicle_demand,
        demand_breakdown=breakdown,
        gate_totals=gate_totals(vehicle_demand, corridor),
    )


def _bpr_link_times(edges: pd.DataFrame, flow: np.ndarray) -> np.ndarray:
    """Return BPR travel time for every directed link at ``flow``."""

    free_flow = pd.to_numeric(
        edges["free_flow_time_min"], errors="coerce"
    ).fillna(1.0).to_numpy(dtype=float)
    capacity = pd.to_numeric(
        edges["capacity_vph"], errors="coerce"
    ).fillna(1.0).clip(lower=1.0).to_numpy(dtype=float)
    alpha = pd.to_numeric(
        edges.get("alpha", pd.Series(0.15, index=edges.index)), errors="coerce"
    ).fillna(0.15).to_numpy(dtype=float)
    beta = pd.to_numeric(
        edges.get("beta", pd.Series(4.0, index=edges.index)), errors="coerce"
    ).fillna(4.0).to_numpy(dtype=float)
    ratio = np.maximum(np.asarray(flow, dtype=float) / capacity, 0.0)
    return np.clip(free_flow * (1.0 + alpha * ratio ** beta), 1e-4, 500.0)


def _sparse_shortest_path_inputs(
    edges: pd.DataFrame,
    edge_times: np.ndarray,
) -> tuple[Any, np.ndarray, dict[int, int], dict[tuple[int, int], int]]:
    """Build a sparse graph while retaining the fastest parallel link."""

    from scipy.sparse import csr_matrix

    sources = pd.to_numeric(edges["source"], errors="raise").to_numpy(dtype=int)
    targets = pd.to_numeric(edges["target"], errors="raise").to_numpy(dtype=int)
    unique_nodes = np.unique(np.concatenate([sources, targets]))
    node_to_position = {int(node): i for i, node in enumerate(unique_nodes)}
    source_positions = np.asarray([node_to_position[int(node)] for node in sources])
    target_positions = np.asarray([node_to_position[int(node)] for node in targets])

    # scipy sums duplicate sparse entries.  Road networks can contain parallel
    # links, so select the currently fastest one before creating the matrix.
    edge_lookup: dict[tuple[int, int], int] = {}
    for edge_position, (source, target) in enumerate(
        zip(source_positions, target_positions)
    ):
        pair = (int(source), int(target))
        previous = edge_lookup.get(pair)
        if previous is None or edge_times[edge_position] < edge_times[previous]:
            edge_lookup[pair] = edge_position
    selected_positions = np.fromiter(edge_lookup.values(), dtype=int)
    graph = csr_matrix(
        (
            np.asarray(edge_times, dtype=float)[selected_positions],
            (
                source_positions[selected_positions],
                target_positions[selected_positions],
            ),
        ),
        shape=(len(unique_nodes), len(unique_nodes)),
    )
    return graph, unique_nodes, node_to_position, edge_lookup


def _all_or_nothing_sparse(
    edges: pd.DataFrame,
    zone_node_map: Mapping[str, int],
    demand: pd.DataFrame,
    edge_times: np.ndarray,
    *,
    od_threshold: float,
) -> tuple[np.ndarray, float, float, int]:
    """Load retained OD demand along the native shortest-path predecessor trees.

    Tree sums avoid retracing every small OD path. Graph construction, directed
    routes and fastest-parallel-link tie-breaking remain unchanged. Link times
    follow the positive BPR contract; only summation order differs numerically.
    """

    from scipy.sparse.csgraph import shortest_path

    graph, _, node_to_position, edge_lookup = _sparse_shortest_path_inputs(
        edges, edge_times
    )
    labels = demand.index.astype(str).to_numpy()
    values = demand.to_numpy(dtype=float)
    retained = np.isfinite(values) & (values > float(od_threshold))
    np.fill_diagonal(retained, False)
    origins, destinations = np.nonzero(retained)

    origin_positions = np.asarray(
        [node_to_position.get(int(zone_node_map.get(labels[i], -1)), -1) for i in origins],
        dtype=int,
    )
    destination_positions = np.asarray(
        [node_to_position.get(int(zone_node_map.get(labels[j], -1)), -1) for j in destinations],
        dtype=int,
    )
    trips = values[origins, destinations]
    valid = (origin_positions >= 0) & (destination_positions >= 0)
    unserved = float(trips[~valid].sum())
    unserved_pairs = int((~valid).sum())
    same_node = valid & (origin_positions == destination_positions)
    # Different zone labels can share one routing connector.  Their demand is
    # served locally without loading any physical road link.
    served = float(trips[same_node].sum())
    routed = valid & ~same_node
    origin_positions = origin_positions[routed]
    destination_positions = destination_positions[routed]
    trips = trips[routed]

    flow = np.zeros(len(edges), dtype=float)
    unique_origins = np.unique(origin_positions)
    if not len(unique_origins):
        return flow, served, unserved, unserved_pairs
    _, predecessors = shortest_path(
        graph, directed=True, indices=unique_origins, return_predecessors=True,
    )
    node_count = graph.shape[0]
    pair_keys = np.fromiter((a * node_count + b for a, b in edge_lookup), dtype=np.int64)
    pair_edges = np.fromiter(edge_lookup.values(), dtype=np.int64)
    edge_order = np.argsort(pair_keys)
    pair_keys, pair_edges = pair_keys[edge_order], pair_edges[edge_order]
    for row, origin in enumerate(unique_origins):
        selected = origin_positions == origin
        destinations_here, trips_here = destination_positions[selected], trips[selected]
        pred = predecessors[row]
        connected = pred[destinations_here] >= 0
        unserved += float(trips_here[~connected].sum())
        unserved_pairs += int((~connected).sum())
        served += float(trips_here[connected].sum())
        if not connected.any():
            continue

        # At jump j, send each node's descendants at depths 0..2**j-1 to
        # its 2**j-th ancestor. Disjoint depth bands count each destination
        # once on every ancestral edge, using logarithmically many array sums.
        node_demand = np.bincount(destinations_here[connected],
                                  weights=trips_here[connected], minlength=node_count)
        ancestor = np.where(pred >= 0, pred, -1)
        for _jump in range(int(np.ceil(np.log2(max(node_count, 2)))) + 1):
            valid_ancestor = ancestor >= 0
            if not valid_ancestor.any():
                break
            node_demand += np.bincount(ancestor[valid_ancestor],
                                      weights=node_demand[valid_ancestor], minlength=node_count)
            next_ancestor = np.full(node_count, -1, dtype=int)
            next_ancestor[valid_ancestor] = ancestor[ancestor[valid_ancestor]]
            ancestor = next_ancestor
        else:
            raise ValueError("Predecessor tree did not terminate at its root.")
        children = np.flatnonzero((pred >= 0) & (node_demand > 0))
        keys = pred[children].astype(np.int64) * node_count + children
        positions = np.searchsorted(pair_keys, keys)
        if np.any(positions >= len(pair_keys)) or np.any(pair_keys[positions] != keys):
            raise ValueError("Predecessor edge is missing from the native edge lookup.")
        np.add.at(flow, pair_edges[positions], node_demand[children])
    return flow, served, unserved, unserved_pairs


def _corridor_shortest_time_table(
    edges: pd.DataFrame,
    zone_node_map: Mapping[str, int],
    edge_times: np.ndarray,
) -> pd.DataFrame:
    """Shortest-path minutes between every internal-zone/gate routing node."""

    from scipy.sparse.csgraph import shortest_path

    graph, _, node_to_position, _ = _sparse_shortest_path_inputs(edges, edge_times)
    labels = [str(label) for label in zone_node_map]
    valid_labels = [
        label for label in labels if int(zone_node_map[label]) in node_to_position
    ]
    node_positions = np.asarray(
        [node_to_position[int(zone_node_map[label])] for label in valid_labels],
        dtype=int,
    )
    distances = shortest_path(
        graph, directed=True, indices=node_positions, return_predecessors=False
    )
    distances = distances[:, node_positions]
    return pd.DataFrame(distances, index=valid_labels, columns=valid_labels)


def _project_corridor_delay_to_full_od(
    context: TransportContext,
    corridor: CorridorContext,
    delay_table: pd.DataFrame,
    labels: pd.Index,
) -> pd.DataFrame:
    """Map node-specific corridor delay back to each full-canton OD pair."""

    internal = set(corridor.zone_ids)
    origin_labels = [
        label if label in internal else corridor.external_zone_to_entry_gate.get(label)
        for label in labels.astype(str)
    ]
    destination_labels = [
        label if label in internal else corridor.external_zone_to_exit_gate.get(label)
        for label in labels.astype(str)
    ]
    lookup = {label: i for i, label in enumerate(delay_table.index.astype(str))}
    origin_positions = np.asarray([lookup.get(label, -1) for label in origin_labels])
    destination_positions = np.asarray([lookup.get(label, -1) for label in destination_labels])
    valid = (origin_positions[:, None] >= 0) & (destination_positions[None, :] >= 0)
    projected = np.zeros((len(labels), len(labels)), dtype=float)
    safe_origins = np.maximum(origin_positions, 0)
    safe_destinations = np.maximum(destination_positions, 0)
    candidates = delay_table.to_numpy(dtype=float)[
        safe_origins[:, None], safe_destinations[None, :]
    ]
    projected[valid] = candidates[valid]
    projected[~np.isfinite(projected)] = 0.0
    projected = np.maximum(projected, 0.0)
    return pd.DataFrame(projected, index=labels, columns=labels)


def run_coupled_corridor_assignment(
    context: TransportContext,
    corridor: CorridorContext,
    mode_result: ModeChoiceResult,
    *,
    demand_multiplier: float = 1.0,
    background_multiplier: float = 1.0,
    passthrough_fraction: float = DEFAULT_PASSTHROUGH_FRACTION,
    drive_occupancy: float | None = None,
    max_iterations: int | None = None,
    min_iterations: int | None = None,
    relative_gap_threshold: float | None = None,
    flow_gap_threshold: float | None = None,
    demand_gap_threshold: float | None = None,
    od_threshold: float | None = None,
    stopping_rule: str | None = None,
    road_gap_threshold: float | None = None,
    check_every: int | None = None,
) -> AssignmentResult:
    """Solve link flows and five-mode demand together in one MSA loop.

    ``flow_gap_threshold``/``demand_gap_threshold`` let a caller require
    different precision for road-flow versus modal-split/demand convergence.
    Each defaults to ``relative_gap_threshold`` when omitted, so passing
    neither reproduces the previous single-threshold behaviour exactly
    (``max(flow_change, demand_change) <= relative_gap_threshold`` is
    equivalent to checking both changes against the same threshold).

    Default ``stopping_rule="road_relative_gap"`` tests the true road
    relative gap against ``road_gap_threshold`` (or ``relative_gap_threshold``).
    Checks start at ``min_iterations`` and repeat every ``check_every`` steps,
    including the final cap. Its feasible OD is the average actually loaded
    onto paths, not the separately averaged current mode-choice demand. This
    rule assesses road assignment only; it does not certify joint modal-demand
    equilibrium. The surrogate builder configures its own bounds/tolerance.

    Each iteration obtains OD-specific corridor delays from the same road graph
    used by assignment.  Those delays update the car skim, the multinomial
    logit model updates car demand, and both flows and demand are averaged with
    the classical ``1 / iteration`` MSA step.  Only delay inside the local
    cordon is fed back; the remainder of every whole-trip skim stays fixed.
    """

    settings = resolve_assignment_settings({name: value for name, value in {
        "drive_occupancy": drive_occupancy, "max_iterations": max_iterations,
        "min_iterations": min_iterations, "relative_gap_threshold": relative_gap_threshold,
        "od_threshold": od_threshold, "stopping_rule": stopping_rule,
        "road_gap_threshold": road_gap_threshold, "check_every": check_every,
        "flow_gap_threshold": flow_gap_threshold, "demand_gap_threshold": demand_gap_threshold,
    }.items() if value is not None})
    drive_occupancy = settings["drive_occupancy"]
    max_iterations, min_iterations = settings["max_iterations"], settings["min_iterations"]
    relative_gap_threshold, od_threshold = settings["relative_gap_threshold"], settings["od_threshold"]
    stopping_rule, road_gap_threshold = settings["stopping_rule"], settings["road_gap_threshold"]
    check_every = settings["check_every"]
    flow_gap_threshold, demand_gap_threshold = settings["flow_gap_threshold"], settings["demand_gap_threshold"]
    if stopping_rule not in {"successive_change", "road_relative_gap"}:
        raise ValueError("Unknown coupled-assignment stopping_rule.")
    road_stopping = stopping_rule == "road_relative_gap"
    road_limit = float(relative_gap_threshold if road_gap_threshold is None else road_gap_threshold)
    if road_stopping and (
        int(check_every) != check_every or check_every < 1
        or int(min_iterations) != min_iterations or int(max_iterations) != max_iterations
        or not 1 <= min_iterations <= max_iterations
        or not np.isfinite(road_limit) or road_limit < 0
    ):
        raise ValueError("Road-gap stopping requires valid iteration bounds, cadence and nonnegative tolerance.")
    started = time.perf_counter()
    edges = corridor.edges.copy().reset_index(drop=True)
    flow = np.zeros(len(edges), dtype=float)
    occupancy = max(float(drive_occupancy), 1e-9)
    passenger_scale = max(float(demand_multiplier), 0.0)
    background_scale = max(float(background_multiplier), 0.0)

    # Sum the five mode-specific matrices, then grow every OD pair uniformly.
    total_od = sum(
        (frame.copy() for frame in mode_result.od_by_mode.values()),
        start=pd.DataFrame(
            0.0,
            index=mode_result.drive_od.index,
            columns=mode_result.drive_od.columns,
        ),
    ) * passenger_scale
    labels = total_od.index.astype(str)
    total_od.index = labels
    total_od.columns = labels

    source_times = mode_result.uncongested_travel_times or mode_result.travel_times
    base_times = {name: frame.copy() for name, frame in source_times.items()}
    lengths = {name: frame.copy() for name, frame in mode_result.lengths.items()}
    drive_key = "drive" if "drive" in base_times else "car"
    if drive_key not in base_times:
        raise KeyError("The mode-choice result has no drive/car travel-time skim.")
    base_drive_time = base_times[drive_key].reindex(index=labels, columns=labels)

    mode_module = context.modules["mode_choice_zurich"]
    travel_time_module = context.modules["travel_times"]
    parameters = mode_module.load_mode_choice_parameters()
    for key in list(parameters):
        if key in mode_result.scenario:
            parameters[key] = mode_result.scenario[key]
    walk_mask = travel_time_module.standalone_walk_mask(lengths, context.zones)

    # Prepared background is fixed with respect to mode choice. The legacy
    # synthetic uplift follows passenger demand and is computed before gates.
    # Passenger and non-passenger road demand have separate causal drivers.
    # Freight/commercial vehicles do not participate in passenger mode choice.
    scaled_background = context.road_background_od * background_scale
    background, background_breakdown = collapse_od_to_gates(
        scaled_background,
        context,
        corridor,
        passthrough_fraction=passthrough_fraction,
    )
    initial_passenger, _ = collapse_od_to_gates(
        mode_result.drive_od * passenger_scale,
        context,
        corridor,
        passthrough_fraction=passthrough_fraction,
    )
    passenger_vehicle_matrix = initial_passenger / occupancy
    synthetic, _ = collapse_od_to_gates(
        _synthetic_assignment_background(
            context, mode_result.drive_od * passenger_scale, scaled_background, occupancy
        ),
        context,
        corridor,
        passthrough_fraction=passthrough_fraction,
    )
    vehicle_demand = passenger_vehicle_matrix + background + synthetic

    free_flow_times = _bpr_link_times(edges, np.zeros(len(edges), dtype=float))
    free_flow_table = _corridor_shortest_time_table(
        edges, corridor.zone_node_map, free_flow_times
    )
    averaged_assigned_od = None
    if road_stopping:
        feasible = np.isfinite(free_flow_table.reindex(
            index=vehicle_demand.index, columns=vehicle_demand.columns
        ).to_numpy(dtype=float))
        np.fill_diagonal(feasible, False)
    road_checks: list[dict[str, float]] = []
    road_check_table = road_check_times = None
    history_rows: list[dict[str, float]] = []
    latest_od_by_mode = {
        name: frame * passenger_scale for name, frame in mode_result.od_by_mode.items()
    }
    latest_drive_time = base_drive_time.copy()
    served = unserved = 0.0
    unserved_pairs = 0

    for iteration in range(1, int(max_iterations) + 1):
        edge_times = _bpr_link_times(edges, flow)
        auxiliary, served, unserved, unserved_pairs = _all_or_nothing_sparse(
            edges,
            corridor.zone_node_map,
            vehicle_demand,
            edge_times,
            od_threshold=float(od_threshold),
        )
        step = 1.0 / float(iteration)
        if road_stopping:
            # AoN above used the PRE-update demand; retain exactly its feasible
            # positive off-diagonal OD, including distinct labels at one node.
            values = vehicle_demand.to_numpy(dtype=float)
            assigned = np.where(feasible & np.isfinite(values) & (values > float(od_threshold)), values, 0.0)
            if averaged_assigned_od is None:
                averaged_assigned_od = assigned.copy()
            else:
                averaged_assigned_od += step * (assigned - averaged_assigned_od)
        updated_flow = flow + step * (auxiliary - flow)
        flow_change = float(np.abs(updated_flow - flow).sum()) / max(
            float(np.abs(flow).sum()), 1e-9
        )

        current_table = _corridor_shortest_time_table(
            edges, corridor.zone_node_map, edge_times
        )
        node_delay = (current_table - free_flow_table).clip(lower=0.0)
        full_delay = _project_corridor_delay_to_full_od(
            context, corridor, node_delay, labels
        )
        latest_drive_time = base_drive_time + full_delay
        updated_times = {name: frame.copy() for name, frame in base_times.items()}
        updated_times[drive_key] = latest_drive_time

        latest_od_by_mode, _ = mode_module.mode_split_aggregated(
            updated_times,
            lengths,
            total_od,
            walk_allowed_mask=walk_mask,
            parameters=parameters,
        )
        passenger_target, _ = collapse_od_to_gates(
            latest_od_by_mode["drive"],
            context,
            corridor,
            passthrough_fraction=passthrough_fraction,
        )
        synthetic_target, _ = collapse_od_to_gates(
            _synthetic_assignment_background(
                context, latest_od_by_mode["drive"], scaled_background, occupancy
            ),
            context,
            corridor,
            passthrough_fraction=passthrough_fraction,
        )
        target_vehicle_demand = passenger_target / occupancy + background + synthetic_target
        updated_demand = vehicle_demand + step * (target_vehicle_demand - vehicle_demand)
        demand_change = float(
            np.abs(updated_demand.to_numpy() - vehicle_demand.to_numpy()).sum()
        ) / max(float(np.abs(vehicle_demand.to_numpy()).sum()), 1e-9)

        mode_summary = _corridor_mode_summary(latest_od_by_mode, corridor.zone_ids)
        shares = mode_summary.set_index("mode")["share"].to_dict()
        trip_weights = total_od.to_numpy(dtype=float)
        mean_delay = float(
            (full_delay.to_numpy(dtype=float) * trip_weights).sum()
            / max(trip_weights.sum(), 1e-9)
        )
        history_rows.append(
            {
                "iteration": iteration,
                "msa_step": step,
                "relative_l1_flow_change": flow_change,
                "relative_l1_demand_change": demand_change,
                "car_share": shares.get("Car (Driver)", np.nan),
                "pt_share": shares.get("Public Transport", np.nan),
                "bike_share": shares.get("Bicycle", np.nan),
                "walk_share": shares.get("Walking", np.nan),
                "demand_weighted_corridor_delay_min": mean_delay,
                "served_vehicles": served,
            }
        )
        passenger_vehicle_matrix += step * (passenger_target / occupancy - passenger_vehicle_matrix)
        synthetic += step * (synthetic_target - synthetic)
        flow, vehicle_demand = updated_flow, updated_demand
        if road_stopping:
            if iteration == int(max_iterations) or (
                iteration >= int(min_iterations)
                and (iteration - int(min_iterations)) % int(check_every) == 0
            ):
                road_check_times = _bpr_link_times(edges, flow)
                road_check_table = _corridor_shortest_time_table(
                    edges, corridor.zone_node_map, road_check_times
                )
                shortest = road_check_table.reindex(
                    index=vehicle_demand.index, columns=vehicle_demand.columns
                ).to_numpy(dtype=float)
                minimum_time = float(np.sum(averaged_assigned_od * np.where(np.isfinite(shortest), shortest, 0.0)))
                actual_time = float(np.dot(flow, road_check_times))
                gap = (actual_time - minimum_time) / max(actual_time, 1e-12)
                if not np.isfinite(gap) or gap < -1e-8:
                    raise ValueError("Road gap is inconsistent with the actually assigned feasible OD.")
                gap = max(gap, 0.0)
                road_checks.append({"iteration": iteration,
                    "road_relative_gap_feasible_averaged_od": gap,
                    "averaged_assigned_feasible_vehicles": float(averaged_assigned_od.sum())})
                history_rows[-1]["road_relative_gap_feasible_averaged_od"] = gap
                if gap <= road_limit:
                    break
            continue
        effective_flow_threshold = (
            float(relative_gap_threshold) if flow_gap_threshold is None
            else float(flow_gap_threshold)
        )
        effective_demand_threshold = (
            float(relative_gap_threshold) if demand_gap_threshold is None
            else float(demand_gap_threshold)
        )
        if (
            iteration >= int(min_iterations)
            and flow_change <= effective_flow_threshold
            and demand_change <= effective_demand_threshold
        ):
            break

    final_times = road_check_times if road_stopping else _bpr_link_times(edges, flow)
    # Re-evaluate the skim and modal split once at the final averaged flow so
    # the returned behavioural result corresponds to the displayed link times.
    final_table = road_check_table if road_stopping else _corridor_shortest_time_table(
        edges, corridor.zone_node_map, final_times
    )
    local_delay = (final_table - free_flow_table).clip(lower=0.0)
    local_delay = local_delay.where(np.isfinite(local_delay), 0.0)
    final_delay = _project_corridor_delay_to_full_od(
        context,
        corridor,
        (final_table - free_flow_table).clip(lower=0.0),
        labels,
    )
    latest_drive_time = base_drive_time + final_delay
    final_mode_times = {name: frame.copy() for name, frame in base_times.items()}
    final_mode_times[drive_key] = latest_drive_time
    latest_od_by_mode, _ = mode_module.mode_split_aggregated(
        final_mode_times,
        lengths,
        total_od,
        walk_allowed_mask=walk_mask,
        parameters=parameters,
    )
    capacity = pd.to_numeric(edges["capacity_vph"], errors="coerce").clip(lower=1.0)
    free_flow = pd.to_numeric(edges["free_flow_time_min"], errors="coerce")
    length_km = pd.to_numeric(edges["length_m"], errors="coerce").fillna(0.0) / 1000.0
    links = edges.copy()
    links["flow_vehicles"] = flow
    links["time_min"] = final_times
    links["volume_capacity_ratio"] = flow / capacity.to_numpy(dtype=float)
    links["delay_min"] = np.maximum(final_times - free_flow.to_numpy(dtype=float), 0.0)
    links["assigned_speed_kph"] = np.divide(
        length_km.to_numpy(dtype=float),
        final_times / 60.0,
        out=np.zeros(len(links), dtype=float),
        where=final_times > 0,
    )
    history = pd.DataFrame(history_rows)
    final_flow_change = float(history.iloc[-1]["relative_l1_flow_change"])
    final_demand_change = float(history.iloc[-1]["relative_l1_demand_change"])
    total_delay_hours = float((flow * links["delay_min"].to_numpy() / 60.0).sum())
    total_vkt = float((flow * length_km.to_numpy()).sum())
    total_vht = float((flow * final_times / 60.0).sum())

    # Report the separately averaged mode-choice demand and its components.
    # Road-gap checks use the actually loaded OD average tracked above; these
    # two averages can still differ at a finite iteration count.
    internal = corridor.zone_ids
    gates = corridor.gates["gate_id"].astype(str).tolist()
    passenger_categories = [
        float(passenger_vehicle_matrix.loc[internal, internal].to_numpy().sum()),
        float(passenger_vehicle_matrix.loc[gates, internal].to_numpy().sum()),
        float(passenger_vehicle_matrix.loc[internal, gates].to_numpy().sum()),
        float(passenger_vehicle_matrix.loc[gates, gates].to_numpy().sum()),
    ]
    synthetic_categories = [
        float(synthetic.loc[internal, internal].to_numpy().sum()),
        float(synthetic.loc[gates, internal].to_numpy().sum()),
        float(synthetic.loc[internal, gates].to_numpy().sum()),
        float(synthetic.loc[gates, gates].to_numpy().sum()),
    ]
    breakdown = pd.DataFrame(
        {
            "category": background_breakdown["category"],
            "passenger_car_trips": np.asarray(passenger_categories) * occupancy,
            "passenger_vehicles": passenger_categories,
            "background_vehicles": background_breakdown["trips"].to_numpy(),
            "synthetic_background_vehicles": synthetic_categories,
        }
    )
    breakdown["assignment_vehicles"] = (
        breakdown["passenger_vehicles"] + breakdown["background_vehicles"]
        + breakdown["synthetic_background_vehicles"]
    )
    updated_times = {name: frame.copy() for name, frame in base_times.items()}
    updated_times[drive_key] = latest_drive_time
    coupled_mode_result = ModeChoiceResult(
        drive_od=latest_od_by_mode["drive"],
        od_by_mode=latest_od_by_mode,
        summary=_corridor_mode_summary(latest_od_by_mode, corridor.zone_ids),
        scenario=dict(mode_result.scenario),
        travel_times=updated_times,
        lengths=lengths,
        cache_key=tuple(mode_result.cache_key) + ("coupled_msa",),
        assigned_edges=links,
        uncongested_travel_times=base_times,
        technology_base_times=getattr(mode_result, "technology_base_times", None),
    )
    diagnostics = {
        "algorithm": "coupled MSA/BPR assignment and five-mode logit",
        "modal_feedback": True,
        "runtime_s": time.perf_counter() - started,
        "iterations": len(history),
        "max_iterations": int(max_iterations),
        "min_iterations": int(min_iterations),
        "relative_l1_threshold": float(relative_gap_threshold),
        "flow_gap_threshold": (
            float(relative_gap_threshold) if flow_gap_threshold is None
            else float(flow_gap_threshold)
        ),
        "demand_gap_threshold": (
            float(relative_gap_threshold) if demand_gap_threshold is None
            else float(demand_gap_threshold)
        ),
        "final_relative_l1_flow_change": final_flow_change,
        "final_relative_l1_demand_change": final_demand_change,
        "stopping_rule_met": bool(
            len(history) >= int(min_iterations)
            and final_flow_change <= (
                float(relative_gap_threshold) if flow_gap_threshold is None
                else float(flow_gap_threshold)
            )
            and final_demand_change <= (
                float(relative_gap_threshold) if demand_gap_threshold is None
                else float(demand_gap_threshold)
            )
        ),
        "od_threshold_veh_h": float(od_threshold),
        "passthrough_fraction": float(passthrough_fraction),
        "drive_occupancy": float(drive_occupancy),
        "passenger_demand_multiplier": passenger_scale,
        "road_freight_multiplier": background_scale,
        "requested_total_vehicles": float(vehicle_demand.to_numpy().sum()),
        "drive_vehicle_trips": float(passenger_vehicle_matrix.to_numpy().sum()),
        "npvm_background_trips": float(background.to_numpy().sum()),
        "external_endpoint_background_trips": float(synthetic.to_numpy().sum()),
        "assignment_vehicle_trips": float(vehicle_demand.to_numpy().sum()),
        "demand_component_scope": "retained corridor OD before assignment cutoff",
        "served_vehicles": float(served),
        "unserved_vehicles": float(unserved),
        "unserved_od_pairs": int(unserved_pairs),
        "total_delay_hours": total_delay_hours,
        "total_vkt": total_vkt,
        "total_vht": total_vht,
        "avg_speed_kph": total_vkt / max(total_vht, 1e-9),
        "corridor_delay_feedback": (
            "OD-specific local-network delay added to the fixed whole-trip drive skim"
        ),
    }
    diagnostics["stopping_rule"] = stopping_rule
    if road_stopping:
        final_gap = float(road_checks[-1]["road_relative_gap_feasible_averaged_od"])
        diagnostics.update(
            road_gap_threshold=road_limit, check_every=int(check_every),
            final_road_relative_gap_feasible_averaged_od=final_gap,
            convergence_check_history=road_checks,
            stopping_rule_met=bool(final_gap <= road_limit),
        )
    diagnostics["termination_reason"] = "converged" if diagnostics["stopping_rule_met"] else "max_iterations"
    _report_unserved_road_demand(diagnostics)
    coupled_mode_result.assigned_metadata = diagnostics
    links.attrs["metadata"] = diagnostics
    return AssignmentResult(
        links=links,
        history=history,
        diagnostics=diagnostics,
        demand_matrix=vehicle_demand,
        demand_breakdown=breakdown,
        gate_totals=gate_totals(vehicle_demand, corridor),
        mode_result=coupled_mode_result,
        local_od_delay_min=local_delay,
    )


def transport_state_from_surrogate_od(
    context: TransportContext,
    corridor: CorridorContext,
    delay_table: pd.DataFrame,
    *,
    stage: int,
    stage_specs: Mapping[int, Mapping[str, Any]],
    passenger_demand_multiplier: float,
    pt_asc_shift: float = 0.0,
    bike_asc_shift: float = 0.0,
    road_freight_multiplier: float = 1.0,
    ebike_share: float | None = None,
    include_welfare: bool = False,
    base_mode_result: ModeChoiceResult | None = None,
    free_flow_table: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Reconstruct native mode choice and matched welfare from road OD delays.

    Rows/columns are ordered directional routing-zone/gate IDs; entries are
    congestion minutes added to the exact stage car skim. Non-car skims and
    all five modal OD quantities stay in the native pipeline. Link-only totals
    cannot be recovered from this representation and are deliberately absent.
    """
    stage = int(stage)
    table = delay_table.copy()
    table.index = table.index.astype(str)
    table.columns = table.columns.astype(str)
    if table.index.has_duplicates or not table.index.equals(table.columns):
        raise ValueError("OD delay rows/columns must be identical unique ordered labels.")
    if free_flow_table is not None:
        expected_labels = free_flow_table.index.astype(str)
        if not expected_labels.equals(free_flow_table.columns.astype(str)):
            raise ValueError("Free-flow OD rows and columns differ.")
    else:
        nodes = set(corridor.edges["source"].astype(int)) | set(corridor.edges["target"].astype(int))
        expected_labels = pd.Index([str(label) for label, node in corridor.zone_node_map.items() if int(node) in nodes])
    if not table.index.equals(expected_labels):
        raise ValueError("OD delay layout does not match the selected stage's routing nodes.")
    raw = table.to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("Predicted OD delay must be finite.")
    clipped = int(np.count_nonzero(raw < 0.0))
    table = table.clip(lower=0.0)
    np.fill_diagonal(table.values, 0.0)
    passenger_scale = float(passenger_demand_multiplier)
    if not np.isfinite(passenger_scale) or passenger_scale < 0:
        raise ValueError("Passenger demand multiplier must be finite and nonnegative.")
    mode = base_mode_result
    if mode is None:
        mode = run_transport_mode_choice(
            context, stage=stage, stage_specs=stage_specs,
            corridor_zone_ids=corridor.zone_ids, demand_multiplier=1.0,
            pt_asc_shift=0.0, bike_asc_shift=0.0,
        )
    if ebike_share is not None:
        mode = mode_choice_with_ebike_share(context, mode, ebike_share, corridor.zone_ids)
    labels = mode.drive_od.index.astype(str)
    base_times = mode.uncongested_travel_times or mode.travel_times
    updated_times = {name: frame.copy() for name, frame in base_times.items()}
    drive_key = "drive" if "drive" in base_times else "car"
    if drive_key not in base_times:
        raise KeyError("The mode-choice result has no drive/car travel-time skim.")
    full_delay = _project_corridor_delay_to_full_od(context, corridor, table, labels)
    updated_times[drive_key] = base_times[drive_key].reindex(index=labels, columns=labels) + full_delay
    total_od = sum(
        (frame.copy() for frame in mode.od_by_mode.values()),
        start=pd.DataFrame(0.0, index=mode.drive_od.index, columns=mode.drive_od.columns),
    ) * passenger_scale
    parameters = context.modules["mode_choice_zurich"].load_mode_choice_parameters()
    for key in list(parameters):
        if key in mode.scenario:
            parameters[key] = mode.scenario[key]
    for key in ("ASC_PT_WALK", "ASC_PT_BIKE"):
        parameters[key] = float(parameters[key]) + float(pt_asc_shift)
    parameters["ASC_BIKE"] = float(parameters["ASC_BIKE"]) + float(bike_asc_shift)
    walk_mask = context.modules["travel_times"].standalone_walk_mask(mode.lengths, context.zones)
    od_by_mode, _ = context.modules["mode_choice_zurich"].mode_split_aggregated(
        updated_times, mode.lengths, total_od,
        walk_allowed_mask=walk_mask, parameters=parameters,
    )
    updated_mode = ModeChoiceResult(
        drive_od=od_by_mode["drive"], od_by_mode=od_by_mode,
        summary=_corridor_mode_summary(od_by_mode, corridor.zone_ids),
        scenario={**dict(mode.scenario), "ASC_PT_WALK": parameters["ASC_PT_WALK"],
                  "ASC_PT_BIKE": parameters["ASC_PT_BIKE"], "ASC_BIKE": parameters["ASC_BIKE"]},
        travel_times=updated_times, lengths=mode.lengths,
        cache_key=tuple(mode.cache_key) + ("surrogate_msa_od_delay",),
        uncongested_travel_times=base_times,
        technology_base_times=getattr(mode, "technology_base_times", None),
        assigned_metadata={"drive_occupancy": float(ASSIGNMENT_SETTINGS.get("drive_occupancy", 1.14)),
                           "passenger_demand_multiplier": passenger_scale,
                           "road_freight_multiplier": float(road_freight_multiplier)},
    )
    state: dict[str, Any] = {
        "schema_version": 1, "source": "surrogate_msa", "stage": stage,
        "representation": "directional internal-zone/gate OD delay minutes",
        "od_delay_clipped_negative_count": clipped,
        "metrics": extract_corridor_metrics(context, updated_mode, corridor.zone_ids,
                                            min_distance_km=stage_specs[stage].get("_min_distance_km")),
    }
    if include_welfare:
        state["welfare_od"] = _compact_welfare_od_sample(context, updated_mode, corridor.zone_ids)
    return state


def _assignment_convergence_run(
    run: int,
    *,
    seed: int,
    passenger_base: pd.DataFrame,
    background_base: pd.DataFrame,
    gate_ids: list[str],
    edges: Any,
    zone_node_map: Mapping[str, int],
    max_iterations: int,
    min_iterations: int,
    road_gap_threshold: float,
    check_every: int,
    od_threshold: float,
) -> pd.DataFrame:
    """Run one independent demand perturbation (parallel-worker helper)."""

    rng = np.random.default_rng(int(seed) + int(run))
    demand_multiplier = float(rng.uniform(0.80, 1.25))
    passthrough = float(rng.uniform(0.02, 0.12))

    passenger = passenger_base.copy()
    background = background_base.copy()
    if gate_ids:
        passenger.loc[gate_ids, gate_ids] *= passthrough
        background.loc[gate_ids, gate_ids] *= passthrough
    passenger *= demand_multiplier * rng.lognormal(
        mean=0.0, sigma=0.12, size=passenger.shape
    )
    background *= demand_multiplier

    _, history, _ = coarse_msa_assignment(
        edges=edges,
        zone_node_map=zone_node_map,
        demand=passenger + background,
        max_iterations=int(max_iterations),
        min_iterations=int(min_iterations),
        stopping_rule="road_relative_gap",
        road_gap_threshold=float(road_gap_threshold),
        check_every=int(check_every),
        od_threshold=float(od_threshold),
    )
    history = history.copy()
    history["run"] = int(run) + 1
    history["demand_multiplier"] = demand_multiplier
    history["passthrough_fraction"] = passthrough
    return history


def assignment_convergence_test(
    context: TransportContext,
    corridor: CorridorContext,
    mode_result: ModeChoiceResult,
    *,
    n_runs: int = 6,
    max_iterations: int | None = None,
    min_iterations: int | None = None,
    road_gap_threshold: float | None = None,
    check_every: int | None = None,
    thresholds: tuple[float, ...] = (0.10, 0.05, 0.02),
    od_threshold: float | None = None,
    seed: int = 42,
    n_jobs: int = 1,
) -> dict[str, pd.DataFrame]:
    """Small Monte Carlo diagnostic of road relative-gap convergence.

    The test perturbs total car demand, individual retained OD cells, and the
    uncertain pass-through share. Each run stops at the configured road gap
    or iteration cap. Summaries contain the runs still observed at each check.
    This is a numerical stability diagnostic, not uncertainty
    analysis for the infrastructure project. Independent runs use separate
    worker processes when ``n_jobs`` is greater than one.
    """

    settings = resolve_assignment_settings({"stopping_rule": "road_relative_gap", **{
        key: value for key, value in {
            "max_iterations": max_iterations, "min_iterations": min_iterations,
            "road_gap_threshold": road_gap_threshold, "check_every": check_every,
            "od_threshold": od_threshold,
        }.items() if value is not None
    }})
    if int(n_runs) != n_runs or int(n_runs) < 1:
        raise ValueError("n_runs must be a positive integer.")
    if int(n_jobs) == 0:
        raise ValueError("n_jobs cannot be zero; use 1 or a negative value for all cores.")

    occupancy = max(float(ASSIGNMENT_SETTINGS["drive_occupancy"]), 1e-9)
    # Include each component's endpoint uplift before collapsing. Workers
    # receive vehicles and only scale the smaller gate-level demand matrices.
    drive_od = mode_result.drive_od.rename(index=str, columns=str)
    road_background = context.road_background_od.rename(index=str, columns=str).reindex(
        index=drive_od.index, columns=drive_od.columns, fill_value=0.0
    )
    passenger_vehicles = drive_od / occupancy
    passenger_vehicles += _synthetic_assignment_background(
        context, drive_od, road_background * 0.0, occupancy
    )
    background_vehicles = road_background + _synthetic_assignment_background(
        context, drive_od * 0.0, road_background, occupancy
    )
    passenger_base, _ = collapse_od_to_gates(
        passenger_vehicles, context, corridor, passthrough_fraction=1.0
    )
    background_base, _ = collapse_od_to_gates(
        background_vehicles, context, corridor, passthrough_fraction=1.0
    )
    worker_kwargs = {
        "seed": int(seed),
        "passenger_base": passenger_base,
        "background_base": background_base,
        "gate_ids": corridor.gates["gate_id"].astype(str).tolist(),
        "edges": corridor.edges,
        "zone_node_map": corridor.zone_node_map,
        **{name: settings[name] for name in (
            "max_iterations", "min_iterations", "road_gap_threshold", "check_every", "od_threshold"
        )},
    }
    if int(n_jobs) == 1:
        rows = [
            _assignment_convergence_run(run, **worker_kwargs)
            for run in range(int(n_runs))
        ]
    else:
        from joblib import Parallel, delayed

        rows = Parallel(n_jobs=int(n_jobs), backend="loky", verbose=10)(
            delayed(_assignment_convergence_run)(run, **worker_kwargs)
            for run in range(int(n_runs))
        )

    runs = pd.concat(rows, ignore_index=True)
    summary = (
        runs.dropna(subset=["road_relative_gap_feasible_averaged_od"])
        .groupby("iteration")["road_relative_gap_feasible_averaged_od"]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            observed_runs="count",
        )
        .reset_index()
    )
    reached_rows = []
    for threshold in thresholds:
        for run, group in runs.groupby("run"):
            reached = group.loc[
                group["road_relative_gap_feasible_averaged_od"] <= float(threshold), "iteration"
            ]
            reached_rows.append(
                {
                    "run": int(run),
                    "threshold": float(threshold),
                    "first_iteration": int(reached.iloc[0]) if len(reached) else np.nan,
                    "reached": bool(len(reached)),
                }
            )
    threshold_reach = pd.DataFrame(reached_rows)
    return {"runs": runs, "summary": summary, "threshold_reach": threshold_reach}


# =============================================================================
# 8. DETAILED OPENSTREETMAP NETWORK ENHANCER
# =============================================================================

_DETAILED_NETWORK_MERGE_POLICY = "osm_identity_connected_attachments_v2"


def _merge_detailed_road_network(
    base_network: Mapping[str, Any],
    detailed_network: Mapping[str, Any],
    zones: Any,
    core_zones: Any,
    polygon: Any,
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace interior roads, retaining verified original routes at the boundary.

    Shared OSM identities join the two graphs. Between shared nodes, original
    road chains are retained only where they connect the exterior to the detail.
    Nearest-node attachment is reserved for model-zone connectors, never roads.
    """
    import geopandas as gpd
    import networkx as nx
    from scipy.spatial import cKDTree
    from shapely.geometry import LineString

    base_nodes = base_network["nodes"].copy()
    base_edges = base_network["edges"].copy()
    detail_edges = detailed_network["edges"].copy()
    detail_edges = detail_edges.loc[detail_edges["edge_id"].astype(str).str.startswith("OSM_")].copy()
    if detail_edges.empty:
        raise ValueError("The detailed road network contains no OSM links.")
    used_detail = set(detail_edges["source"]) | set(detail_edges["target"])
    detail_nodes = detailed_network["nodes"].loc[
        detailed_network["nodes"]["node_id"].isin(used_detail)
    ].copy()

    def osm_identity(value: Any) -> str:
        text = str(value)
        return text[1:] if text.startswith("N") and text[1:].isdigit() else text

    road_identity_nodes = base_nodes.loc[~base_nodes["is_zone"].astype(bool)]
    old_osm = dict(zip(road_identity_nodes["nodeID"].map(osm_identity), road_identity_nodes["node_id"].astype(int)))
    identities = dict(zip(detail_edges["source"], detail_edges["source_id"].map(osm_identity)))
    identities.update(zip(detail_edges["target"], detail_edges["target_id"].map(osm_identity)))
    next_id = int(base_nodes["node_id"].max()) + 1
    remap = {}
    for node_id in detail_nodes["node_id"]:
        identity = identities[node_id]
        if identity in old_osm:
            remap[node_id] = old_osm[identity]
        else:
            remap[node_id] = next_id
            next_id += 1
    detail_nodes["nodeID"] = detail_nodes["node_id"].map(lambda value: "N" + identities[value])
    detail_nodes["node_id"] = detail_nodes["node_id"].map(remap).astype(int)
    detail_edges["source"] = detail_edges["source"].map(remap).astype(int)
    detail_edges["target"] = detail_edges["target"].map(remap).astype(int)
    shared = set(base_nodes["node_id"]) & set(detail_nodes["node_id"])
    if not shared:
        raise ValueError("Detailed and base roads have no shared OSM nodes; a verified boundary connection is required.")

    roads = base_edges.loc[~base_edges["highway"].eq("manual_connector")].copy()
    # Removing shared anchors partitions the original roads into replacement
    # chains. Keep chains touching the exterior, including their inside tails.
    remaining = nx.Graph()
    remaining.add_edges_from(zip(roads["source"], roads["target"]))
    remaining.remove_nodes_from(shared)
    inside = set(base_nodes.loc[base_nodes.geometry.intersects(polygon), "node_id"])
    exterior_nodes = set()
    for component in nx.connected_components(remaining):
        if component.difference(inside):
            exterior_nodes.update(component)
    retain = roads["source"].isin(exterior_nodes) | roads["target"].isin(exterior_nodes)
    # A link can leave and re-enter the footprint with both endpoints shared.
    # Preserve such excursions only if the detailed graph has not retained them.
    detail_pairs = set(zip(detail_edges["source"], detail_edges["target"]))
    boundary_excursions = roads["source"].isin(shared) & roads["target"].isin(shared) & ~roads.geometry.within(polygon)
    retain |= boundary_excursions & np.asarray([
        (source, target) not in detail_pairs for source, target in zip(roads["source"], roads["target"])
    ])
    kept_roads = roads.loc[retain].copy()

    # Detailed coordinates take precedence for nodes present in both inputs.
    nodes = gpd.GeoDataFrame(pd.concat([base_nodes, detail_nodes], ignore_index=True),
                             geometry="geometry", crs=base_nodes.crs).drop_duplicates("node_id", keep="last")
    node_geometry = nodes.set_index("node_id").geometry
    # Keep physical roads, but attach demand only where vehicles can travel
    # in both directions through the merged network. One-way spurs are not
    # valid zone anchors even when they lie closest to the zone centroid.
    road_graph = nx.DiGraph()
    road_graph.add_edges_from(zip(kept_roads["source"], kept_roads["target"]))
    road_graph.add_edges_from(zip(detail_edges["source"], detail_edges["target"]))
    main_road_core = max(nx.strongly_connected_components(road_graph), key=lambda part: (len(part), -min(part)))
    attachment_nodes = detail_nodes.loc[detail_nodes["node_id"].isin(main_road_core)].sort_values("node_id")
    if attachment_nodes.empty:
        raise ValueError("Detailed roads have no anchors in the main connected road component. Check the source network and corridor boundary.")
    tree = cKDTree(attachment_nodes[["x", "y"]].to_numpy(dtype=float))
    connectors = base_edges.loc[base_edges["highway"].eq("manual_connector")].copy()
    # Preserve non-core centroids and valid attachments. Repair an attachment
    # that was replaced or cannot reach the main roads, preserving its direction.
    repaired_connectors = 0
    for index, row in connectors.iterrows():
        source, target = int(row["source"]), int(row["target"])
        zone_source = str(row["source_id"]) in base_network["zone_node_map"]
        road_node, zone_node = (target, source) if zone_source else (source, target)
        if road_node not in main_road_core:
            point = node_geometry.loc[zone_node]
            _, position = tree.query([point.x, point.y])
            replacement = int(attachment_nodes.iloc[int(position)]["node_id"])
            road_point = node_geometry.loc[replacement]
            field, id_field = ("target", "target_id") if zone_source else ("source", "source_id")
            connectors.at[index, field] = replacement
            connectors.at[index, id_field] = str(attachment_nodes.iloc[int(position)]["nodeID"])
            connectors.at[index, "geometry"] = LineString([point, road_point] if zone_source else [road_point, point])
            length = max(float(point.distance(road_point)), 1.0)
            connectors.at[index, "length_m"] = length
            connectors.at[index, "free_flow_time_min"] = length / 1000.0 / float(row["speed_kph"]) * 60.0
            repaired_connectors += 1

    zone_map = dict(base_network["zone_node_map"])
    centroids = core_zones["centroid"]
    _, positions = tree.query(np.column_stack((centroids.x, centroids.y)))
    for zone, position in zip(core_zones["grid_id"].astype(str), positions):
        zone_map[zone] = int(attachment_nodes.iloc[int(position)]["node_id"])
    # Core zones are attached directly to roads; their obsolete centroid links
    # must not create artificial shortcuts through zone centroids.
    core_node_ids = {base_network["zone_node_map"][zone] for zone in core_zones["grid_id"].astype(str)}
    connectors = connectors.loc[~connectors["source"].isin(core_node_ids) & ~connectors["target"].isin(core_node_ids)]
    edges = gpd.GeoDataFrame(pd.concat([kept_roads, detail_edges, connectors], ignore_index=True),
                            geometry="geometry", crs=base_edges.crs)
    used = set(edges["source"]) | set(edges["target"])
    nodes = nodes.loc[nodes["node_id"].isin(used)].copy()
    if not set(zone_map.values()).issubset(used):
        raise ValueError("A model zone has no road attachment after the detailed-network merge.")
    zone_graph = road_graph.copy()
    zone_graph.add_edges_from(zip(connectors["source"], connectors["target"]))
    anchor = min(main_road_core)
    reachable = (nx.descendants(zone_graph, anchor) & nx.ancestors(zone_graph, anchor)) | {anchor}
    unreachable_zones = [zone for zone, node in zone_map.items() if node not in reachable]
    if unreachable_zones:
        raise ValueError("Model zones lack a two-way road attachment: " + ", ".join(map(str, unreachable_zones[:8])) + ". Check their source-network connectors; no zones were dropped.")
    return {"edges": edges, "nodes": nodes, "zone_node_map": zone_map, "metadata": {
        **metadata, "merge_policy": _DETAILED_NETWORK_MERGE_POLICY,
        "shared_osm_nodes": len(shared), "retained_base_road_links": len(kept_roads),
        "repaired_zone_connectors": repaired_connectors,
        "eligible_detailed_attachment_nodes": len(attachment_nodes),
        "road_nodes_outside_attachment_core": road_graph.number_of_nodes() - len(main_road_core),
        "zone_road_reachability": "mutually reachable",
    }}


def load_detailed_corridor_network(
    context: TransportContext,
    corridor_municipalities: list[str],
    *,
    cache_path: str | Path | None = None,
    buffer_m: float = 800.0,
    allow_download: bool = True,
    force_regenerate: bool | None = None,
) -> TransportContext:
    """
    Enhance the macroscopic road assignment network with high-detail OpenStreetMap roads
    for any specified project corridor (e.g. MehrSpur, Limmattal, or Forch).

    Reuses the configured cache after checking its corridor scope. A network
    already loaded in this context is reused without a second disk read.
    Otherwise, if allow_download is True, it queries OSM via osmnx, converts to traffic
    assignment schema, stitches zone centroids, and caches the result.
    """
    import pickle
    import geopandas as gpd
    import numpy as np

    if force_regenerate is None:
        force_regenerate = ASSIGNMENT_SETTINGS.get(
            "force_regenerate_corridor_network"
        )

    if cache_path is None:
        cache_path = configured_detailed_network_path(context.project_root, must_exist=False)
    elif not Path(cache_path).is_absolute():
        cache_path = context.project_root / cache_path
    cache_path = Path(cache_path).resolve() if cache_path is not None else None

    if not force_regenerate and cache_path is not None and cache_path.exists():
        existing_source = context.assignment_network.get("metadata", {}).get("_cache_source")
        if existing_source == _detailed_network_file_identity(cache_path) and context.assignment_network.get("metadata", {}).get("merge_policy") == _DETAILED_NETWORK_MERGE_POLICY:
            _validate_detailed_network(
                context.assignment_network, corridor_municipalities=corridor_municipalities, buffer_m=buffer_m,
            )
            print(f"[OK] Reusing detailed corridor network from {cache_path.name}")
            return context
        detailed_network = _read_detailed_network(
            cache_path, corridor_municipalities=corridor_municipalities, buffer_m=buffer_m,
        )
        if detailed_network.get("metadata", {}).get("merge_policy") != _DETAILED_NETWORK_MERGE_POLICY:
            base_network = _load_assignment_network(
                context.project_root / "data/transport/prepared", configured_input_paths(context.project_root)["network"]
            )
            core_zones = _get_core_zones(context.zones, corridor_municipalities)
            polygon = core_zones.geometry.union_all().buffer(float(buffer_m))
            detailed_network = _merge_detailed_road_network(
                base_network, detailed_network, context.zones, core_zones, polygon,
                metadata={key: value for key, value in detailed_network["metadata"].items() if key != "_cache_source"},
            )
            temporary = cache_path.with_suffix(".tmp.pkl")
            with temporary.open("wb") as handle:
                pickle.dump(detailed_network, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(cache_path)
            detailed_network["metadata"]["_cache_source"] = _detailed_network_file_identity(cache_path)
            print("[OK] Updated detailed-network boundary connections and zone attachments from existing inputs.")
        print(f"[OK] Loaded cached detailed corridor network from {Path(cache_path).name}")
        return TransportContext(
            project_root=context.project_root,
            model_dir=context.model_dir,
            zones=context.zones,
            baseline_od=context.baseline_od,
            road_background_od=context.road_background_od,
            travel_times=context.travel_times,
            lengths=context.lengths,
            assignment_network=detailed_network,
            modules=context.modules,
        )

    if not allow_download:
        if cache_path is not None and not cache_path.is_file():
            raise FileNotFoundError(f"Detailed network does not exist: {cache_path}. Prepare it in Notebook 02.")
        print("Detailed network cache not found and allow_download=False. Falling back to coarse macroscopic network.")
        return context

    try:
        import osmnx as ox
        import networkx as nx
        # Increase timeout and memory limits for large queries
        try:
            ox.settings.timeout = 600
            ox.settings.memory = 1073741824
        except Exception:
            pass
    except ImportError:
        print("The 'osmnx' library is not installed. Falling back to coarse macroscopic network.")
        return context

    zones = context.zones.copy()
    core_zones = _get_core_zones(zones, corridor_municipalities)
    if core_zones.empty:
        raise ValueError(f"None of the specified corridor municipalities {corridor_municipalities} were found in zoning data.")

    cordon_polygon = core_zones.geometry.union_all().buffer(float(buffer_m))
    cordon_polygon_wgs84 = gpd.GeoSeries([cordon_polygon], crs=zones.crs).to_crs(4326).iloc[0]

    # 1. Fetch OSM network (with selected-zone groups if the full polygon fails)
    try:
        print(f"Downloading detailed OSM road network for {len(core_zones)} selected zones (this may take 1-3 minutes)...")
        G = ox.graph_from_polygon(cordon_polygon_wgs84, network_type="drive", simplify=False, retain_all=True)
    except Exception as e_full:
        print(f"Single polygon query had an issue ({e_full}). Trying chunked per-municipality download...")
        graphs = []
        failed_municipalities = []
        for muni, m_zones in core_zones.groupby("municipality_name", sort=False):
            m_poly = m_zones.geometry.union_all().buffer(float(buffer_m))
            m_poly_w84 = gpd.GeoSeries([m_poly], crs=zones.crs).to_crs(4326).iloc[0]
            try:
                print(f"  [DOWNLOAD] OSM roads for {muni}...")
                g_muni = ox.graph_from_polygon(m_poly_w84, network_type="drive", simplify=False, retain_all=True)
                if len(g_muni) > 0:
                    graphs.append(g_muni)
                else:
                    failed_municipalities.append(str(muni))
            except Exception as m_err:
                failed_municipalities.append(str(muni))
                print(f"    Warning: Could not fetch OSM for {muni}: {m_err}")

        if failed_municipalities or not graphs:
            raise RuntimeError(
                "Detailed road download incomplete for: " + ", ".join(failed_municipalities)
                + ". Retry the download; an incomplete corridor has not been cached."
            ) from e_full

        G = nx.compose_all(graphs)

    # Simplify after composing downloads so their overlap has shared junctions.
    # Keep physical roads; the merge selects mutually reachable zone anchors.
    G = ox.simplify_graph(G)

    osm_nodes_gdf, osm_edges_gdf = ox.graph_to_gdfs(G)

    osm_nodes_gdf = osm_nodes_gdf.to_crs(zones.crs)
    osm_edges_gdf = osm_edges_gdf.to_crs(zones.crs)

    base_network = _load_assignment_network(
        context.project_root / "data/transport/prepared", configured_input_paths(context.project_root)["network"]
    )
    if not base_network:
        raise ValueError("The original prepared assignment network is required to build detailed roads.")
    base_nodes = base_network["nodes"].copy()
    # Format OSM nodes; the merge reconciles their identities with the base graph.
    start_id = int(base_nodes["node_id"].max()) + 1
    osm_osmid_to_id = {osmid: start_id + idx for idx, osmid in enumerate(osm_nodes_gdf.index)}

    new_nodes_records = []
    for osmid, row in osm_nodes_gdf.iterrows():
        new_nodes_records.append({
            "node_id": osm_osmid_to_id[osmid],
            "x": float(row.geometry.x),
            "y": float(row.geometry.y),
            "geometry": row.geometry,
            "is_zone": False,
            "zone_id": "",
        })
    new_nodes_gdf = gpd.GeoDataFrame(new_nodes_records, geometry="geometry", crs=zones.crs)

    # 4. Format OSM edges with speed & capacity (self-contained to avoid pandana dependency)
    capacity_lookup = {
        "motorway": {1: 1700.0, 2: 4000.0, 3: 5800.0, 4: 7850.0},
        "trunk": {1: 1700.0, 2: 4000.0, 3: 5800.0},
        "motorway_link": {1: 1000.0, 2: 1700.0, 3: 2800.0},
        "trunk_link": {1: 1000.0, 2: 1700.0, 3: 2800.0},
        "primary": {1: 1200.0, 2: 2100.0, 3: 3200.0},
        "primary_link": {1: 1200.0, 2: 2100.0, 3: 3200.0},
        "secondary": {1: 1000.0, 2: 1600.0, 3: 2600.0},
        "secondary_link": {1: 1000.0, 2: 1600.0, 3: 2600.0},
        "tertiary": {1: 900.0, 2: 1500.0, 3: 2400.0},
        "tertiary_link": {1: 900.0, 2: 1500.0, 3: 2400.0},
        "residential": {1: 700.0, 2: 1100.0},
        "unclassified": {1: 700.0, 2: 1100.0},
        "living_street": {1: 400.0},
    }
    speed_lookup = {
        "motorway": 100.0, "trunk": 90.0, "motorway_link": 80.0, "trunk_link": 80.0,
        "primary": 60.0, "primary_link": 60.0, "secondary": 55.0, "secondary_link": 55.0,
        "tertiary": 45.0, "tertiary_link": 45.0, "residential": 35.0, "unclassified": 45.0,
        "living_street": 20.0,
    }
    # OSM simplification can return several classes in an unordered list.
    # Use the highest road class consistently across processes and platforms.
    road_class_rank = {name: rank for rank, name in enumerate(capacity_lookup)}

    new_edges_records = []
    for idx, row in osm_edges_gdf.reset_index().iterrows():
        u = row["u"]
        v = row["v"]
        key = row.get("key", 0)
        highway_val = row.get("highway", "unclassified")
        if isinstance(highway_val, (list, tuple, set)):
            highway_val = min(
                (str(value).strip().lower() for value in highway_val),
                key=lambda value: (road_class_rank.get(value, len(road_class_rank)), value),
                default="unclassified",
            )
        highway = str(highway_val or "unclassified").strip().lower()

        lanes_raw = row.get("lanes", 1.0)
        lanes = pd.to_numeric(pd.Series([lanes_raw]), errors="coerce").fillna(1.0).iloc[0]
        lanes = max(float(lanes), 1.0)

        length_m = float(row.geometry.length)
        if length_m <= 0:
            continue

        speed_raw = row.get("maxspeed", np.nan)
        speed_kph = pd.to_numeric(pd.Series([speed_raw]), errors="coerce").fillna(np.nan).iloc[0]
        if not np.isfinite(speed_kph) or speed_kph <= 0:
            speed_kph = float(speed_lookup.get(highway, 45.0))

        free_flow_time_min = length_m / 1000.0 / speed_kph * 60.0

        # Calculate capacity
        cap_table = capacity_lookup.get(highway, {1: 700.0, 2: 1100.0})
        lane_keys = sorted(cap_table.keys())
        cap_vals = [cap_table[k] for k in lane_keys]
        capacity_vph = float(np.interp(lanes, lane_keys, cap_vals))

        src_id = osm_osmid_to_id.get(u)
        tgt_id = osm_osmid_to_id.get(v)
        if src_id is None or tgt_id is None:
            continue

        new_edges_records.append({
            "source": int(src_id),
            "target": int(tgt_id),
            "source_id": str(u),
            "target_id": str(v),
            "length_m": length_m,
            "free_flow_time_min": free_flow_time_min,
            "capacity_vph": capacity_vph,
            "speed_kph": speed_kph,
            "lanes": float(lanes),
            "highway": highway,
            "alpha": 0.15,
            "beta": 4.0,
            "edge_id": f"OSM_{u}_{v}_{key}",
            "geometry": row.geometry,
        })

    new_edges_gdf = gpd.GeoDataFrame(new_edges_records, geometry="geometry", crs=zones.crs)

    detailed_network = _merge_detailed_road_network(
        base_network, {"edges": new_edges_gdf, "nodes": new_nodes_gdf},
        zones, core_zones, cordon_polygon,
        metadata={
            "corridor_municipalities": corridor_municipalities,
            "corridor_scope": _detailed_network_scope(corridor_municipalities, buffer_m),
            "added_osm_nodes": len(new_nodes_gdf),
            "added_osm_edges": len(new_edges_gdf),
        },
    )

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp.pkl")
        with temporary.open("wb") as f:
            pickle.dump(detailed_network, f, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(cache_path)
        detailed_network["metadata"]["_cache_source"] = _detailed_network_file_identity(cache_path)

    return TransportContext(
        project_root=context.project_root,
        model_dir=context.model_dir,
        zones=context.zones,
        baseline_od=context.baseline_od,
        road_background_od=context.road_background_od,
        travel_times=context.travel_times,
        lengths=context.lengths,
        assignment_network=detailed_network,
        modules=context.modules,
    )


# =============================================================================
# 9. CORRIDOR FLOW MAP & DESIRE-LINE EXPLORER
# =============================================================================


def aggregate_od_by_municipality(
    context: TransportContext,
    municipalities: list[str] | tuple[str, ...] | set[str],
    *,
    matrix: pd.DataFrame | None = None,
    zone_ids: Sequence[str] | set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate an FSM zone-level OD matrix into directed area flows.

    Names may refer to municipalities or, when available, city quartiers.
    The caller may replace the cantonal baseline with any square FSM OD matrix.
    Returns both the area matrix and a long directional-link table.
    """
    requested = list(dict.fromkeys(map(str, municipalities)))
    if not requested:
        raise ValueError("At least one municipality or quartier must be supplied.")

    source = context.baseline_od if matrix is None else matrix
    if not isinstance(source, pd.DataFrame) or source.empty:
        raise ValueError("The source OD matrix must be a non-empty DataFrame.")
    source = source.copy()
    source.index = source.index.astype(str)
    source.columns = source.columns.astype(str)
    if source.index.has_duplicates or source.columns.has_duplicates:
        raise ValueError("The source OD matrix must have unique zone labels.")

    required_columns = {"grid_id", "municipality_name"}
    missing_columns = required_columns.difference(context.zones.columns)
    if missing_columns:
        raise ValueError(f"Zone data are missing columns: {sorted(missing_columns)}")
    area_zone_ids = resolve_area_zone_ids(
        context.zones,
        requested,
        allowed_zone_ids=zone_ids,
    )
    zone_to_area = {
        zone_id: area_name
        for area_name, selected_ids in area_zone_ids.items()
        for zone_id in selected_ids
        if zone_id in source.index and zone_id in source.columns
    }
    selected_zone_ids = list(zone_to_area)
    if not selected_zone_ids:
        raise ValueError("No FSM zones matched the selected municipalities or quartiers.")
    zone_to_area_series = pd.Series(zone_to_area)
    selected_od = source.loc[selected_zone_ids, selected_zone_ids].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0)
    if (selected_od.to_numpy(dtype=float) < 0).any():
        raise ValueError("OD demand cannot contain negative trip values.")

    # Aggregate origins, transpose, then aggregate destinations. This avoids
    # the deprecated ``groupby(axis=1)`` API and works across supported pandas versions.
    area_od = selected_od.groupby(zone_to_area_series, sort=False).sum()
    area_od = area_od.T.groupby(zone_to_area_series, sort=False).sum().T
    area_order = [name for name in requested if name in area_od.index]
    area_od = area_od.reindex(
        index=area_order,
        columns=area_order,
        fill_value=0.0,
    )

    directed_links = area_od.rename_axis("origin").reset_index().melt(
        id_vars="origin",
        var_name="destination",
        value_name="peak-hour passenger trips",
    )
    directed_links = directed_links.loc[
        directed_links["origin"] != directed_links["destination"]
    ].copy()
    directed_links["OD relation"] = (
        directed_links["origin"] + " -> " + directed_links["destination"]
    )
    directed_links = directed_links.sort_values(
        "peak-hour passenger trips", ascending=False
    ).reset_index(drop=True)
    return area_od, directed_links


# =============================================================================
# 10. MODAL SPLIT & OD RELATION SUMMARIES
# =============================================================================

def corridor_modal_split(
    context: TransportContext,
    mode_result: ModeChoiceResult,
    corridor_municipalities: list[str] | None = None,
    zone_ids: list[str] | set[str] | None = None,
    *,
    detailed_pt: bool = False,
) -> pd.DataFrame:
    """
    Compute internal-to-internal (Binnenverkehr) modal split within the specified corridor.

    Parameters:
        context: Loaded transport context.
        mode_result: Output from run_transport_mode_choice.
        corridor_municipalities: Municipalities to include in the internal corridor.
        detailed_pt: If False (default), aggregates into 4 modes (Car, PT, Bike, Walk).
                     If True, breaks PT into PT (Walk access) and PT (Bike access).
    """
    if zone_ids is None and corridor_municipalities is not None:
        zone_ids = get_zone_ids_for_municipalities(context, corridor_municipalities)
    elif zone_ids is None:
        zone_ids = list(context.zones["grid_id"].astype(str))

    od_by_mode = mode_result.od_by_mode
    reference = next(iter(od_by_mode.values()))
    labels = reference.index.astype(str)
    internal = np.isin(labels, list(map(str, zone_ids)))
    internal_to_internal = internal[:, None] & internal[None, :]

    car_total = float(od_by_mode["drive"].to_numpy(dtype=float)[internal_to_internal].sum()) if "drive" in od_by_mode else 0.0
    pt_walk_total = float(od_by_mode["pt_walk"].to_numpy(dtype=float)[internal_to_internal].sum()) if "pt_walk" in od_by_mode else 0.0
    pt_bike_total = float(od_by_mode["pt_bike"].to_numpy(dtype=float)[internal_to_internal].sum()) if "pt_bike" in od_by_mode else 0.0
    bike_total = float(od_by_mode["bike"].to_numpy(dtype=float)[internal_to_internal].sum()) if "bike" in od_by_mode else 0.0
    walk_total = float(od_by_mode["walk"].to_numpy(dtype=float)[internal_to_internal].sum()) if "walk" in od_by_mode else 0.0

    if detailed_pt:
        rows = [
            {"mode": "Car (Driver)", "trips": car_total},
            {"mode": "PT (Walk access)", "trips": pt_walk_total},
            {"mode": "PT (Bike access)", "trips": pt_bike_total},
            {"mode": "Bicycle", "trips": bike_total},
            {"mode": "Walking", "trips": walk_total},
        ]
    else:
        rows = [
            {"mode": "Car (Driver)", "trips": car_total},
            {"mode": "Public Transport", "trips": pt_walk_total + pt_bike_total},
            {"mode": "Bicycle", "trips": bike_total},
            {"mode": "Walking", "trips": walk_total},
        ]

    df = pd.DataFrame(rows)
    total_trips = float(df["trips"].sum())
    df["share"] = df["trips"] / max(total_trips, 1e-9)
    return df


def corridor_od_summary(
    context: TransportContext,
    mode_result: ModeChoiceResult,
    pairs: list[tuple[str, str]],
    macro_regions: dict[str, list[str]] | None = None,
    *,
    detailed_pt: bool = False,
) -> pd.DataFrame:
    """
    Summarize total passenger trips and modal split shares for specific macro-region or municipality pairs.

    Parameters:
        detailed_pt: If False (default), outputs 4-mode shares (Car, PT, Bike, Walk).
                     If True, splits PT into pt_walk and pt_bike.
    """
    zones = context.zones
    # If macro_regions is provided and not empty, map by region; otherwise map directly by municipality name
    if macro_regions:
        first_list = next(iter(macro_regions.values()), [])
        if first_list and any(str(x).isdigit() and len(str(x)) >= 6 for x in first_list):
            # Dict maps region name -> list of zone IDs
            zone_to_region = {str(z): r for r, z_list in macro_regions.items() for z in z_list}
        else:
            # Dict maps region name -> list of municipality names
            muni_to_region = {m: r for r, munis in macro_regions.items() for m in munis}
            zone_to_region = (
                zones.set_index("grid_id")["municipality_name"]
                .astype(str)
                .map(muni_to_region)
                .fillna("Rest of Canton")
                .to_dict()
            )
    else:
        # Direct municipality-level OD mapping
        zone_to_region = zones.set_index("grid_id")["municipality_name"].astype(str).to_dict()

    od_by_mode = mode_result.od_by_mode
    reference = next(iter(od_by_mode.values()))
    labels = reference.index.astype(str)

    origin_regions = np.array([zone_to_region.get(str(z), "Other") for z in labels])
    dest_regions = np.array([zone_to_region.get(str(z), "Other") for z in labels])

    car_mat = od_by_mode["drive"].to_numpy(dtype=float) if "drive" in od_by_mode else np.zeros((len(labels), len(labels)))
    pt_walk_mat = od_by_mode["pt_walk"].to_numpy(dtype=float) if "pt_walk" in od_by_mode else np.zeros((len(labels), len(labels)))
    pt_bike_mat = od_by_mode["pt_bike"].to_numpy(dtype=float) if "pt_bike" in od_by_mode else np.zeros((len(labels), len(labels)))
    bike_mat = od_by_mode["bike"].to_numpy(dtype=float) if "bike" in od_by_mode else np.zeros((len(labels), len(labels)))
    walk_mat = od_by_mode["walk"].to_numpy(dtype=float) if "walk" in od_by_mode else np.zeros((len(labels), len(labels)))

    rows = []
    for orig, dest in pairs:
        mask = (origin_regions[:, None] == orig) & (dest_regions[None, :] == dest)
        c_trips = float(car_mat[mask].sum())
        pw_trips = float(pt_walk_mat[mask].sum())
        pb_trips = float(pt_bike_mat[mask].sum())
        b_trips = float(bike_mat[mask].sum())
        w_trips = float(walk_mat[mask].sum())
        tot = c_trips + pw_trips + pb_trips + b_trips + w_trips

        if detailed_pt:
            rows.append({
                "OD relation": f"{orig} → {dest}",
                "total_trips": tot,
                "car_trips": c_trips,
                "pt_walk_trips": pw_trips,
                "pt_bike_trips": pb_trips,
                "bike_trips": b_trips,
                "walk_trips": w_trips,
                "car_share": c_trips / max(tot, 1e-9),
                "pt_walk_share": pw_trips / max(tot, 1e-9),
                "pt_bike_share": pb_trips / max(tot, 1e-9),
                "bike_share": b_trips / max(tot, 1e-9),
                "walk_share": w_trips / max(tot, 1e-9),
            })
        else:
            p_trips = pw_trips + pb_trips
            rows.append({
                "OD relation": f"{orig} → {dest}",
                "total_trips": tot,
                "car_trips": c_trips,
                "pt_trips": p_trips,
                "bike_trips": b_trips,
                "walk_trips": w_trips,
                "car_share": c_trips / max(tot, 1e-9),
                "pt_share": p_trips / max(tot, 1e-9),
                "bike_share": b_trips / max(tot, 1e-9),
                "walk_share": w_trips / max(tot, 1e-9),
            })

    return pd.DataFrame(rows)


# =============================================================================
# 11. MULTI-STAGE HORIZON PERFORMANCE COMPARISON HELPER
# =============================================================================

def _stage_snapshot_content_digest(value: Any) -> str:
    """Hash table values rather than pandas' mutable internal caches/layout."""
    import hashlib
    from joblib import hash as content_hash

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, pd.DataFrame):
            digest.update(b"frame")
            update(tuple(item.columns))
            try:
                values = pd.util.hash_pandas_object(item, index=True).to_numpy()
                digest.update(values.tobytes())
            except (TypeError, ValueError):
                # Some prepared edge tables contain list-valued OSM IDs.
                update(tuple(item.index))
                for row in item.itertuples(index=False, name=None):
                    update(row)
        elif isinstance(item, Mapping):
            digest.update(b"mapping")
            for key in sorted(item, key=repr):
                update(key)
                update(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(f"sequence:{len(item)}:".encode())
            for child in item:
                update(child)
        else:
            digest.update(content_hash(item).encode())

    update(value)
    return digest.hexdigest()


def pt_reporting_cohort(
    reference_mode_result: ModeChoiceResult,
    corridor_zone_ids: Sequence[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Keep positive baseline PT OD/submode weights for comparable display means.

    Both PT access submodes retain their own skims. Journeys have at least one
    endpoint in the reporting corridor; no small-flow cutoff is applied.
    This reporting cohort does not enter welfare or transport calculations.
    """
    import hashlib
    import json

    labels = pd.Index(sorted(reference_mode_result.od_by_mode["pt_walk"].index.astype(str)))
    if labels.has_duplicates:
        raise ValueError("PT reporting requires unique OD labels.")
    scope = None if corridor_zone_ids is None else tuple(sorted(map(str, corridor_zone_ids)))
    inside = np.isin(labels, scope) if scope is not None else np.ones(len(labels), dtype=bool)
    mask = inside[:, None] | inside[None, :]
    modes = {}
    total = 0.0
    digest = hashlib.sha256(json.dumps({"labels": list(labels), "scope": scope}).encode("utf-8"))
    for mode in ("pt_walk", "pt_bike"):
        frame = reference_mode_result.od_by_mode[mode].rename(index=str, columns=str)
        values = frame.reindex(index=labels, columns=labels).to_numpy(dtype=float)
        if not np.isfinite(values[mask]).all() or (values[mask] < 0).any():
            raise ValueError("PT reporting requires finite, nonnegative baseline passenger quantities.")
        row, col = np.nonzero(mask & (values > 0))
        quantity = values[row, col]
        modes[mode] = {"row": row, "col": col, "weights": quantity}
        total += float(quantity.sum())
        digest.update(mode.encode("ascii"))
        for array in (row.astype("<i8"), col.astype("<i8"), quantity.astype("<f8")):
            digest.update(array.tobytes())
    for selected in modes.values():
        selected["weights"] = selected["weights"] / total if total > 0 else selected["weights"]
    return {"signature": digest.hexdigest(), "labels": tuple(labels), "modes": modes,
            "pt_reference_trips_peak": total, "reporting_zone_ids": scope}


def pt_component_report(mode_result: ModeChoiceResult, cohort: Mapping[str, Any]) -> dict[str, Any]:
    """Fixed-cohort mean PT minutes, keeping invalid/missing components unknown."""
    from transport_core.config import NO_PATH_TIME_MIN

    labels = pd.Index(cohort["labels"])
    times = mode_result.travel_times
    result = {"cohort_signature": cohort["signature"],
              "pt_reference_trips_peak": float(cohort["pt_reference_trips_peak"])}
    components = {"wait": ("initial_wait", "transfer_wait"), "access": ("access",),
                  "egress": ("egress",), "transfer_walk": ("transfer_physical",)}
    for component, prefixes in components.items():
        mean = 0.0 if result["pt_reference_trips_peak"] > 0 else np.nan
        for mode, selected in cohort["modes"].items():
            if not len(selected["weights"]):
                continue
            for prefix in prefixes:
                frame = times.get(f"{prefix}_{mode}")
                if frame is None:
                    mean = np.nan
                    continue
                rows = frame.index.astype(str).get_indexer(labels)[selected["row"]]
                cols = frame.columns.astype(str).get_indexer(labels)[selected["col"]]
                if (rows < 0).any() or (cols < 0).any():
                    mean = np.nan
                    continue
                # Select existing arrays directly instead of copying full OD
                # matrices for each component in a reporting-only calculation.
                sampled = frame.to_numpy(dtype=float, copy=False)[rows, cols]
                if not (np.isfinite(sampled) & (sampled >= 0) & (sampled < NO_PATH_TIME_MIN)).all():
                    mean = np.nan
                    continue
                mean += float(np.dot(selected["weights"], sampled))
        result[f"pt_{component}_min"] = mean
    return result


def stage_year_snapshot(
    context: TransportContext,
    mode_result: ModeChoiceResult,
    *,
    stage_spec: Mapping[str, Any],
    corridor_context: CorridorContext,
    year_idx: int,
    g_cum: float,
    params: Mapping[str, Any],
    assignment_settings: Mapping[str, Any] | None = None,
    snapshots: dict | None = None,
    transport_state: Mapping[str, Any] | None = None,
    compute_if_missing: bool = True,
    reporting_cohort: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Reuse one year on the corridor's fixed reporting scope without retaining OD arrays.

    ``transport_state`` may seed a freshly solved state for these exact inputs;
    its caller is responsible for matching the year, physical inputs and solver.
    The returned entry contains ``transport_state``, ``annual`` and ``cache_key``.
    Pass its compact state to ``simulate_year`` to reprice a plan without another
    transport solve. The optional mapping is an in-kernel cache, not a persisted
    model cache: restart after changing model source or prepared input files.

    Keys hash content, including mutable road attributes and gate mappings, not
    object identity. Uncongested skims allow an assigned Year-1 mode result and
    its original mode result to reuse the same snapshot. Hashing reads existing
    arrays but the mapping retains only aggregate metrics and diagnostics.
    ``compute_if_missing=False`` returns None when a new assignment is needed,
    allowing callers to schedule missing states together.
    ``reporting_cohort`` adds comparable PT component means when an existing
    full mode result is available; missing display data never triggers a solve.
    """
    import parameters as p
    import simulation_engine as m

    stage = int(mode_result.stage)
    if int(year_idx) != year_idx or year_idx < 0:
        raise ValueError("year_idx must be a nonnegative integer.")
    year_idx = int(year_idx)
    settings = resolve_assignment_settings(assignment_settings)
    year_params = m.resolved_parameters(dict(params))
    corridor = ensure_stage_road_context(corridor_context, stage_spec)
    scope = tuple(sorted(map(str, corridor.zone_ids)))
    if not scope:
        raise ValueError("An annual snapshot requires a nonempty corridor reporting scope.")

    mode_key = tuple(mode_result.cache_key)
    while mode_key[-1:] == ("coupled_msa",):
        mode_key = mode_key[:-1]
    definitions = {
        "schema": 1, "stage_spec": dict(stage_spec), "mode_key": mode_key,
        "scenario": mode_result.scenario,
        "uncongested_times": mode_result.uncongested_travel_times or mode_result.travel_times,
        "technology_base_times": mode_result.technology_base_times,
        "lengths": mode_result.lengths,
        "baseline_od": context.baseline_od, "background_od": context.road_background_od,
        "zones": context.zones, "reporting_zone_ids": scope,
        "edges": corridor.edges, "gates": corridor.gates,
        "zone_node_map": corridor.zone_node_map,
        "entry_gates": corridor.external_zone_to_entry_gate,
        "exit_gates": corridor.external_zone_to_exit_gate,
        "year_params": year_params, "growth": float(g_cum), "solver": settings,
        "passthrough_fraction": DEFAULT_PASSTHROUGH_FRACTION,
        "section": getattr(p, "SECTION", None),
        "external_flow": getattr(p, "EXTERNAL_FLOW", None),
        "native_preferences": context.modules["mode_choice_zurich"].load_mode_choice_parameters(),
    }
    key = (stage, year_idx, _stage_snapshot_content_digest(definitions))
    if snapshots is not None and key in snapshots:
        entry = snapshots[key]
        if reporting_cohort is not None and entry.get("reporting", {}).get("cohort_signature") != reporting_cohort["signature"]:
            entry.pop("reporting", None)
            supplied_mode = transport_state.get("mode_result") if transport_state is not None else None
            if supplied_mode is not None:
                entry["reporting"] = pt_component_report(supplied_mode, reporting_cohort)
        return entry
    if transport_state is None and not compute_if_missing:
        return None

    if transport_state is None:
        transport_state = acquire_native_transport_state(
            context, mode_result, stage=stage,
            corridor_municipalities=None, corridor_context=corridor,
            passenger_demand_multiplier=1.0 + float(g_cum),
            pt_asc_shift=year_params["PT_ASC_SHIFT"],
            bike_asc_shift=year_params["BIKE_ASC_SHIFT"],
            ebike_share=year_params["EBIKE_SHARE"],
            road_freight_multiplier=max(0.0, 1.0 + year_params["ROAD_FREIGHT_GROWTH"]),
            assignment_settings=settings,
        )
    if int(transport_state.get("stage", stage)) != stage:
        raise ValueError("The supplied transport state has a different stage.")
    if not isinstance(transport_state.get("metrics"), Mapping):
        raise ValueError("The supplied transport state must contain aggregate metrics.")
    # Reaggregate supplied native states on precisely the same scope as new
    # solves. Only these aggregate values survive in the cache.
    supplied_mode = transport_state.get("mode_result")
    metrics = (
        extract_corridor_metrics(context, supplied_mode, scope,
                                 min_distance_km=stage_spec.get("_min_distance_km"))
        if supplied_mode is not None else dict(transport_state["metrics"])
    )
    compact_state = {
        "schema_version": 1, "stage": stage,
        "source": transport_state.get("source", "coupled_msa"),
        "metrics": dict(metrics),
        "assignment_diagnostics": deepcopy(transport_state.get("assignment_diagnostics", {})),
        "reporting_zone_ids": scope,
    }
    annual = m.simulate_year(
        compact_state["metrics"], stage=stage, year_idx=year_idx, g_cum=float(g_cum),
        params=year_params, transport_state=compact_state,
        assignment_settings=settings, return_details=True,
    )
    entry = {"cache_key": key, "transport_state": compact_state, "annual": annual}
    if reporting_cohort is not None and supplied_mode is not None:
        entry["reporting"] = pt_component_report(supplied_mode, reporting_cohort)
    if snapshots is not None:
        # Retain only the latest definition for a stage/year; repeated editor
        # changes must not accumulate obsolete snapshots in a long-lived kernel.
        for old_key in list(snapshots):
            if isinstance(old_key, tuple) and old_key[:2] == key[:2]:
                del snapshots[old_key]
        snapshots[key] = entry
    return entry


def _calculate_stage_year_snapshot(request: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
    """Return a compact stage/year result from an independent native solve."""
    started = time.perf_counter()
    return stage_year_snapshot(**request), time.perf_counter() - started


def compare_stages_summary(
    context: TransportContext,
    stage_specs: Mapping[int, Mapping[str, Any]] | Any,
    corridor_zone_ids: list[str] | set[str] | None = None,
    corridor_municipalities: list[str] | None = None,
    *,
    stage_results: dict[int, Any] | None = None,
    g_d_base: float | None = None,
    n_years: int | None = None,
    stage_names: dict[int, str] | None = None,
    corridor_contexts_by_stage: Mapping[int, CorridorContext] | None = None,
    assignment_settings: Mapping[str, Any] | None = None,
    year_snapshots: dict | None = None,
    display_tables: bool = True,
    n_jobs: int = 1,
    progress: bool = True,
    reporting_references: Mapping[int, ModeChoiceResult] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compare physical indicators across infrastructure stages for Year 1 and Year 40.

    Generates and optionally displays the 4 standard mode-specific performance tables:
      1. Car Passenger Performance
      2. Public Transport Performance
      3. Active Mobility (Bike & Walk) Performance
      4. Overall Multimodal System Summary

    Both years use the same corridor.zone_ids reporting scope. The legacy
    ``corridor_zone_ids`` argument is checked against that scope, not used to
    silently give Year 1 a different measurement boundary. ``year_snapshots``
    reuses and receives lightweight entries from ``stage_year_snapshot``.
    Missing stage/year assignments run independently on up to ``n_jobs`` workers.
    Optional ``reporting_references`` contains already-solved baseline modes by
    zero-based year, used only for fixed-passenger PT component displays.
    """
    import simulation_engine as m
    import parameters as p
    from additional import uncertainty as uc
    from IPython.display import display
    import os
    from joblib import Parallel, delayed, parallel_config
    from threadpoolctl import threadpool_limits

    if int(n_jobs) != n_jobs or int(n_jobs) < 1:
        raise ValueError("n_jobs must be a positive integer.")

    # 1. Resolve stage specifications
    if hasattr(stage_specs, "get_stages"):
        specs = stage_specs.get_stages(vars(p))
    elif isinstance(stage_specs, dict):
        specs = stage_specs
    else:
        specs = {0: {"name": "Stage 0"}, 1: {"name": "Stage 1"}, 2: {"name": "Stage 2"}}

    stage_keys = sorted(specs.keys())

    # 2. Planning horizon growth
    ny = n_years if n_years is not None else getattr(p, "N_YEARS", 40)
    if g_d_base is not None:
        g_cum_40 = (1.0 + float(g_d_base)) ** (ny - 1) - 1.0
    elif hasattr(p, "PASSENGER_DEMAND_GROWTH_Y40"):
        g_cum_40 = float(p.PASSENGER_DEMAND_GROWTH_Y40)
    else:
        g_cum_40 = (1.0 + getattr(p, "G_D_BASE", 0.006)) ** (ny - 1) - 1.0
    scale_y1 = 1.0
    settings = resolve_assignment_settings(assignment_settings)
    snapshots = year_snapshots if year_snapshots is not None else {}
    paths = uc.nominal_paths()
    y1_params = uc.year_parameters(paths, 0, params=p.NOMINAL_PARAMS)
    y40_params = uc.year_parameters(paths, ny - 1, params=p.NOMINAL_PARAMS)

    if stage_names is None:
        stage_names = {}

    car_data = []
    pt_data = []
    active_data = []
    system_data = []

    c_munis = corridor_municipalities if corridor_municipalities is not None else getattr(p, "CORRIDOR_MUNICIPALITIES", None)
    reporting_scope = None
    annual_results = {}
    pending = []
    reporting_cohorts = {}

    for s in stage_keys:
        st_name = stage_names.get(s, specs[s].get("name", f"Stage {s}"))

        stage_corridor = (
            corridor_contexts_by_stage.get(s)
            if corridor_contexts_by_stage is not None else None
        )
        if stage_corridor is None:
            stage_corridor = build_corridor_context(
                context, corridor_municipalities=list(c_munis or []),
                name="configured project corridor",
            )
        stage_corridor = ensure_stage_road_context(stage_corridor, specs[s])
        scope = tuple(sorted(map(str, stage_corridor.zone_ids)))
        if corridor_zone_ids is not None and set(map(str, corridor_zone_ids)) != set(scope):
            raise ValueError(
                "Stage comparisons require corridor_zone_ids to match corridor.zone_ids "
                "for both years; pass the configured corridor's reporting zones."
            )
        if reporting_scope is not None and scope != reporting_scope:
            raise ValueError("All compared stages must use the same reporting-zone scope.")
        reporting_scope = scope
        if reporting_references is not None and not reporting_cohorts:
            reporting_cohorts = {
                year: pt_reporting_cohort(reference, scope)
                for year, reference in reporting_references.items()
            }

        if stage_results is not None and s in stage_results and stage_results[s] is not None:
            res_s = stage_results[s]
        else:
            res_s = run_transport_mode_choice(
                context,
                stage=s,
                stage_specs=specs,
                corridor_zone_ids=scope,
                corridor_municipalities=c_munis,
            )
        existing_y1 = (
            transport_state_from_mode(context, res_s, scope)
            if assignment_settings_match(
                res_s.assigned_metadata or {}, settings, modal_feedback=True,
                passthrough_fraction=DEFAULT_PASSTHROUGH_FRACTION,
            ) else None
        )
        for label, year_idx, growth, year_params, supplied in (
            ("Year 1", 0, 0.0, y1_params, existing_y1),
            ("Year 40", ny - 1, g_cum_40, y40_params, None),
        ):
            if s == 0 and reporting_references is not None and year_idx in reporting_references:
                # Baseline year assignments already exist in the notebook.
                supplied = transport_state_from_mode(context, reporting_references[year_idx], scope)
            request = dict(context=context, mode_result=res_s, stage_spec=specs[s],
                           corridor_context=stage_corridor, year_idx=year_idx,
                           g_cum=growth, params=year_params, assignment_settings=settings)
            if year_idx in reporting_cohorts:
                request["reporting_cohort"] = reporting_cohorts[year_idx]
            entry = stage_year_snapshot(**request, snapshots=snapshots,
                                        transport_state=supplied, compute_if_missing=False)
            if entry is not None:
                annual_results[(s, year_idx)] = entry["annual"]
                if progress:
                    print(f"{st_name}, {label}: reused calculated transport state.", flush=True)
            else:
                pending.append(request)

    workers = min(int(n_jobs), len(pending), max(1, (os.cpu_count() or 2) - 1))
    if progress:
        print(f"Stage comparison: {len(pending)} missing year-state(s), {workers} worker(s).", flush=True)

    def record(result):
        entry, elapsed = result
        key = entry["cache_key"]
        for old_key in list(snapshots):
            if isinstance(old_key, tuple) and old_key[:2] == key[:2]:
                del snapshots[old_key]
        snapshots[key] = entry
        annual_results[key[:2]] = entry["annual"]
        if progress:
            diagnostics = entry["transport_state"].get("assignment_diagnostics", {})
            gap = diagnostics.get("final_road_relative_gap_feasible_averaged_od", np.nan)
            gap_text = f"{gap:.2%}" if np.isfinite(gap) else "unavailable"
            name = stage_names.get(key[0], specs[key[0]].get("name", f"Stage {key[0]}"))
            print(f"{name}, Year {key[1] + 1}: {diagnostics.get('iterations', '?')} iterations, "
                  f"road gap {gap_text}, {elapsed:.1f}s.", flush=True)

    if pending and workers <= 1:
        with threadpool_limits(limits=1):
            for request in pending:
                record(_calculate_stage_year_snapshot(request))
    elif workers > 1:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            completed = Parallel(n_jobs=workers, return_as="generator_unordered", batch_size=1,
                                 pre_dispatch=workers)(
                delayed(_calculate_stage_year_snapshot)(request) for request in pending)
            for result in completed:
                record(result)

    for s in stage_keys:
        st_name = stage_names.get(s, specs[s].get("name", f"Stage {s}"))
        y1 = annual_results[(s, 0)]
        y40 = annual_results[(s, ny - 1)]

        # Peak hour trips
        metrics = y1["mode_metrics"]
        car_y1 = metrics.get("car_trips", 0.0)
        pt_y1 = metrics.get("pt_trips", 0.0)
        bike_y1 = metrics.get("bike_trips", 0.0)
        walk_y1 = metrics.get("walk_trips", 0.0)
        total_y1 = metrics.get("total_trips", 0.0)

        # Extract recalculated base metrics natively from the simulation engine's mode details
        y40_metrics = y40.get("mode_metrics", metrics)
        car_y40 = y40_metrics.get("car_trips", 0.0)
        pt_y40 = y40_metrics.get("pt_trips", 0.0)
        bike_y40 = y40_metrics.get("bike_trips", 0.0)
        walk_y40 = y40_metrics.get("walk_trips", 0.0)
        total_y40 = y40_metrics.get("total_trips", 0.0)

        # Delay & CO2 (peak hour)
        factors_y1 = m._annualizers(y1_params)
        factors_y40 = m._annualizers(y40_params)
        delay_y1_peak = y1.get("congestion_delay_hours", 0.0) / factors_y1["congestion"]
        delay_y40_peak = y40.get("congestion_delay_hours", 0.0) / factors_y40["congestion"]
        co2_y1_peak = y1["car_co2_tonnes"] / factors_y1["car"]
        co2_y40_peak = y40["car_co2_tonnes"] / factors_y40["car"]

        # Mode Travel Times
        car_tt_y1 = ((metrics.get("car_tt_hours", 0.0) * scale_y1 + delay_y1_peak) / max(car_y1, 1e-6)) * 60.0
        car_tt_y40 = ((y40_metrics.get("car_tt_hours", 0.0) + delay_y40_peak) / max(car_y40, 1e-6)) * 60.0
        pt_tt_y1 = (metrics.get("pt_tt_hours", 0.0) / max(pt_y1, 1e-6)) * 60.0
        pt_tt_y40 = (y40_metrics.get("pt_tt_hours", 0.0) / max(pt_y40, 1e-6)) * 60.0

        # --- Mode 1: Car ---
        car_data.append({
            "Stage": st_name,
            "Year 1 Trips": car_y1,
            "Year 40 Trips": car_y40,
            "Year 1 Trip Share": car_y1 / max(total_y1, 1e-9),
            "Year 40 Trip Share": car_y40 / max(total_y40, 1e-9),
            "Year 1 PKM Share (>5km)": metrics.get("car_share", car_y1 / max(total_y1, 1e-9)),
            "Year 40 PKM Share (>5km)": y40_metrics.get("car_share", car_y40 / max(total_y40, 1e-9)),
            "Year 1 TT": car_tt_y1,
            "Year 40 TT": car_tt_y40,
            "Year 1 Delay": delay_y1_peak,
            "Year 40 Delay": delay_y40_peak,
            "Year 1 CO2": co2_y1_peak,
            "Year 40 CO2": co2_y40_peak,
        })

        # --- Mode 2: Public Transport ---
        pt_data.append({
            "Stage": st_name,
            "Year 1 Trips": pt_y1,
            "Year 40 Trips": pt_y40,
            "Year 1 Trip Share": pt_y1 / max(total_y1, 1e-9),
            "Year 40 Trip Share": pt_y40 / max(total_y40, 1e-9),
            "Year 1 PKM Share (>5km)": metrics.get("pt_share", pt_y1 / max(total_y1, 1e-9)),
            "Year 40 PKM Share (>5km)": y40_metrics.get("pt_share", pt_y40 / max(total_y40, 1e-9)),
            "Year 1 TT": pt_tt_y1,
            "Year 40 TT": pt_tt_y40,
        })

        # --- Mode 3: Active Mobility (Bike & Walk) ---
        active_data.append({
            "Stage": st_name,
            "Year 1 Bike Trips": bike_y1,
            "Year 40 Bike Trips": bike_y40,
            "Bike Trip Share": bike_y1 / max(total_y1, 1e-9),
            "Bike PKM Share": metrics.get("bike_share", bike_y1 / max(total_y1, 1e-9)),
            "Year 1 Walk Trips": walk_y1,
            "Year 40 Walk Trips": walk_y40,
            "Walk Trip Share": walk_y1 / max(total_y1, 1e-9),
            "Walk PKM Share": metrics.get("walk_share", walk_y1 / max(total_y1, 1e-9)),
        })

        # --- Mode 4: Overall System ---
        system_data.append({
            "Stage": st_name,
            "Year 1 Total Demand": total_y1,
            "Year 40 Total Demand": total_y40,
            "Year 1 System Avg TT": y1.get("avg_tt_min", 0.0),
            "Year 40 System Avg TT": y40.get("avg_tt_min", 0.0),
            "Year 1 Peak Delay": delay_y1_peak,
            "Year 40 Peak Delay": delay_y40_peak,
            "Year 1 Peak CO2": co2_y1_peak,
            "Year 40 Peak CO2": co2_y40_peak,
        })

    # Convert to DataFrames
    df_car = pd.DataFrame(car_data).set_index("Stage")
    df_pt = pd.DataFrame(pt_data).set_index("Stage")
    df_active = pd.DataFrame(active_data).set_index("Stage")
    df_system = pd.DataFrame(system_data).set_index("Stage")

    if display_tables:
        print(f"Reporting scope: {len(reporting_scope or ())} zones; journeys with either endpoint in scope.")
        print("TT: Car/PT in-vehicle minutes (car includes delay); delay: passenger-hours; peak CO2: passenger-car tonnes.")
        print("PKM shares use the common car-distance proxy at >=5 km; active-mode shares below are Year 1.")
        print("[CAR] 1. CAR PASSENGER PERFORMANCE (Single Evening Peak Hour)")
        display(df_car.style.format({
            "Year 1 Trips": "{:,.0f} trips",
            "Year 40 Trips": "{:,.0f} trips",
            "Year 1 Trip Share": "{:.1%}", "Year 40 Trip Share": "{:.1%}",
            "Year 1 PKM Share (>5km)": "{:.1%}", "Year 40 PKM Share (>5km)": "{:.1%}",
            "Year 1 TT": "{:.1f} min",
            "Year 40 TT": "{:.1f} min",
            "Year 1 Delay": "{:,.0f} h",
            "Year 40 Delay": "{:,.0f} h",
            "Year 1 CO2": "{:,.1f} t",
            "Year 40 CO2": "{:,.1f} t",
        }))

        print("\n[PT] 2. PUBLIC TRANSPORT PERFORMANCE (Single Evening Peak Hour)")
        display(df_pt.style.format({
            "Year 1 Trips": "{:,.0f} trips",
            "Year 40 Trips": "{:,.0f} trips",
            "Year 1 Trip Share": "{:.1%}", "Year 40 Trip Share": "{:.1%}",
            "Year 1 PKM Share (>5km)": "{:.1%}", "Year 40 PKM Share (>5km)": "{:.1%}",
            "Year 1 TT": "{:.1f} min",
            "Year 40 TT": "{:.1f} min",
        }))

        print("\n[ACTIVE] 3. ACTIVE MOBILITY (BIKE & WALK) (Single Evening Peak Hour)")
        display(df_active.style.format({
            "Year 1 Bike Trips": "{:,.0f} trips",
            "Year 40 Bike Trips": "{:,.0f} trips",
            "Bike Trip Share": "{:.1%}",
            "Bike PKM Share": "{:.1%}",
            "Year 1 Walk Trips": "{:,.0f} trips",
            "Year 40 Walk Trips": "{:,.0f} trips",
            "Walk Trip Share": "{:.1%}",
            "Walk PKM Share": "{:.1%}",
        }))

        print("\n[SYSTEM] 4. OVERALL MULTIMODAL SYSTEM SUMMARY (Single Evening Peak Hour)")
        display(df_system.style.format({
            "Year 1 Total Demand": "{:,.0f} trips",
            "Year 40 Total Demand": "{:,.0f} trips",
            "Year 1 System Avg TT": "{:.1f} min",
            "Year 40 System Avg TT": "{:.1f} min",
            "Year 1 Peak Delay": "{:,.0f} h",
            "Year 40 Peak Delay": "{:,.0f} h",
            "Year 1 Peak CO2": "{:,.1f} t",
            "Year 40 Peak CO2": "{:,.1f} t",
        }))

    return df_car, df_pt, df_active, df_system


# =============================================================================
# 12. PROJECT-AGNOSTIC PARAMETER PLAYGROUND AND SENSITIVITY SWEEPS
# =============================================================================

def parameter_playground_specs(
    stage_spec: Mapping[str, Any] | None = None,
    *,
    demand_growth_y40: float | None = None,
    pt_asc_shift_y40: float | None = None,
    nominal_params: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return the supported controls and defensible teaching ranges."""

    import parameters as p
    nominal = {**p.NOMINAL_PARAMS, **dict(nominal_params or {})}
    if demand_growth_y40 is None:
        demand_growth_y40 = nominal["PASSENGER_DEMAND_GROWTH_Y40"]
    if pt_asc_shift_y40 is None:
        pt_asc_shift_y40 = nominal["PT_ASC_SHIFT_Y40"]
    stage_spec = dict(stage_spec or {})

    def stage_value(name: str, default: float) -> float:
        return float(stage_spec.get(name, default))

    return {
        "PASSENGER_DEMAND_GROWTH_Y40": {
            "value": float(demand_growth_y40), "min": 0.0, "max": 0.80,
            "step": 0.05, "readout_format": ".0%", "group": "Demand",
            "description": "Cumulative demand growth by Year 40",
        },
        "PT_ASC_SHIFT_Y40": {
            "value": float(pt_asc_shift_y40), "min": -1.0, "max": 1.0,
            "step": 0.05, "readout_format": ".2f", "group": "Mode preferences",
            "description": "Common additive PT ASC preference shift by Year 40",
        },
        "EBIKE_SHARE": {
            "value": float(nominal["EBIKE_SHARE_Y40"]), "min": 0.0, "max": 1.0,
            "step": 0.05, "readout_format": ".0%", "group": "Cycling",
            "description": "Share of bicycles represented by e-bikes in Year 40",
        },
        "EBIKE_SPEED_MULTIPLIER": {
            "value": stage_value("EBIKE_SPEED_MULTIPLIER", 1.5),
            "min": 1.0, "max": 2.5, "step": 0.1, "readout_format": ".1f",
            "group": "Cycling", "description": "E-bike speed multiplier",
        },
        **{
            name: {
                "value": 0.0, "min": 0.0, "max": 50.0, "step": 5.0,
                "readout_format": ".0f", "group": "PT infrastructure and service",
                "description": description,
            }
            for name, description in {
                "travel_time_reduction_pct": "PT in-vehicle time reduction (%)",
                "initial_wait_reduction_pct": "Initial waiting-time reduction (%)",
                "transfer_wait_reduction_pct": "Transfer waiting-time reduction (%)",
                "transfer_time_reduction_pct": "Physical transfer-time reduction (%)",
                "access_time_reduction_pct": "Station access-time reduction (%)",
                "egress_time_reduction_pct": "Station egress-time reduction (%)",
            }.items()
        },
    }


def _playground_stage_spec(stage_spec, selected, corridor_municipalities):
    """Apply the playground's physical controls to a fresh stage definition."""
    import itertools

    result = deepcopy(dict(stage_spec))
    result["EBIKE_SPEED_MULTIPLIER"] = selected["EBIKE_SPEED_MULTIPLIER"]
    effects = {name: selected[name] for name in (
        "travel_time_reduction_pct", "initial_wait_reduction_pct",
        "transfer_wait_reduction_pct", "transfer_time_reduction_pct",
        "access_time_reduction_pct", "egress_time_reduction_pct",
    )}
    if any(value > 0.0 for value in effects.values()):
        result["railway_expansions"] = [*result.get("railway_expansions", []), {
            "name": "Exploratory corridor PT intervention",
            "area_pairs": [
                {"origin": {"municipality_name": origin},
                 "destination": {"municipality_name": destination}}
                for origin, destination in itertools.combinations(corridor_municipalities, 2)
            ],
            "both_directions": True, "effects": effects,
        }]
    return result


def _playground_reference_matches(mode_result, stage_spec, stage):
    """Reject stale stage interventions before seeding a playground cache."""
    import json
    from additional.section_flows import section_config

    if mode_result is None or int(mode_result.stage) != int(stage):
        return False
    expected = {key: value for key, value in stage_spec.items() if key != "name"}
    for key in ("railway_expansions", "mobility_hubs", "bike_highways", "road_capacity"):
        expected[key] = expected.get(key, [])
    expected["section_config"] = section_config(stage_spec.get("section_config"))
    actual = {key: mode_result.scenario.get(key, [] if isinstance(value, list) else None)
              for key, value in expected.items()}
    return json.dumps(expected, sort_keys=True, default=str) == json.dumps(actual, sort_keys=True, default=str)


def evaluate_parameter_playground(
    context: TransportContext,
    stage_specs: Mapping[int, Mapping[str, Any]],
    *,
    corridor_zone_ids: list[str] | set[str],
    corridor_municipalities: list[str],
    values: Mapping[str, float] | None = None,
    slider_specs: Mapping[str, Mapping[str, Any]] | None = None,
    stage: int = 0,
    year40_index: int = 39,
    simulation_params: Mapping[str, Any] | None = None,
    calculation_cache: dict[str, Any] | None = None,
    corridor_context: CorridorContext | None = None,
    parallel_years: bool = False,
    precomputed_year_assignments: Mapping[str, Any] | None = None,
    simulation_assignment_settings: Mapping[str, Any] | None = None,
    years: Sequence[str] = ("year1", "year40"),
) -> dict[str, Any]:
    """Evaluate one configuration for the requested nominal years.

    Both the interactive dashboard and the sensitivity sweep call this
    function, preventing either view from silently ignoring a control.
    """

    import json
    import parameters as p
    import simulation_engine as simulation

    stage = int(stage)
    requested_years = tuple(dict.fromkeys(years))
    if not requested_years or set(requested_years).difference(("year1", "year40")):
        raise ValueError("years must contain 'year1', 'year40', or both.")
    if stage not in stage_specs:
        raise KeyError(f"Unknown stage {stage}; available stages: {list(stage_specs)}")
    if len(corridor_municipalities) < 2:
        raise ValueError("At least two corridor municipalities are required.")

    specs = dict(
        slider_specs
        or parameter_playground_specs(stage_specs[stage], nominal_params=simulation_params)
    )
    selected = {name: float(spec["value"]) for name, spec in specs.items()}
    unknown = set(values or {}).difference(specs)
    if unknown:
        raise KeyError(f"Unsupported playground parameters: {sorted(unknown)}")
    selected.update({name: float(value) for name, value in (values or {}).items()})

    stage_spec = _playground_stage_spec(stage_specs[stage], selected, corridor_municipalities)
    dynamic_specs = {stage: stage_spec}

    # Demand and PT preference growth affect only Year 40. Excluding them from this
    # key means a growth-only slider change reuses both mode choice and the Year 1 solve.
    transport_key = (
        stage,
        tuple(
            (name, selected[name])
            for name in specs
            if name not in ("PASSENGER_DEMAND_GROWTH_Y40", "PT_ASC_SHIFT_Y40", "EBIKE_SHARE")
        ),
    )
    nominal_transport_key = (
        stage,
        tuple(
            (name, float(specs[name]["value"]))
            for name in specs
            if name not in ("PASSENGER_DEMAND_GROWTH_Y40", "PT_ASC_SHIFT_Y40", "EBIKE_SHARE")
        ),
    )
    active_assignment_settings = resolve_assignment_settings(simulation_assignment_settings)
    assignment_key = tuple(
        sorted((str(name), repr(value)) for name, value in active_assignment_settings.items())
    )
    year1_key = (transport_key, assignment_key)
    cache = calculation_cache if calculation_cache is not None else {}
    effective_params = {**p.NOMINAL_PARAMS, **dict(simulation_params or {})}
    from additional import uncertainty as uc
    nominal_paths = uc.nominal_paths(nominal=effective_params)

    def year_parameters(year_index, growth, pt_shift):
        params = uc.year_parameters(nominal_paths, year_index, params=effective_params)
        params["PASSENGER_DEMAND_GROWTH"] = growth
        params["PT_ASC_SHIFT"] = pt_shift
        if year_index > 0:
            params["EBIKE_SHARE"] = selected["EBIKE_SHARE"]
        return params

    physical_scope = (
        id(context), id(corridor_context), tuple(sorted(map(str, corridor_zone_ids))),
        tuple(corridor_municipalities), json.dumps(stage_specs, sort_keys=True, default=str),
        tuple((name, effective_params.get(name, 0.0)) for name in (
            "BIKE_ASC_SHIFT", "BIKE_ASC_SHIFT_Y40", "ROAD_FREIGHT_GROWTH",
            "ROAD_FREIGHT_GROWTH_Y40", "EBIKE_SHARE",
        )),
    )
    if cache.get("physical_scope", physical_scope) != physical_scope:
        cache.clear()
    cache["physical_scope"] = physical_scope
    mode_cache = cache.setdefault("mode_choice", {})
    cached_mode = mode_cache.get(transport_key)
    if cached_mode is not None and not _playground_reference_matches(cached_mode[0], stage_spec, stage):
        mode_cache.pop(transport_key)
        cache.pop("year1", None)
        cache.pop("year40", None)
    year1_cache = cache.setdefault("year1", {})
    year40_cache = cache.setdefault("year40", {})
    cache_hits = {
        "mode_choice": transport_key in mode_cache,
        "year1": year1_key in year1_cache,
    }

    if transport_key not in mode_cache:
        mode_result = run_transport_mode_choice(
            context,
            stage=stage,
            stage_specs=dynamic_specs,
            corridor_zone_ids=corridor_zone_ids,
        )
        metrics = extract_corridor_metrics(
            context, mode_result, corridor_zone_ids=corridor_zone_ids
        )
        mode_cache[transport_key] = (mode_result, metrics)
        # Mode results contain full OD matrices, so keep only recent scenarios.
        while len(mode_cache) > 3:
            removable = next(
                (key for key in mode_cache if key != nominal_transport_key), None
            )
            if removable is None:
                break
            mode_cache.pop(removable)
    else:
        mode_result, metrics = mode_cache[transport_key]

    common = {
        "base_metrics": metrics,
        "stage": stage,
        "context": context,
        "mode_result": mode_result,
        "corridor_municipalities": corridor_municipalities,
        "corridor_context": corridor_context,
        "return_details": True,
        "assignment_settings": active_assignment_settings,
    }
    common["params"] = effective_params

    pt_asc_shift_y40 = float(selected.get("PT_ASC_SHIFT_Y40", 0.0))
    year40_key = (
        transport_key, assignment_key,
        selected["PASSENGER_DEMAND_GROWTH_Y40"], pt_asc_shift_y40,
        selected["EBIKE_SHARE"], int(year40_index),
    )
    cache_hits["year40"] = year40_key in year40_cache
    missing_years = []
    if "year1" in requested_years and year1_key not in year1_cache:
        missing_years.append(("year1", 0, 0.0, 0.0))
    if "year40" in requested_years and year40_key not in year40_cache:
        missing_years.append(
            ("year40", int(year40_index), selected["PASSENGER_DEMAND_GROWTH_Y40"], pt_asc_shift_y40)
        )

    def simulate(item: tuple[str, int, float, float]) -> tuple[str, dict[str, Any]]:
        name, year_index, growth, pt_asc_shift = item
        year_params = year_parameters(year_index, growth, pt_asc_shift)
        reusable = precomputed_year_assignments or {}
        assignment_result = reusable.get(
            name, reusable.get("Year 1" if name == "year1" else "Year 40")
        )
        if assignment_result is not None:
            candidate_mode = getattr(assignment_result, "mode_result", None)
            if candidate_mode is None:
                assignment_result = None
            else:
                base_cache_key = tuple(getattr(mode_result, "cache_key", ()))
                candidate_cache_key = tuple(getattr(candidate_mode, "cache_key", ()))
                expected_cache_key = base_cache_key
                if abs(float(mode_result.scenario.get("ebike_share", 0.0)) - year_params["EBIKE_SHARE"]) > 1e-12:
                    expected_cache_key += ("ebike_share", float(year_params["EBIKE_SHARE"]))
                same_scenario = (
                    candidate_cache_key == (*expected_cache_key, "coupled_msa")
                    and transport_key == nominal_transport_key
                    and assignment_settings_match(
                        assignment_result.diagnostics, active_assignment_settings,
                        modal_feedback=True, passthrough_fraction=DEFAULT_PASSTHROUGH_FRACTION,
                    )
                )
                changing_keys = {"name", "ASC_PT_WALK", "ASC_PT_BIKE", "ASC_BIKE", "ebike_share"}
                physical_scenario = lambda mode: {key: value for key, value in mode.scenario.items()
                                                  if key not in changing_keys}
                same_scenario = same_scenario and (
                    json.dumps(physical_scenario(candidate_mode), sort_keys=True, default=str)
                    == json.dumps(physical_scenario(mode_result), sort_keys=True, default=str)
                )
                for asc, shift in (("ASC_PT_WALK", year_params["PT_ASC_SHIFT"]),
                                   ("ASC_PT_BIKE", year_params["PT_ASC_SHIFT"]),
                                   ("ASC_BIKE", year_params["BIKE_ASC_SHIFT"])):
                    same_scenario = same_scenario and np.isclose(
                        candidate_mode.scenario.get(asc, np.nan),
                        mode_result.scenario.get(asc, np.nan) + shift,
                    )
                same_scenario = same_scenario and np.isclose(
                    candidate_mode.scenario.get("ebike_share", np.nan), year_params["EBIKE_SHARE"]
                ) and np.isclose(
                    assignment_result.diagnostics.get("road_freight_multiplier", np.nan),
                    1.0 + year_params["ROAD_FREIGHT_GROWTH"],
                )
                base_od = sum(mode_result.od_by_mode.values())
                candidate_od = sum(candidate_mode.od_by_mode.values())
                same_demand = (
                    base_od.index.equals(candidate_od.index)
                    and base_od.columns.equals(candidate_od.columns)
                    and np.allclose(candidate_od.to_numpy(), base_od.to_numpy() * (1.0 + float(growth)),
                                    rtol=1e-6, atol=1e-9)
                )
                if not same_scenario or not same_demand:
                    assignment_result = None
        year_common = dict(common)
        year_common["params"] = year_params
        snapshot = simulation.simulate_year(
            **year_common, year_idx=year_index, g_cum=growth,
            assignment_result=assignment_result,
        )
        snapshot["reused_assignment"] = assignment_result is not None
        return name, snapshot

    if len(missing_years) == 2 and parallel_years:
        # The two years share inputs but have independent MSA trajectories.
        # Processes provide real multi-core execution for NetworkX routing.
        from joblib import Parallel, delayed
        calculated = Parallel(n_jobs=2, backend="loky")(
            delayed(simulate)(item) for item in missing_years
        )
    else:
        calculated = [simulate(item) for item in missing_years]

    for name, value in calculated:
        if name == "year1":
            year1_cache[year1_key] = value
        else:
            year40_cache[year40_key] = value
    while len(year1_cache) > 6:
        year1_cache.pop(next(iter(year1_cache)))
    while len(year40_cache) > 10:
        year40_cache.pop(next(iter(year40_cache)))

    def reprice(snapshot, year_index, growth, pt_shift):
        # Annual caches retain solved physical states; valuation is inexpensive
        # and always reflects the current parameters.
        year_params = year_parameters(year_index, growth, pt_shift)
        physical = {"schema_version": 1, "source": "coupled_msa", "stage": stage,
                    "metrics": snapshot["mode_metrics"],
                    "assignment_diagnostics": snapshot.get("assignment_diagnostics", {})}
        result = simulation.simulate_year(
            physical["metrics"], stage=stage, year_idx=year_index, g_cum=growth,
            params=year_params, transport_state=physical, return_details=True,
        )
        result["assignment_diagnostics"] = physical["assignment_diagnostics"]
        result["reused_assignment"] = bool(snapshot.get("reused_assignment", False))
        return result

    annual_results = {}
    if "year1" in requested_years:
        annual_results["year1"] = reprice(year1_cache[year1_key], 0, 0.0, 0.0)
    if "year40" in requested_years:
        annual_results["year40"] = reprice(year40_cache[year40_key], int(year40_index), selected["PASSENGER_DEMAND_GROWTH_Y40"], pt_asc_shift_y40)
    return {
        "parameters": selected,
        "stage_specs": dynamic_specs,
        "mode_result": mode_result,
        "metrics": metrics,
        **annual_results,
        "cache_hits": cache_hits,
        "calculation_method": (
            "coupled_msa"
            if active_assignment_settings.get("method") == "MSA"
            and active_assignment_settings.get("modal_feedback", True)
            else "native_msa"
        ),
    }


def _playground_peak_value(snapshot: Mapping[str, Any], key: str) -> float:
    """Recover peak physical quantities with their own annualization factors."""
    import parameters as p
    import simulation_engine as m

    if key == "total_demand":
        return sum(float(snapshot[f"{mode}_trips"]) for mode in ("car", "pt", "bike", "walk"))
    if key in ("congestion_delay_hours", "co2_tonnes"):
        mode = "congestion" if key == "congestion_delay_hours" else "car"
        factor = snapshot.get(f"welfare_p2a_{mode}", m._annualizers(p.NOMINAL_PARAMS)[mode])
        value = snapshot["car_co2_tonnes"] if key == "co2_tonnes" else snapshot[key]
        return float(value) / float(factor)
    return float(snapshot[key])


def parameter_objective_sweep(
    context: TransportContext,
    stage_specs: Mapping[int, Mapping[str, Any]],
    parameter_name: str,
    *,
    corridor_zone_ids: list[str] | set[str],
    corridor_municipalities: list[str],
    slider_specs: Mapping[str, Mapping[str, Any]] | None = None,
    n_points: int = 7,
    stage: int = 0,
    simulation_params: Mapping[str, Any] | None = None,
    assignment_method: str = "MSA",
    reference_mode_result: ModeChoiceResult | None = None,
    corridor_context: CorridorContext | None = None,
    precomputed_year_assignments: Mapping[str, Any] | None = None,
    simulation_assignment_settings: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Sweep one control at Year 40 using native MSA with congestion feedback."""
    return parameter_objective_sweeps(
        context, stage_specs, [parameter_name], n_jobs=1,
        corridor_zone_ids=corridor_zone_ids, corridor_municipalities=corridor_municipalities,
        slider_specs=slider_specs, n_points=n_points, stage=stage,
        simulation_params=simulation_params, assignment_method=assignment_method,
        reference_mode_result=reference_mode_result, corridor_context=corridor_context,
        precomputed_year_assignments=precomputed_year_assignments,
        simulation_assignment_settings=simulation_assignment_settings,
    )[parameter_name]


def _parameter_sweep_point(shared: Mapping[str, Any], point: Mapping[str, Any]) -> dict[str, Any]:
    """Return only scalar Year-40 indicators from one independent physical solve."""
    started = time.perf_counter()
    cache = {"mode_choice": {shared["nominal_transport_key"]: shared["reference"]}}
    result = evaluate_parameter_playground(
        shared["context"], shared["stage_specs"],
        values=point["values"], calculation_cache=cache,
        years=("year40",), parallel_years=False,
        precomputed_year_assignments=shared.get("precomputed_year_assignments"),
        **shared["options"],
    )
    snapshot = result["year40"]
    metrics = snapshot.get("mode_metrics", result["metrics"])
    diagnostics = snapshot.get("assignment_diagnostics", {})
    row = {
        "year40__avg_travel_time_min": snapshot["avg_tt_min"],
        "year40__congestion_delay_hours": _playground_peak_value(snapshot, "congestion_delay_hours"),
        "year40__co2_tonnes": _playground_peak_value(snapshot, "co2_tonnes"),
        "year40__total_demand": _playground_peak_value(snapshot, "total_demand"),
        **{f"year40__{mode}_share_trips": metrics[f"{mode}_share_trips"]
           for mode in ("car", "pt", "bike", "walk")},
        "year40__car_trips": metrics["car_trips"],
        "year40__car_vkt": metrics["car_dist_km"],
        "year40__assignment_iterations": int(diagnostics.get("iterations", 0)),
        "year40__road_relative_gap": float(diagnostics.get("final_road_relative_gap_feasible_averaged_od", np.nan)),
        "year40__reused_assignment": bool(snapshot.get("reused_assignment", False)),
        "year40__elapsed_seconds": time.perf_counter() - started,
    }
    return {"key": point["key"], "label": point["label"], "row": row}


def parameter_objective_sweeps(
    context: TransportContext,
    stage_specs: Mapping[int, Mapping[str, Any]],
    parameter_names: list[str],
    *,
    n_jobs: int = 2,
    assignment_method: str = "MSA",
    reference_mode_result: ModeChoiceResult | None = None,
    corridor_context: CorridorContext | None = None,
    precomputed_year_assignments: Mapping[str, Any] | None = None,
    simulation_assignment_settings: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """Schedule unique Year-40 points, sharing one nominal case across all curves."""
    import os
    import parameters as p
    from joblib import Parallel, delayed, parallel_config
    from threadpoolctl import threadpool_limits

    if not parameter_names:
        return {}
    method = str(assignment_method).upper()
    if method != "MSA":
        raise ValueError("Physical parameter sweeps require assignment_method='MSA'.")
    if int(n_jobs) != n_jobs or int(n_jobs) < 1:
        raise ValueError("n_jobs must be a positive integer.")
    allowed = {"corridor_zone_ids", "corridor_municipalities", "slider_specs", "n_points", "stage", "simulation_params"}
    if set(kwargs).difference(allowed):
        raise TypeError(f"Unsupported sweep options: {sorted(set(kwargs).difference(allowed))}")
    stage = int(kwargs.get("stage", 0))
    n_points = kwargs.get("n_points", 7)
    if int(n_points) != n_points or int(n_points) < 2:
        raise ValueError("n_points must be an integer of at least two.")
    simulation_params = {**p.NOMINAL_PARAMS, **dict(kwargs.get("simulation_params") or {})}
    specs = dict(kwargs.get("slider_specs") or parameter_playground_specs(stage_specs[stage], nominal_params=simulation_params))
    unknown = set(parameter_names).difference(specs)
    if unknown:
        raise KeyError(f"Unknown playground parameters: {sorted(unknown)}")
    settings = resolve_assignment_settings({"method": method, "modal_feedback": True,
                                           **dict(simulation_assignment_settings or {})})
    if not settings["modal_feedback"]:
        raise ValueError("Parameter sweeps require congestion-to-mode-choice feedback.")
    corridor_municipalities = list(kwargs["corridor_municipalities"])
    if corridor_context is None:
        corridor_context = build_corridor_context(
            context, corridor_municipalities=corridor_municipalities,
            name="parameter sweep corridor",
        )
    corridor_zone_ids = list(kwargs["corridor_zone_ids"])
    nominal = {name: float(spec["value"]) for name, spec in specs.items()}

    def point_key(values):
        return tuple((name, values[name]) for name in specs)

    nominal_key = point_key(nominal)
    points = {nominal_key: {"key": nominal_key, "values": nominal, "label": "nominal reference"}}
    requests = {}
    for parameter_name in dict.fromkeys(parameter_names):
        spec = specs[parameter_name]
        values = np.linspace(spec["min"], spec["max"], int(n_points))
        values[np.argmin(np.abs(values - nominal[parameter_name]))] = nominal[parameter_name]
        requests[parameter_name] = []
        for value in np.unique(values):
            selected = {**nominal, parameter_name: float(value)}
            key = point_key(selected)
            requests[parameter_name].append((float(value), key))
            points.setdefault(key, {"key": key, "values": selected,
                                    "label": f"{parameter_name}={value:g}"})

    nominal_stage_spec = _playground_stage_spec(stage_specs[stage], nominal, corridor_municipalities)
    if not _playground_reference_matches(reference_mode_result, nominal_stage_spec, stage):
        reference_mode_result = run_transport_mode_choice(
            context,
            stage=stage,
            stage_specs={stage: nominal_stage_spec},
            corridor_zone_ids=corridor_zone_ids,
        )
    shared = {
        "context": context, "stage_specs": stage_specs,
        "nominal_transport_key": (stage, tuple((name, nominal[name]) for name in specs
            if name not in ("PASSENGER_DEMAND_GROWTH_Y40", "PT_ASC_SHIFT_Y40", "EBIKE_SHARE"))),
        "reference": (reference_mode_result, extract_corridor_metrics(
            context, reference_mode_result, corridor_zone_ids=corridor_zone_ids)),
        "options": {
            "corridor_zone_ids": corridor_zone_ids, "corridor_municipalities": corridor_municipalities,
            "slider_specs": specs, "stage": stage, "year40_index": int(simulation_params["N_YEARS"]) - 1,
            "simulation_params": simulation_params, "corridor_context": corridor_context,
            "simulation_assignment_settings": settings,
        },
    }
    workers = min(len(points), int(n_jobs), max(1, (os.cpu_count() or 2) - 1))
    print(f"Year-40 MSA sweep: {len(requests)} parameters, {n_points} points each; "
          f"{len(points)} unique cases, {workers} worker(s).", flush=True)
    rows = {}

    def record(result):
        row = result["row"]
        rows[result["key"]] = row
        gap = row["year40__road_relative_gap"]
        gap_text = f"{gap:.2%}" if np.isfinite(gap) else "unavailable"
        reuse = " (reused assignment)" if row["year40__reused_assignment"] else ""
        print(f"[{len(rows)}/{len(points)}] {result['label']}: "
              f"{row['year40__assignment_iterations']} iterations, road gap {gap_text}, "
              f"{row['year40__elapsed_seconds']:.1f}s{reuse}", flush=True)

    # Reuse the already solved nominal assignment in the parent; workers do not
    # need another full set of its mode-choice matrices in their shared inputs.
    if precomputed_year_assignments:
        with threadpool_limits(limits=1):
            record(_parameter_sweep_point(
                {**shared, "precomputed_year_assignments": precomputed_year_assignments}, points[nominal_key]
            ))
    pending = [point for key, point in points.items() if key not in rows]
    workers = min(workers, len(pending))
    if workers <= 1:
        with threadpool_limits(limits=1):
            for point in pending:
                record(_parameter_sweep_point(shared, point))
    else:
        # Joblib shares large task-input arrays through memory maps. Each
        # finished point returns only its scalar indicators.
        with parallel_config(backend="loky", inner_max_num_threads=1):
            completed = Parallel(
                n_jobs=workers, return_as="generator_unordered", batch_size=1,
                pre_dispatch=workers,
            )(delayed(_parameter_sweep_point)(shared, point) for point in pending)
            for result in completed:
                record(result)
    results = {}
    for name, selections in requests.items():
        frame = pd.DataFrame([{"parameter": name, "value": value, **rows[key]} for value, key in selections])
        frame = frame.sort_values("value").reset_index(drop=True)
        frame.attrs.update(assignment_method=method, year_index=shared["options"]["year40_index"],
                           unique_year40_cases=len(points),
                           screening_note="Year-40 native MSA with congestion-to-mode-choice feedback.")
        results[name] = frame
    return results


# =============================================================================
# STAGE AND LINK-INTERVENTION EXPLORERS
# =============================================================================

def stage_specification_tables(
    stage_specs: Mapping[int, Mapping[str, Any]],
    params: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return compact, project-neutral tables describing the current stages.

    The notebook only displays these outputs; all parsing of intervention
    scopes and effects stays here so custom student stages appear automatically.
    """

    import parameters as p
    from stages import construction_emissions_tonnes, stage_capacity, stage_components, stage_headway

    params = {**p.NOMINAL_PARAMS, **(params or {})}
    intervention_keys = (
        "railway_expansions", "bike_highways", "mobility_hubs", "road_capacity"
    )
    def scope_text(intervention: Mapping[str, Any]) -> str:
        parts: list[str] = []
        for key, label in (
            ("area_pairs", "area pairs"), ("od_pairs", "OD pairs"),
            ("edge_ids", "directed links"),
        ):
            if intervention.get(key):
                parts.append(f"{len(intervention[key])} {label}")
        if intervention.get("zones"):
            zone_labels = []
            for selector in intervention["zones"]:
                zone_labels.append(
                    ", ".join(f"{k}={v}" for k, v in selector.items())
                    if isinstance(selector, Mapping) else str(selector)
                )
            parts.append("; ".join(zone_labels))
        if intervention.get("both_directions"):
            parts.append("both directions")
        return " | ".join(parts) or "No spatial selector"

    overview_rows, intervention_rows = [], []
    for stage_id, spec in sorted(stage_specs.items()):
        stations_active, tunnel_active = stage_components(stage_id)
        capital = (float(params["C_INV_STAGE1"]) * stations_active
                   + float(params["C_INV_STAGE2"]) * tunnel_active)
        overview_rows.append({
            "Stage": int(stage_id),
            "Name": spec.get("name", f"Stage {stage_id}"),
            "Rail": len(spec.get("railway_expansions", [])),
            "Bike": len(spec.get("bike_highways", [])),
            "Hubs": len(spec.get("mobility_hubs", [])),
            "Road": len(spec.get("road_capacity", [])),
            "Headway (min)": stage_headway(stage_id, params),
            "Section saving (min)": float(spec.get("section_time_saving_min", 0.0)),
            "PT capacity": stage_capacity(stage_id, params),
            "Incremental CAPEX (CHF)": capital,
            "Incremental OPEX (CHF/year)": capital * float(params["OPEX_RATE"]),
            "Construction emissions (total tCO2e)": (
                construction_emissions_tonnes(1) * stations_active
                + construction_emissions_tonnes(2) * tunnel_active),
        })
        for intervention_type in intervention_keys:
            for intervention in spec.get(intervention_type, []):
                effects = ", ".join(
                    f"{key}={value}"
                    for key, value in intervention.get("effects", {}).items()
                    if float(value) != 0.0
                ) or "No numerical effects"
                intervention_rows.append({
                    "Stage": int(stage_id),
                    "Type": intervention_type,
                    "Intervention": intervention.get("name", "Unnamed"),
                    "Spatial scope": scope_text(intervention),
                    "Effects": effects,
                })

    overview = pd.DataFrame(overview_rows).set_index("Stage")
    details = pd.DataFrame(
        intervention_rows,
        columns=["Stage", "Type", "Intervention", "Spatial scope", "Effects"],
    )
    return overview, details


def _prepare_road_edges(edges: Any) -> Any:
    """Return an editable edge copy with stable keys and baseline attributes."""

    prepared = edges.copy().reset_index(drop=True)
    has_unique_ids = (
        "edge_id" in prepared
        and prepared["edge_id"].notna().all()
        and not prepared["edge_id"].astype(str).duplicated().any()
    )
    if has_unique_ids:
        prepared["_link_key"] = prepared["edge_id"].astype(str)
    else:
        prepared["_link_key"] = [
            f"{source}->{target}#{position}"
            for position, (source, target) in enumerate(
                zip(prepared["source"], prepared["target"])
            )
        ]
    for column in ("capacity_vph", "lanes", "speed_kph", "free_flow_time_min"):
        if column not in prepared:
            prepared[column] = np.nan
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        baseline = f"baseline_{column}"
        if baseline not in prepared:
            prepared[baseline] = prepared[column]
    return prepared


def _road_change_columns(edges: Any) -> Any:
    """Update common before/after fields used by maps, tables, and assignment."""

    edges = edges.copy()
    base_capacity = edges["baseline_capacity_vph"].replace(0.0, np.nan)
    base_speed = edges["baseline_speed_kph"].replace(0.0, np.nan)
    edges["capacity_change_pct"] = (
        100.0 * (edges["capacity_vph"] / base_capacity - 1.0)
    ).fillna(0.0)
    edges["speed_change_pct"] = (
        100.0 * (edges["speed_kph"] / base_speed - 1.0)
    ).fillna(0.0)
    changed = np.zeros(len(edges), dtype=bool)
    for column in ("capacity_vph", "lanes", "speed_kph", "free_flow_time_min"):
        current = edges[column].fillna(-1.0).to_numpy(dtype=float)
        baseline = edges[f"baseline_{column}"].fillna(-1.0).to_numpy(dtype=float)
        changed |= ~np.isclose(current, baseline)
    edges["road_stage_modified"] = changed
    edges["intervention_status"] = np.where(changed, "modified", "baseline")
    return edges


def apply_road_capacity_stage(
    corridor: CorridorContext,
    stage_spec: Mapping[str, Any] | None,
) -> tuple[CorridorContext, pd.DataFrame]:
    """Apply documented ``road_capacity`` entries to a corridor copy.

    Supported selectors are ``edge_ids`` and directional ``source_target_pairs``.
    Supported effects are absolute or percentage capacity/speed changes and
    absolute or incremental lane counts. Missing selectors fail loudly: a road
    intervention must never expand silently to the entire network.
    """

    edges = _prepare_road_edges(corridor.edges)
    configured = (stage_spec or {}).get("road_capacity", []) or []
    interventions = [configured] if isinstance(configured, Mapping) else list(configured)
    intervention_name = pd.Series("", index=edges.index, dtype="object")

    for intervention in interventions:
        selected = pd.Series(False, index=edges.index)
        configured_ids = {str(value) for value in intervention.get("edge_ids", [])}
        if configured_ids:
            known_ids = set(edges.get("edge_id", pd.Series(dtype=str)).astype(str))
            missing = configured_ids.difference(known_ids)
            if missing:
                raise ValueError(f"Road intervention contains unknown edge_ids: {sorted(missing)}")
            selected |= edges["edge_id"].astype(str).isin(configured_ids)

        pairs = intervention.get(
            "source_target_pairs", intervention.get("node_pairs", [])
        )
        for pair in pairs:
            if isinstance(pair, Mapping):
                source, target = pair.get("source"), pair.get("target")
            elif len(pair) == 2:
                source, target = pair
            else:
                raise ValueError(f"Invalid source-target pair: {pair}")
            forward = edges["source"].astype(str).eq(str(source)) & edges["target"].astype(str).eq(str(target))
            selected |= forward
            if intervention.get("both_directions", False):
                selected |= edges["source"].astype(str).eq(str(target)) & edges["target"].astype(str).eq(str(source))

        if not selected.any():
            raise ValueError(
                f"Road intervention '{intervention.get('name', 'Unnamed')}' selected no links."
            )

        effects = dict(intervention.get("effects", {}))
        if "capacity_vph" in effects:
            edges.loc[selected, "capacity_vph"] = float(effects["capacity_vph"])
        elif "capacity_increase_pct" in effects:
            factor = 1.0 + float(effects["capacity_increase_pct"]) / 100.0
            if factor <= 0.0:
                raise ValueError("capacity_increase_pct must be greater than -100%.")
            edges.loc[selected, "capacity_vph"] *= factor

        if "lanes" in effects:
            edges.loc[selected, "lanes"] = float(effects["lanes"])
        elif "lanes_change" in effects:
            edges.loc[selected, "lanes"] = np.maximum(
                edges.loc[selected, "lanes"] + float(effects["lanes_change"]), 1.0
            )

        speed_changed = False
        if "speed_kph" in effects:
            edges.loc[selected, "speed_kph"] = float(effects["speed_kph"])
            speed_changed = True
        elif "speed_increase_pct" in effects:
            speed_factor = 1.0 + float(effects["speed_increase_pct"]) / 100.0
            if speed_factor <= 0.0:
                raise ValueError("speed_increase_pct must be greater than -100%.")
            edges.loc[selected, "speed_kph"] *= speed_factor
            speed_changed = True
        if speed_changed:
            edges.loc[selected, "free_flow_time_min"] = (
                edges.loc[selected, "length_m"] / 1000.0
                / edges.loc[selected, "speed_kph"].clip(lower=1.0) * 60.0
            )
        intervention_name.loc[selected] = intervention.get("name", "Road intervention")

    edges = _road_change_columns(edges)
    edges["road_intervention"] = intervention_name
    audit_columns = [
        "_link_key", "edge_id", "source", "target", "highway", "length_m",
        "baseline_lanes", "lanes", "baseline_capacity_vph", "capacity_vph",
        "capacity_change_pct", "baseline_speed_kph", "speed_kph",
        "speed_change_pct", "baseline_free_flow_time_min", "free_flow_time_min",
        "road_intervention",
    ]
    audit = edges.loc[
        edges["road_stage_modified"],
        [column for column in audit_columns if column in edges],
    ].copy()
    metadata = {
        **corridor.metadata,
        "modified_directed_links": int(edges["road_stage_modified"].sum()),
        "applied_road_capacity": deepcopy(interventions),
    }
    return replace(corridor, edges=edges, metadata=metadata), audit


def ensure_stage_road_context(
    corridor: CorridorContext,
    stage_spec: Mapping[str, Any] | None,
) -> CorridorContext:
    """Apply a stage once, preserving its subsequent interactive road edits."""
    configured = (stage_spec or {}).get("road_capacity", []) or []
    interventions = [configured] if isinstance(configured, Mapping) else list(configured)
    previous = corridor.metadata.get("applied_road_capacity")
    if previous == interventions or (previous is None and not interventions):
        return corridor
    if previous:
        # A context reused for a different stage starts from its original links.
        edges = corridor.edges.copy()
        for column in ("capacity_vph", "lanes", "speed_kph", "free_flow_time_min"):
            edges[column] = edges[f"baseline_{column}"]
        corridor = replace(corridor, edges=edges)
    return apply_road_capacity_stage(corridor, stage_spec)[0]


# =============================================================================
# MULTIMODAL STAGE COMPARISON AND EDITING
# =============================================================================




# Notebook presentation API (implementations are loaded only when requested).

def readiness_message(status: pd.DataFrame) -> str:
    """See additional.transport_display.readiness_message."""
    from additional.transport_display import readiness_message as render
    return render(status)


def mode_choice_dashboard(context: TransportContext, stage_specs: Mapping[int, Mapping[str, Any]], corridor_zone_ids: list[str] | set[str] | None=None) -> tuple[Any, dict[str, Any]]:
    """See additional.transport_display.mode_choice_dashboard."""
    from additional.transport_display import mode_choice_dashboard as render
    return render(context, stage_specs, corridor_zone_ids)


def network_explorer(edges: Any, metrics: Mapping[str, Mapping[str, Any]], *, nodes: Any | None=None, gates: Any | None=None, polygon: Any | None=None, zone_polygons: Any | None=None, boundary_level: str='FSM zone', show_zone_boundaries: bool=False, width_column: str | None=None, title: str='Network explorer', simplify_m: float=3.0, height: int=620) -> Any:
    """See additional.transport_display.network_explorer."""
    from additional.transport_display import network_explorer as render
    return render(edges, metrics, nodes=nodes, gates=gates, polygon=polygon, zone_polygons=zone_polygons, boundary_level=boundary_level, show_zone_boundaries=show_zone_boundaries, width_column=width_column, title=title, simplify_m=simplify_m, height=height)


def corridor_explorer(corridor: CorridorContext) -> Any:
    """See additional.transport_display.corridor_explorer."""
    from additional.transport_display import corridor_explorer as render
    return render(corridor)


def corridor_network_explorer(context: TransportContext, corridor_municipalities: list[str] | None=None) -> Any:
    """See additional.transport_display.corridor_network_explorer."""
    from additional.transport_display import corridor_network_explorer as render
    return render(context, corridor_municipalities)


def full_network_explorer(context: TransportContext, corridor_municipalities: list[str] | None=None) -> Any:
    """See additional.transport_display.full_network_explorer."""
    from additional.transport_display import full_network_explorer as render
    return render(context, corridor_municipalities)


def corridor_detail_table(edges: pd.DataFrame) -> pd.DataFrame:
    """See additional.transport_display.corridor_detail_table."""
    from additional.transport_display import corridor_detail_table as render
    return render(edges)


def corridor_dashboard(context: TransportContext, corridor: CorridorContext, *, demand_matrix: pd.DataFrame | None=None, passthrough_fraction: float=DEFAULT_PASSTHROUGH_FRACTION) -> Any:
    """See additional.transport_display.corridor_dashboard."""
    from additional.transport_display import corridor_dashboard as render
    return render(context, corridor, demand_matrix=demand_matrix, passthrough_fraction=passthrough_fraction)


def assignment_explorer(assignment: AssignmentResult | Any, corridor: CorridorContext | None=None) -> Any:
    """See additional.transport_display.assignment_explorer."""
    from additional.transport_display import assignment_explorer as render
    return render(assignment, corridor)


def static_network_plot(context: TransportContext, mode_result: Any=None, corridor_municipalities: list[str] | None=None, project_name: str='Corridor', assigned_edges: Any=None) -> Any:
    """See additional.transport_display.static_network_plot."""
    from additional.transport_display import static_network_plot as render
    return render(context, mode_result, corridor_municipalities, project_name, assigned_edges)


def assignment_dashboard(context: TransportContext, corridor: CorridorContext, mode_state: dict[str, Any], *, modal_feedback: bool=False) -> tuple[Any, dict[str, Any]]:
    """See additional.transport_display.assignment_dashboard."""
    from additional.transport_display import assignment_dashboard as render
    return render(context, corridor, mode_state, modal_feedback=modal_feedback)


def plot_assignment_convergence_test(test_result: Mapping[str, pd.DataFrame], *, thresholds: tuple[float, ...]=(0.1, 0.05, 0.02)) -> Any:
    """See additional.transport_display.plot_assignment_convergence_test."""
    from additional.transport_display import plot_assignment_convergence_test as render
    return render(test_result, thresholds=thresholds)


def od_matrix_explorer(matrix: pd.DataFrame, corridor: CorridorContext, *, height: int=650) -> Any:
    """See additional.transport_display.od_matrix_explorer."""
    from additional.transport_display import od_matrix_explorer as render
    return render(matrix, corridor, height=height)


def corridor_flow_map_explorer(matrix: pd.DataFrame, corridor: CorridorContext, *, n_bins: int=5, max_flows: int=250, min_trips: float=0.0, height: int=680) -> Any:
    """See additional.transport_display.corridor_flow_map_explorer."""
    from additional.transport_display import corridor_flow_map_explorer as render
    return render(matrix, corridor, n_bins=n_bins, max_flows=max_flows, min_trips=min_trips, height=height)


def corridor_od_explorer(matrix: pd.DataFrame, corridor: CorridorContext, **flow_map_kwargs: Any) -> Any:
    """See additional.transport_display.corridor_od_explorer."""
    from additional.transport_display import corridor_od_explorer as render
    return render(matrix, corridor, **flow_map_kwargs)


def municipality_od_explorer(context: TransportContext, municipalities: list[str] | tuple[str, ...] | set[str], *, matrix: pd.DataFrame | None=None, zone_ids: list[str] | set[str] | None=None, height: int=700) -> Any:
    """See additional.transport_display.municipality_od_explorer."""
    from additional.transport_display import municipality_od_explorer as render
    return render(context, municipalities, matrix=matrix, zone_ids=zone_ids, height=height)


def municipality_flow_map_explorer(context: TransportContext, municipalities: list[str] | tuple[str, ...] | set[str], *, matrix: pd.DataFrame | None=None, zone_ids: list[str] | set[str] | None=None, n_bins: int=5, trip_bin_edges: list[float] | None=None, min_trips: float=0.0, max_flows: int=150, height: int=700) -> Any:
    """See additional.transport_display.municipality_flow_map_explorer."""
    from additional.transport_display import municipality_flow_map_explorer as render
    return render(context, municipalities, matrix=matrix, zone_ids=zone_ids, n_bins=n_bins, trip_bin_edges=trip_bin_edges, min_trips=min_trips, max_flows=max_flows, height=height)


def flow_map_explorer(context: TransportContext, mode_result: ModeChoiceResult | None=None, corridor_municipalities: list[str] | None=None, *, matrix: pd.DataFrame | None=None, buffer_m: float=DEFAULT_CORRIDOR_BUFFER_M, n_bins: int=5, max_flows: int=250, pass_through_factor: float=0.05, min_trips: float=0.0) -> Any:
    """See additional.transport_display.flow_map_explorer."""
    from additional.transport_display import flow_map_explorer as render
    return render(context, mode_result, corridor_municipalities, matrix=matrix, buffer_m=buffer_m, n_bins=n_bins, max_flows=max_flows, pass_through_factor=pass_through_factor, min_trips=min_trips)


def plot_parameter_playground_result(result: Mapping[str, Any]) -> Any:
    """See additional.transport_display.plot_parameter_playground_result."""
    from additional.transport_display import plot_parameter_playground_result as render
    return render(result)


def plot_parameter_playground_impacts(reference: Mapping[str, Any], scenario: Mapping[str, Any], *, year: str='year40') -> Any:
    """See additional.transport_display.plot_parameter_playground_impacts."""
    from additional.transport_display import plot_parameter_playground_impacts as render
    return render(reference, scenario, year=year)


def parameter_playground_dashboard(context: TransportContext, stage_specs: Mapping[int, Mapping[str, Any]], *, corridor_zone_ids: list[str] | set[str], corridor_municipalities: list[str], demand_growth_y40: float | None=None, stage: int=0, simulation_params: Mapping[str, Any] | None=None, reference_mode_result: ModeChoiceResult | None=None, reference_year_assignments: Mapping[str, Any] | None=None, control_groups: Mapping[str, Sequence[str]] | None=None) -> tuple[Any, dict[str, Any]]:
    """See additional.transport_display.parameter_playground_dashboard."""
    from additional.transport_display import parameter_playground_dashboard as render
    return render(context, stage_specs, corridor_zone_ids=corridor_zone_ids, corridor_municipalities=corridor_municipalities, demand_growth_y40=demand_growth_y40, stage=stage, simulation_params=simulation_params, reference_mode_result=reference_mode_result, reference_year_assignments=reference_year_assignments, control_groups=control_groups)


def plot_parameter_objective_sweeps(sweep_results: Mapping[str, pd.DataFrame], slider_specs: Mapping[str, Mapping[str, Any]]) -> Any:
    """See additional.transport_display.plot_parameter_objective_sweeps."""
    from additional.transport_display import plot_parameter_objective_sweeps as render
    return render(sweep_results, slider_specs)


def stage_intervention_scope_map(context: TransportContext, corridor: CorridorContext, stage_spec: Mapping[str, Any], *, stage_id: int | None=None, height: int=620, boundary_level: str='Municipality', show_zone_boundaries: bool=False, show_centroids: bool=True) -> Any:
    """See additional.transport_display.stage_intervention_scope_map."""
    from additional.transport_display import stage_intervention_scope_map as render
    return render(context, corridor, stage_spec, stage_id=stage_id, height=height, boundary_level=boundary_level, show_zone_boundaries=show_zone_boundaries, show_centroids=show_centroids)


def stage_map_dashboard(context: TransportContext, corridor: CorridorContext, stage_specs: Mapping[int, Mapping[str, Any]], *, height: int=620) -> tuple[Any, dict[str, Any]]:
    """See additional.transport_display.stage_map_dashboard."""
    from additional.transport_display import stage_map_dashboard as render
    return render(context, corridor, stage_specs, height=height)


def road_link_stage_editor(context: TransportContext, corridor: CorridorContext, stage_specs: Mapping[int, Mapping[str, Any]], *, height: int=590, stage_selector: Any | None=None, defer_map: bool=False) -> tuple[Any, dict[str, Any]]:
    """See additional.transport_display.road_link_stage_editor."""
    from additional.transport_display import road_link_stage_editor as render
    return render(context, corridor, stage_specs, height=height, stage_selector=stage_selector, defer_map=defer_map)


def road_network_difference_dashboard(baseline: CorridorContext, modified: CorridorContext, *, height: int=590) -> Any:
    """See additional.transport_display.road_network_difference_dashboard."""
    from additional.transport_display import road_network_difference_dashboard as render
    return render(baseline, modified, height=height)


def stage_value_variation_dashboard(context: TransportContext, corridor: CorridorContext, stage_specs: Mapping[int, Mapping[str, Any]], *, road_contexts: Mapping[int, CorridorContext] | None=None, height: int=560, original_stage_specs: Mapping | None=None, original_road_contexts: Mapping | None=None, editor_state: dict | None=None) -> tuple[Any, dict[str, Any]]:
    """See additional.transport_display.stage_value_variation_dashboard."""
    from additional.transport_display import stage_value_variation_dashboard as render
    return render(context, corridor, stage_specs, road_contexts=road_contexts, height=height, original_stage_specs=original_stage_specs, original_road_contexts=original_road_contexts, editor_state=editor_state)


def desire_line_stage_editor(context: TransportContext, corridor: CorridorContext, stage_specs: dict[int, dict[str, Any]], *, height: int=560, stage_selector: Any | None=None, fixed_intervention_type: str | None=None, defer_map: bool=False) -> tuple[Any, dict[str, Any]]:
    """See additional.transport_display.desire_line_stage_editor."""
    from additional.transport_display import desire_line_stage_editor as render
    return render(context, corridor, stage_specs, height=height, stage_selector=stage_selector, fixed_intervention_type=fixed_intervention_type, defer_map=defer_map)


def stage_intervention_editor(context: TransportContext, corridor: CorridorContext, stage_specs: Mapping[int, Mapping[str, Any]], *, height: int=560, packages: Mapping | None=None, combined_effects: Mapping | None=None, assemble_stages: Any | None=None, intervention_types: Sequence[str] | None=None) -> tuple[Any, dict[str, Any]]:
    """See additional.transport_display.stage_intervention_editor."""
    from additional.transport_display import stage_intervention_editor as render
    return render(context, corridor, stage_specs, height=height, packages=packages, combined_effects=combined_effects, assemble_stages=assemble_stages, intervention_types=intervention_types)
