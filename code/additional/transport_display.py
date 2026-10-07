"""Notebook maps, widgets and intervention editors.

Numerical transport functions remain in transport_model_interface; its public
presentation functions delegate here so notebook imports stay unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence
import importlib
import sys
import time
import numpy as np
import pandas as pd
from IPython.display import display
from IPython.core.display_functions import clear_output

import transport_model_interface as tmi
from additional import notebook_exports as exports


def readiness_message(status: pd.DataFrame) -> str:
    """Return a styled HTML readiness callout for notebook display."""
    missing = status.loc[status["required"] & ~status["present"], "relative_path"].tolist()
    if not missing:
        return (
            "<div style='padding:10px;border-left:5px solid #2e7d32;background:#edf7ed'>"
            "<b>Transport model ready.</b> 1,223 Canton Zürich zones and skims loaded.</div>"
        )
    items = "".join(f"<li><code>{value}</code></li>" for value in missing)
    return (
        "<div style='padding:10px;border-left:5px solid #c47f00;background:#fff8e1'>"
        f"<b>Missing prepared inputs:</b><ul>{items}</ul></div>"
    )


def _scale_dashboard_mode_choice(result: Any, multiplier: float, zone_ids: Any) -> Any:
    """Scale fixed-skim mode-choice quantities without repeating the same logit."""
    if multiplier == 1.0:
        return result
    quantities = {mode: matrix * multiplier for mode, matrix in result.od_by_mode.items()}
    return replace(
        result,
        drive_od=quantities["drive"],
        od_by_mode=quantities,
        summary=tmi._corridor_mode_summary(quantities, zone_ids),
        scenario={**result.scenario, "demand_multiplier": multiplier},
        cache_key=(result.cache_key[0], round(multiplier, 6), *result.cache_key[2:]),
    )


def mode_choice_dashboard(
    context: tmi.TransportContext,
    stage_specs: Mapping[int, Mapping[str, Any]],
    corridor_zone_ids: list[str] | set[str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Create a fast, responsive interactive Jupyter widget for mode-choice exploration."""
    from io import BytesIO
    from html import escape

    import ipywidgets as widgets
    import matplotlib.pyplot as plt

    state: dict[str, Any] = {
        "result": None,
        "metrics": None,
        "cache_key": None,
        "rendered_key": None,
        "rendering": False,
        "pending_values": None,
        "base_key": None,
        "base_result": None,
        "closed": False,
    }

    controls = {
        "stage": widgets.Dropdown(
            options=[(f"{spec.get('name', f'Stage {s}')}", s) for s, spec in sorted(stage_specs.items())],
            value=0 if 0 in stage_specs else min(stage_specs),
            description="Stage:",
            layout=widgets.Layout(width="340px"),
            style={"description_width": "60px"},
        ),
        "demand_multiplier": widgets.FloatSlider(
            value=1.0, min=0.75, max=1.50, step=0.05,
            description="Demand ×:", continuous_update=False, readout_format=".2f",
            tooltip="Multiply every baseline passenger OD cell by this factor.",
            layout=widgets.Layout(width="280px"),
            style={"description_width": "65px"},
        ),
        "pt_asc_shift": widgets.FloatSlider(
            value=0.0, min=-1.0, max=1.0, step=0.05,
            description="PT ASC +:", continuous_update=False, readout_format=".2f",
            tooltip="Add this preference shift to both PT alternatives; positive values favour PT.",
            layout=widgets.Layout(width="290px"),
            style={"description_width": "75px"},
        ),
    }

    # A single Image value is replaced atomically after each calculation.
    # This avoids the stale/duplicated rich-display messages that some
    # Jupyter frontends retain inside an Output widget.
    figure_image = widgets.Image(
        format="png",
        layout=widgets.Layout(width="100%", max_width="1150px"),
    )
    status = widgets.HTML()
    refresh = widgets.Button(description="Refresh", icon="refresh")
    state["controls"] = controls
    state["status"] = status

    def update(**values: Any) -> None:
        if state["closed"]:
            return
        key = tuple(values[name] for name in controls)
        # Some widget/front-end combinations deliver the same trait event more
        # than once. Do not repeat an unchanged calculation or output render.
        if state["rendering"]:
            state["pending_values"] = values
            return
        if state["rendered_key"] == key:
            return
        state["rendering"] = True
        refresh.disabled = True
        status.value = "<b>Updating mode choice and trip totals…</b>"
        started = time.perf_counter()
        fig = None
        try:
            if state["cache_key"] != key:
                stage_val = int(values["stage"])
                base_key = (stage_val, float(values["pt_asc_shift"]))
                if state["base_key"] != base_key:
                    state["base_result"] = tmi.run_transport_mode_choice(
                        context, stage=stage_val, demand_multiplier=1.0,
                        pt_asc_shift=base_key[1],
                        corridor_zone_ids=corridor_zone_ids, stage_specs=stage_specs,
                    )
                    state["base_key"] = base_key
                state["result"] = _scale_dashboard_mode_choice(
                    state["base_result"], float(values["demand_multiplier"]), corridor_zone_ids,
                )
                min_dist_km = stage_specs[stage_val].get("_min_distance_km") if stage_specs and stage_val in stage_specs else None
                state["metrics"] = tmi.extract_corridor_metrics(
                    context,
                    state["result"],
                    corridor_zone_ids=corridor_zone_ids,
                    min_distance_km=min_dist_km,
                )
                state["cache_key"] = key

            result = state["result"]
            metrics = state["metrics"]

            # 2. Side-by-Side: Table and Chart
            fig, (ax_table, ax_bar) = plt.subplots(
                1, 2, figsize=(11.5, 2.7),
                gridspec_kw={"width_ratios": [1.15, 1.25]}
            )

            # Build unified comparison data
            mode_rows = [
                ("Car passengers", metrics["car_trips"], metrics["car_share_trips"], metrics["car_share"]),
                ("Public Transport", metrics["pt_trips"], metrics["pt_share_trips"], metrics["pt_share"]),
                ("Bicycle", metrics["bike_trips"], metrics["bike_share_trips"], metrics["bike_share"]),
                ("Walking", metrics["walk_trips"], metrics["walk_share_trips"], metrics["walk_share"]),
            ]

            # Render clean table in left axes
            ax_table.axis("off")
            table_data = [["Mode", "Trips/peak hour", "Trip Share", "PKM share"]]
            for m_name, m_trips, m_t_share, m_p_share in mode_rows:
                table_data.append([m_name, f"{m_trips:,.0f}", f"{m_t_share:.1%}", f"{m_p_share:.1%}"])

            table = ax_table.table(
                cellText=table_data,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9.0)
            table.scale(1.0, 1.40)
            for (row_idx, col_idx), cell in table.get_celld().items():
                if row_idx == 0:
                    cell.set_facecolor("#edf2f7")
                    cell.set_text_props(weight="bold", color="#2d3748")
                cell.set_edgecolor("#cbd5e0")

            # Render Side-by-Side Bar Chart in right axes
            modes = [r[0] for r in mode_rows]
            trip_shares = [r[2] * 100 for r in mode_rows]
            pkm_shares = [r[3] * 100 for r in mode_rows]
            colors = ["#4C78A8", "#F28E2B", "#54A24B", "#76B7B2"]

            x = np.arange(len(modes))
            w = 0.35

            bars1 = ax_bar.bar(x - w/2, trip_shares, w, label="Trip Share", color=colors, alpha=0.45, edgecolor="black", linewidth=0.8)
            bars2 = ax_bar.bar(x + w/2, pkm_shares, w, label="Strategic PKM share", color=colors, hatch="//", edgecolor="black", linewidth=0.8)

            max_s = max(max(trip_shares), max(pkm_shares), 50.0)
            ax_bar.set_ylim(0, max_s * 1.25)
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(["Car", "PT", "Bike", "Walk"], fontsize=9)
            ax_bar.set_ylabel("Modal share (%)", fontsize=9.0)
            ax_bar.set_title(f"Corridor Modal Split (Stage {int(values['stage'])})", fontsize=10.0, fontweight="bold", pad=8)
            ax_bar.grid(axis="y", linestyle="--", alpha=0.3)
            ax_bar.legend(loc="upper right", fontsize=8, frameon=True)
            ax_bar.set_axisbelow(True)

            for b in bars1:
                h = b.get_height()
                ax_bar.text(b.get_x() + b.get_width()/2.0, h + 0.8, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5)
            for b in bars2:
                h = b.get_height()
                ax_bar.text(b.get_x() + b.get_width()/2.0, h + 0.8, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

            plt.tight_layout()
            buffer = BytesIO()
            fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
            plt.close(fig)
            figure_image.value = buffer.getvalue()
            exports.save_outputs(
                "1_5_mode_choice", png=figure_image.value,
                tables={
                    "modal_split": pd.DataFrame(mode_rows, columns=[
                        "Mode", "Trips/peak hour", "Trip Share", "PKM share",
                    ]),
                    "settings": pd.Series(values, name="value"),
                },
            )
            total_trips = sum(row[1] for row in mode_rows)
            status.value = (
                f"<b>Demand ×{values['demand_multiplier']:.2f}: {total_trips:,.0f} passenger trips/peak hour.</b> "
                f"PT ASC shift {values['pt_asc_shift']:+.2f}. "
                f"<small>Updated in {time.perf_counter() - started:.2f} s.</small><br>"
                "With fixed travel times, uniform demand scaling changes trip totals. "
                "The PT ASC shift changes modal shares."
            )
            state["rendered_key"] = key

        except Exception as error:
            state["result"] = None
            state["metrics"] = None
            state["cache_key"] = None
            state["rendered_key"] = None
            status.value = (
                "<span style='color:#b71c1c'><b>Mode-choice dashboard failed:</b> "
                f"<code>{escape(type(error).__name__ + ': ' + str(error))}</code> "
                "Correct the input and click Refresh to retry.</span>"
            )
        finally:
            if fig is not None:
                plt.close(fig)
            state["rendering"] = False
            refresh.disabled = False
            pending = state["pending_values"]
            state["pending_values"] = None
            if pending is not None:
                update(**pending)

    def update_from_controls(_: Any = None) -> None:
        """Evaluate exactly once using the current control values.

        ``widgets.interactive_output`` maintains a second, hidden Output widget
        and can deliver duplicate initial updates in some Jupyter frontends.
        Direct observers keep one visible output and one callback per change.
        """

        update(**{name: control.value for name, control in controls.items()})

    for control in controls.values():
        control.observe(update_from_controls, names="value")

    def refresh_view(_: Any = None) -> None:
        state["rendered_key"] = None
        update_from_controls()

    refresh.on_click(refresh_view)

    # Clean top control bar layout
    row1 = widgets.HBox(
        [*controls.values(), refresh],
        layout=widgets.Layout(margin="0 0 6px 0", flex_flow="row wrap"),
    )
    control_panel = widgets.VBox([row1], layout=widgets.Layout(
        background_color="#ffffff",
        padding="10px 14px",
        border="1px solid #e2e8f0",
        border_radius="8px",
        margin="0 0 10px 0"
    ))

    ui = widgets.VBox(
        [control_panel, status, figure_image],
        layout=widgets.Layout(overflow="visible", width="100%"),
    )

    def close() -> None:
        if state["closed"]:
            return
        state["closed"] = True
        for control in controls.values():
            control.unobserve(update_from_controls, names="value")
        refresh.on_click(refresh_view, remove=True)
        for widget in [*controls.values(), refresh, row1, control_panel, status, figure_image, ui]:
            widget.close()
        state.update(result=None, metrics=None, base_result=None, pending_values=None)

    state["close"] = close
    # Populate the dashboard once after all observers and visible containers
    # have been constructed. No hidden widget performs a second initial call.
    update_from_controls()
    return ui, state


def _get_or_generate_corridor(
    context: tmi.TransportContext,
    corridor_municipalities: list[str] | None = None,
    buffer_m: float = tmi.DEFAULT_CORRIDOR_BUFFER_M,
    max_gates: int = tmi.DEFAULT_MAX_GATES,
) -> tuple[Any, Any]:
    """Helper to extract the clipped corridor network and cordon gates."""
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    zones = context.zones.copy()
    if not corridor_municipalities:
        import parameters as p
        corridor_municipalities = getattr(p, "CORRIDOR_MUNICIPALITIES", None)
        if not corridor_municipalities and context.assignment_network and "metadata" in context.assignment_network:
            corridor_municipalities = context.assignment_network["metadata"].get("corridor_municipalities", [])
        if not corridor_municipalities:
            return context.assignment_network["edges"].copy(), None

    try:
        import parameters as p
        max_gates = getattr(p, "MAX_GATES", max_gates)
    except Exception:
        pass

    core_zones = tmi._get_core_zones(zones, corridor_municipalities)
    if core_zones.empty:
        return context.assignment_network["edges"].copy(), None

    polygon = core_zones.geometry.union_all().buffer(float(buffer_m))
    network_edges = context.assignment_network["edges"]
    network_nodes = context.assignment_network["nodes"]

    node_inside = network_nodes.geometry.intersects(polygon)
    inside_by_node = dict(zip(network_nodes["node_id"].astype(int), node_inside.astype(bool)))
    source_inside = network_edges["source"].map(inside_by_node).fillna(False).astype(bool)
    target_inside = network_edges["target"].map(inside_by_node).fillna(False).astype(bool)

    local_edges = network_edges.loc[source_inside & target_inside].copy()
    used_nodes = set(local_edges["source"].astype(int)) | set(local_edges["target"].astype(int))
    local_nodes = network_nodes.loc[network_nodes["node_id"].astype(int).isin(used_nodes)]

    crossing = network_edges.loc[source_inside ^ target_inside].copy()
    crossing["inside_node"] = np.where(source_inside.loc[crossing.index], crossing["source"], crossing["target"]).astype(int)
    crossing["can_exit"] = source_inside.loc[crossing.index].to_numpy(dtype=bool)
    crossing["can_enter"] = target_inside.loc[crossing.index].to_numpy(dtype=bool)
    crossing = crossing.loc[crossing["inside_node"].isin(used_nodes)]

    node_geometry = local_nodes.set_index("node_id").geometry
    candidate_rows = []
    for node_id, group in crossing.groupby("inside_node", sort=False):
        point = node_geometry.get(node_id)
        if point is None:
            continue
        candidate_rows.append({
            "node_id": int(node_id),
            "geometry": point,
            "crossing_count": int(len(group)),
            "can_enter": bool(group["can_enter"].any()),
            "can_exit": bool(group["can_exit"].any()),
        })

    # Robust fallback: If road network was pre-clipped and has no crossing edges,
    # detect entry/exit gates from perimeter boundary nodes of the clipped network
    if not candidate_rows and not local_edges.empty:
        deg = local_edges["source"].value_counts().add(local_edges["target"].value_counts(), fill_value=0)
        perimeter_ids = deg[deg <= 2].index.astype(int)
        perim_nodes = local_nodes.loc[local_nodes["node_id"].astype(int).isin(perimeter_ids)]
        for _, r in perim_nodes.iterrows():
            candidate_rows.append({
                "node_id": int(r["node_id"]),
                "geometry": r["geometry"],
                "crossing_count": 1,
                "can_enter": True,
                "can_exit": True,
            })

    if not candidate_rows:
        return local_edges, gpd.GeoDataFrame(columns=["gate_id", "node_id", "geometry", "can_enter", "can_exit"], crs=local_edges.crs)

    candidate_gates = gpd.GeoDataFrame(candidate_rows, crs=network_nodes.crs)
    candidate_gates = candidate_gates.sort_values(
        by=["crossing_count", "can_enter", "can_exit"],
        ascending=[False, False, False],
    )

    # Spatial deduplication: distribute gates around perimeter
    selected_gates = []
    min_dist = float(tmi.DEFAULT_GATE_SEPARATION_M)
    for _, row in candidate_gates.iterrows():
        if len(selected_gates) >= max_gates:
            break
        point = row["geometry"]
        if not selected_gates or all(point.distance(s["geometry"]) >= min_dist for s in selected_gates):
            selected_gates.append(row)

    if not selected_gates and not candidate_gates.empty:
        selected_gates = [row for _, row in candidate_gates.head(max_gates).iterrows()]

    gates = gpd.GeoDataFrame(selected_gates, crs=network_nodes.crs).reset_index(drop=True)
    gates["gate_id"] = [f"G_{i+1:02d}" for i in range(len(gates))]
    return local_edges, gates


_CONTINUOUS_PALETTE = np.asarray(
    [
        [44, 123, 182, 230],
        [102, 194, 165, 230],
        [255, 255, 191, 230],
        [253, 174, 97, 230],
        [215, 25, 28, 235],
    ],
    dtype=np.uint8,
)


_CATEGORY_PALETTE = np.asarray(
    [
        [76, 120, 168, 230],
        [245, 133, 24, 230],
        [84, 162, 75, 230],
        [228, 87, 86, 230],
        [178, 121, 162, 230],
        [114, 183, 178, 230],
        [255, 157, 167, 230],
        [156, 117, 95, 230],
        [186, 176, 172, 230],
        [237, 201, 72, 230],
    ],
    dtype=np.uint8,
)


def _numeric_style(
    values: pd.Series,
    *,
    breaks: list[float] | None = None,
    palette: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Return link colours and an HTML legend for one numeric metric."""
    full_palette = _CONTINUOUS_PALETTE if palette is None else np.asarray(palette, dtype=np.uint8)
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        colors = np.repeat([[160, 160, 160, 210]], len(values), axis=0).astype(np.uint8)
        return colors, "<i>No finite values.</i>"

    if breaks is None:
        breaks_array = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, len(full_palette) + 1)))
        if len(breaks_array) < 3:
            minimum, maximum = float(finite.min()), float(finite.max())
            if np.isclose(minimum, maximum):
                maximum = minimum + 1.0
            breaks_array = np.linspace(minimum, maximum, len(full_palette) + 1)
    else:
        breaks_array = np.asarray(breaks, dtype=float)
        if len(breaks_array) < 2:
            raise ValueError("Numeric map breaks must contain at least two values.")

    n_colors = min(len(full_palette), max(len(breaks_array) - 1, 1))
    selected_palette = full_palette[:n_colors]
    indices = np.digitize(numeric, breaks_array[1:-1], right=False)
    indices = np.clip(indices, 0, n_colors - 1)
    colors = selected_palette[indices].copy()
    colors[~np.isfinite(numeric)] = [160, 160, 160, 180]

    swatches = []
    for index in range(n_colors):
        low, high = breaks_array[index], breaks_array[index + 1]
        if breaks is not None and index == 0:
            interval = f"&lt; {high:,.2f}"
        elif breaks is not None and index == n_colors - 1:
            interval = f"≥ {low:,.2f}"
        else:
            interval = f"{low:,.2f}–{high:,.2f}"
        rgba = selected_palette[index]
        colour = f"rgb({rgba[0]},{rgba[1]},{rgba[2]})"
        swatches.append(
            "<span style='display:inline-block;margin-right:14px'>"
            f"<span style='display:inline-block;width:14px;height:10px;background:{colour};"
            f"margin-right:4px'></span>{interval}</span>"
        )
    return colors.astype(np.uint8), "".join(swatches)


def _categorical_style(values: pd.Series) -> tuple[np.ndarray, str]:
    """Return stable colours and an HTML legend for categorical values."""
    labels = values.fillna("missing").astype(str)
    order = labels.value_counts().index.tolist()
    if len(order) > len(_CATEGORY_PALETTE):
        retained = order[: len(_CATEGORY_PALETTE) - 1]
        labels = labels.where(labels.isin(retained), "other")
        order = retained + ["other"]
    colour_lookup = {
        label: _CATEGORY_PALETTE[position % len(_CATEGORY_PALETTE)]
        for position, label in enumerate(order)
    }
    colors = np.vstack([colour_lookup[label] for label in labels]).astype(np.uint8)
    swatches = []
    for label in order:
        rgba = colour_lookup[label]
        colour = f"rgb({rgba[0]},{rgba[1]},{rgba[2]})"
        swatches.append(
            "<span style='display:inline-block;margin-right:14px'>"
            f"<span style='display:inline-block;width:14px;height:10px;background:{colour};"
            f"margin-right:4px'></span>{label}</span>"
        )
    return colors, "".join(swatches)


def _metric_colours(frame: pd.DataFrame, spec: Mapping[str, Any]) -> tuple[np.ndarray, str]:
    """Dispatch map styling for a metric specification."""
    column = str(spec["column"])
    if column not in frame:
        raise KeyError(f"Map metric column not found: {column}")
    if spec.get("kind", "numeric") == "categorical":
        return _categorical_style(frame[column])
    return _numeric_style(frame[column], breaks=spec.get("breaks"), palette=spec.get("palette"))


def _slim_geodataframe(frame: Any, columns: list[str], *, simplify_m: float = 0.0) -> Any:
    """Keep browser-facing fields only, simplify a copy, and reproject to WGS84."""
    keep = [column for column in columns if column in frame.columns]
    if "geometry" not in keep:
        keep.append("geometry")
    result = frame[keep].copy()
    if result.crs is None:
        raise ValueError("Map data need a declared CRS.")
    if simplify_m > 0 and not result.crs.is_geographic:
        result.geometry = result.geometry.simplify(float(simplify_m), preserve_topology=True)
    return result.to_crs(4326)


def _prepare_zone_boundaries(
    zones: Any,
    *,
    zone_ids: list[str] | set[str] | None = None,
    level: str = "FSM zone",
    simplify_m: float = 25.0,
) -> Any:
    """Return lightweight WGS84 zone or municipality polygons for map overlays."""

    import geopandas as gpd

    if zones is None or len(zones) == 0:
        return gpd.GeoDataFrame(columns=["area", "geometry"], geometry="geometry", crs=4326)
    if zones.crs is None:
        raise ValueError("Zone geometries need a declared CRS.")

    selected = zones.copy()
    if zone_ids is not None:
        allowed = set(map(str, zone_ids))
        selected = selected.loc[selected["grid_id"].astype(str).isin(allowed)].copy()
    selected = selected.dropna(subset=["geometry"])
    selected = selected.loc[~selected.geometry.is_empty].copy()

    if level == "Municipality":
        selected["area"] = selected["municipality_name"].fillna("Unknown").astype(str)
        selected = selected[["area", "geometry"]].dissolve(by="area").reset_index()
    else:
        selected["area"] = selected["grid_id"].astype(str)
        selected = selected[["area", "geometry"]]

    if simplify_m > 0 and not selected.crs.is_geographic:
        selected.geometry = selected.geometry.simplify(
            float(simplify_m), preserve_topology=True
        )
    return selected.to_crs(4326).reset_index(drop=True)


def _plotly_zone_boundary_trace(
    boundaries: Any,
    *,
    visible: bool | str = "legendonly",
    name: str = "Zone boundaries",
) -> Any | None:
    """Build one low-overhead Plotly map trace containing polygon outlines."""

    if boundaries is None or len(boundaries) == 0:
        return None
    import plotly.graph_objects as go

    lon: list[float | None] = []
    lat: list[float | None] = []
    for geometry in boundaries.geometry:
        polygons = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
        for polygon in polygons:
            if not hasattr(polygon, "exterior"):
                continue
            x, y = polygon.exterior.xy
            lon.extend([*map(float, x), None])
            lat.extend([*map(float, y), None])
    if not lon:
        return None
    return go.Scattermap(
        lon=lon,
        lat=lat,
        mode="lines",
        line={"color": "rgba(70,70,70,0.55)", "width": 1},
        hoverinfo="skip",
        name=name,
        visible=visible,
        showlegend=True,
    )


def _lonboard_map(
    layers: list[Any],
    *,
    height: int = 620,
    show_side_panel: bool = True,
) -> Any:
    """Construct a clickable Lonboard map across supported API versions."""
    from lonboard import Map

    kwargs = {
        "show_tooltip": True,
        "show_side_panel": bool(show_side_panel),
        "picking_radius": 5,
    }
    try:
        return Map(layers, height=height, **kwargs)
    except TypeError:
        result = Map(layers, **kwargs)
        if hasattr(result, "_height"):
            result._height = int(height)
        try:
            result.layout.height = f"{int(height)}px"
        except Exception:
            pass
        return result


def network_explorer(
    edges: Any,
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    nodes: Any | None = None,
    gates: Any | None = None,
    polygon: Any | None = None,
    zone_polygons: Any | None = None,
    boundary_level: str = "FSM zone",
    show_zone_boundaries: bool = False,
    width_column: str | None = None,
    title: str = "Network explorer",
    simplify_m: float = 3.0,
    height: int = 620,
) -> Any:
    """Create one clickable, metric-selectable network widget for notebooks.

    Only lightweight WGS84 copies are sent to the browser. Model data remain
    in their projected CRS and are never modified. Link attributes appear on
    hover or, after a click, in Lonboard's persistent side panel.
    """
    import ipywidgets as widgets
    from lonboard import PathLayer, PolygonLayer, ScatterplotLayer, SolidPolygonLayer

    if not metrics:
        raise ValueError("At least one map metric must be supplied.")
    if edges is None or len(edges) == 0:
        raise ValueError("The network explorer received no road links.")

    metric_columns = [str(spec["column"]) for spec in metrics.values()]
    popup_columns = [
        "edge_id", "source", "target", "source_id", "target_id", "highway",
        "drive_area", "length_m", "speed_kph", "lanes", "capacity_vph",
        "free_flow_time_min", "flow_vehicles", "volume_capacity_ratio",
        "time_min", "delay_min", "assigned_speed_kph", "stage2_lane_converted",
        *metric_columns,
    ]
    links_map = _slim_geodataframe(
        edges, list(dict.fromkeys(popup_columns)), simplify_m=simplify_m
    ).dropna(subset=["geometry"])
    links_map = links_map.loc[~links_map.geometry.is_empty].reset_index(drop=True)
    if links_map.empty:
        raise ValueError("The network explorer received no valid link geometry.")

    first_label = next(iter(metrics))
    link_colours, legend_html = _metric_colours(links_map, metrics[first_label])
    if width_column and width_column in links_map:
        width_values = pd.to_numeric(links_map[width_column], errors="coerce").fillna(0.0)
        upper = max(float(width_values.quantile(0.98)), 1e-9)
        link_width = 1.0 + 5.0 * np.sqrt(np.clip(width_values / upper, 0.0, 1.0))
    else:
        link_width = np.full(len(links_map), 1.6)

    link_layer = PathLayer.from_geopandas(
        links_map,
        get_color=link_colours,
        get_width=np.asarray(link_width, dtype=np.float32),
        width_units="pixels",
        width_min_pixels=1,
        width_max_pixels=7,
        auto_highlight=True,
        # pyrefly: ignore [unexpected-keyword]
        highlight_color=[20, 20, 20, 180],
        pickable=True,
    )

    layers: list[Any] = []
    if polygon is not None and len(polygon):
        polygon_map = _slim_geodataframe(polygon, ["name"], simplify_m=5.0)
        layers.append(
            SolidPolygonLayer.from_geopandas(
                polygon_map,
                get_fill_color=[35, 120, 180, 25],
                get_line_color=[35, 120, 180, 180],
                filled=True,
                pickable=True,
            )
        )
    zone_boundary_layer = None
    if zone_polygons is not None and len(zone_polygons):
        zone_boundary_map = _prepare_zone_boundaries(
            zone_polygons, level=boundary_level, simplify_m=max(10.0, simplify_m)
        )
        if len(zone_boundary_map):
            zone_boundary_layer = PolygonLayer.from_geopandas(
                zone_boundary_map,
                get_fill_color=[0, 0, 0, 0],
                get_line_color=[70, 70, 70, 135],
                filled=False,
                stroked=True,
                line_width_min_pixels=1,
                pickable=False,
            )
            zone_boundary_layer.visible = bool(show_zone_boundaries)
            layers.append(zone_boundary_layer)
    layers.append(link_layer)

    node_layer = None
    if nodes is not None and len(nodes):
        node_columns = ["node_id", "nodeID", "zone_id", "is_zone", "x", "y", "geometry"]
        nodes_map = _slim_geodataframe(nodes, node_columns).reset_index(drop=True)
        node_layer = ScatterplotLayer.from_geopandas(
            nodes_map,
            get_fill_color=[48, 92, 160, 180],
            get_line_color=[255, 255, 255, 230],
            get_radius=3.0,
            radius_units="pixels",
            radius_min_pixels=2,
            radius_max_pixels=6,
            stroked=True,
            auto_highlight=True,
            pickable=True,
        )
        layers.append(node_layer)

    gate_layer = None
    if gates is not None and len(gates):
        gate_columns = [
            "gate_id", "node_id", "direction", "can_enter", "can_exit",
            "capacity_vph", "highway", "geometry",
        ]
        gates_map = _slim_geodataframe(gates, gate_columns).reset_index(drop=True)
        gate_layer = ScatterplotLayer.from_geopandas(
            gates_map,
            get_fill_color=[128, 0, 128, 235],
            get_line_color=[255, 255, 255, 255],
            get_radius=7.0,
            radius_units="pixels",
            radius_min_pixels=5,
            radius_max_pixels=11,
            line_width_min_pixels=2,
            stroked=True,
            auto_highlight=True,
            pickable=True,
        )
        layers.append(gate_layer)

    map_widget = _lonboard_map(layers, height=height)
    selector = widgets.Dropdown(
        options=list(metrics), value=first_label, description="Colour:",
        layout=widgets.Layout(width="420px"),
    )
    legend = widgets.HTML(
        value=f"<b>{first_label}</b><br>{legend_html}",
        layout=widgets.Layout(width="100%"),
    )
    toggles = []
    if zone_boundary_layer is not None:
        boundary_toggle = widgets.Checkbox(
            value=bool(show_zone_boundaries), description="Show zone boundaries"
        )
        boundary_toggle.observe(
            lambda change: setattr(zone_boundary_layer, "visible", bool(change["new"])),
            names="value",
        )
        toggles.append(boundary_toggle)
    if node_layer is not None:
        node_toggle = widgets.Checkbox(value=True, description="Show nodes/centroids")
        node_toggle.observe(
            lambda change: setattr(node_layer, "visible", bool(change["new"])),
            names="value",
        )
        toggles.append(node_toggle)
    if gate_layer is not None:
        gate_toggle = widgets.Checkbox(value=True, description="Show gates")
        gate_toggle.observe(
            lambda change: setattr(gate_layer, "visible", bool(change["new"])),
            names="value",
        )
        toggles.append(gate_toggle)

    def recolour(change: dict[str, Any]) -> None:
        label = change["new"]
        colours, html = _metric_colours(links_map, metrics[label])
        link_layer.get_color = colours
        legend.value = f"<b>{label}</b><br>{html}"

    selector.observe(recolour, names="value")
    controls = widgets.HBox([selector, *toggles])
    help_text = widgets.HTML(
        "<span style='color:#555'>Hover for a tooltip; click a segment or "
        "point to keep its attributes in the map side panel.</span>"
    )
    heading = widgets.HTML(f"<h4 style='margin:4px 0'>{title}</h4>")
    return widgets.VBox([heading, controls, legend, help_text, map_widget])


def _network_characteristic_metrics() -> dict[str, dict[str, Any]]:
    """Metric catalogue shared by full-network and corridor maps."""
    return {
        "Road class": {"column": "highway", "kind": "categorical"},
        "Free-flow speed (km/h)": {"column": "speed_kph"},
        "Directional capacity (veh/h)": {"column": "capacity_vph"},
        "Lanes": {"column": "lanes"},
        "Link length (m)": {"column": "length_m"},
        "Free-flow time (min)": {"column": "free_flow_time_min"},
    }


def corridor_explorer(corridor: tmi.CorridorContext) -> Any:
    """Show retained links, internal zone nodes, cordon gates, and attributes."""
    zone_nodes = corridor.nodes
    if "is_zone" in zone_nodes:
        zone_nodes = zone_nodes.loc[zone_nodes["is_zone"].astype(bool)].copy()
    return network_explorer(
        corridor.edges,
        _network_characteristic_metrics(),
        nodes=zone_nodes,
        gates=corridor.gates,
        polygon=corridor.polygon_gdf,
        zone_polygons=corridor.zones,
        title="MehrSpur topology and cordon gates (no assignment)",
        simplify_m=2.0,
        height=650,
    )


def corridor_network_explorer(
    context: tmi.TransportContext,
    corridor_municipalities: list[str] | None = None,
) -> Any:
    """Backward-compatible shortcut for the richer corridor explorer."""
    try:
        corridor = tmi.build_corridor_context(
            context,
            corridor_municipalities=corridor_municipalities,
            name="MehrSpur",
        )
        return corridor_explorer(corridor)
    except ImportError:
        print("Lonboard or ipywidgets is not installed; interactive map skipped.")
        return None


def full_network_explorer(
    context: tmi.TransportContext,
    corridor_municipalities: list[str] | None = None,
) -> Any:
    """Clickable metric explorer for the full network or a selected corridor."""
    if corridor_municipalities is not None:
        return corridor_network_explorer(context, corridor_municipalities)
    if context.assignment_network is None or "edges" not in context.assignment_network:
        print("No assignment network loaded in context.")
        return None
    try:
        nodes = context.assignment_network["nodes"]
        # Drawing every intersection marker on a large graph is costly and
        # obscures the roads, so the canton-wide view keeps zone nodes only.
        if "is_zone" in nodes and len(nodes) > 20_000:
            nodes = nodes.loc[nodes["is_zone"].astype(bool)].copy()
        return network_explorer(
            context.assignment_network["edges"],
            _network_characteristic_metrics(),
            nodes=nodes,
            zone_polygons=context.zones,
            title="Canton Zurich motor-vehicle network - characteristics only (no assignment)",
            simplify_m=4.0,
            height=650,
        )
    except ImportError:
        print("Lonboard or ipywidgets is not installed; interactive map skipped.")
        return None


def corridor_detail_table(edges: pd.DataFrame) -> pd.DataFrame:
    """Summarise retained link count and length by road class."""
    if edges is None or len(edges) == 0:
        return pd.DataFrame(
            columns=["highway", "edges", "total_length_km", "mean_length_m", "share_%"]
        )
    frame = edges.copy()
    highway_values = frame.get("highway", pd.Series("unknown", index=frame.index))
    length_values = frame.get("length_m", pd.Series(0.0, index=frame.index))
    frame["highway"] = highway_values.fillna("unknown").astype(str)
    frame["length_m"] = pd.to_numeric(length_values, errors="coerce").fillna(0.0)
    table = (
        frame.groupby("highway", dropna=False)
        .agg(
            edges=("length_m", "size"),
            total_length_km=("length_m", "sum"),
            mean_length_m=("length_m", "mean"),
        )
        .sort_values("edges", ascending=False)
    )
    table["total_length_km"] = (table["total_length_km"] / 1000.0).round(1)
    table["mean_length_m"] = table["mean_length_m"].round(0)
    table["share_%"] = (100.0 * table["edges"] / table["edges"].sum()).round(1)
    return table.reset_index()


def corridor_dashboard(
    context: tmi.TransportContext,
    corridor: tmi.CorridorContext,
    *,
    demand_matrix: pd.DataFrame | None = None,
    passthrough_fraction: float = tmi.DEFAULT_PASSTHROUGH_FRACTION,
) -> Any:
    """Package the corridor map and audit tables into one tabbed widget.

    This keeps notebook cells short while leaving every generated table
    available through the standalone functions above for later calculations.
    """
    import ipywidgets as widgets
    from IPython.display import display

    def table_output(value: Any) -> Any:
        """Render a table as static HTML.

        A ``widgets.Output`` holding a one-shot ``display()`` call can replay
        its buffered messages when a Tab view reconnects (e.g. switching
        tabs), duplicating the table. A single-trait HTML widget cannot.
        """
        table_html = value.to_html() if hasattr(value, "to_html") else pd.DataFrame(value).to_html()
        return widgets.HTML(
            value=f"<div style='overflow-x:auto'>{table_html}</div>",
            layout=widgets.Layout(width="100%"),
        )

    gate_columns = [
        column for column in
        ["gate_id", "node_id", "direction", "highway", "capacity_vph"]
        if column in corridor.gates
    ]
    metadata = pd.Series(corridor.metadata, name="value").to_frame()
    gates = corridor.gates[gate_columns]
    road_detail = corridor_detail_table(corridor.edges)
    tables = {"overview": metadata, "gates": gates, "road_detail": road_detail}
    overview = widgets.VBox(
        [
            table_output(metadata),
            table_output(gates),
        ]
    )
    children = [corridor_explorer(corridor), overview, table_output(road_detail)]
    titles = ["Interactive map", "Overview and gates", "Road detail"]

    selected_demand = context.baseline_od if demand_matrix is None else demand_matrix
    if isinstance(selected_demand, pd.DataFrame) and not selected_demand.empty:
        reduced, breakdown = tmi.collapse_od_to_gates(
            selected_demand,
            context,
            corridor,
            passthrough_fraction=passthrough_fraction,
        )
        gate_demand = tmi.gate_totals(reduced, corridor)
        tables.update(demand_breakdown=breakdown, gate_demand=gate_demand)
        children.append(widgets.VBox([table_output(breakdown), table_output(gate_demand)]))
        titles.append("Cordon demand")

    tabs = widgets.Tab(children=children)
    for index, title in enumerate(titles):
        tabs.set_title(index, title)
    exports.save_outputs("1_3_corridor", tables=tables)
    return tabs


def assignment_explorer(
    assignment: tmi.AssignmentResult | Any,
    corridor: tmi.CorridorContext | None = None,
) -> Any:
    """Interactive assignment map with selectable link-performance metrics.

    Passing a bare link GeoDataFrame is retained for older notebooks.  A full
    ``AssignmentResult`` together with its corridor additionally displays the
    modelling cordon and direction-aware gates.
    """

    links = assignment.links if hasattr(assignment, "diagnostics") and hasattr(assignment, "links") else assignment
    if links is None or len(links) == 0:
        raise ValueError("No assigned road links are available to display.")

    metrics = {
        "Assigned load (veh/h)": {"column": "flow_vehicles"},
        "Volume / capacity": {
            "column": "volume_capacity_ratio",
            "breaks": [0.0, 0.50, 0.80, 1.00, 1.20, 2.50],
        },
        "Congested time (min)": {"column": "time_min"},
        "Delay (min)": {"column": "delay_min"},
        "Assigned speed (km/h)": {"column": "assigned_speed_kph"},
        "Directional capacity (veh/h)": {"column": "capacity_vph"},
    }
    return network_explorer(
        links,
        metrics,
        gates=corridor.gates if corridor is not None else None,
        polygon=corridor.polygon_gdf if corridor is not None else None,
        zone_polygons=corridor.zones if corridor is not None else None,
        width_column="flow_vehicles",
        title="Corridor road assignment - link width represents assigned flow",
        simplify_m=1.0,
        height=650,
    )


def static_network_plot(
    context: tmi.TransportContext,
    mode_result: Any = None,
    corridor_municipalities: list[str] | None = None,
    project_name: str = "Corridor",
    assigned_edges: Any = None,
) -> Any:
    """Render a publication-ready static Matplotlib map of the assigned corridor network."""
    import matplotlib.pyplot as plt
    import geopandas as gpd

    # Extract edges to plot flexibly
    edges_plot = None
    gates_plot = None

    if assigned_edges is not None and isinstance(assigned_edges, gpd.GeoDataFrame):
        edges_plot = assigned_edges.copy()
        gates_plot = assigned_edges.attrs.get("metadata", {}).get("gates", None)
    elif mode_result is not None and isinstance(mode_result, gpd.GeoDataFrame):
        edges_plot = mode_result.copy()
        gates_plot = mode_result.attrs.get("metadata", {}).get("gates", None)
    elif mode_result is not None and hasattr(mode_result, "assigned_edges") and mode_result.assigned_edges is not None:
        edges_plot = mode_result.assigned_edges.copy()
        if hasattr(mode_result, "assigned_metadata") and mode_result.assigned_metadata is not None:
            gates_plot = mode_result.assigned_metadata.get("gates", None)
    else:
        edges_plot, gates_plot = _get_or_generate_corridor(context, corridor_municipalities)

    # Fallback to ensure gates are ALWAYS available for plotting
    if gates_plot is None or (isinstance(gates_plot, gpd.GeoDataFrame) and gates_plot.empty):
        _, gates_plot = _get_or_generate_corridor(context, corridor_municipalities)

    if edges_plot is None or len(edges_plot) == 0:
        print("No assigned edges available to plot.")
        return None

    munis = corridor_municipalities or []
    zones_plot = tmi._get_core_zones(context.zones, munis) if munis else gpd.GeoDataFrame()

    road_colours = {
        "motorway": "#d73027", "motorway_link": "#fc8d59",
        "trunk": "#f46d43", "trunk_link": "#fdae61",
        "primary": "#fee08b", "primary_link": "#fff2a8",
        "secondary": "#91bfdb", "secondary_link": "#abd9e9",
        "tertiary": "#74add1", "tertiary_link": "#a6cee3",
        "residential": "#969696", "unclassified": "#bdbdbd",
        "living_street": "#d9d9d9",
    }

    fig, ax = plt.subplots(figsize=(14, 8))

    if not zones_plot.empty:
        zones_plot.plot(
            ax=ax, facecolor="#3182bd", edgecolor="#08519c",
            linewidth=1.8, alpha=0.12, zorder=1,
        )

    if "highway" in edges_plot.columns:
        hw_col = edges_plot["highway"].fillna("unclassified").astype(str)
    else:
        hw_col = pd.Series("unclassified", index=edges_plot.index)
    plot_df = edges_plot.assign(_hw_group=hw_col)

    for highway, group in plot_df.groupby("_hw_group"):
        highway = str(highway).lower()
        colour = road_colours.get(highway, "#bdbdbd")
        linewidth = 1.8 if highway in {"motorway", "trunk", "primary"} else 0.7
        group.plot(ax=ax, color=colour, linewidth=linewidth, alpha=0.85, label=highway, zorder=3)

    if gates_plot is not None and not gates_plot.empty:
        gates_plot.plot(
            ax=ax, color="#7a0177", edgecolor="white", linewidth=1.2,
            markersize=80, zorder=5, label="Cordon gates",
        )
        for gate in gates_plot.itertuples(index=False):
            ax.annotate(
                gate.gate_id, xy=(gate.geometry.x, gate.geometry.y),
                xytext=(4, 4), textcoords="offset points",
                fontsize=8.5, fontweight="bold", color="#4a004a", zorder=6,
            )

    ax.set_title(f"Detailed {project_name} Road Network & Cordon Gates", fontsize=14, fontweight="bold")
    ax.set_xlabel("Swiss projected coordinate — Easting (m)")
    ax.set_ylabel("Swiss projected coordinate — Northing (m)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        unique_legend = dict(zip(labels, handles))
        ax.legend(
            unique_legend.values(), unique_legend.keys(),
            title="Road class", loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9,
        )
    plt.tight_layout()
    exports.save_outputs("1_3_baseline_network", figure=fig)
    plt.show()
    plt.close(fig)
    return None


def assignment_dashboard(
    context: tmi.TransportContext,
    corridor: tmi.CorridorContext,
    mode_state: dict[str, Any],
    *,
    modal_feedback: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Explicit-button MSA dashboard with cached settings combinations.

    ``modal_feedback=False`` holds the car OD matrix fixed, which isolates the
    assignment algorithm.  ``modal_feedback=True`` recomputes modal split from
    the OD-specific congested drive skim within the same MSA iteration.
    """

    from io import BytesIO

    import ipywidgets as widgets
    import matplotlib.pyplot as plt

    state: dict[str, Any] = {
        "result": None,
        "cache": {},
        "max_cached_runs": 3,
        "context": context,
        "corridor": corridor,
        "mode_cache_key": None,
        "running": False,
        "rendered_key": None,
        "run_count": 0,
    }
    run_button = widgets.Button(
        description=(
            "Run coupled MSA + mode choice"
            if modal_feedback
            else "Run corridor MSA assignment"
        ),
        button_style="primary",
        icon="play",
    )
    road_gap = widgets.Dropdown(
        options=[
            ("1% (stricter)", 0.01),
            ("2% (default)", 0.02),
            ("5% (coarser)", 0.05),
        ],
        value=float(tmi.ASSIGNMENT_SETTINGS["road_gap_threshold"]),
        description="Road gap:",
        style={"description_width": "95px"},
    )
    max_iterations = widgets.IntSlider(
        value=int(tmi.ASSIGNMENT_SETTINGS["max_iterations"]),
        min=int(tmi.ASSIGNMENT_SETTINGS["min_iterations"]),
        max=max(128, int(tmi.ASSIGNMENT_SETTINGS["max_iterations"])),
        step=1,
        description="Max iter.:",
        continuous_update=False,
    )
    od_threshold = widgets.Dropdown(
        options=[("0 (all flows, default)", 0.0), ("0.1 veh/h", 0.1), ("1.0 veh/h", 1.0)],
        value=float(tmi.ASSIGNMENT_SETTINGS["od_threshold"]),
        description="OD cutoff:",
    )
    passthrough = widgets.FloatSlider(
        value=tmi.DEFAULT_PASSTHROUGH_FRACTION,
        min=0.0,
        max=0.25,
        step=0.01,
        description="Pass-through:",
        continuous_update=False,
        readout_format=".0%",
    )
    status = widgets.HTML(
        "<i>Assignment has not been run. Adjust the controls and click the button.</i>"
    )
    results = widgets.VBox()
    state["controls"] = {"road_gap_threshold": road_gap, "max_iterations": max_iterations,
                         "od_threshold": od_threshold, "passthrough_fraction": passthrough}

    def table_panel(*sections: tuple[str, Any]) -> Any:
        """Create ordinary HTML tables without Jupyter rich-output capture.

        DataFrame ``display()`` messages inside an Output widget may be replayed
        whenever a tab view reconnects. Static HTML widgets have one trait
        value each, so every table can exist only once in the dashboard.
        """

        children = []
        for title, value in sections:
            if hasattr(value, "to_html"):
                table_html = value.to_html()
            else:
                table_html = pd.DataFrame(value).to_html()
            children.append(widgets.HTML(
                value=(
                    f"<h4 style='margin:8px 0 5px'>{title}</h4>"
                    f"<div style='overflow-x:auto'>{table_html}</div>"
                ),
                layout=widgets.Layout(width="100%"),
            ))
        return widgets.VBox(children)

    export_name = "1_6_assignment_coupled" if modal_feedback else "1_6_assignment_fixed"

    def figure_image(figure: Any, suffix: str) -> Any:
        """Render one Matplotlib figure into one replaceable widget value."""

        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
        plt.close(figure)
        exports.save_outputs(f"{export_name}_{suffix}", png=buffer.getvalue())
        return widgets.Image(
            value=buffer.getvalue(), format="png",
            layout=widgets.Layout(width="100%", max_width="1150px"),
        )

    def render_result(result: tmi.AssignmentResult, threshold: float) -> Any:
        """Build a new tab set for one completed assignment calculation."""

        diagnostic_table = pd.DataFrame({
            "diagnostic": list(result.diagnostics),
            "value": [
                value if not isinstance(value, (dict, list)) else str(value)
                for value in result.diagnostics.values()
            ],
        })
        overview = table_panel(("Assignment diagnostics", diagnostic_table))
        demand_values: list[tuple[str, Any]] = [
            ("Demand retained by cordon category", result.demand_breakdown),
            ("Demand by cordon gate", result.gate_totals),
        ]
        if result.mode_result is not None:
            demand_values.append((
                "Updated modal split",
                result.mode_result.summary.style.format({
                    "trips": "{:,.1f}", "share": "{:.1%}",
                }),
            ))
        demand = table_panel(*demand_values)

        convergence_children: list[Any] = []
        if len(result.history):
            figure, axis = plt.subplots(figsize=(7.2, 3.2))
            checked = result.history.dropna(subset=["road_relative_gap_feasible_averaged_od"])
            axis.plot(
                checked["iteration"],
                checked["road_relative_gap_feasible_averaged_od"],
                marker="o", label="road relative gap",
            )
            axis.axhline(
                float(threshold), color="crimson", linestyle="--",
                label="threshold",
            )
            axis.set_yscale("symlog", linthresh=0.001)
            axis.set_xlabel("MSA iteration")
            axis.set_ylabel("Road relative gap (fraction)")
            axis.set_title("Road assignment convergence")
            axis.legend()
            figure.tight_layout()
            convergence_children.append(figure_image(figure, "convergence"))

            if result.mode_result is not None:
                figure, axes = plt.subplots(1, 2, figsize=(10.8, 3.2))
                for column, label in (
                    ("car_share", "Car"),
                    ("pt_share", "Public transport"),
                    ("bike_share", "Bicycle"),
                    ("walk_share", "Walking"),
                ):
                    axes[0].plot(
                        result.history["iteration"],
                        100.0 * result.history[column],
                        marker="o", label=label,
                    )
                axes[0].set_xlabel("MSA iteration")
                axes[0].set_ylabel("Corridor trip share (%)")
                axes[0].set_title("Modal response to congestion")
                axes[0].legend(fontsize=8)
                axes[1].plot(
                    result.history["iteration"],
                    result.history["demand_weighted_corridor_delay_min"],
                    color="crimson", marker="o",
                )
                axes[1].set_xlabel("MSA iteration")
                axes[1].set_ylabel("Minutes per trip")
                axes[1].set_title("Demand-weighted local road delay")
                figure.tight_layout()
                convergence_children.append(figure_image(figure, "modal_response"))
        if not convergence_children:
            convergence_children.append(widgets.HTML("<i>No iteration history is available.</i>"))

        tabs = widgets.Tab(children=[
            overview,
            demand,
            widgets.VBox(convergence_children),
            assignment_explorer(result, corridor),
        ])
        for index, title in enumerate((
            "Diagnostics", "Demand", "Convergence", "Assignment map"
        )):
            tabs.set_title(index, title)
        tables = {
            "diagnostics": diagnostic_table,
            "demand_breakdown": result.demand_breakdown,
            "gate_totals": result.gate_totals,
            "history": result.history,
            "settings": pd.Series({
                "modal_feedback": bool(modal_feedback),
                **{name: control.value for name, control in state["controls"].items()},
                **dict(mode_state["result"].scenario),
            }, name="value"),
        }
        if result.mode_result is not None:
            tables["modal_split"] = result.mode_result.summary
        exports.save_outputs(export_name, tables=tables)
        return tabs

    def run(_: Any) -> None:
        # Ignore a second click while the first calculation is still running.
        if state["running"]:
            return
        state["running"] = True
        run_button.disabled = True
        try:
            selected_mode_result = mode_state.get("result")
            if selected_mode_result is None:
                status.value = (
                    "<span style='color:#b71c1c'><b>Assignment not started:</b> "
                    "run the mode-choice dashboard first.</span>"
                )
                return
            key = (
                selected_mode_result.cache_key,
                bool(modal_feedback),
                float(road_gap.value),
                int(max_iterations.value),
                float(od_threshold.value),
                float(passthrough.value),
            )
            # Ignore duplicate button messages for a result that is already
            # visible. A changed control creates a different key and reruns.
            if state["rendered_key"] == key and state["result"] is not None:
                status.value = "<i>The current assignment result is already displayed.</i>"
                return
            status.value = "<b>Running MSA on the reduced corridor network...</b>"
            if key not in state["cache"]:
                runner = (
                    tmi.run_coupled_corridor_assignment
                    if modal_feedback
                    else tmi.run_corridor_assignment
                )
                state["cache"][key] = runner(
                    context,
                    corridor,
                    selected_mode_result,
                    passthrough_fraction=float(passthrough.value),
                    max_iterations=int(max_iterations.value),
                    min_iterations=int(tmi.ASSIGNMENT_SETTINGS["min_iterations"]),
                    stopping_rule="road_relative_gap",
                    road_gap_threshold=float(road_gap.value),
                    check_every=int(tmi.ASSIGNMENT_SETTINGS["check_every"]),
                    od_threshold=float(od_threshold.value),
                )
                while len(state["cache"]) > int(state["max_cached_runs"]):
                    state["cache"].pop(next(iter(state["cache"])))
            result = state["cache"][key]
            state["result"] = result
            state["mode_cache_key"] = tuple(selected_mode_result.cache_key)
            results.children = [render_result(result, float(road_gap.value))]
            state["rendered_key"] = key
            state["run_count"] += 1
            final_gap = float(result.diagnostics["final_road_relative_gap_feasible_averaged_od"])
            stopping_note = "target reached" if result.diagnostics["stopping_rule_met"] else "iteration limit reached"
            status.value = (
                f"<span style='color:#1b5e20'><b>Assignment complete.</b> "
                f"{len(result.history)} MSA iteration(s); road gap {final_gap:.2%}, {stopping_note}.</span>"
            )
        except Exception as error:
            state["rendered_key"] = None
            status.value = (
                "<span style='color:#b71c1c'><b>Assignment failed:</b> "
                f"<code>{type(error).__name__}: {error}</code></span>"
            )
        finally:
            state["running"] = False
            run_button.disabled = False

    run_button.on_click(run)
    controls = widgets.VBox(
        [road_gap, max_iterations, od_threshold, passthrough, run_button]
    )
    note = widgets.HTML(
        (
            "<b>Coupled calculation:</b> congestion updates OD-specific car "
            "travel times and modal split during MSA."
            if modal_feedback
            else "<b>Fixed-demand diagnostic:</b> MSA updates routes and link "
            "times while the selected car OD matrix remains fixed."
        )
    )
    return widgets.VBox([note, controls, status, results]), state


def plot_assignment_convergence_test(
    test_result: Mapping[str, pd.DataFrame],
    *,
    thresholds: tuple[float, ...] = (0.10, 0.05, 0.02),
) -> Any:
    """Plot median/IQR MSA convergence and threshold-reach frequencies."""

    import matplotlib.pyplot as plt

    summary = test_result["summary"]
    reached = test_result["threshold_reach"]
    trace = summary

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].plot(trace["iteration"], trace["median"], marker="o", label="median")
    axes[0].fill_between(
        trace["iteration"].to_numpy(dtype=float),
        trace["q25"].to_numpy(dtype=float),
        trace["q75"].to_numpy(dtype=float),
        alpha=0.25,
        label="interquartile range",
    )
    for threshold in thresholds:
        axes[0].axhline(float(threshold), linestyle="--", linewidth=1, label=f"{threshold:.0%}")
    axes[0].set_yscale("symlog", linthresh=0.001)
    axes[0].set_xlabel("MSA iteration")
    axes[0].set_ylabel("Road relative gap (fraction)")
    axes[0].set_title("Convergence across perturbed demand runs")
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].annotate(
        "Runs stop at the configured gap; later checks summarize\n"
        "only the runs still executing (see observed_runs in the table).",
        xy=(0, -0.32), xycoords="axes fraction", fontsize=7.5, color="#555555",
    )

    rates = reached.groupby("threshold")["reached"].mean().reindex(thresholds)
    axes[1].bar([f"{value:.0%}" for value in rates.index], rates.values * 100.0)
    axes[1].set_ylim(0, 100)
    axes[1].set_xlabel("Road relative-gap threshold")
    axes[1].set_ylabel("Runs reaching threshold (%)")
    axes[1].set_title("Threshold reached within iteration limit")
    fig.tight_layout()
    # Close so matplotlib's inline backend does not auto-display this figure
    # a second time at the end of the cell; the caller's own display(fig)
    # call still renders the (already-drawn) figure once.
    plt.close(fig)
    return fig


SPECTRUM_STOPS = [
    (0.0, (0x21, 0x66, 0xac)),   # blue  (lowest bin)
    (0.5, (0x1a, 0x98, 0x50)),   # green (middle bin)
    (1.0, (0xd7, 0x30, 0x27)),   # red   (highest bin)
]


def _spectrum_color(t: float) -> str:
    t = min(max(t, 0.0), 1.0)
    for (t0, c0), (t1, c1) in zip(SPECTRUM_STOPS, SPECTRUM_STOPS[1:]):
        if t0 <= t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            r, g, b = (round(c0[i] + f * (c1[i] - c0[i])) for i in range(3))
            return f"rgb({r},{g},{b})"
    r, g, b = SPECTRUM_STOPS[-1][1]
    return f"rgb({r},{g},{b})"


def _bearing_deg(lon0: float, lat0: float, lon1: float, lat1: float) -> float:
    """Initial compass bearing (degrees clockwise from north), for marker.angle."""
    import math
    phi0, phi1 = math.radians(lat0), math.radians(lat1)
    dlon = math.radians(lon1 - lon0)
    x = math.sin(dlon) * math.cos(phi1)
    y = math.cos(phi0) * math.sin(phi1) - math.sin(phi0) * math.cos(phi1) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


SEQUENTIAL_BLUE = [
    [0.00, "#cde2fb"], [0.15, "#9ec5f4"], [0.30, "#6da7ec"],
    [0.45, "#3987e5"], [0.60, "#256abf"], [0.75, "#184f95"],
    [1.00, "#0d366b"],
]


def _ordered_cordon_matrix(
    matrix: pd.DataFrame,
    corridor: tmi.CorridorContext,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Validate and order a reduced OD matrix as internal zones then gates."""
    if not isinstance(matrix, pd.DataFrame) or matrix.empty:
        raise ValueError("The cordon OD matrix must be a non-empty DataFrame.")
    normalised = matrix.copy()
    normalised.index = normalised.index.astype(str)
    normalised.columns = normalised.columns.astype(str)
    if normalised.index.has_duplicates or normalised.columns.has_duplicates:
        raise ValueError("The cordon OD matrix must have unique origin and destination labels.")

    internal_ids = [str(zone) for zone in corridor.zone_ids if str(zone) in normalised.index]
    gate_ids = [
        str(gate) for gate in corridor.gates["gate_id"].astype(str)
        if str(gate) in normalised.index
    ]
    ordered = internal_ids + gate_ids
    missing_columns = [label for label in ordered if label not in normalised.columns]
    if not ordered:
        raise ValueError("The matrix contains no labels belonging to this corridor.")
    if missing_columns:
        raise ValueError(f"Cordon destinations missing from the matrix: {missing_columns}")

    result = normalised.loc[ordered, ordered].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if (result.to_numpy(dtype=float) < 0).any():
        raise ValueError("OD demand cannot contain negative trip values.")
    return result, internal_ids, gate_ids


def _ignore_private_plotly_relayout_properties(figure: Any) -> None:
    """Ignore frontend-only map state unsupported by Plotly's Python schema.

    Some Jupyter frontends send ``map._derived`` (or the legacy
    ``mapbox._derived``) whenever a FigureWidget map is panned or resized.
    These are private browser-calculated values, but certain Plotly version
    combinations try to validate them as public layout properties and raise a
    ``ValueError``. This instance-local handler drops only those private keys;
    ordinary zoom, centre, selection, and click events continue to work.
    """
    trait_name = "_js2py_relayout"
    notifiers = figure._trait_notifiers.get(trait_name, {}).get("change", [])
    default_handlers = [
        handler for handler in list(notifiers)
        if getattr(handler, "name", "") == "_handler_js2py_relayout"
    ]
    if not default_handlers:
        return
    for handler in default_handlers:
        figure.unobserve(handler, names=trait_name)

    def filtered_relayout(change: dict[str, Any]) -> None:
        message = change["new"]
        if not message:
            return
        relayout_data = {
            key: value
            for key, value in dict(message.get("relayout_data", {})).items()
            if key != "lastInputTime"
            and not key.endswith("._derived")
            and "._derived." not in key
        }
        if relayout_data:
            figure.plotly_relayout(
                relayout_data=relayout_data,
                source_view_id=message.get("source_view_id"),
            )
        figure.set_trait(trait_name, None)

    figure.observe(filtered_relayout, names=trait_name)


def od_matrix_explorer(
    matrix: pd.DataFrame,
    corridor: tmi.CorridorContext,
    *,
    height: int = 650,
) -> Any:
    """Interactive heatmap of a reduced corridor OD matrix.

    Quadrant controls distinguish internal and gate movements. Hover always
    reports raw trips, even when logarithmic colour scaling is enabled, and a
    clicked cell remains visible in the detail panel.
    """
    import ipywidgets as widgets
    import plotly.graph_objects as go

    mat, internal_ids, gate_ids = _ordered_cordon_matrix(matrix, corridor)
    ordered = internal_ids + gate_ids
    values = mat.to_numpy(dtype=float)
    is_internal = np.asarray([label in set(internal_ids) for label in ordered])
    row_internal = is_internal[:, None]
    column_internal = is_internal[None, :]
    quadrants = {
        "internal -> internal": row_internal & column_internal,
        "gate -> internal": ~row_internal & column_internal,
        "internal -> gate": row_internal & ~column_internal,
        "gate -> gate (pass-through)": ~row_internal & ~column_internal,
    }
    active = {label: True for label in quadrants}
    state = {"log": False}

    def visible_values() -> np.ndarray:
        mask = np.zeros_like(values, dtype=bool)
        for label, enabled in active.items():
            if enabled:
                mask |= quadrants[label]
        visible = np.where(mask, values, np.nan)
        return np.log1p(visible) if state["log"] else visible

    heatmap = go.Heatmap(
        z=visible_values(),
        x=ordered,
        y=ordered,
        customdata=values,
        colorscale=SEQUENTIAL_BLUE,
        colorbar={"title": "Trips"},
        hovertemplate=(
            "Origin: %{y}<br>Destination: %{x}<br>"
            "Trips: %{customdata:,.1f}<extra></extra>"
        ),
        zmin=0.0,
    )
    figure = go.FigureWidget(data=[heatmap])
    figure.update_layout(
        title="Cordon OD matrix - internal zones and gates",
        xaxis={"title": "Destination", "showticklabels": False, "showgrid": False},
        yaxis={
            "title": "Origin", "showticklabels": False,
            "showgrid": False, "autorange": "reversed",
        },
        height=int(height),
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    if internal_ids and gate_ids:
        boundary = len(internal_ids) - 0.5
        figure.add_shape(
            type="line", x0=boundary, x1=boundary, y0=-0.5, y1=len(ordered) - 0.5,
            line={"color": "#52514e", "width": 1, "dash": "dot"},
        )
        figure.add_shape(
            type="line", y0=boundary, y1=boundary, x0=-0.5, x1=len(ordered) - 0.5,
            line={"color": "#52514e", "width": 1, "dash": "dot"},
        )

    scale_label = widgets.HTML()

    def redraw() -> None:
        visible = visible_values()
        finite = visible[np.isfinite(visible)]
        with figure.batch_update():
            figure.data[0].z = visible
            figure.data[0].zmax = float(finite.max()) if finite.size else 1.0
        scale_label.value = (
            "<i>Colours use log(1 + trips); hover values remain unscaled.</i>"
            if state["log"] else "<i>Colours show trips on a linear scale.</i>"
        )

    checkboxes = {
        label: widgets.Checkbox(value=True, description=label, indent=False)
        for label in quadrants
    }
    for label, checkbox in checkboxes.items():
        def on_quadrant_toggle(change: dict[str, Any], label: str = label) -> None:
            active[label] = bool(change["new"])
            redraw()
        checkbox.observe(on_quadrant_toggle, names="value")

    log_toggle = widgets.Checkbox(value=False, description="Log colour scale", indent=False)

    def on_log_toggle(change: dict[str, Any]) -> None:
        state["log"] = bool(change["new"])
        redraw()

    log_toggle.observe(on_log_toggle, names="value")
    detail = widgets.HTML(value="<i>Click a cell to pin its value here.</i>")

    def on_click(trace: Any, points: Any, selector: Any) -> None:
        if not points.xs:
            return
        destination, origin = str(points.xs[0]), str(points.ys[0])
        detail.value = (
            f"<b>{origin} -> {destination}</b><br>"
            f"Trips: {float(mat.loc[origin, destination]):,.1f}"
        )

    figure.data[0].on_click(on_click)
    redraw()
    controls = widgets.VBox(
        [widgets.HTML("<b>Show:</b>"), *checkboxes.values(),
         widgets.HTML("<br><b>Scale:</b>"), log_toggle, scale_label]
    )
    side = widgets.VBox(
        [controls, detail], layout=widgets.Layout(width="280px", padding="8px")
    )
    exports.save_outputs("1_4_cordon_od_matrix", tables={"": mat})
    return widgets.HBox([figure, side])


def corridor_flow_map_explorer(
    matrix: pd.DataFrame,
    corridor: tmi.CorridorContext,
    *,
    n_bins: int = 5,
    max_flows: int = 250,
    min_trips: float = 0.0,
    height: int = 680,
) -> Any:
    """Interactive, project-neutral desire-line map for a reduced OD matrix.

    The matrix is assumed to have already been reduced by
    :func:`collapse_od_to_gates`; pass-through demand is therefore not scaled
    again. Quantile bins and their legend are recomputed after each category
    filter, while ``max_flows`` limits browser rendering rather than demand.
    """
    import geopandas as gpd
    import ipywidgets as widgets
    import plotly.graph_objects as go

    mat, internal_ids, gate_ids = _ordered_cordon_matrix(matrix, corridor)
    internal_set = set(internal_ids)
    zone_rows = corridor.zones.loc[
        corridor.zones["grid_id"].astype(str).isin(internal_ids)
    ].copy()
    zone_centroids = gpd.GeoSeries(zone_rows["centroid"], crs=zone_rows.crs).to_crs(4326)
    zone_xy = {
        str(row.grid_id): (point.x, point.y)
        for row, point in zip(zone_rows.itertuples(index=False), zone_centroids)
    }
    gates_wgs84 = corridor.gates.set_geometry("geometry").to_crs(4326)
    gate_xy = {
        str(gate): (point.x, point.y)
        for gate, point in zip(corridor.gates["gate_id"].astype(str), gates_wgs84.geometry)
    }
    coordinates = {**zone_xy, **gate_xy}
    polygon_wgs84 = gpd.GeoSeries(
        [corridor.polygon], crs=corridor.zones.crs
    ).to_crs(4326).iloc[0]
    polygons = (
        list(polygon_wgs84.geoms)
        if hasattr(polygon_wgs84, "geoms") else [polygon_wgs84]
    )
    zone_boundaries = _prepare_zone_boundaries(
        zone_rows, zone_ids=internal_ids, level="FSM zone"
    )
    layer_visibility = {"boundaries": False, "centroids": True}

    records = []
    for origin, row in mat.iterrows():
        if origin not in coordinates:
            continue
        for destination, value in row.items():
            value = float(value)
            if destination == origin or destination not in coordinates or value <= float(min_trips):
                continue
            origin_internal = origin in internal_set
            destination_internal = destination in internal_set
            if origin_internal and destination_internal:
                category = "internal -> internal"
            elif not origin_internal and destination_internal:
                category = "gate -> internal"
            elif origin_internal and not destination_internal:
                category = "internal -> gate"
            else:
                category = "gate -> gate (pass-through)"
            records.append((origin, destination, value, category))

    flows = pd.DataFrame(records, columns=["origin", "destination", "value", "category"])
    active_category = {label: True for label in flows["category"].unique()} if len(flows) else {}
    active_bin = {index: True for index in range(max(int(n_bins), 1))}
    all_lons = [coordinate[0] for coordinate in coordinates.values()]
    all_lats = [coordinate[1] for coordinate in coordinates.values()]
    center = {"lon": float(np.mean(all_lons)), "lat": float(np.mean(all_lats))}

    figure = go.FigureWidget()
    _ignore_private_plotly_relayout_properties(figure)
    figure.update_layout(
        title="Cordon OD flows - desire lines",
        map={"style": "open-street-map", "center": center, "zoom": 11.0},
        height=int(height),
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        showlegend=False,
    )
    detail = widgets.HTML(value="<i>Click a flow line to pin its value here.</i>")
    flow_count_label = widgets.HTML()
    bin_controls_box = widgets.VBox()

    def click_handler(origin: str, destination: str, value: float) -> Any:
        def handler(trace: Any, points: Any, selector: Any) -> None:
            if points.point_inds:
                detail.value = f"<b>{origin} -> {destination}</b><br>Trips: {value:,.1f}"
        return handler

    def redraw() -> None:
        category_mask = (
            flows["category"].map(active_category).fillna(False)
            if len(flows) else pd.Series(dtype=bool)
        )
        binned = flows.loc[category_mask].copy() if len(flows) else flows.copy()
        bins_info = []
        if len(binned):
            codes, _ = pd.qcut(
                binned["value"], q=max(int(n_bins), 1), labels=False,
                retbins=True, duplicates="drop",
            )
            binned["bin"] = codes.astype(int)
            actual_bins = int(binned["bin"].max()) + 1
            for bin_index in range(actual_bins):
                bin_values = binned.loc[binned["bin"] == bin_index, "value"]
                position = bin_index / (actual_bins - 1) if actual_bins > 1 else 0.5
                bins_info.append({
                    "index": bin_index,
                    "count": int(len(bin_values)),
                    "minimum": float(bin_values.min()),
                    "maximum": float(bin_values.max()),
                    "color": _spectrum_color(position),
                })
        else:
            actual_bins = 0

        bin_rows = []
        for info in bins_info:
            bin_index = info["index"]
            swatch = widgets.HTML(
                value=(
                    f'<div style="width:14px;height:14px;background:{info["color"]};'
                    'border-radius:3px;margin-top:3px;"></div>'
                )
            )
            checkbox = widgets.Checkbox(
                value=active_bin.get(bin_index, True),
                description=(
                    f'{info["minimum"]:,.1f}-{info["maximum"]:,.1f} trips '
                    f'(n={info["count"]})'
                ),
                indent=False,
                layout=widgets.Layout(width="240px"),
            )
            def on_bin_toggle(change: dict[str, Any], bin_index: int = bin_index) -> None:
                active_bin[bin_index] = bool(change["new"])
                redraw()
            checkbox.observe(on_bin_toggle, names="value")
            bin_rows.append(widgets.HBox([swatch, checkbox]))
        bin_controls_box.children = bin_rows

        subset = (
            binned.loc[binned["bin"].map(lambda value: active_bin.get(int(value), True))]
            if actual_bins else binned
        )
        selected_count = len(subset)
        subset = subset.sort_values("value", ascending=False).head(int(max_flows))
        base_traces = []
        for part in polygons:
            longitudes, latitudes = part.exterior.xy
            base_traces.append(go.Scattermap(
                lon=list(longitudes), lat=list(latitudes), fill="toself",
                fillcolor="rgba(35,120,180,0.12)",
                line={"color": "rgba(35,120,180,0.8)", "width": 2},
                hoverinfo="skip", showlegend=False,
            ))
        boundary_trace = _plotly_zone_boundary_trace(
            zone_boundaries,
            visible=layer_visibility["boundaries"],
        )
        if boundary_trace is not None:
            boundary_trace.showlegend = False
            base_traces.append(boundary_trace)
        base_traces.append(go.Scattermap(
            lon=[coordinates[zone][0] for zone in internal_ids if zone in coordinates],
            lat=[coordinates[zone][1] for zone in internal_ids if zone in coordinates],
            mode="markers", marker={"size": 5, "color": "#898781"},
            hoverinfo="skip", showlegend=False,
            visible=layer_visibility["centroids"],
        ))
        visible_gates = [gate for gate in gate_ids if gate in coordinates]
        base_traces.append(go.Scattermap(
            lon=[coordinates[gate][0] for gate in visible_gates],
            lat=[coordinates[gate][1] for gate in visible_gates],
            mode="markers", marker={"size": 11, "color": "#7a0177"},
            text=visible_gates, hovertemplate="Gate %{text}<extra></extra>", showlegend=False,
        ))

        width_upper = max(float(subset["value"].quantile(0.98)), 1e-9) if len(subset) else 1.0
        colours = {info["index"]: info["color"] for info in bins_info}
        flow_traces, trace_metadata = [], []
        arrow_lon, arrow_lat, arrow_angle, arrow_colour, arrow_size = [], [], [], [], []
        for flow in subset.itertuples(index=False):
            lon0, lat0 = coordinates[flow.origin]
            lon1, lat1 = coordinates[flow.destination]
            colour = colours.get(int(flow.bin), "rgb(128,128,128)")
            width = 1.0 + 4.0 * np.sqrt(min(float(flow.value) / width_upper, 1.0))
            flow_traces.append(go.Scattermap(
                lon=[lon0, lon1], lat=[lat0, lat1], mode="lines",
                line={"color": colour, "width": width},
                text=[
                    f"{flow.origin} -> {flow.destination}<br>{flow.category}<br>"
                    f"Trips: {float(flow.value):,.1f}"
                ] * 2,
                hovertemplate="%{text}<extra></extra>", showlegend=False,
            ))
            trace_metadata.append((flow.origin, flow.destination, float(flow.value)))
            arrow_lon.append(lon0 + 0.85 * (lon1 - lon0))
            arrow_lat.append(lat0 + 0.85 * (lat1 - lat0))
            arrow_angle.append(_bearing_deg(lon0, lat0, lon1, lat1))
            arrow_colour.append(colour)
            arrow_size.append(8 + 6 * min(float(flow.value) / width_upper, 1.0))
        arrow_trace = go.Scattermap(
            lon=arrow_lon, lat=arrow_lat, mode="markers",
            marker={
                "symbol": "triangle", "size": arrow_size,
                "angle": arrow_angle, "color": arrow_colour,
            },
            hoverinfo="skip", showlegend=False,
        )

        with figure.batch_update():
            figure.data = []
            figure.add_traces(base_traces)
            figure.add_traces(flow_traces)
            flow_start = len(base_traces)
            figure.add_trace(arrow_trace)
        for trace, metadata in zip(
            figure.data[flow_start:flow_start + len(flow_traces)], trace_metadata
        ):
            trace.on_click(click_handler(*metadata))
        flow_count_label.value = (
            f"Showing {len(subset)} of {selected_count} selected flows "
            f"(capped at {int(max_flows)})."
        )

    category_checkboxes = {}
    for label in sorted(active_category):
        checkbox = widgets.Checkbox(value=True, description=label, indent=False)
        def on_category_toggle(change: dict[str, Any], label: str = label) -> None:
            active_category[label] = bool(change["new"])
            redraw()
        checkbox.observe(on_category_toggle, names="value")
        category_checkboxes[label] = checkbox
    boundary_toggle = widgets.Checkbox(
        value=False, description="Show zone boundaries", indent=False
    )
    centroid_toggle = widgets.Checkbox(
        value=True, description="Show centroids", indent=False
    )

    def on_layer_toggle(change: dict[str, Any], key: str) -> None:
        layer_visibility[key] = bool(change["new"])
        redraw()

    boundary_toggle.observe(
        lambda change: on_layer_toggle(change, "boundaries"), names="value"
    )
    centroid_toggle.observe(
        lambda change: on_layer_toggle(change, "centroids"), names="value"
    )
    redraw()
    controls = widgets.VBox([
        widgets.HTML("<b>Map layers:</b>"), boundary_toggle, centroid_toggle,
        widgets.HTML("<b>Show category:</b>"), *category_checkboxes.values(),
        widgets.HTML("<br><b>Show range (equal-count bins):</b>"),
        bin_controls_box, widgets.HTML("<br>"), flow_count_label,
    ])
    side = widgets.VBox(
        [controls, detail], layout=widgets.Layout(width="320px", padding="8px")
    )
    return widgets.HBox([figure, side])


def corridor_od_explorer(
    matrix: pd.DataFrame,
    corridor: tmi.CorridorContext,
    **flow_map_kwargs: Any,
) -> Any:
    """Combine the OD heatmap and desire-line map in one compact tab widget."""
    import ipywidgets as widgets

    tabs = widgets.Tab(children=[
        od_matrix_explorer(matrix, corridor),
        corridor_flow_map_explorer(matrix, corridor, **flow_map_kwargs),
    ])
    tabs.set_title(0, "OD matrix")
    tabs.set_title(1, "Desire lines")
    return tabs


def municipality_od_explorer(
    context: tmi.TransportContext,
    municipalities: list[str] | tuple[str, ...] | set[str],
    *,
    matrix: pd.DataFrame | None = None,
    zone_ids: list[str] | set[str] | None = None,
    height: int = 700,
) -> Any:
    """Return a static Plotly heatmap of directed municipality/quartier demand.

    A regular ``Figure`` is intentional: it keeps hover and zoom in the browser
    without FigureWidget relayout callbacks, avoiding frontend/backend Plotly
    version conflicts involving private ``map._derived`` state.
    """
    import plotly.graph_objects as go

    municipality_od, _ = tmi.aggregate_od_by_municipality(
        context,
        municipalities,
        matrix=matrix,
        zone_ids=zone_ids,
    )
    values = municipality_od.to_numpy(dtype=float)
    figure = go.Figure(go.Heatmap(
        z=values,
        x=municipality_od.columns,
        y=municipality_od.index,
        colorscale="Blues",
        colorbar={"title": "Passenger trips"},
        text=np.round(values, 1),
        texttemplate="%{text:,.1f}",
        hovertemplate=(
            "Origin: %{y}<br>Destination: %{x}<br>"
            "Peak-hour passenger trips: %{z:,.1f}<extra></extra>"
        ),
    ))
    figure.update_layout(
        title="Cantonal-reference OD demand by corridor area",
        xaxis_title="Destination area",
        yaxis_title="Origin area",
        yaxis_autorange="reversed",
        height=int(height),
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    exports.save_outputs("1_4_municipality_od", tables={"": municipality_od})
    return figure


def municipality_flow_map_explorer(
    context: tmi.TransportContext,
    municipalities: list[str] | tuple[str, ...] | set[str],
    *,
    matrix: pd.DataFrame | None = None,
    zone_ids: list[str] | set[str] | None = None,
    n_bins: int = 5,
    trip_bin_edges: list[float] | None = None,
    min_trips: float = 0.0,
    max_flows: int = 150,
    height: int = 700,
) -> Any:
    """Map directional corridor-area OD desire lines from any FSM matrix.

    Area centroids anchor straight desire lines; they are not assigned
    routes. A regular Plotly ``Figure`` keeps the map independent of Jupyter
    widget relayout callbacks. Fixed trip intervals may be supplied through
    ``trip_bin_edges``; otherwise equal-count quantile bins are used.
    """
    import geopandas as gpd
    import plotly.graph_objects as go

    _, links = tmi.aggregate_od_by_municipality(
        context, municipalities, matrix=matrix, zone_ids=zone_ids
    )
    links = links.loc[
        links["peak-hour passenger trips"] > float(min_trips)
    ].copy()
    if links.empty:
        raise ValueError("No corridor-area OD flows exceed min_trips.")

    requested = list(dict.fromkeys(map(str, municipalities)))
    area_zone_ids = tmi.resolve_area_zone_ids(
        context.zones,
        requested,
        allowed_zone_ids=zone_ids,
    )
    zone_to_area = {
        zone_id: area_name
        for area_name, selected_ids in area_zone_ids.items()
        for zone_id in selected_ids
    }
    zones = context.zones.loc[
        context.zones["grid_id"].astype(str).isin(zone_to_area),
        ["grid_id", "municipality_name", "geometry"],
    ].copy()
    if zones.crs is None:
        raise ValueError("Zone geometries need a declared CRS.")
    zones["area_name"] = zones["grid_id"].astype(str).map(zone_to_area)
    area_shapes = zones.dissolve(by="area_name")
    centres_projected = area_shapes.geometry.centroid
    centres_wgs84 = gpd.GeoSeries(
        centres_projected, index=area_shapes.index, crs=zones.crs
    ).to_crs(4326)
    coordinates = {
        str(name): (float(point.x), float(point.y))
        for name, point in centres_wgs84.items()
    }
    links = links.loc[
        links["origin"].isin(coordinates)
        & links["destination"].isin(coordinates)
    ].copy()
    # A valid attribute name keeps ``itertuples`` access independent of how
    # pandas sanitises the human-readable column containing spaces and hyphens.
    links["value"] = links["peak-hour passenger trips"].astype(float)

    values = links["value"]
    if trip_bin_edges is not None:
        edges = np.asarray(trip_bin_edges, dtype=float)
        if len(edges) < 2 or not np.all(np.diff(edges) > 0):
            raise ValueError(
                "trip_bin_edges must contain at least two strictly increasing values."
            )
        links["bin"] = pd.cut(
            values, bins=edges, labels=False, include_lowest=True, right=False
        )
        links = links.dropna(subset=["bin"]).copy()
        links["bin"] = links["bin"].astype(int)
        bin_labels = {
            index: (
                f"{edges[index]:,.0f}+ trips"
                if np.isinf(edges[index + 1])
                else f"{edges[index]:,.0f}-{edges[index + 1]:,.0f} trips"
            )
            for index in sorted(links["bin"].unique())
        }
    else:
        requested_bins = max(1, min(int(n_bins), len(links)))
        links["bin"] = pd.qcut(
            values, q=requested_bins, labels=False, duplicates="drop"
        ).astype(int)
        bin_labels = {}
        for index in sorted(links["bin"].unique()):
            bin_values = links.loc[
                links["bin"] == index, "peak-hour passenger trips"
            ]
            bin_labels[index] = (
                f"{float(bin_values.min()):,.0f}-{float(bin_values.max()):,.0f} trips"
            )

    # Plotly draws later traces on top of earlier ones. Sorting ascending here
    # ensures the largest flows are added last and therefore render above the
    # thinner lines, without altering the underlying data or flow values.
    links = links.sort_values(
        "peak-hour passenger trips", ascending=True
    ).head(int(max_flows))
    populated_bins = sorted(links["bin"].unique())
    colours = {
        bin_index: _spectrum_color(
            position / (len(populated_bins) - 1) if len(populated_bins) > 1 else 0.5
        )
        for position, bin_index in enumerate(populated_bins)
    }
    width_upper = max(
        float(links["peak-hour passenger trips"].quantile(0.98)), 1e-9
    )

    figure = go.Figure()
    boundary_trace = _plotly_zone_boundary_trace(
        _prepare_zone_boundaries(zones, level="Municipality"),
        visible="legendonly",
        name="Municipality boundaries",
    )
    if boundary_trace is not None:
        figure.add_trace(boundary_trace)
    arrow_lon, arrow_lat, arrow_angle, arrow_colour, arrow_size = [], [], [], [], []
    for flow in links.itertuples(index=False):
        lon0, lat0 = coordinates[flow.origin]
        lon1, lat1 = coordinates[flow.destination]
        value = float(flow.value)
        bin_index = int(flow.bin)
        colour = colours[bin_index]
        width = 1.0 + 5.0 * np.sqrt(min(value / width_upper, 1.0))
        figure.add_trace(go.Scattermap(
            lon=[lon0, lon1], lat=[lat0, lat1], mode="lines",
            line={"color": colour, "width": width},
            text=[
                f"{flow.origin} -> {flow.destination}<br>"
                f"Peak-hour passenger trips: {value:,.1f}"
            ] * 2,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))
        arrow_lon.append(lon0 + 0.85 * (lon1 - lon0))
        arrow_lat.append(lat0 + 0.85 * (lat1 - lat0))
        arrow_angle.append(_bearing_deg(lon0, lat0, lon1, lat1))
        arrow_colour.append(colour)
        arrow_size.append(8 + 6 * min(value / width_upper, 1.0))

    figure.add_trace(go.Scattermap(
        lon=arrow_lon, lat=arrow_lat, mode="markers",
        marker={
            "symbol": "triangle", "size": arrow_size,
            "angle": arrow_angle, "color": arrow_colour,
        },
        hoverinfo="skip", showlegend=False,
    ))
    names = [name for name in requested if name in coordinates]
    figure.add_trace(go.Scattermap(
        lon=[coordinates[name][0] for name in names],
        lat=[coordinates[name][1] for name in names],
        mode="markers+text",
        marker={"size": 10, "color": "#5b2c83"},
        text=names,
        textposition="top center",
        hovertemplate="%{text}<extra></extra>",
        name="Corridor areas",
    ))
    for bin_index in populated_bins:
        figure.add_trace(go.Scattermap(
            lon=[None], lat=[None], mode="markers",
            marker={"size": 10, "color": colours[bin_index]},
            name=bin_labels[bin_index], hoverinfo="skip",
        ))
    figure.update_layout(
        title=(
            "Cantonal-reference corridor-area OD desire lines "
            f"(showing {len(links)} flows)"
        ),
        map={
            "style": "open-street-map",
            "center": {
                "lon": float(np.mean([coordinates[name][0] for name in names])),
                "lat": float(np.mean([coordinates[name][1] for name in names])),
            },
            "zoom": 9.3,
        },
        height=int(height),
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        legend={"title": "Layers and flow range", "groupclick": "toggleitem"},
    )
    return figure


def flow_map_explorer(
    context: tmi.TransportContext,
    mode_result: tmi.ModeChoiceResult | None = None,
    corridor_municipalities: list[str] | None = None,
    *,
    matrix: pd.DataFrame | None = None,
    buffer_m: float = tmi.DEFAULT_CORRIDOR_BUFFER_M,
    n_bins: int = 5,
    max_flows: int = 250,
    pass_through_factor: float = 0.05,
    min_trips: float = 0.0,
) -> Any:
    """
    Desire-line map on OpenStreetMap basemap: OD flows as arrowed lines, split into
    n_bins equal-count quantile bins with interactive category and range filtering.
    """
    import math
    import geopandas as gpd
    import ipywidgets as widgets
    import plotly.graph_objects as go

    zones = context.zones.copy()
    if corridor_municipalities is None:
        corridor_municipalities = list(zones["municipality_name"].unique())

    core_zones = tmi._get_core_zones(zones, corridor_municipalities)
    if core_zones.empty:
        raise ValueError("No matching corridor municipalities found.")

    polygon = core_zones.geometry.union_all().buffer(float(buffer_m))

    # Check for gates in mode_result or generate them
    gates_gdf = None
    if mode_result is not None and mode_result.assigned_metadata is not None and "gates" in mode_result.assigned_metadata:
        gates_gdf = mode_result.assigned_metadata["gates"].copy()
    else:
        # Generate gates using helper
        _, gates_gdf = _get_or_generate_corridor(context, corridor_municipalities, buffer_m=buffer_m, max_gates=tmi.DEFAULT_MAX_GATES)

    # Zone centroids and gate coordinates in WGS84
    zone_centroids = gpd.GeoSeries(zones["centroid"], crs=zones.crs)
    internal_mask = zone_centroids.intersects(polygon)
    internal_zones = zones.loc[internal_mask].copy()
    internal_ids = set(internal_zones["grid_id"].astype(str))

    int_centroids_wgs84 = gpd.GeoSeries(internal_zones["centroid"], crs=zones.crs).to_crs(4326)
    zone_xy = {
        str(row.grid_id): (pt.x, pt.y)
        for row, pt in zip(internal_zones.itertuples(index=False), int_centroids_wgs84)
    }

    gates_wgs84 = gates_gdf.set_geometry("geometry").to_crs(4326)
    gate_xy = {
        str(gid): (pt.x, pt.y)
        for gid, pt in zip(gates_gdf["gate_id"].astype(str), gates_wgs84.geometry)
    }
    coords = {**zone_xy, **gate_xy}

    # Polygon in WGS84
    polygon_wgs84 = gpd.GeoSeries([polygon], crs=zones.crs).to_crs(4326).iloc[0]
    polygons = list(polygon_wgs84.geoms) if hasattr(polygon_wgs84, "geoms") else [polygon_wgs84]
    zone_boundaries = _prepare_zone_boundaries(
        internal_zones, zone_ids=internal_ids, level="FSM zone"
    )
    layer_visibility = {"boundaries": False, "centroids": True}

    # Build or use matrix
    if matrix is None:
        if mode_result is None:
            raise ValueError("Either matrix or mode_result must be provided.")
        # Map external demand to nearest gate
        external = zones.loc[~zones["grid_id"].astype(str).isin(internal_ids)].copy()
        external_centroids = gpd.GeoSeries(external["centroid"], crs=zones.crs)
        ext_xy = np.column_stack((external_centroids.x, external_centroids.y))

        def map_gates(eligible_gates):
            if eligible_gates.empty: return {}
            gate_xy_loc = np.column_stack((eligible_gates.geometry.x, eligible_gates.geometry.y))
            nearest = np.argmin(((ext_xy[:, None, 0] - gate_xy_loc[None, :, 0])**2 + (ext_xy[:, None, 1] - gate_xy_loc[None, :, 1])**2), axis=1)
            return dict(zip(external["grid_id"].astype(str), eligible_gates.iloc[nearest]["gate_id"].astype(str)))

        origin_map = {z: z for z in internal_ids}
        origin_map.update(map_gates(gates_gdf.loc[gates_gdf["can_enter"].astype(bool)]))

        dest_map = {z: z for z in internal_ids}
        dest_map.update(map_gates(gates_gdf.loc[gates_gdf["can_exit"].astype(bool)]))

        total_od = sum(mode_result.od_by_mode.values()).copy()
        total_od.index = total_od.index.astype(str).map(lambda x: origin_map.get(x, None))
        total_od.columns = total_od.columns.astype(str).map(lambda x: dest_map.get(x, None))
        total_od = total_od.loc[total_od.index.notna(), total_od.columns.notna()]
        matrix = total_od.groupby(level=0).sum().groupby(level=0, axis=1).sum()

    gate_ids = set(gates_gdf["gate_id"].astype(str)) if gates_gdf is not None else set()

    records = []
    for origin in matrix.index:
        orig_str = str(origin)
        if orig_str not in coords:
            continue
        for destination in matrix.columns:
            dest_str = str(destination)
            if dest_str == orig_str or dest_str not in coords:
                continue
            value = float(matrix.loc[origin, destination])
            if value <= min_trips:
                continue
            origin_internal = orig_str in internal_ids
            destination_internal = dest_str in internal_ids
            if origin_internal and destination_internal:
                category = "internal → internal"
            elif not origin_internal and destination_internal:
                category = "gate → internal"
            elif origin_internal and not destination_internal:
                category = "internal → gate"
            else:
                category = "gate → gate (pass-through)"
                value = value * float(pass_through_factor)
                if value <= min_trips:
                    continue
            records.append((orig_str, dest_str, value, category))

    flows = pd.DataFrame(records, columns=["origin", "destination", "value", "category"])
    active_category = {key: True for key in flows["category"].unique()} if len(flows) else {}
    active_bin = {i: True for i in range(n_bins)}

    all_lons = [c[0] for c in coords.values()]
    all_lats = [c[1] for c in coords.values()]
    center = dict(lon=float(np.mean(all_lons)), lat=float(np.mean(all_lats)))

    fig = go.FigureWidget()
    _ignore_private_plotly_relayout_properties(fig)
    fig.update_layout(
        title="Corridor OD Flows — Desire Lines",
        map=dict(style="open-street-map", center=center, zoom=11.0),
        height=680,
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False,
    )

    detail = widgets.HTML(value="<i>Click a flow line to pin its value here.</i>")
    flow_count_label = widgets.HTML()
    bin_controls_box = widgets.VBox()

    def on_click(origin, destination, value):
        def handler(trace, points, sel_state):
            if points.point_inds:
                detail.value = f"<b>{origin} → {destination}</b><br>Trips: {value:,.1f}"
        return handler

    def redraw():
        category_mask = flows["category"].map(active_category).fillna(False) if len(flows) else pd.Series(dtype=bool)
        by_category = flows.loc[category_mask] if len(flows) else flows

        bins_info = []
        binned = by_category.copy()
        if len(by_category):
            codes, edges = pd.qcut(
                by_category["value"], q=n_bins, labels=False, retbins=True, duplicates="drop"
            )
            binned["bin"] = codes.values
            actual_bins = int(binned["bin"].max()) + 1
            for b in range(actual_bins):
                bin_values = binned.loc[binned["bin"] == b, "value"]
                t = b / (actual_bins - 1) if actual_bins > 1 else 0.5
                bins_info.append({
                    "index": b,
                    "count": int(len(bin_values)),
                    "vmin": float(bin_values.min()),
                    "vmax": float(bin_values.max()),
                    "color": _spectrum_color(t),
                })
        else:
            actual_bins = 0

        bin_rows = []
        for info in bins_info:
            b = info["index"]
            swatch = widgets.HTML(
                value=f'<div style="width:14px;height:14px;background:{info["color"]};'
                      f'border-radius:3px;margin-top:3px;"></div>'
            )
            cb = widgets.Checkbox(
                value=active_bin.get(b, True),
                description=f'{info["vmin"]:,.1f}–{info["vmax"]:,.1f} trips (n={info["count"]})',
                indent=False,
                layout=widgets.Layout(width="240px"),
            )
            def on_bin_toggle(change, b=b):
                active_bin[b] = change["new"]
                redraw()
            cb.observe(on_bin_toggle, names="value")
            bin_rows.append(widgets.HBox([swatch, cb]))
        bin_controls_box.children = bin_rows

        subset = (
            binned.loc[binned["bin"].map(lambda b: active_bin.get(b, True))]
            if actual_bins else binned.assign(bin=[])
        )
        subset = subset.sort_values("value", ascending=False).head(max_flows)

        base_traces = []
        for part in polygons:
            lons, lats = part.exterior.xy
            base_traces.append(go.Scattermap(
                lon=list(lons), lat=list(lats), fill="toself",
                fillcolor="rgba(35,120,180,0.12)",
                line=dict(color="rgba(35,120,180,0.8)", width=2),
                hoverinfo="skip", showlegend=False,
            ))
        boundary_trace = _plotly_zone_boundary_trace(
            zone_boundaries, visible=layer_visibility["boundaries"]
        )
        if boundary_trace is not None:
            boundary_trace.showlegend = False
            base_traces.append(boundary_trace)
        base_traces.append(go.Scattermap(
            lon=[coords[z][0] for z in internal_ids if z in coords], lat=[coords[z][1] for z in internal_ids if z in coords],
            mode="markers", marker=dict(size=5, color="#898781"),
            hoverinfo="skip", showlegend=False,
            visible=layer_visibility["centroids"],
        ))
        base_traces.append(go.Scattermap(
            lon=[coords[g][0] for g in gate_ids if g in coords], lat=[coords[g][1] for g in gate_ids if g in coords],
            mode="markers", marker=dict(size=11, color="#7a0177"),
            text=list(gate_ids), hovertemplate="Gate %{text}<extra></extra>", showlegend=False,
        ))

        width_upper = max(float(subset["value"].quantile(0.98)), 1e-9) if len(subset) else 1.0
        colors_by_bin = {info["index"]: info["color"] for info in bins_info}

        new_traces, meta = [], []
        arrow_lon, arrow_lat, arrow_angle, arrow_color, arrow_size = [], [], [], [], []
        for row in subset.itertuples(index=False):
            lon0, lat0 = coords[row.origin]
            lon1, lat1 = coords[row.destination]
            colour = colors_by_bin.get(int(row.bin), "rgb(128,128,128)")
            width = 1.0 + 4.0 * math.sqrt(min(row.value / width_upper, 1.0))
            new_traces.append(go.Scattermap(
                lon=[lon0, lon1], lat=[lat0, lat1], mode="lines",
                line=dict(color=colour, width=width),
                text=[f"{row.origin} → {row.destination}<br>{row.category}<br>Trips: {row.value:,.1f}"] * 2,
                hovertemplate="%{text}<extra></extra>", showlegend=False,
            ))
            meta.append((row.origin, row.destination, row.value))

            tlon, tlat = lon0 + 0.85 * (lon1 - lon0), lat0 + 0.85 * (lat1 - lat0)
            bearing = _bearing_deg(lon0, lat0, lon1, lat1)
            arrow_lon.append(tlon); arrow_lat.append(tlat)
            arrow_angle.append(bearing); arrow_color.append(colour)
            arrow_size.append(8 + 6 * min(row.value / width_upper, 1.0))

        arrow_trace = go.Scattermap(
            lon=arrow_lon, lat=arrow_lat, mode="markers",
            marker=dict(symbol="triangle", size=arrow_size, angle=arrow_angle, color=arrow_color),
            hoverinfo="skip", showlegend=False,
        )

        with fig.batch_update():
            fig.data = []
            fig.add_traces(base_traces)
            fig.add_traces(new_traces)
            line_trace_start = len(base_traces)
            fig.add_traces([arrow_trace])

        for trace, (origin, destination, value) in zip(
            fig.data[line_trace_start:line_trace_start + len(new_traces)], meta
        ):
            trace.on_click(on_click(origin, destination, value))

        flow_count_label.value = (
            f"Showing {len(subset)} of {int(binned['bin'].map(lambda b: active_bin.get(b, True)).sum()) if actual_bins else 0} "
            f"selected-bin flows (capped at {max_flows})."
        )

    category_checkboxes = {}
    for key in sorted(active_category):
        cb = widgets.Checkbox(value=True, description=key, indent=False)
        def on_category_toggle(change, key=key):
            active_category[key] = change["new"]
            redraw()
        cb.observe(on_category_toggle, names="value")
        category_checkboxes[key] = cb

    boundary_toggle = widgets.Checkbox(
        value=False, description="Show zone boundaries", indent=False
    )
    centroid_toggle = widgets.Checkbox(
        value=True, description="Show centroids", indent=False
    )

    def on_layer_toggle(change: dict[str, Any], key: str) -> None:
        layer_visibility[key] = bool(change["new"])
        redraw()

    boundary_toggle.observe(
        lambda change: on_layer_toggle(change, "boundaries"), names="value"
    )
    centroid_toggle.observe(
        lambda change: on_layer_toggle(change, "centroids"), names="value"
    )

    redraw()

    controls = widgets.VBox([
        widgets.HTML("<b>Map layers:</b>"), boundary_toggle, centroid_toggle,
        widgets.HTML("<b>Show category:</b>"), *category_checkboxes.values(),
        widgets.HTML("<br><b>Show range (equal-count bins):</b>"), bin_controls_box,
        widgets.HTML("<br>"), flow_count_label,
    ])
    side = widgets.VBox([controls, detail], layout=widgets.Layout(width="320px", padding="8px"))
    return widgets.HBox([fig, side])


_PLAYGROUND_OBJECTIVES = (
    ("avg_tt_min", "Average travel time", "min/trip", 1.0),
    ("congestion_delay_hours", "Passenger road delay", "person-hours/peak hour", None),
    ("co2_tonnes", "Tailpipe CO2", "tonnes/peak hour", None),
    ("total_demand", "Passenger demand", "trips/peak hour", None),
)
_PLAYGROUND_LABELS = {key: title for key, title, _, _ in _PLAYGROUND_OBJECTIVES}
_PLAYGROUND_LOWER_IS_BETTER = {
    _PLAYGROUND_LABELS[key]
    for key in ("avg_tt_min", "congestion_delay_hours", "co2_tonnes")
}


def plot_parameter_playground_result(
    result: Mapping[str, Any],
) -> Any:
    """Plot the four standard physical indicators for two time snapshots."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(_PLAYGROUND_OBJECTIVES), figsize=(16, 3.8))
    for axis, (key, title, unit, scale) in zip(axes, _PLAYGROUND_OBJECTIVES):
        values = [tmi._playground_peak_value(result[year], key) for year in ("year1", "year40")]
        axis.bar(
            ["Year 1", "Year 40"], values,
            color=["#2b5c8f", "#d95f02"], width=0.55,
        )
        axis.set(title=title, ylabel=unit)
        axis.set_ylim(0.0, max(max(values) * 1.20, 1e-9))
        axis.grid(axis="y", linestyle="--", alpha=0.4)
        for position, value in enumerate(values):
            axis.text(position, value * 1.03, f"{value:,.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    return fig


def _playground_objective_table(
    reference: Mapping[str, Any],
    scenario: Mapping[str, Any],
    year: str,
) -> pd.DataFrame:
    """Return absolute and relative changes against the nominal case."""

    rows = []
    for key, title, unit, scale in _PLAYGROUND_OBJECTIVES:
        nominal = tmi._playground_peak_value(reference[year], key)
        selected = tmi._playground_peak_value(scenario[year], key)
        delta = selected - nominal
        relative = 100.0 * delta / abs(nominal) if nominal else np.nan
        rows.append({
            "Indicator": title,
            "Nominal": nominal,
            "Selected": selected,
            "Absolute change": delta,
            "Relative change (%)": relative,
            "Unit": unit,
        })
    return pd.DataFrame(rows).set_index("Indicator")


def _playground_mode_shift_table(
    reference: Mapping[str, Any],
    scenario: Mapping[str, Any],
    year: str,
) -> pd.DataFrame:
    """Return coupled-MSA trip-share changes in percentage points."""

    modes = (
        ("Car", "car_share_trips"),
        ("Public transport", "pt_share_trips"),
        ("Bicycle", "bike_share_trips"),
        ("Walking", "walk_share_trips"),
    )
    reference_metrics = reference[year].get("mode_metrics", {})
    scenario_metrics = scenario[year].get("mode_metrics", {})
    rows = []
    for label, key in modes:
        nominal = float(reference_metrics.get(key, np.nan))
        selected = float(scenario_metrics.get(key, np.nan))
        rows.append({
            "Mode": label,
            "Nominal share": nominal,
            "Selected share": selected,
            "Change (pp)": 100.0 * (selected - nominal),
        })
    return pd.DataFrame(rows).set_index("Mode")


def plot_parameter_playground_impacts(
    reference: Mapping[str, Any],
    scenario: Mapping[str, Any],
    *,
    year: str = "year40",
) -> Any:
    """Plot scenario impacts and modal shifts relative to the nominal case."""

    import matplotlib.pyplot as plt

    impacts = _playground_objective_table(
        reference, scenario, year
    )
    modes = _playground_mode_shift_table(reference, scenario, year)
    objective_values = impacts["Relative change (%)"].fillna(0.0)
    mode_values = modes["Change (pp)"].fillna(0.0)
    lower_is_better = _PLAYGROUND_LOWER_IS_BETTER
    colours = []
    for label, value in objective_values.items():
        if label not in lower_is_better or abs(value) < 1e-12:
            colours.append("#4c78a8" if label not in lower_is_better else "#9e9e9e")
        else:
            colours.append("#2e8b57" if value < 0.0 else "#c94c4c")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.2))
    axes[0].barh(objective_values.index, objective_values, color=colours)
    axes[0].axvline(0.0, color="#444444", linewidth=0.8)
    axes[0].set(title="Impact against nominal", xlabel="Relative change (%)")
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.25)
    for index, value in enumerate(objective_values):
        axes[0].annotate(
            f"{value:+.1f}%", (value, index), xytext=(4 if value >= 0 else -4, 0),
            textcoords="offset points", ha="left" if value >= 0 else "right",
            va="center", fontsize=9,
        )

    mode_colours = ["#c94c4c" if value > 0 else "#2e8b57" for value in mode_values]
    axes[1].barh(mode_values.index, mode_values, color=mode_colours)
    axes[1].axvline(0.0, color="#444444", linewidth=0.8)
    axes[1].set(title="Modal shift against nominal", xlabel="Change (percentage points)")
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.25)
    for index, value in enumerate(mode_values):
        axes[1].annotate(
            f"{value:+.2f} pp", (value, index), xytext=(4 if value >= 0 else -4, 0),
            textcoords="offset points", ha="left" if value >= 0 else "right",
            va="center", fontsize=9,
        )
    fig.suptitle(f"Selected parameters vs nominal — {'Year 1' if year == 'year1' else 'Year 40'}")
    fig.tight_layout()
    # Widget callbacks explicitly display the returned figure. Leaving it
    # registered with pyplot also lets notebook backends flush it implicitly,
    # rendering the same chart again after every callback.
    plt.close(fig)
    return fig


def _parameter_playground_summary_html(
    reference: Mapping[str, Any],
    scenario: Mapping[str, Any],
    specs: Mapping[str, Mapping[str, Any]],
    *,
    year: str,
) -> str:
    """Build compact changed-input, KPI and causal-summary cards."""

    import html

    nominal_values = reference["parameters"]
    selected_values = scenario["parameters"]
    changed = [
        name for name in specs
        if not np.isclose(nominal_values[name], selected_values[name])
    ]

    def parameter_value(name: str, value: float) -> str:
        if name in {"PASSENGER_DEMAND_GROWTH_Y40", "EBIKE_SHARE"}:
            return f"{value:.0%}"
        if name.endswith("_pct"):
            return f"{value:.0f}%"
        return f"{value:.2f}"

    if changed:
        chips = "".join(
            "<span style='display:inline-block;margin:3px;padding:6px 9px;"
            "border-radius:12px;background:#eef3f8'>"
            f"<b>{html.escape(name)}</b>: "
            f"{parameter_value(name, nominal_values[name])} &rarr; "
            f"{parameter_value(name, selected_values[name])}</span>"
            for name in changed
        )
    else:
        chips = "<i>No assumptions differ from the nominal configuration.</i>"

    impacts = _playground_objective_table(
        reference, scenario, year
    )
    cards = []
    lower_is_better = _PLAYGROUND_LOWER_IS_BETTER
    for label, row in impacts.iterrows():
        delta, relative = row["Absolute change"], row["Relative change (%)"]
        if label == _PLAYGROUND_LABELS["total_demand"]:
            colour = "#2b5c8f"
        elif abs(delta) < 1e-12:
            colour = "#666666"
        else:
            colour = "#2e7d32" if (delta < 0) == (label in lower_is_better) else "#b71c1c"
        cards.append(
            "<div style='flex:1;min-width:185px;padding:10px;border-top:4px solid "
            f"{colour};background:#f7f7f7'><b>{html.escape(label)}</b><br>"
            f"<span style='font-size:20px'>{row['Selected']:,.1f}</span> "
            f"<small>{html.escape(str(row['Unit']))}</small><br>"
            f"<span style='color:{colour}'>{delta:+,.1f} ({relative:+.1f}%)</span>"
            "<br><small>against nominal</small></div>"
        )

    mode_shift = _playground_mode_shift_table(reference, scenario, year)
    car_shift = float(mode_shift.loc["Car", "Change (pp)"])
    delay_change = float(impacts.loc[_PLAYGROUND_LABELS["congestion_delay_hours"], "Relative change (%)"])
    time_change = float(impacts.loc[_PLAYGROUND_LABELS["avg_tt_min"], "Relative change (%)"])
    co2_change = float(impacts.loc[_PLAYGROUND_LABELS["co2_tonnes"], "Relative change (%)"])
    method_text = "The coupled MSA reports"
    qualification = "This includes route assignment and congestion-to-mode-choice feedback."
    causal = (
        f"For the selected year, the changed assumptions produce a car-share shift of "
        f"<b>{car_shift:+.2f} percentage points</b>. {method_text} "
        f"<b>{delay_change:+.1f}%</b> road delay, <b>{time_change:+.1f}%</b> average "
        f"travel time and <b>{co2_change:+.1f}%</b> tailpipe CO2 relative to nominal. "
        f"{qualification} "
        "These are combined model effects; they are not a separate causal attribution "
        "to each slider when several parameters change together."
    )
    return (
        "<div style='font-family:Arial,sans-serif'>"
        "<h4 style='margin-bottom:4px'>Changed assumptions</h4>" + chips +
        "<h4 style='margin-bottom:7px'>Impact summary</h4>"
        "<div style='display:flex;flex-wrap:wrap;gap:9px'>" + "".join(cards) + "</div>"
        "<div style='margin-top:10px;padding:10px;border-left:4px solid #546e7a;"
        "background:#f3f6f7'><b>What happened in the model?</b><br>" + causal + "</div></div>"
    )


def _parameter_playground_groups(
    specs: Mapping[str, Mapping[str, Any]],
    control_groups: Mapping[str, Sequence[str]] | None,
) -> tuple[dict[str, list[str]], list[str]]:
    """Validate notebook-selected controls against the wired parameter catalogue."""
    if control_groups is None:
        groups: dict[str, list[str]] = {}
        for name, spec in specs.items():
            if name != "EBIKE_SPEED_MULTIPLIER":
                groups.setdefault(str(spec["group"]), []).append(name)
        return groups, []
    if not isinstance(control_groups, Mapping):
        return {}, ["control_groups must map tab names to lists of supported parameter keys."]
    groups, errors, seen = {}, [], set()
    for title, names in control_groups.items():
        if not isinstance(title, str) or not title.strip():
            errors.append("Each tab needs a non-empty name.")
            continue
        if not isinstance(names, (list, tuple)) or not names:
            errors.append(f"{title}: supply a non-empty list of parameter keys.")
            continue
        valid = []
        for name in names:
            if not isinstance(name, str) or name not in specs:
                errors.append(f"{title}: unsupported parameter {name!r}.")
            elif name in seen:
                errors.append(f"{title}: {name} is listed more than once.")
            else:
                valid.append(name)
                seen.add(name)
        if valid:
            groups[title] = valid
    if not groups:
        errors.append("Choose at least one supported control.")
    return groups, errors


def parameter_playground_dashboard(
    context: tmi.TransportContext,
    stage_specs: Mapping[int, Mapping[str, Any]],
    *,
    corridor_zone_ids: list[str] | set[str],
    corridor_municipalities: list[str],
    demand_growth_y40: float | None = None,
    stage: int = 0,
    simulation_params: Mapping[str, Any] | None = None,
    reference_mode_result: tmi.ModeChoiceResult | None = None,
    reference_year_assignments: Mapping[str, Any] | None = None,
    control_groups: Mapping[str, Sequence[str]] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build a cached parameter dashboard; omitted controls retain nominal values."""

    import html
    from io import BytesIO
    import ipywidgets as widgets
    import matplotlib.pyplot as plt

    specs = tmi.parameter_playground_specs(
        stage_specs[int(stage)], demand_growth_y40=demand_growth_y40,
        nominal_params=simulation_params,
    )
    groups, configuration_errors = _parameter_playground_groups(specs, control_groups)
    visible_names = [name for names in groups.values() for name in names]
    widget_keys = {"value", "min", "max", "step", "readout_format"}
    sliders = {
        name: widgets.FloatSlider(
            **{key: spec[key] for key in widget_keys},
            description="", continuous_update=False,
            layout=widgets.Layout(width="260px"),
        )
        for name, spec in specs.items() if name in visible_names
    }

    sections = []
    for names in groups.values():
        boxes = [
            widgets.VBox([
                widgets.HTML(
                    f"<small><b>{name}</b><br>{specs[name]['description']}</small>"
                ),
                sliders[name],
            ])
            for name in names
        ]
        sections.append(widgets.GridBox(
            boxes,
            layout=widgets.Layout(
                grid_template_columns="repeat(2, 290px)", grid_gap="10px 18px"
            ),
        ))
    accordion = widgets.Accordion(children=sections)
    for index, title in enumerate(groups):
        accordion.set_title(index, title)

    run_button = widgets.Button(
        description="Run coupled MSA", icon="play", button_style="primary",
        disabled=bool(configuration_errors),
    )
    reset_button = widgets.Button(description="Reset", icon="refresh")
    year_selector = widgets.ToggleButtons(
        options=[("Year 1", "year1"), ("Year 40", "year40")],
        value="year40", description="Show:", button_style="info",
    )
    pending = widgets.HTML()
    status = widgets.HTML()
    summary_output = widgets.HTML()
    figure_output = widgets.Image(format="png", layout=widgets.Layout(max_width="100%"))
    details_output = widgets.HTML(layout=widgets.Layout(width="100%"))
    details = widgets.Accordion(children=[details_output], selected_index=None)
    details.set_title(0, "Numerical details")
    output = widgets.VBox([summary_output, figure_output, details], layout=widgets.Layout(display="none"))
    state: dict[str, Any] = {
        "result": None, "reference": None, "running": False, "closed": False, "cache": {},
        "calculation_cache": {}, "corridor": None, "max_cached_runs": 12,
        "sliders": sliders, "slider_specs": specs,
        "calculation_method": None,
        "control_groups": groups, "configuration_errors": configuration_errors,
    }

    # The notebook may already have calculated the nominal Stage 0 mode choice.
    # Seed the cache with it instead of repeating the initial FSM calculation.
    if reference_mode_result is not None:
        nominal_values = {name: float(spec["value"]) for name, spec in specs.items()}
        nominal_transport_key = (
            int(stage),
            tuple(
                (name, nominal_values[name])
                for name in specs
                if name not in ("PASSENGER_DEMAND_GROWTH_Y40", "PT_ASC_SHIFT_Y40", "EBIKE_SHARE")
            ),
        )
        reference_metrics = tmi.extract_corridor_metrics(
            context,
            reference_mode_result,
            corridor_zone_ids=corridor_zone_ids,
        )
        state["calculation_cache"].setdefault("mode_choice", {})[
            nominal_transport_key
        ] = (reference_mode_result, reference_metrics)

    def selected_values() -> dict[str, float]:
        return {
            **{name: float(spec["value"]) for name, spec in specs.items()},
            **{name: float(slider.value) for name, slider in sliders.items()},
        }

    def configuration_key(values: Mapping[str, float]) -> tuple[Any, ...]:
        import json
        import parameters as p
        return (tuple((name, values[name]) for name in specs),
                json.dumps({**p.NOMINAL_PARAMS, **dict(simulation_params or {})}, sort_keys=True, default=str),
                json.dumps(stage_specs, sort_keys=True, default=str),
                repr(tmi.resolve_assignment_settings()))

    def update_pending(_: Any = None) -> None:
        changed = [
            name for name, value in selected_values().items()
            if not np.isclose(value, float(specs[name]["value"]))
        ]
        pending.value = (
            "<small><b>Pending changes:</b> " + ", ".join(changed) +
            ". Select <b>Run coupled MSA</b> to calculate the effects.</small>"
            if changed else
            "<small>Current controls match the nominal configuration.</small>"
        )

    def render() -> None:
        if state["closed"] or state["result"] is None or state["reference"] is None:
            return
        year = str(year_selector.value)
        impacts = _playground_objective_table(
            state["reference"], state["result"], year
        )
        modes = _playground_mode_shift_table(
            state["reference"], state["result"], year
        )
        summary_html = _parameter_playground_summary_html(
            state["reference"], state["result"], specs, year=year,
        )
        figure = None
        try:
            figure = plot_parameter_playground_impacts(
                state["reference"], state["result"],
                year=year,
            )
            buffer = BytesIO()
            figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
            figure_bytes = buffer.getvalue()
        finally:
            if figure is not None:
                plt.close(figure)
        impact_table = impacts.style.format({
            "Nominal": "{:,.2f}", "Selected": "{:,.2f}",
            "Absolute change": "{:+,.2f}",
            "Relative change (%)": "{:+.2f}%",
        }).to_html()
        mode_table = modes.style.format({
            "Nominal share": "{:.2%}", "Selected share": "{:.2%}",
            "Change (pp)": "{:+.2f} pp",
        }).to_html()
        summary_output.value = summary_html
        figure_output.value = figure_bytes
        details_output.value = impact_table + mode_table
        output.layout.display = ""
        exports.save_outputs(
            f"2_2_parameter_playground_{year}", png=figure_bytes,
            tables={
                "impacts": impacts,
                "modal_split": modes,
                "parameters": pd.DataFrame({
                    "Nominal": state["reference"]["parameters"],
                    "Selected": state["result"]["parameters"],
                }),
                "settings": pd.Series({"stage": stage, "year": year}, name="value"),
            },
        )

    def run(_: Any = None) -> None:
        if state["closed"] or state["running"] or configuration_errors:
            return
        method = "MSA"
        state["running"] = True
        run_button.disabled = True
        selected = selected_values()
        nominal = {name: float(spec["value"]) for name, spec in specs.items()}
        selected_key = (method, configuration_key(selected))
        nominal_key = (method, configuration_key(nominal))
        try:
            method_label = "coupled MSA"
            status.value = (
                "<span style='color:#2b5c8f'><b>Calculating...</b> "
                f"Cached components will be reused ({method_label}).</span>"
            )
            if state["corridor"] is None:
                state["corridor"] = tmi.build_corridor_context(
                    context,
                    corridor_municipalities=corridor_municipalities,
                    name="parameter playground corridor",
                )

            playground_assignment_settings = {"method": "MSA", "modal_feedback": True}
            parallel_years = True

            # Compute the nominal reference once. It remains fixed while the
            # user explores alternative slider combinations.
            if nominal_key not in state["cache"]:
                state["cache"][nominal_key] = tmi.evaluate_parameter_playground(
                    context, stage_specs,
                    corridor_zone_ids=corridor_zone_ids,
                    corridor_municipalities=corridor_municipalities,
                    values=nominal, slider_specs=specs, stage=stage,
                    simulation_params=simulation_params,
                    calculation_cache=state["calculation_cache"],
                    corridor_context=state["corridor"], parallel_years=parallel_years,
                    precomputed_year_assignments=reference_year_assignments,
                    simulation_assignment_settings=playground_assignment_settings,
                )
            state["reference"] = state["cache"][nominal_key]

            if selected_key not in state["cache"]:
                state["cache"][selected_key] = tmi.evaluate_parameter_playground(
                    context, stage_specs,
                    corridor_zone_ids=corridor_zone_ids,
                    corridor_municipalities=corridor_municipalities,
                    values=selected, slider_specs=specs, stage=stage,
                    simulation_params=simulation_params,
                    calculation_cache=state["calculation_cache"],
                    corridor_context=state["corridor"], parallel_years=parallel_years,
                    simulation_assignment_settings=playground_assignment_settings,
                )
                while len(state["cache"]) > int(state["max_cached_runs"]):
                    removable = next(
                        (key for key in state["cache"] if key != nominal_key), None
                    )
                    if removable is None:
                        break
                    state["cache"].pop(removable)
            state["result"] = state["cache"][selected_key]
            state["calculation_method"] = state["result"]["calculation_method"]
            hits = state["result"].get("cache_hits", {})
            reused = [name.replace("_", " ") for name, hit in hits.items() if hit]
            status.value = (
                "<span style='color:#2e7d32'><b>Ready.</b>"
                f" Method: {method_label}."
                + (" Reused: " + ", ".join(reused) + "." if reused else "")
                + " Change the year selector without recalculating.</span>"
            )
            render()
        except Exception as error:
            status.value = (
                "<span style='color:#b71c1c'><b>Parameter comparison failed:</b> "
                f"<code>{type(error).__name__}: {error}</code></span>"
            )
        finally:
            state["running"] = False
            run_button.disabled = bool(configuration_errors)

    def reset(_: Any) -> None:
        for name, slider in sliders.items():
            slider.value = specs[name]["value"]
        update_pending()

    run_button.on_click(run)
    reset_button.on_click(reset)
    def change_year(_: Any) -> None:
        render()

    year_selector.observe(change_year, names="value")
    for slider in sliders.values():
        slider.observe(update_pending, names="value")
    update_pending()
    note = widgets.HTML(
        "Coupled MSA evaluates the selected physical changes against the nominal "
        "configuration. Completed calculations are cached. "
        "Controls omitted from control_groups keep their nominal values."
    )
    configuration_notice = widgets.HTML(
        "<div style='color:#b71c1c'><b>Check control_groups in the notebook:</b><ul>"
        + "".join(f"<li>{html.escape(message)}</li>" for message in configuration_errors)
        + "</ul>Correct the configuration and rerun this dashboard cell.</div>"
        if configuration_errors else ""
    )
    catalog_frame = pd.DataFrame([
        {"Key": name, "Default group": spec["group"], "Meaning": spec["description"],
         "Nominal": spec["value"], "Visible": name in visible_names}
        for name, spec in specs.items()
    ])
    state["supported_controls"] = catalog_frame
    catalog = widgets.Accordion(children=[widgets.HTML(catalog_frame.to_html(index=False))])
    catalog.set_title(0, "Supported controls and nominal values")
    catalog.selected_index = None
    exports.save_outputs("2_2_parameter_playground_controls", tables={"": catalog_frame})
    controls = widgets.HBox([
        run_button, reset_button, year_selector
    ])
    ui = widgets.VBox([
        note, configuration_notice, accordion, catalog, pending, controls, status, output,
    ])

    def close() -> None:
        state["closed"] = True
        run_button.on_click(run, remove=True)
        reset_button.on_click(reset, remove=True)
        year_selector.unobserve(change_year, names="value")
        for slider in sliders.values():
            slider.unobserve(update_pending, names="value")
        state["cache"].clear()
        state["calculation_cache"].clear()
        state["result"] = state["reference"] = state["corridor"] = None
        _close_stage_widgets(ui)

    state.update(close=close, render=render, output=output,
                 controls={"run": run_button, "reset": reset_button, "year": year_selector})
    return ui, state


def plot_parameter_objective_sweeps(
    sweep_results: Mapping[str, pd.DataFrame],
    slider_specs: Mapping[str, Mapping[str, Any]],
) -> Any:
    """Show Year-40 sensitivity and a selectable detailed response curve."""

    import io

    import ipywidgets as widgets
    import matplotlib.pyplot as plt

    def figure_png(figure: Any) -> bytes:
        buffer = io.BytesIO()
        try:
            figure.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
            return buffer.getvalue()
        finally:
            plt.close(figure)

    parameters = list(sweep_results)
    if not parameters:
        raise ValueError("No parameter sweep results were supplied.")
    objectives = [
        ("avg_travel_time_min", "Average travel time", 1.0, "min"),
        ("congestion_delay_hours", "Passenger congestion delay", 1.0, "person-h/peak hour"),
        ("co2_tonnes", "Tailpipe CO2", 1.0, "t/peak hour"),
        ("total_demand", "Passenger demand", 1.0, "trips/peak hour"),
        ("car_trips", "Car trips", 1.0, "trips"),
        ("car_vkt", "Car vehicle-kilometres", 1.0, "veh-km"),
        ("car_share_trips", "Car share", 100.0, "%"),
        ("pt_share_trips", "Public-transport share", 100.0, "%"),
        ("bike_share_trips", "Bicycle share", 100.0, "%"),
        ("walk_share_trips", "Walking share", 100.0, "%"),
    ]
    method = str(next(iter(sweep_results.values())).attrs.get(
        "assignment_method", "MSA"
    ))

    # Compare each Year-40 curve with its nominal Year-40 point.
    sensitivity = np.zeros((len(objectives), len(parameters)), dtype=float)
    for column, parameter in enumerate(parameters):
        frame = sweep_results[parameter]
        nominal = float(slider_specs[parameter]["value"])
        nominal_row = frame.iloc[(frame["value"] - nominal).abs().argmin()]
        for row, (key, _, scale, _) in enumerate(objectives):
            values = frame[f"year40__{key}"].to_numpy(dtype=float) * scale
            baseline = float(nominal_row[f"year40__{key}"]) * scale
            denominator = max(abs(baseline), 1e-9)
            sensitivity[row, column] = np.max(np.abs(100.0 * (values - baseline) / denominator))

    fig, axis = plt.subplots(figsize=(max(12.0, 1.05 * len(parameters)), 7.0))
    image = axis.imshow(sensitivity, cmap="YlOrRd", aspect="auto", vmin=0.0)
    axis.set_xticks(range(len(parameters)), labels=parameters, rotation=55, ha="right")
    axis.set_yticks(range(len(objectives)), labels=[item[1] for item in objectives])
    axis.set_title(
        "Year 40: maximum response across each range (% of nominal Stage 0)"
    )
    for row in range(sensitivity.shape[0]):
        for column in range(sensitivity.shape[1]):
            value = sensitivity[row, column]
            axis.text(
                column, row, f"{value:.1f}", ha="center", va="center",
                fontsize=8, color="white" if value > sensitivity.max() * 0.55 else "black",
            )
    fig.colorbar(
        image, ax=axis, label="Maximum absolute change from Stage 0 default (%)"
    )
    fig.tight_layout()
    heatmap = widgets.Image(value=figure_png(fig), format="png", layout=widgets.Layout(width="100%"))
    exports.save_outputs(
        "2_3_parameter_sensitivity", png=heatmap.value,
        tables={
            "maximum_change_pct": pd.DataFrame(sensitivity, index=[item[1] for item in objectives], columns=parameters),
            **{f"sweep_{parameter}": frame for parameter, frame in sweep_results.items()},
        },
    )

    selector = widgets.Dropdown(
        options=parameters, value=parameters[0], description="Inspect:",
        layout=widgets.Layout(width="430px"),
    )
    detail_image = widgets.Image(format="png", layout=widgets.Layout(width="100%"))

    def draw_detail(change: Any = None) -> None:
        parameter = str(selector.value)
        frame = sweep_results[parameter]
        nominal = float(slider_specs[parameter]["value"])
        fig, axes = plt.subplots(3, 4, figsize=(14.0, 9.0), squeeze=False)
        for axis, (key, label, scale, unit) in zip(axes.flat, objectives):
            axis.plot(
                frame["value"], frame[f"year40__{key}"] * scale,
                marker="o", color="#F58518", label="Year 40",
            )
            axis.axvline(
                nominal, color="black", linestyle=":", alpha=0.7,
                label="Stage 0 default" if axis is axes.flat[0] else None,
            )
            axis.set(title=label, xlabel=parameter, ylabel=unit)
            axis.grid(alpha=0.25)
        for axis in axes.flat[len(objectives):]:
            axis.set_visible(False)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.suptitle(
            f"Year-40 response to {parameter}",
            x=0.01, y=0.995, ha="left", fontweight="bold",
        )
        fig.legend(
            handles, labels, loc="upper right",
            bbox_to_anchor=(0.99, 0.995), ncol=2, frameon=False,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        detail_image.value = figure_png(fig)
        exports.save_outputs(
            "2_3_parameter_response_year40", png=detail_image.value,
            tables={"": frame, "settings": pd.Series({
                "parameter": parameter, "nominal": nominal,
                "year": 40, "assignment_method": method,
            }, name="value")},
        )

    selector.observe(draw_detail, names="value")
    draw_detail()
    note = widgets.HTML(
        f"<b>Year-40 {method} analysis:</b> congestion, updated skims and modal feedback are included."
    )
    return widgets.VBox([note, heatmap, selector, detail_image])


def _stage_selector_point(zones: Any, selector: Any) -> Any | None:
    """Resolve one stage selector to a WGS84 representative point."""

    import geopandas as gpd

    selector = selector if isinstance(selector, Mapping) else {"grid_id": selector}
    selected = np.ones(len(zones), dtype=bool)
    for column, requested in selector.items():
        if column not in zones:
            return None
        values = requested if isinstance(requested, (list, tuple, set)) else [requested]
        selected &= zones[column].astype(str).isin([str(value) for value in values]).to_numpy()
    matched = zones.loc[selected]
    if matched.empty:
        return None
    point = matched.geometry.union_all().centroid
    return gpd.GeoSeries([point], crs=zones.crs).to_crs(4326).iloc[0]


def stage_intervention_scope_map(
    context: tmi.TransportContext,
    corridor: tmi.CorridorContext,
    stage_spec: Mapping[str, Any],
    *,
    stage_id: int | None = None,
    height: int = 620,
    boundary_level: str = "Municipality",
    show_zone_boundaries: bool = False,
    show_centroids: bool = True,
) -> Any:
    """Map the OD scopes and station locations encoded in one stage.

    Straight lines describe affected OD relations, not assigned routes. Road
    interventions remain link-based and are shown by the road-network view.
    """

    import geopandas as gpd
    import plotly.graph_objects as go

    colours = {
        "railway_expansions": "#2864b7",
        "bike_highways": "#2b9348",
        "mobility_hubs": "#7a2e8e",
    }
    labels = {
        "railway_expansions": "PT OD intervention",
        "bike_highways": "Bicycle OD intervention",
        "mobility_hubs": "Mobility hub",
    }
    zones = context.zones
    figure = go.Figure()
    # `_stage_selector_point` unions every matched zone polygon from scratch
    # on each call. Stage interventions can repeat the same municipality/zone
    # selector many times (e.g. a fully-connected area_pairs list), so cache
    # the resolved point per selector for the lifetime of this map build.
    _point_cache: dict[Any, Any] = {}

    def selector_point(selector: Any) -> Any:
        key = tuple(sorted(selector.items())) if isinstance(selector, Mapping) else selector
        if key not in _point_cache:
            _point_cache[key] = _stage_selector_point(zones, selector)
        return _point_cache[key]

    boundaries = _prepare_zone_boundaries(
        zones, zone_ids=corridor.zone_ids, level=boundary_level
    )
    boundary_trace = _plotly_zone_boundary_trace(
        boundaries, visible=bool(show_zone_boundaries)
    )
    if boundary_trace is not None:
        boundary_trace.showlegend = False
        figure.add_trace(boundary_trace)

    # A light corridor polygon provides spatial context without implying that
    # every road or OD pair inside it is modified.
    polygon = gpd.GeoSeries([corridor.polygon], crs=corridor.zones.crs).to_crs(4326).iloc[0]
    polygon_parts = list(polygon.geoms) if hasattr(polygon, "geoms") else [polygon]
    for part in polygon_parts:
        lon, lat = part.exterior.xy
        figure.add_trace(go.Scattermap(
            lon=list(lon), lat=list(lat), fill="toself",
            fillcolor="rgba(80,100,120,0.08)",
            line={"color": "rgba(80,100,120,0.55)", "width": 1},
            hoverinfo="skip", showlegend=False,
        ))

    legend_seen: set[str] = set()
    for intervention_type in ("railway_expansions", "bike_highways"):
        for intervention in stage_spec.get(intervention_type, []):
            effect_text = ", ".join(
                f"{key}: {value}%" for key, value in intervention.get("effects", {}).items()
                if float(value) != 0.0
            ) or "No numerical effect"
            for pair in intervention.get("area_pairs", []):
                origin = selector_point(pair.get("origin", {}))
                destination = selector_point(pair.get("destination", {}))
                if origin is None or destination is None:
                    continue
                label = labels[intervention_type]
                figure.add_trace(go.Scattermap(
                    lon=[origin.x, destination.x], lat=[origin.y, destination.y],
                    mode="lines+markers" if show_centroids else "lines",
                    line={"color": colours[intervention_type], "width": 3},
                    marker={"size": 6, "color": colours[intervention_type]},
                    text=[
                        f"{intervention.get('name', label)}<br>{effect_text}<br>"
                        f"Both directions: {bool(intervention.get('both_directions', True))}"
                    ] * 2,
                    hovertemplate="%{text}<extra></extra>",
                    name=label, showlegend=label not in legend_seen,
                ))
                legend_seen.add(label)
            for pair in intervention.get("od_pairs", []):
                if len(pair) != 2:
                    continue
                origin = selector_point(pair[0])
                destination = selector_point(pair[1])
                if origin is None or destination is None:
                    continue
                label = labels[intervention_type]
                figure.add_trace(go.Scattermap(
                    lon=[origin.x, destination.x], lat=[origin.y, destination.y],
                    mode="lines+markers" if show_centroids else "lines",
                    line={"color": colours[intervention_type], "width": 3},
                    text=[f"{intervention.get('name', label)}<br>{effect_text}"] * 2,
                    hovertemplate="%{text}<extra></extra>",
                    name=label, showlegend=label not in legend_seen,
                ))
                legend_seen.add(label)

    for intervention in stage_spec.get("mobility_hubs", []):
        points, hover = [], []
        effect_text = ", ".join(
            f"{key}: {value}%" for key, value in intervention.get("effects", {}).items()
            if float(value) != 0.0
        ) or "No numerical effect"
        for selector in intervention.get("zones", []):
            point = selector_point(selector)
            if point is not None:
                points.append(point)
                hover.append(f"{intervention.get('name', 'Mobility hub')}<br>{effect_text}")
        if points:
            label = labels["mobility_hubs"]
            figure.add_trace(go.Scattermap(
                lon=[point.x for point in points], lat=[point.y for point in points],
                mode="markers", marker={"size": 14, "color": colours["mobility_hubs"]},
                text=hover, hovertemplate="%{text}<extra></extra>",
                name=label, showlegend=label not in legend_seen,
            ))
            legend_seen.add(label)

    centre = polygon.centroid
    title = stage_spec.get("name", f"Stage {stage_id}" if stage_id is not None else "Stage")
    if not legend_seen:
        figure.add_annotation(
            text="No OD-based or hub intervention is defined for this stage.",
            x=0.5, y=0.98, xref="paper", yref="paper", showarrow=False,
        )
    figure.update_layout(
        title=f"{title} — intervention scopes (straight desire lines)",
        map={"style": "open-street-map", "center": {"lon": centre.x, "lat": centre.y}, "zoom": 9.2},
        height=int(height), margin={"l": 0, "r": 0, "t": 55, "b": 0},
        legend={"title": "Stage component"},
    )
    return figure


def stage_map_dashboard(
    context: tmi.TransportContext,
    corridor: tmi.CorridorContext,
    stage_specs: Mapping[int, Mapping[str, Any]],
    *,
    height: int = 620,
) -> tuple[Any, dict[str, Any]]:
    """Interactive stage selector with road-network and OD-scope views."""

    import ipywidgets as widgets
    from IPython.display import clear_output, display

    # Open on the first intervention stage so students immediately see an
    # informative map; Stage 0 remains available from the selector.
    first_stage = next(
        (stage_id for stage_id in sorted(stage_specs) if int(stage_id) != 0),
        min(stage_specs),
    )
    state = {"stage": first_stage, "view": "OD intervention scopes"}
    stage = widgets.Dropdown(
        options=[(spec.get("name", f"Stage {sid}"), sid) for sid, spec in sorted(stage_specs.items())],
        value=first_stage, description="Stage:", layout=widgets.Layout(width="520px"),
    )
    view = widgets.ToggleButtons(
        options=["OD intervention scopes", "Road network"],
        value="OD intervention scopes", description="View:",
    )
    show_boundaries = widgets.Checkbox(
        value=False, description="Show zone boundaries", indent=False
    )
    show_centroids = widgets.Checkbox(
        value=True, description="Show centroids", indent=False
    )
    output = widgets.Output()

    def redraw(_: Any = None) -> None:
        state.update(stage=int(stage.value), view=str(view.value))
        with output:
            clear_output(wait=True)
            if view.value == "Road network":
                stage_corridor, audit = tmi.apply_road_capacity_stage(
                    corridor, stage_specs[int(stage.value)]
                )
                display(corridor_explorer(stage_corridor))
                if len(audit):
                    display(audit)
            else:
                display(stage_intervention_scope_map(
                    context, corridor, stage_specs[int(stage.value)],
                    stage_id=int(stage.value), height=height,
                    show_zone_boundaries=bool(show_boundaries.value),
                    show_centroids=bool(show_centroids.value),
                ))

    stage.observe(redraw, names="value")
    view.observe(redraw, names="value")
    show_boundaries.observe(redraw, names="value")
    show_centroids.observe(redraw, names="value")
    redraw()
    return widgets.VBox([
        widgets.HBox([stage, view]),
        widgets.HBox([show_boundaries, show_centroids]),
        output,
    ]), state


def _close_stage_widgets(widget: Any) -> None:
    """Dispose an owned map/panel and its layer models, including observers."""
    seen: set[int] = set()

    def close_owned(item: Any) -> None:
        if item is None or id(item) in seen:
            return
        seen.add(id(item))
        for child in (*getattr(item, "children", ()), *getattr(item, "layers", ())):
            close_owned(child)
        try:
            if hasattr(item, "unobserve_all"):
                item.unobserve_all()
            if hasattr(item, "close"):
                item.close()
        except Exception:
            # An already disconnected frontend must not prevent other owned
            # layer models from being released or a replacement being shown.
            pass

    close_owned(widget)


def _stage_editor_label(stage_id: int, specification: Mapping[str, Any]) -> str:
    return str(specification.get("name", f"Stage {stage_id}"))


def _stage_preview_table(frame: pd.DataFrame, *, empty: str, limit: int = 200) -> str:
    """Bound browser DOM size without discarding the full Python-side table."""
    if frame.empty:
        return f"<i>{empty}</i>"
    note = (
        f"<p>Showing the first {limit:,} of {len(frame):,} rows; the full table remains in the dashboard state.</p>"
        if len(frame) > limit else ""
    )
    return note + frame.head(limit).to_html()


def _road_editor_map(
    corridor: tmi.CorridorContext,
    selected_keys: set[str],
    on_link_click: Any,
    *,
    height: int = 590,
    show_zone_boundaries: bool = False,
    show_centroids: bool = True,
) -> tuple[Any, dict[str, list[Any]]]:
    """Clickable Lonboard road map used by the student link editor."""

    import geopandas as gpd
    from lonboard import PathLayer, PolygonLayer, ScatterplotLayer, SolidPolygonLayer

    columns = [
        "_link_key", "edge_id", "source", "target", "source_id", "target_id",
        "highway", "length_m", "lanes", "capacity_vph", "speed_kph",
        "free_flow_time_min", "capacity_change_pct", "intervention_status",
    ]
    links = _slim_geodataframe(
        corridor.edges, [column for column in columns if column in corridor.edges],
        simplify_m=2.0,
    ).dropna(subset=["geometry"]).reset_index(drop=True)
    selected = links["_link_key"].astype(str).isin(selected_keys).to_numpy()
    modified = links["intervention_status"].eq("modified").to_numpy()
    colours = np.tile(np.array([145, 148, 152, 145], dtype=np.uint8), (len(links), 1))
    colours[modified] = np.array([230, 126, 34, 220], dtype=np.uint8)
    colours[selected] = np.array([32, 92, 175, 245], dtype=np.uint8)
    widths = np.where(selected, 6.0, np.where(modified, 4.0, 1.4)).astype(np.float32)

    link_layer = PathLayer.from_geopandas(
        links, get_color=colours, get_width=widths, width_units="pixels",
        width_min_pixels=1, width_max_pixels=8, pickable=True,
        auto_highlight=True,
        # pyrefly: ignore [unexpected-keyword]
        highlight_color=[20, 20, 20, 220],
    )

    def selected_link(change: dict[str, Any]) -> None:
        index = change.get("new")
        if index is None:
            return
        try:
            position = int(index)
        except (TypeError, ValueError):
            return
        if 0 <= position < len(links):
            on_link_click(str(links.iloc[position]["_link_key"]))

    link_layer.observe(selected_link, names="selected_index")
    layers: list[Any] = []
    layer_groups: dict[str, list[Any]] = {
        "corridor": [], "boundaries": [], "links": [link_layer],
        "centroids": [], "gates": [],
        "link_keys": links["_link_key"].astype(str).tolist(),
        "modified_mask": modified.tolist(),
    }
    if corridor.polygon_gdf is not None and len(corridor.polygon_gdf):
        polygon = _slim_geodataframe(corridor.polygon_gdf, ["name"], simplify_m=5.0)
        polygon_layer = SolidPolygonLayer.from_geopandas(
            polygon, get_fill_color=[40, 110, 170, 18],
            get_line_color=[40, 110, 170, 150], filled=True, pickable=False,
        )
        layers.append(polygon_layer)
        layer_groups["corridor"].append(polygon_layer)
    boundaries = _prepare_zone_boundaries(corridor.zones, level="FSM zone")
    if len(boundaries):
        boundary_layer = PolygonLayer.from_geopandas(
            boundaries,
            get_fill_color=[0, 0, 0, 0],
            get_line_color=[70, 70, 70, 135],
            filled=False,
            stroked=True,
            line_width_min_pixels=1,
            pickable=False,
        )
        boundary_layer.visible = bool(show_zone_boundaries)
        layers.append(boundary_layer)
        layer_groups["boundaries"].append(boundary_layer)
    layers.append(link_layer)
    if corridor.zones is not None and len(corridor.zones):
        zone_points = corridor.zones[["grid_id", "geometry"]].copy()
        zone_points.geometry = zone_points.geometry.centroid
        zone_points = _slim_geodataframe(zone_points, ["grid_id"])
        centroid_layer = ScatterplotLayer.from_geopandas(
            zone_points,
            get_fill_color=[70, 70, 70, 170],
            get_radius=3.0,
            radius_units="pixels",
            radius_min_pixels=2,
            radius_max_pixels=5,
            pickable=True,
        )
        centroid_layer.visible = bool(show_centroids)
        layers.append(centroid_layer)
        layer_groups["centroids"].append(centroid_layer)
    if corridor.gates is not None and len(corridor.gates):
        gates = _slim_geodataframe(
            corridor.gates,
            [column for column in ["gate_id", "direction", "capacity_vph"] if column in corridor.gates],
        ).reset_index(drop=True)
        gate_layer = ScatterplotLayer.from_geopandas(
            gates, get_fill_color=[122, 45, 140, 230], get_radius=6.0,
            radius_units="pixels", radius_min_pixels=4, radius_max_pixels=9,
            pickable=True,
        )
        layers.append(gate_layer)
        layer_groups["gates"].append(gate_layer)
    return (
        _lonboard_map(layers, height=height, show_side_panel=False),
        layer_groups,
    )


def road_link_stage_editor(
    context: tmi.TransportContext,
    corridor: tmi.CorridorContext,
    stage_specs: Mapping[int, Mapping[str, Any]],
    *,
    height: int = 590,
    stage_selector: Any | None = None,
    defer_map: bool = True,
    on_change: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Edit directed road links, with an optional map preview.

    Clicking the Lonboard layer identifies a directed edge. Students may add
    its reverse direction, set absolute capacity/lane/speed values, and inspect
    a live audit table. Every stage owns a corridor copy; Stage 0 is immutable.
    """

    import html

    import ipywidgets as widgets

    baseline = replace(corridor, edges=tmi._prepare_road_edges(corridor.edges))
    editable_stages = [stage for stage in sorted(stage_specs) if int(stage) != 0]
    if not editable_stages:
        raise ValueError("The road editor requires at least one stage after Stage 0.")

    original_specs = deepcopy(stage_specs)
    corridors, audits = {0: baseline}, {0: pd.DataFrame()}
    selected_by_stage: dict[int, set[str]] = {}
    catalogue: dict[int, list[dict[str, Any]]] = {}

    def road_definitions(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
        values = spec.get("road_capacity", []) or []
        return deepcopy([values] if isinstance(values, Mapping) else list(values))
    state: dict[str, Any] = {
        "corridors": corridors, "audits": audits,
        "selected": selected_by_stage, "clicked_link": None,
        "active_stage": editable_stages[0],
        "map_active": not defer_map, "map": None, "map_layers": {},
        "rendering": False, "editing": False, "refreshing": False,
        "catalogue": catalogue,
    }

    def initialize_stage(stage_id: int) -> None:
        """Prepare road state before a shared selector can activate a stage."""

        stage_id = int(stage_id)
        if stage_id == 0 or stage_id in corridors:
            return
        if stage_id not in stage_specs:
            raise KeyError(f"Stage {stage_id} is not present in the shared stage specifications.")
        corridors[stage_id], audits[stage_id] = tmi.apply_road_capacity_stage(
            baseline, stage_specs[stage_id]
        )
        selected_by_stage[stage_id] = set()
        catalogue[stage_id] = [{"item": item, "enabled": True}
                               for item in road_definitions(stage_specs[stage_id])]

    for stage_id in editable_stages:
        initialize_stage(stage_id)
    state["initialize_stage"] = initialize_stage

    owns_stage_selector = stage_selector is None
    stage = stage_selector or widgets.Dropdown(
        options=[(_stage_editor_label(s, stage_specs[s]), s) for s in editable_stages],
        value=editable_stages[0], description="Edit stage:",
        layout=widgets.Layout(width="520px"),
    )
    if stage.value is None or int(stage.value) not in corridors:
        raise ValueError("The shared stage selector references a stage not initialized by the road editor.")
    state["stage_selector"] = stage
    show_boundaries = widgets.Checkbox(
        value=False, description="Show zone boundaries", indent=False
    )
    show_centroids = widgets.Checkbox(
        value=True, description="Show centroids", indent=False
    )
    road_classes = sorted(
        baseline.edges.get(
            "highway", pd.Series("unknown", index=baseline.edges.index)
        ).fillna("unknown").astype(str).unique()
    )
    road_class = widgets.Dropdown(
        options=["All", *road_classes], value="All", description="Road class:",
        layout=widgets.Layout(width="330px"),
    )
    search = widgets.Text(
        description="Find link:", placeholder="edge_id, source, target or road class",
        layout=widgets.Layout(width="520px"), continuous_update=False,
    )
    browser = widgets.Dropdown(description="Result:", layout=widgets.Layout(width="100%"))
    selected_list = widgets.SelectMultiple(
        description="Selected:", rows=5, layout=widgets.Layout(width="100%"),
    )
    clicked_detail = widgets.HTML("<i>Click a road link on the map or choose a search result.</i>")
    add_clicked = widgets.Button(description="Add clicked link", icon="plus")
    add_reverse = widgets.Button(description="Add reverse direction", icon="exchange")
    remove_selected = widgets.Button(description="Remove selected", icon="minus")
    clear_selected = widgets.Button(description="Clear selection", icon="trash")

    change_capacity = widgets.Checkbox(value=True, description="Set capacity")
    capacity = widgets.FloatText(value=1000.0, description="veh/h:")
    change_lanes = widgets.Checkbox(value=False, description="Set lanes")
    lanes = widgets.FloatText(value=1.0, description="lanes:")
    change_speed = widgets.Checkbox(value=False, description="Set speed")
    speed = widgets.FloatText(value=50.0, description="km/h:")
    apply_button = widgets.Button(description="Apply to selected links", button_style="success", icon="check")
    reset_button = widgets.Button(
        description="Reset road link edits", button_style="warning", icon="undo"
    )
    definition = widgets.Dropdown(
        options=[("New road definition", None)], description="Definition:",
        layout=widgets.Layout(width="520px"),
    )
    definition_enabled = widgets.Checkbox(value=True, description="Enabled", disabled=True)
    remove_definition = widgets.Button(description="Remove definition", icon="trash", disabled=True)
    status = widgets.HTML()
    map_status = widgets.HTML()
    retry_map = widgets.Button(description="Preview map", icon="map")
    stage_summary = widgets.HTML(layout=widgets.Layout(width="100%"))
    map_output = widgets.VBox(layout=widgets.Layout(width="100%"))
    # Static HTML rather than an Output+display() pair: an Output holding a
    # display() call can replay its buffered messages when a Tab view
    # reconnects (e.g. switching tabs), duplicating the table.
    audit_output = widgets.HTML(layout=widgets.Layout(width="100%"))

    def current_stage() -> int:
        return int(stage.value)

    def current_edges() -> Any:
        return state["corridors"][current_stage()].edges

    def link_row(key: str) -> pd.Series:
        rows = current_edges().loc[current_edges()["_link_key"].astype(str).eq(str(key))]
        if rows.empty:
            raise KeyError(f"Unknown link: {key}")
        return rows.iloc[0]

    def show_clicked(key: str) -> None:
        row = link_row(key)
        state["clicked_link"] = key
        capacity.value = float(row["capacity_vph"]) if pd.notna(row["capacity_vph"]) else 0.0
        lanes.value = float(row["lanes"]) if pd.notna(row["lanes"]) else 0.0
        speed.value = float(row["speed_kph"]) if pd.notna(row["speed_kph"]) else 0.0
        clicked_detail.value = (
            f"<b>{key}</b><br>{row.get('highway', 'unknown')} | "
            f"{row['source']} → {row['target']} | {row['capacity_vph']:,.0f} veh/h | "
            f"{row['lanes']:.1f} lanes | {row['speed_kph']:.1f} km/h"
        )

    def refresh_browser(_: Any = None) -> None:
        edges = current_edges()
        previous = browser.value
        mask = pd.Series(True, index=edges.index)
        if road_class.value != "All":
            mask &= edges["highway"].fillna("unknown").astype(str).eq(road_class.value)
        query = search.value.strip().lower()
        if query:
            columns = [column for column in ("_link_key", "edge_id", "source", "target", "source_id", "target_id", "highway") if column in edges]
            text = edges[columns].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()
            mask &= text.str.contains(query, regex=False)
        options = []
        for _, row in edges.loc[mask].head(500).iterrows():
            key = str(row["_link_key"])
            options.append((
                f"{key} | {row.get('highway', 'unknown')} | "
                f"{row.get('source')}→{row.get('target')} | "
                f"{row.get('capacity_vph', np.nan):,.0f} veh/h",
                key,
            ))
        browser.options = options or [("No matching links", "")]
        available = [value for _, value in browser.options]
        browser.value = previous if previous in available else available[0]
        # A stage switch can retain the same dropdown value without emitting
        # a value event. Refresh the selected record for the new stage anyway.
        if browser.value:
            show_clicked(str(browser.value))
        else:
            state["clicked_link"] = None
            clicked_detail.value = "<i>No matching road link. Change the search or road class.</i>"

    def refresh_selection() -> None:
        keys = sorted(selected_by_stage[current_stage()])
        selected_list.options = [(key, key) for key in keys]
        groups = state.get("map_layers", {})
        if groups.get("links") and "link_keys" in groups:
            selected = np.isin(groups["link_keys"], keys)
            modified = np.asarray(groups["modified_mask"], dtype=bool)
            colours = np.tile(np.array([145, 148, 152, 145], dtype=np.uint8), (len(selected), 1))
            colours[modified] = [230, 126, 34, 220]
            colours[selected] = [32, 92, 175, 245]
            groups["links"][0].get_color = colours
            groups["links"][0].get_width = np.where(selected, 6.0, np.where(modified, 4.0, 1.4)).astype(np.float32)

    def refresh_audit() -> None:
        audit = state["audits"][current_stage()]
        audit_output.value = _stage_preview_table(audit, empty="No road links are modified in this stage.")
        exports.save_outputs(
            f"3_2_road_edit_{current_stage()}",
            tables={"audit": audit.drop(columns="geometry", errors="ignore"),
                    "settings": pd.Series({"stage_or_package": current_stage(),
                        "name": _stage_editor_label(current_stage(), stage_specs[current_stage()])}, name="value")},
        )

    def refresh_stage_summary() -> None:
        stage_id = current_stage()
        definitions = catalogue[stage_id]
        stage_name = html.escape(_stage_editor_label(stage_id, stage_specs[stage_id]))
        stage_summary.value = (
            f"<b>{stage_name}</b> — {sum(item['enabled'] for item in definitions)} of {len(definitions)} road definitions enabled; "
            f"{len(state['audits'][stage_id])} modified directed link(s)."
        )

    def selected_definition() -> dict[str, Any] | None:
        index = definition.value
        values = catalogue[current_stage()]
        return values[index] if index is not None and 0 <= index < len(values) else None

    def load_definition(_: Any = None) -> None:
        if state["refreshing"]:
            return
        state["refreshing"] = True
        try:
            entry = selected_definition()
            definition_enabled.disabled = remove_definition.disabled = entry is None
            definition_enabled.value = bool(entry["enabled"]) if entry else True
            if entry:
                item, edges = entry["item"], current_edges()
                selected = edges["edge_id"].astype(str).isin(map(str, item.get("edge_ids", [])))
                for pair in item.get("source_target_pairs", item.get("node_pairs", [])):
                    source, target = (pair.get("source"), pair.get("target")) if isinstance(pair, Mapping) else pair
                    selected |= edges["source"].astype(str).eq(str(source)) & edges["target"].astype(str).eq(str(target))
                    if item.get("both_directions", False):
                        selected |= edges["source"].astype(str).eq(str(target)) & edges["target"].astype(str).eq(str(source))
                selected_by_stage[current_stage()] = set(edges.loc[selected, "_link_key"].astype(str))
                effects = item.get("effects", {})
                for checked, field, widget in ((change_capacity, "capacity_vph", capacity),
                                               (change_lanes, "lanes", lanes),
                                               (change_speed, "speed_kph", speed)):
                    checked.value = field in effects
                    if field in effects:
                        widget.value = float(effects[field])
                refresh_selection()
            apply_button.description = "Update selected definition" if entry else "Apply to selected links"
        finally:
            state["refreshing"] = False

    def refresh_definitions(selected: int | None = None) -> None:
        state["refreshing"] = True
        try:
            values = catalogue[current_stage()]
            definition.options = [("New road definition", None)] + [
                (f"{'On' if item['enabled'] else 'Off'} · {item['item'].get('name', 'Road definition')}", i)
                for i, item in enumerate(values)
            ]
            definition.value = selected if selected is not None and selected < len(values) else None
        finally:
            state["refreshing"] = False
        load_definition()

    def close_map() -> None:
        """Release the hidden WebGL widget before another page is activated."""

        old_children = map_output.children
        map_output.children = ()
        state["map"] = None
        state["map_layers"] = {}
        for child in old_children:
            _close_stage_widgets(child)

    def refresh_map(_: Any = None, *, force: bool = False) -> None:
        state["active_stage"] = current_stage()
        if not force:
            map_status.value = "<i>Preview is optional. Click Preview map to show current links.</i>"
            return
        if state["rendering"]:
            return
        state["rendering"] = True
        retry_map.disabled = True
        map_status.value = "<i>Loading road links…</i>"
        try:
            figure, layer_groups = _road_editor_map(
                state["corridors"][current_stage()],
                selected_by_stage[current_stage()], show_clicked, height=height,
                show_zone_boundaries=bool(show_boundaries.value),
                show_centroids=bool(show_centroids.value),
            )
            previous = map_output.children
            state["map"] = figure
            state["map_layers"] = layer_groups
            map_output.children = (figure,)
            for widget in previous:
                _close_stage_widgets(widget)
            state.pop("last_error", None)
            map_status.value = ""
        except Exception as error:
            state["last_error"] = f"{type(error).__name__}: {error}"
            map_status.value = (
                "<span style='color:#b71c1c'><b>Road map unavailable.</b> "
                f"{html.escape(state['last_error'])} Your edits are retained; use Preview map to retry.</span>"
            )
        finally:
            state["rendering"] = False
            retry_map.disabled = False

    def set_layer_visibility(change: dict[str, Any], group: str) -> None:
        """Toggle an existing road-map layer without rebuilding the network map."""

        for layer in state.get("map_layers", {}).get(group, []):
            layer.visible = bool(change["new"])

    def activate() -> None:
        if state["map_active"] and state.get("map") is not None:
            return
        state["map_active"] = True

    def deactivate() -> None:
        state["map_active"] = False
        close_map()
        map_output.children = (
            widgets.HTML("<i>Map paused while another editor page is open.</i>"),
        )

    state["activate"] = activate
    state["deactivate"] = deactivate

    def publish() -> None:
        stage_id = current_stage()
        values = [deepcopy(entry["item"]) for entry in catalogue[stage_id] if entry["enabled"]]
        updated = {**stage_specs[stage_id], "road_capacity": values}
        new_corridor, new_audit = tmi.apply_road_capacity_stage(baseline, updated)
        stage_specs[stage_id]["road_capacity"] = values
        state["corridors"][stage_id], state["audits"][stage_id] = new_corridor, new_audit
        if on_change is not None:
            on_change(stage_id, "road_capacity")

    def refresh_values(selected: int | None = None) -> None:
        refresh_selection(); refresh_browser(); refresh_audit(); refresh_stage_summary()
        refresh_definitions(selected)
        refresh_map()

    def toggle_definition(_: Any = None) -> None:
        entry = selected_definition()
        if entry is None:
            return
        index = definition.value
        entry["enabled"] = bool(definition_enabled.value)
        publish()
        refresh_values(index)
        status.value = "Enabled road definitions are included in the working stages; disabled definitions are omitted."

    def delete_definition(_: Any = None) -> None:
        if selected_definition() is None:
            return
        item = catalogue[current_stage()].pop(int(definition.value))
        publish()
        selected_by_stage[current_stage()].clear()
        refresh_values()
        status.value = f"Removed {html.escape(item['item'].get('name', 'road definition'))} from the working stage."

    def add_one(_: Any = None) -> None:
        key = state.get("clicked_link")
        if not key:
            status.value = (
                "<span style='color:#b71c1c'>Click a link on the map or pick one "
                "from the search results first.</span>"
            )
            return
        selected_by_stage[current_stage()].add(str(key))
        refresh_selection()

    def add_other_direction(_: Any = None) -> None:
        key = state.get("clicked_link")
        if not key:
            status.value = (
                "<span style='color:#b71c1c'>Click a link on the map or pick one "
                "from the search results first.</span>"
            )
            return
        row, edges = link_row(key), current_edges()
        reverse = edges.loc[
            edges["source"].astype(str).eq(str(row["target"]))
            & edges["target"].astype(str).eq(str(row["source"])), "_link_key"
        ].astype(str)
        selected_by_stage[current_stage()].update([str(key), *reverse.tolist()])
        status.value = f"<b>{len(reverse)}</b> reverse-direction link(s) found."
        refresh_selection()

    def remove_many(_: Any = None) -> None:
        selected_by_stage[current_stage()].difference_update(map(str, selected_list.value))
        refresh_selection()

    def clear_many(_: Any = None) -> None:
        selected_by_stage[current_stage()].clear()
        refresh_selection()

    def apply_values(_: Any = None) -> None:
        keys = selected_by_stage[current_stage()]
        if not keys:
            raise ValueError("Select at least one link.")
        if not any((change_capacity.value, change_lanes.value, change_speed.value)):
            raise ValueError("Select an attribute to change.")
        effects = {}
        for checked, field, widget in ((change_capacity, "capacity_vph", capacity),
                                       (change_lanes, "lanes", lanes),
                                       (change_speed, "speed_kph", speed)):
            if checked.value:
                if not np.isfinite(widget.value) or widget.value <= 0:
                    raise ValueError(f"{field} must be finite and greater than zero.")
                effects[field] = float(widget.value)
        edges = current_edges()
        mask = edges["_link_key"].astype(str).isin(keys)
        edge_ids = sorted(edges.loc[mask, "edge_id"].astype(str))
        values = catalogue[current_stage()]
        match = definition.value
        if match is None:
            match = next((i for i, entry in enumerate(values)
                          if entry["item"].get("name", "").startswith("Interactive road edit")
                          and sorted(entry["item"].get("edge_ids", [])) == edge_ids), None)
        item = deepcopy(values[match]["item"]) if match is not None else {
            "name": f"Interactive road edit {len(values) + 1}", "effects": {},
        }
        for old_scope in ("source_target_pairs", "node_pairs", "both_directions"):
            item.pop(old_scope, None)
        item["edge_ids"] = edge_ids
        combined = {**item.get("effects", {}), **effects}
        for absolute, relative in (("capacity_vph", "capacity_increase_pct"),
                                   ("lanes", "lanes_change"), ("speed_kph", "speed_increase_pct")):
            if absolute in effects:
                combined.pop(relative, None)
        item["effects"] = combined
        entry = {"item": item, "enabled": True}
        if match is None:
            values.append(entry)
            match = len(values) - 1
        else:
            values[match] = entry
        publish()
        refresh_values(match)
        stage_name = html.escape(_stage_editor_label(current_stage(), stage_specs[current_stage()]))
        status.value = f"<span style='color:#1b5e20'><b>{len(edge_ids)} directed link(s) updated for {stage_name}.</b></span>"

    def reset_stage(_: Any = None) -> None:
        stage_id = current_stage()
        catalogue[stage_id] = [{"item": item, "enabled": True}
                               for item in road_definitions(original_specs[stage_id])]
        publish()
        selected_by_stage[stage_id].clear()
        refresh_values()
        status.value = f"{html.escape(_stage_editor_label(stage_id, stage_specs[stage_id]))} reset to its loaded definition."

    def choose_browser(change: dict[str, Any]) -> None:
        if change.get("new"):
            show_clicked(str(change["new"]))

    def change_stage(_: Any = None) -> None:
        initialize_stage(current_stage())
        state["clicked_link"] = None
        clicked_detail.value = "<i>Click a road link on the map or choose a search result.</i>"
        status.value = ""
        refresh_values()

    stage.observe(change_stage, names="value")
    show_boundaries.observe(
        lambda change: set_layer_visibility(change, "boundaries"), names="value"
    )
    show_centroids.observe(
        lambda change: set_layer_visibility(change, "centroids"), names="value"
    )
    road_class.observe(refresh_browser, names="value")
    search.observe(refresh_browser, names="value")
    browser.observe(choose_browser, names="value")
    def guarded_action(action: Any) -> Any:
        def run(trigger: Any = None) -> None:
            if state["editing"] or state["refreshing"]:
                return
            stage_id = current_stage()
            before_catalogue = deepcopy(catalogue[stage_id])
            before_spec = deepcopy(stage_specs[stage_id])
            before_corridor, before_audit = corridors[stage_id], audits[stage_id]
            before_selected = set(selected_by_stage[stage_id])
            before_definition = definition.value
            state["editing"] = True
            button = trigger if isinstance(trigger, widgets.Button) else None
            if button is not None:
                button.disabled = True
            state.pop("last_action_error", None)
            try:
                action(trigger)
            except Exception as error:
                catalogue[stage_id] = before_catalogue
                stage_specs[stage_id].clear(); stage_specs[stage_id].update(before_spec)
                corridors[stage_id], audits[stage_id] = before_corridor, before_audit
                selected_by_stage[stage_id] = before_selected
                refresh_values(before_definition)
                state["last_action_error"] = f"{type(error).__name__}: {error}"
                status.value = (
                    "<span style='color:#b71c1c'><b>Could not complete this road edit.</b> "
                    f"{html.escape(state['last_action_error'])} The previous definitions are retained.</span>"
                )
            finally:
                state["editing"] = False
                if button is not None:
                    button.disabled = False
                remove_definition.disabled = definition_enabled.disabled = selected_definition() is None
        return run

    definition.observe(load_definition, names="value")
    definition_enabled.observe(guarded_action(toggle_definition), names="value")
    remove_definition.on_click(guarded_action(delete_definition))

    add_clicked.on_click(guarded_action(add_one))
    add_reverse.on_click(guarded_action(add_other_direction))
    remove_selected.on_click(guarded_action(remove_many))
    clear_selected.on_click(guarded_action(clear_many))
    apply_button.on_click(guarded_action(apply_values))
    reset_button.on_click(guarded_action(reset_stage))
    retry_map.on_click(lambda _: refresh_map(force=True))

    editor = widgets.VBox([
        widgets.HBox([definition, definition_enabled, remove_definition]),
        widgets.HTML("Select an existing definition to enable, disable or update it. Applying values uses the currently selected links."),
        widgets.HTML("<b>Select links</b> — orange links are already modified; blue links are selected."),
        widgets.HBox([road_class, search]), browser, clicked_detail,
        widgets.HBox([add_clicked, add_reverse, remove_selected, clear_selected]),
        selected_list,
        widgets.HTML("<b>Enter new absolute values</b> — unchecked attributes remain unchanged."),
        widgets.HBox([change_capacity, capacity, change_lanes, lanes, change_speed, speed]),
        widgets.HBox([apply_button, reset_button]), status,
    ])
    tabs = widgets.Tab(children=[editor, audit_output])
    tabs.set_title(0, "Edit links")
    tabs.set_title(1, "Before/after audit")
    refresh_selection(); refresh_browser(); refresh_audit(); refresh_stage_summary(); refresh_definitions()
    if state["map_active"]:
        refresh_map()
    else:
        map_output.children = (
            widgets.HTML("<i>Map is optional. Use the search above, or click Preview map.</i>"),
        )
    ui_children = ([stage] if owns_stage_selector else []) + [
        stage_summary,
        widgets.HBox([retry_map, map_status]),
        widgets.HBox([show_boundaries, show_centroids]),
        map_output,
        tabs,
    ]
    ui = widgets.VBox(ui_children)

    def close() -> None:
        state["map_active"] = False
        close_map()
        stage.unobserve(change_stage, names="value")
        _close_stage_widgets(ui)

    state["close"] = close
    state["preview_map"] = lambda: refresh_map(force=True)
    state["controls"] = {"preview": retry_map, "apply": apply_button, "reset": reset_button,
                         "search": search, "browser": browser, "capacity": capacity, "status": status,
                         "definition": definition, "definition_enabled": definition_enabled,
                         "remove_definition": remove_definition, "change_capacity": change_capacity,
                         "change_lanes": change_lanes, "lanes": lanes, "change_speed": change_speed, "speed": speed}
    return ui, state


def road_network_difference_dashboard(
    baseline: tmi.CorridorContext,
    modified: tmi.CorridorContext,
    *,
    height: int = 590,
) -> Any:
    """PTV-Visum-style bandwidth map and audit tables for network changes."""

    import ipywidgets as widgets
    from IPython.display import clear_output, display
    from lonboard import PathLayer, PolygonLayer, ScatterplotLayer, SolidPolygonLayer

    base = tmi._prepare_road_edges(baseline.edges).set_index("_link_key", drop=False)
    after = tmi._prepare_road_edges(modified.edges).set_index("_link_key", drop=False)
    missing = set(base.index).symmetric_difference(after.index)
    if missing:
        raise ValueError("Baseline and modified networks do not contain the same directed links.")

    differences = base.copy()
    metrics = {
        "Capacity (veh/h)": ("capacity_vph", True),
        "Lanes": ("lanes", True),
        "Free-flow speed (km/h)": ("speed_kph", True),
        "Free-flow time (min)": ("free_flow_time_min", False),
    }
    for column, _ in metrics.values():
        differences[f"before_{column}"] = base[column]
        differences[f"after_{column}"] = after.loc[base.index, column]
        differences[f"delta_{column}"] = differences[f"after_{column}"] - differences[f"before_{column}"]
    changed = np.zeros(len(differences), dtype=bool)
    for column, _ in metrics.values():
        changed |= ~np.isclose(
            differences[f"before_{column}"].fillna(-1.0),
            differences[f"after_{column}"].fillna(-1.0),
        )
    differences["changed"] = changed
    changed_table = differences.loc[changed].copy()

    metric = widgets.Dropdown(options=list(metrics), description="Difference:", layout=widgets.Layout(width="430px"))
    show_boundaries = widgets.Checkbox(
        value=False, description="Show zone boundaries", indent=False
    )
    show_centroids = widgets.Checkbox(
        value=True, description="Show centroids", indent=False
    )
    map_output = widgets.Output()

    def draw(_: Any = None) -> None:
        column, positive_is_improvement = metrics[metric.value]
        with map_output:
            clear_output(wait=True)
            background = _slim_geodataframe(differences, ["_link_key"], simplify_m=2.0)
            base_layer = PathLayer.from_geopandas(
                background, get_color=[145, 148, 152, 80], get_width=1.0,
                width_units="pixels", width_min_pixels=1, pickable=False,
            )
            layers: list[Any] = []
            if baseline.polygon_gdf is not None and len(baseline.polygon_gdf):
                polygon = _slim_geodataframe(baseline.polygon_gdf, ["name"], simplify_m=5.0)
                layers.append(SolidPolygonLayer.from_geopandas(
                    polygon, get_fill_color=[40, 110, 170, 15],
                    get_line_color=[40, 110, 170, 120], filled=True, pickable=False,
                ))
            boundaries = _prepare_zone_boundaries(baseline.zones, level="FSM zone")
            if len(boundaries):
                boundary_layer = PolygonLayer.from_geopandas(
                    boundaries,
                    get_fill_color=[0, 0, 0, 0],
                    get_line_color=[70, 70, 70, 135],
                    filled=False,
                    stroked=True,
                    line_width_min_pixels=1,
                    pickable=False,
                )
                boundary_layer.visible = bool(show_boundaries.value)
                layers.append(boundary_layer)
            layers.append(base_layer)
            if len(changed_table):
                delta_column = f"delta_{column}"
                popup = [
                    "_link_key", "edge_id", "source", "target", "highway",
                    f"before_{column}", f"after_{column}", delta_column,
                ]
                links = _slim_geodataframe(
                    changed_table, [value for value in popup if value in changed_table],
                    simplify_m=1.0,
                )
                delta = links[delta_column].fillna(0.0).to_numpy(dtype=float)
                improvement = delta >= 0.0 if positive_is_improvement else delta <= 0.0
                colours = np.where(
                    improvement[:, None],
                    np.array([35, 155, 86, 235], dtype=np.uint8),
                    np.array([205, 55, 55, 235], dtype=np.uint8),
                )
                magnitude = np.abs(delta)
                upper = max(float(np.quantile(magnitude, 0.95)), 1e-9)
                widths = (2.5 + 6.0 * np.sqrt(np.clip(magnitude / upper, 0.0, 1.0))).astype(np.float32)
                layers.append(PathLayer.from_geopandas(
                    links, get_color=colours, get_width=widths,
                    width_units="pixels", width_min_pixels=2, width_max_pixels=9,
                    pickable=True, auto_highlight=True,
                    # pyrefly: ignore [unexpected-keyword]
                    highlight_color=[20, 20, 20, 230],
                ))
            if baseline.zones is not None and len(baseline.zones):
                zone_points = baseline.zones[["grid_id", "geometry"]].copy()
                zone_points.geometry = zone_points.geometry.centroid
                zone_points = _slim_geodataframe(zone_points, ["grid_id"])
                centroid_layer = ScatterplotLayer.from_geopandas(
                    zone_points,
                    get_fill_color=[70, 70, 70, 170],
                    get_radius=3.0,
                    radius_units="pixels",
                    radius_min_pixels=2,
                    radius_max_pixels=5,
                    pickable=True,
                )
                centroid_layer.visible = bool(show_centroids.value)
                layers.append(centroid_layer)
            display(_lonboard_map(layers, height=height))
            display(widgets.HTML(
                "<span style='color:#239b56'><b>Green:</b> improvement</span> &nbsp; "
                "<span style='color:#cd3737'><b>Red:</b> deterioration</span>. "
                "Band width represents the absolute change. Click a changed link for values."
            ))

    metric.observe(draw, names="value")
    show_boundaries.observe(draw, names="value")
    show_centroids.observe(draw, names="value")
    draw()
    summary = pd.DataFrame({
        "indicator": ["Modified directed links", "Modified length (km)"],
        "value": [int(changed.sum()), float(changed_table.get("length_m", pd.Series(dtype=float)).sum()) / 1000.0],
    })
    # Static HTML rather than an Output+display() pair: an Output holding a
    # one-shot display() can replay its buffered messages when a Tab view
    # reconnects (e.g. switching tabs), duplicating the table.
    tables = widgets.VBox([
        widgets.HTML(value=summary.to_html(), layout=widgets.Layout(width="100%")),
        widgets.HTML(
            value=changed_table.drop(columns="geometry", errors="ignore").to_html(),
            layout=widgets.Layout(width="100%"),
        ),
    ])
    tabs = widgets.Tab(children=[
        widgets.VBox([metric, widgets.HBox([show_boundaries, show_centroids]), map_output]),
        tables,
    ])
    tabs.set_title(0, "Difference map")
    tabs.set_title(1, "Changed-link table")
    return tabs


_STAGE_OD_ATTRIBUTES = {
    "Public transport": {
        "Total PT journey time (min)": ("time", "pt_walk", "min"),
        "In-vehicle time (min)": ("time", "ivt_pt_walk", "min"),
        "Initial wait (min)": ("time", "initial_wait_pt_walk", "min"),
        "Transfer wait (min)": ("time", "transfer_wait_pt_walk", "min"),
        "Physical transfer (min)": ("time", "transfer_physical_pt_walk", "min"),
        "Access time (min)": ("time", "access_pt_walk", "min"),
        "Egress time (min)": ("time", "egress_pt_walk", "min"),
        "Distance (km)": ("length", "pt_walk", "km"),
    },
    "Bicycle": {
        "Travel time (min)": ("time", "bike", "min"),
        "Distance (km)": ("length", "bike", "km"),
    },
    "Walking": {
        "Travel time (min)": ("time", "walk", "min"),
        "Distance (km)": ("length", "walk", "km"),
    },
    "PT access and hubs": {
        "Access time (min)": ("time", "access_pt_walk", "min"),
        "Egress time (min)": ("time", "egress_pt_walk", "min"),
        "Transfer time (min)": ("time", "transfer_physical_pt_walk", "min"),
    },
}


def _stage_skim_snapshot(
    context: tmi.TransportContext,
    stage_spec: Mapping[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Apply one stage to fresh skim copies using the normal FSM pipeline."""

    modules = context.modules
    travel_times = modules["travel_times"].apply_perceived_time_policies(
        context.travel_times, context.zones
    )
    lengths = {key: value.copy() for key, value in context.lengths.items()}
    scenario = dict(stage_spec)
    if any(scenario.get(key) for key in (
        "railway_expansions", "bike_highways", "mobility_hubs", "road_capacity"
    )):
        travel_times, lengths, _ = modules["interventions"].apply_interventions(
            travel_times, lengths, scenario, zones=context.zones
        )
    travel_times, _ = modules["travel_times"].apply_ebike_share(
        travel_times,
        float(scenario.get("ebike_share", 0.0)),
        speed_multiplier=float(scenario.get("EBIKE_SPEED_MULTIPLIER", 1.5)),
    )
    from additional.section_flows import apply_section_time_saving
    travel_times, _ = apply_section_time_saving(context, travel_times, scenario)
    return travel_times, lengths


def _stage_area_points(
    context: tmi.TransportContext,
    corridor: tmi.CorridorContext,
    level: str,
) -> tuple[pd.DataFrame, dict[str, str], Any]:
    """Return map centroids, a zone-to-area lookup, and matching boundaries."""

    import geopandas as gpd

    zones = context.zones.copy()
    zones["grid_id"] = zones["grid_id"].astype(str)
    allowed = {str(value) for value in corridor.zone_ids}
    zones = zones.loc[zones["grid_id"].isin(allowed)].copy()
    if zones.crs is None:
        raise ValueError("The zone table needs a coordinate reference system.")
    boundaries = _prepare_zone_boundaries(
        zones, zone_ids=allowed, level=level
    )

    if level == "Municipality":
        zones["area"] = zones["municipality_name"].fillna("Unknown").astype(str)
        area_geometry = zones[["area", "geometry"]].dissolve(by="area").reset_index()
        points = area_geometry.copy()
        points["geometry"] = points.geometry.centroid
        zone_to_area = zones.set_index("grid_id")["area"].to_dict()
    else:
        zones["area"] = zones["grid_id"]
        points = zones[["area", "municipality_name", "geometry"]].copy()
        points["geometry"] = points.geometry.centroid
        zone_to_area = zones.set_index("grid_id")["area"].to_dict()

    points = gpd.GeoDataFrame(points, geometry="geometry", crs=zones.crs).to_crs(4326)
    points["lon"] = points.geometry.x
    points["lat"] = points.geometry.y
    return points.set_index("area"), zone_to_area, boundaries


def stage_value_variation_dashboard(
    context: tmi.TransportContext,
    corridor: tmi.CorridorContext,
    stage_specs: Mapping[int, Mapping[str, Any]],
    *,
    road_contexts: Mapping[int, tmi.CorridorContext] | None = None,
    original_stage_specs: Mapping[int, Mapping[str, Any]] | None = None,
    original_road_contexts: Mapping[int, tmi.CorridorContext] | None = None,
    editor_state: dict[str, Any] | None = None,
    height: int = 560,
) -> tuple[Any, dict[str, Any]]:
    """Compare input values on demand, with a table and optional static map."""

    from collections import OrderedDict
    from html import escape
    import ipywidgets as widgets

    stage_ids = [sid for sid in sorted(stage_specs) if int(sid) != 0]
    if not stage_ids:
        raise ValueError("The stage comparison requires at least one stage after Stage 0.")
    originals = deepcopy(original_stage_specs) if original_stage_specs is not None else None
    export_name = "3_2_edited_stage_values" if originals is not None else "3_1_stage_values"
    stage = widgets.Dropdown(
        options=[(_stage_editor_label(s, stage_specs[s]), s) for s in stage_ids],
        value=stage_ids[0], description="Stage:", layout=widgets.Layout(width="520px"),
    )
    reference = widgets.Dropdown(
        options=([("Original selected stage", "original")] if originals is not None else [])
        + [("Stage 0", "baseline")], description="Compare with:",
        style={"description_width": "initial"}, layout=widgets.Layout(width="340px"),
    )
    mode = widgets.Dropdown(description="Mode:", layout=widgets.Layout(width="300px"))
    attribute = widgets.Dropdown(description="Attribute:", layout=widgets.Layout(width="390px"))
    level = widgets.ToggleButtons(options=["Municipality", "FSM zone"], value="Municipality", description="OD level:")
    compare_button = widgets.Button(description="Compare values", icon="table", button_style="primary")
    preview_button = widgets.Button(description="Preview map", icon="map")
    map_view = widgets.Dropdown(
        options=[("Difference", "change"), ("Original/reference", "baseline"), ("Edited stage", "stage")],
        value="change", description="Map:", layout=widgets.Layout(width="300px"),
    )
    table_output = widgets.HTML("<i>Select a stage and attribute, then Compare values.</i>")
    status = widgets.HTML("No transport calculation has been started.")
    map_status = widgets.HTML("The map is optional. Preview draws a static image of changed relations or links.")
    map_panel = widgets.VBox()
    # Keep only corridor matrices for three stage configurations. Changing a
    # displayed attribute reuses these rather than repeating intervention work.
    cache: OrderedDict[str, dict[tuple[str, str, str], pd.DataFrame]] = OrderedDict()
    area_cache: dict[str, Any] = {}
    state: dict[str, Any] = {
        "stage": stage.value, "mode": None, "relations": None,
        "rendering": False, "closed": False, "stale": True,
        "updating_controls": False, "cache": cache,
    }
    road_attributes = {
        "Capacity (veh/h)": "capacity_vph", "Lanes": "lanes",
        "Free-flow speed (km/h)": "speed_kph", "Free-flow time (min)": "free_flow_time_min",
    }
    od_attributes = {**_STAGE_OD_ATTRIBUTES, "Car": {
        "Travel time (min)": ("time", "drive", "min"),
        "Distance (km)": ("length", "drive", "km"),
    }}

    def invalidate(*_: Any) -> None:
        if state["closed"]:
            return
        state["stale"] = True
        status.value = "<i>Selection or interventions changed. Use Compare values to update the table.</i>"
        if map_panel.children:
            map_status.value = "<i>The map shows the previous comparison. Use Preview map to update it.</i>"

    def refresh_attributes(_: Any = None) -> None:
        state["updating_controls"] = True
        try:
            options = road_attributes if mode.value == "Road network" else od_attributes[mode.value]
            previous = attribute.value
            attribute.options = list(options)
            attribute.value = previous if previous in options else next(iter(options))
            level.disabled = mode.value == "Road network"
        finally:
            state["updating_controls"] = False
        invalidate()

    def refresh_modes(*_: Any) -> None:
        if state["closed"]:
            return
        sid = int(stage.value)
        specs = [stage_specs[sid]]
        if originals is not None and sid in originals:
            specs.append(originals[sid])
        families = {
            "Public transport": "railway_expansions", "Bicycle": "bike_highways",
            "PT access and hubs": "mobility_hubs", "Road network": "road_capacity",
        }
        def has_effect(spec: Mapping[str, Any], key: str) -> bool:
            for item in spec.get(key, []):
                effects = item.get("effects", {})
                if any(float(value) != 0.0 for value in effects.values() if np.isscalar(value)):
                    return True
                if any(float(item.get(effect, 0.0)) != 0.0 for effect in (
                    "section_time_saving_min", "headway_reduction_min", "capacity_increase"
                )):
                    return True
            return False

        available = [name for name, key in families.items() if any(has_effect(spec, key) for spec in specs)]
        if "PT access and hubs" in available and "Public transport" not in available:
            available.append("Public transport")
        for spec in specs:
            if spec.get("section_time_saving_min", 0):
                from additional.section_flows import section_config
                section = section_config(spec.get("section_config"))
                section_name = {"PT": "Public transport", "CAR": "Car", "BIKE": "Bicycle", "WALK": "Walking"}.get(section["mode"]) if section["active"] else None
                if section_name and section_name not in available:
                    available.append(section_name)
        if not available:
            available = [*_STAGE_OD_ATTRIBUTES, "Road network"]
        previous = mode.value
        state["updating_controls"] = True
        try:
            mode.options = available
            mode.value = previous if previous in available else available[0]
        finally:
            state["updating_controls"] = False
        refresh_attributes()

    def edited(*_: Any) -> None:
        refresh_modes()

    def compact_snapshot(spec: Mapping[str, Any]) -> dict[tuple[str, str, str], pd.DataFrame]:
        fingerprint = repr(spec)
        if fingerprint in cache:
            cache.move_to_end(fingerprint)
            return cache[fingerprint]
        travel_times, lengths = _stage_skim_snapshot(context, spec)
        matrices = {}
        allowed = {str(value) for value in corridor.zone_ids}
        attributes = {item for choices in od_attributes.values() for item in choices.values()}
        for item in attributes:
            family, key, unit = item
            source = travel_times if family == "time" else lengths
            if key not in source:
                continue
            frame = source[key]
            rows = frame.index.astype(str).isin(allowed)
            columns = frame.columns.astype(str).isin(allowed)
            compact = frame.loc[rows, columns].apply(pd.to_numeric, errors="coerce")
            compact.index = compact.index.astype(str)
            compact.columns = compact.columns.astype(str)
            matrices[item] = compact / 1000.0 if unit == "km" else compact
        cache[fingerprint] = matrices
        while len(cache) > 3:
            cache.popitem(last=False)
        return matrices

    def od_table(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
        after = after.reindex(index=before.index, columns=before.columns)
        a, b = before.to_numpy(dtype=float), after.to_numpy(dtype=float)
        valid = np.isfinite(a) & np.isfinite(b) & (a < 999.0) & (b < 999.0)
        rows, columns = np.where(valid & ~np.isclose(a, b))
        table = pd.DataFrame({
            "origin": before.index.to_numpy()[rows], "destination": before.columns.to_numpy()[columns],
            "baseline": a[rows, columns], "stage": b[rows, columns],
        })
        if level.value == "Municipality" and len(table):
            zones = context.zones
            names = dict(zip(zones["grid_id"].astype(str), zones["municipality_name"].fillna("Unknown").astype(str)))
            table["origin"] = table["origin"].map(names)
            table["destination"] = table["destination"].map(names)
            table = table.dropna(subset=["origin", "destination"]).groupby(
                ["origin", "destination"], as_index=False
            )[["baseline", "stage"]].mean()
        return differences(table)

    def differences(table: pd.DataFrame) -> pd.DataFrame:
        table["change"] = table["stage"] - table["baseline"]
        table["change_pct"] = np.divide(
            100.0 * table["change"].to_numpy(dtype=float), table["baseline"].to_numpy(dtype=float),
            out=np.full(len(table), np.nan), where=table["baseline"].abs().to_numpy() > 1e-9,
        )
        return table.sort_values("change", key=lambda values: values.abs(), ascending=False).reset_index(drop=True)

    def selection_key() -> tuple[Any, ...]:
        sid = int(stage.value)
        return (sid, reference.value, mode.value, attribute.value, level.value, repr(stage_specs[sid]))

    def compare(_: Any = None) -> bool:
        if state["rendering"] or state["closed"]:
            return False
        state["rendering"] = True
        compare_button.disabled = preview_button.disabled = True
        status.value = "<i>Preparing input values; no traffic assignment is run.</i>"
        sid = int(stage.value)
        try:
            before_sid = sid if reference.value == "original" else min(stage_specs)
            before_spec = originals[before_sid] if reference.value == "original" else stage_specs[before_sid]
            before_label = "Original selected stage" if reference.value == "original" else "Stage 0"
            road_base = None
            if mode.value == "Road network":
                contexts = original_road_contexts if reference.value == "original" else road_contexts
                before_context = contexts.get(before_sid) if contexts is not None else None
                if before_context is None:
                    before_context = tmi.apply_road_capacity_stage(corridor, before_spec)[0]
                after_context = road_contexts.get(sid) if road_contexts is not None else None
                if after_context is None:
                    after_context = tmi.apply_road_capacity_stage(corridor, stage_specs[sid])[0]
                road_base = tmi._prepare_road_edges(before_context.edges).set_index("_link_key", drop=False)
                after_edges = tmi._prepare_road_edges(after_context.edges).set_index("_link_key", drop=False).reindex(road_base.index)
                column = road_attributes[attribute.value]
                a, b = pd.to_numeric(road_base[column], errors="coerce"), pd.to_numeric(after_edges[column], errors="coerce")
                changed = np.isfinite(a) & np.isfinite(b) & ~np.isclose(a, b)
                table = differences(pd.DataFrame({
                    "link": road_base.index[changed], "baseline": a.loc[changed].to_numpy(), "stage": b.loc[changed].to_numpy(),
                }))
                unit = attribute.value
            else:
                selected = od_attributes[mode.value][attribute.value]
                before = compact_snapshot(before_spec)
                after = compact_snapshot(stage_specs[sid])
                if selected not in before or selected not in after:
                    raise KeyError(f"No prepared matrix for {attribute.value}.")
                table = od_table(before[selected], after[selected])
                unit = selected[2]
            summary = ""
            stats = pd.DataFrame(columns=["Reference", "Edited stage", "Difference"])
            if len(table):
                stats = table[["baseline", "stage", "change"]].agg(["min", "mean", "max"]).rename(
                    columns={"baseline": "Reference", "stage": "Edited stage", "change": "Difference"}
                )
                summary = "<b>Changed values: minimum, unweighted mean and maximum</b>" + stats.to_html(float_format=lambda value: f"{value:,.3f}")
            table_output.value = summary + _stage_preview_table(
                table.rename(columns={"baseline": "reference", "stage": "edited_stage"}),
                empty="No changed values for this attribute and spatial level.",
            )
            state.update(relations=table, stage=sid, mode=mode.value, unit=unit, road_base=road_base,
                         comparison_key=selection_key(), stale=False, reference_label=before_label,
                         attribute=attribute.value, level=level.value)
            state["export_settings"] = pd.Series({
                "stage": sid, "stage_name": _stage_editor_label(sid, stage_specs[sid]),
                "reference": before_label, "mode": mode.value,
                "attribute": attribute.value, "level": level.value, "unit": unit,
            }, name="value")
            exports.save_outputs(export_name, tables={
                "changes": table.rename(columns={"baseline": "reference", "stage": "edited_stage"}),
                "statistics": stats, "settings": state["export_settings"],
            })
            state.pop("last_error", None)
            status.value = (
                f"<b>{escape(_stage_editor_label(sid, stage_specs[sid]))}</b>: {len(table):,} changed "
                f"{'links' if mode.value == 'Road network' else 'OD relations'} versus {escape(before_label)}. "
                + ("Municipality values are unweighted averages of changed OD cells. " if level.value == "Municipality" and mode.value != "Road network" else "")
            )
            if map_panel.children:
                map_status.value = "<i>Table updated. Use Preview map to update the previous image.</i>"
            return True
        except Exception as error:
            state["last_error"] = f"{type(error).__name__}: {error}"
            state["stale"] = True
            status.value = f"<span style='color:#b71c1c'><b>Comparison failed.</b> {escape(state['last_error'])} Previous results are retained.</span>"
            return False
        finally:
            compare_button.disabled = preview_button.disabled = False
            state["rendering"] = False

    def preview(_: Any = None) -> None:
        if state["rendering"] or state["closed"]:
            return
        if state["stale"] or state.get("comparison_key") != selection_key():
            if not compare():
                return
        state["rendering"] = True
        preview_button.disabled = compare_button.disabled = True
        map_status.value = "<i>Drawing optional map...</i>"
        figure = None
        try:
            from io import BytesIO
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.collections import LineCollection
            from matplotlib.colors import Normalize, TwoSlopeNorm

            figure = Figure(figsize=(10, max(3.5, height / 110)), constrained_layout=True)
            FigureCanvasAgg(figure)
            axis = figure.subplots()
            table = state["relations"]
            # Limit map clutter only; the comparison table keeps every change.
            visible = table.head(300).sort_values("change", key=lambda values: values.abs(), kind="stable")
            value = str(map_view.value)
            segments, values = [], []
            if state["mode"] == "Road network":
                geometry = state["road_base"]["geometry"]
                for row in visible.itertuples(index=False):
                    shape = geometry.loc[row.link]
                    parts = list(shape.geoms) if hasattr(shape, "geoms") else [shape]
                    for part in parts:
                        if part is not None and not part.is_empty:
                            segments.append(np.asarray(part.coords)[:, :2])
                            values.append(getattr(row, value))
            else:
                selected_level = state["level"]
                if selected_level not in area_cache:
                    area_cache[selected_level] = _stage_area_points(context, corridor, selected_level)
                points, _, _ = area_cache[selected_level]
                axis.scatter(points["lon"], points["lat"], color="#adb5bd", s=10, zorder=2)
                for row in visible.itertuples(index=False):
                    if row.origin in points.index and row.destination in points.index:
                        first, last = points.loc[row.origin], points.loc[row.destination]
                        segments.append([[first.lon, first.lat], [last.lon, last.lat]])
                        values.append(getattr(row, value))
                if selected_level == "Municipality":
                    for name, point in points.iterrows():
                        axis.annotate(str(name), (point.lon, point.lat), xytext=(3, 3), textcoords="offset points", fontsize=7)
                axis.set_xlabel("Longitude")
                axis.set_ylabel("Latitude")
                if len(points):
                    latitude = float(points["lat"].mean())
                    axis.set_aspect(1.0 / max(np.cos(np.radians(latitude)), 0.1))
            if segments:
                numbers = np.asarray(values, dtype=float)
                if value == "change":
                    limit = max(float(np.max(np.abs(numbers))), 1e-9)
                    norm, cmap = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), "RdBu"
                else:
                    low, high = float(numbers.min()), float(numbers.max())
                    norm, cmap = Normalize(vmin=low, vmax=high if high > low else low + 1), "viridis"
                lines = LineCollection(segments, array=numbers, cmap=cmap, norm=norm,
                                       linewidths=3 if value == "change" else 2, alpha=0.95, zorder=3)
                axis.add_collection(lines)
                axis.autoscale_view()
                figure.colorbar(lines, ax=axis, label=f"{'Difference' if value == 'change' else 'Value'} ({state['unit']})")
            else:
                axis.text(0.5, 0.5, "No changed relations or links for this selection", transform=axis.transAxes, ha="center")
            # Keep narrow/straight corridors legible without distorting geography.
            aspect = (1.0 / max(np.cos(np.radians(float(points["lat"].mean()))), 0.1)
                      if state["mode"] != "Road network" and len(points) else 1.0)
            if state["mode"] == "Road network":
                axis.set_aspect(1.0)
            x0, x1 = axis.get_xlim()
            y0, y1 = axis.get_ylim()
            width, height_span = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
            padded_width = max(width, height_span * aspect * 0.35)
            padded_height = max(height_span, width / aspect * 0.35)
            axis.set_xlim((x0 + x1 - padded_width) / 2, (x0 + x1 + padded_width) / 2)
            axis.set_ylim((y0 + y1 - padded_height) / 2, (y0 + y1 + padded_height) / 2)
            axis.ticklabel_format(style="plain", useOffset=False)
            axis.set_title(f"{_stage_editor_label(state['stage'], stage_specs[state['stage']])}\n{map_view.label}: {state['attribute']}")
            buffer = BytesIO()
            figure.savefig(buffer, format="png", dpi=110)
            image = widgets.Image(value=buffer.getvalue(), format="png", layout=widgets.Layout(max_width="100%"))
            previous = map_panel.children
            map_panel.children = (image,)
            for widget in previous:
                _close_stage_widgets(widget)
            state.pop("map_error", None)
            map_status.value = f"Showing {len(visible):,} of {len(table):,} changed relations/links. The map is a static preview."
            exports.save_outputs(f"{export_name}_preview", png=buffer.getvalue(), tables={
                "settings": pd.concat([state["export_settings"], pd.Series({
                    "map_view": value, "displayed_relations": len(visible),
                    "total_changed_relations": len(table),
                }, name="value")]),
            })
        except Exception as error:
            state["map_error"] = f"{type(error).__name__}: {error}"
            map_status.value = f"<span style='color:#b71c1c'><b>Preview failed.</b> {escape(state['map_error'])} The table and previous image are retained.</span>"
        finally:
            if figure is not None:
                figure.clear()
            state["rendering"] = False
            preview_button.disabled = compare_button.disabled = False

    def change_attribute(_: Any = None) -> None:
        if not state["updating_controls"]:
            invalidate()

    mode.observe(lambda change: refresh_attributes() if not state["updating_controls"] else None, names="value")
    stage.observe(refresh_modes, names="value")
    reference.observe(invalidate, names="value")
    attribute.observe(change_attribute, names="value")
    level.observe(invalidate, names="value")
    map_view.observe(lambda _: setattr(map_status, "value", "Use Preview map to draw the selected view."), names="value")
    compare_button.on_click(compare)
    preview_button.on_click(preview)
    refresh_modes()
    if editor_state is not None:
        editor_state.setdefault("listeners", []).append(edited)
    ui = widgets.VBox([
        widgets.HBox([stage, reference]), widgets.HBox([mode, attribute]), level,
        widgets.HBox([compare_button, status]),
        widgets.HBox([map_view, preview_button]), map_status, map_panel,
        table_output,
    ])

    def close() -> None:
        state["closed"] = True
        if editor_state is not None and edited in editor_state.get("listeners", []):
            editor_state["listeners"].remove(edited)
        cache.clear()
        area_cache.clear()
        state["relations"] = state["road_base"] = None
        _close_stage_widgets(ui)

    state.update(
        close=close, refresh=compare, compare=compare, preview=preview, invalidate=edited,
        controls={"stage": stage, "reference": reference, "mode": mode, "attribute": attribute,
                  "level": level, "compare": compare_button, "preview": preview_button, "map_view": map_view},
        table=table_output, status=status, map_status=map_status, map_panel=map_panel,
    )
    return ui, state


_DESIRE_LINE_EFFECTS = {
    "Public transport": {
        "In-vehicle time reduction (%)": "travel_time_reduction_pct",
        "Speed increase (%)": "speed_increase_pct",
        "Distance reduction (%)": "distance_reduction_pct",
        "Initial-wait reduction (%)": "initial_wait_reduction_pct",
        "Transfer-wait reduction (%)": "transfer_wait_reduction_pct",
        "Physical-transfer reduction (%)": "transfer_time_reduction_pct",
        "Access-time reduction (%)": "access_time_reduction_pct",
        "Egress-time reduction (%)": "egress_time_reduction_pct",
    },
    "Bicycle": {
        "Travel-time reduction (%)": "travel_time_reduction_pct",
        "Speed increase (%)": "speed_increase_pct",
        "Distance reduction (%)": "distance_reduction_pct",
    },
    "PT access and hubs": {
        "Access-time reduction (%)": "access_time_reduction_pct",
        "Egress-time reduction (%)": "egress_time_reduction_pct",
        "Initial-wait reduction (%)": "initial_wait_reduction_pct",
        "Transfer-wait reduction (%)": "transfer_wait_reduction_pct",
        "Physical-transfer reduction (%)": "transfer_time_reduction_pct",
    },
}


def _stage_intervention_editor_map(
    context: tmi.TransportContext,
    corridor: tmi.CorridorContext,
    stage_spec: Mapping[str, Any],
    *,
    intervention_key: str,
    boundary_level: str,
    on_intervention_click: Any,
    on_area_click: Any,
    height: int,
    show_interventions: bool = True,
    show_zone_boundaries: bool = False,
    show_area_nodes: bool = True,
    show_corridor: bool = True,
    original_interventions: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[Any, dict[str, list[Any]]]:
    """Build the clickable Lonboard map used by one OD/hub editor page."""

    import geopandas as gpd
    from lonboard import PathLayer, PolygonLayer, ScatterplotLayer, SolidPolygonLayer
    from shapely.geometry import LineString

    colours = {
        "railway_expansions": [40, 100, 183, 230],
        "bike_highways": [43, 147, 72, 230],
        "mobility_hubs": [122, 46, 142, 235],
    }
    component_colour = colours[intervention_key]
    layers: list[Any] = []
    layer_groups: dict[str, list[Any]] = {
        "corridor": [], "boundaries": [], "interventions": [], "area_nodes": [],
    }

    def selector_text(selector: Any) -> str:
        if isinstance(selector, Mapping):
            return str(
                selector.get(
                    "municipality_name",
                    selector.get("grid_id", selector.get("Name", "Selected area")),
                )
            )
        return str(selector)

    if corridor.polygon_gdf is not None and len(corridor.polygon_gdf):
        polygon = _slim_geodataframe(corridor.polygon_gdf, ["name"], simplify_m=5.0)
    else:
        polygon = gpd.GeoDataFrame(
            {"name": ["Corridor extent"], "geometry": [corridor.polygon]},
            geometry="geometry", crs=corridor.zones.crs,
        ).to_crs(4326)
    corridor_layer = SolidPolygonLayer.from_geopandas(
        polygon,
        get_fill_color=[40, 110, 170, 18],
        get_line_color=[40, 110, 170, 150],
        filled=True,
        pickable=False,
    )
    corridor_layer.visible = bool(show_corridor)
    layers.append(corridor_layer)
    layer_groups["corridor"].append(corridor_layer)

    boundaries = _prepare_zone_boundaries(
        context.zones,
        zone_ids=corridor.zone_ids,
        level=boundary_level,
    )
    boundary_layer = None
    if len(boundaries):
        boundary_layer = PolygonLayer.from_geopandas(
            boundaries,
            get_fill_color=[0, 0, 0, 0],
            get_line_color=[70, 70, 70, 135],
            filled=False,
            stroked=True,
            line_width_min_pixels=1,
            pickable=False,
        )
        boundary_layer.visible = bool(show_zone_boundaries)
        layers.append(boundary_layer)
        layer_groups["boundaries"].append(boundary_layer)

    interventions = stage_spec.get(intervention_key, []) or []
    changed_positions = {
        i for i, item in enumerate(interventions)
        if original_interventions is not None and item not in original_interventions
    }
    foreground_layers = []
    background_interventions = []
    records: list[dict[str, Any]] = []
    zones = context.zones
    # `_stage_selector_point` unions every matched zone polygon from scratch
    # on each call. A single intervention can repeat the same municipality
    # selector dozens of times (e.g. a fully-connected area_pairs list), so
    # cache the resolved point per selector for the lifetime of this map build.
    _point_cache: dict[Any, Any] = {}

    def selector_point(selector: Any) -> Any:
        key = tuple(sorted(selector.items())) if isinstance(selector, Mapping) else selector
        if key not in _point_cache:
            _point_cache[key] = _stage_selector_point(zones, selector)
        return _point_cache[key]

    def add_intervention_layers(frame: Any, *, hubs: bool = False) -> None:
        for changed in (False, True):
            selected = frame["intervention_position"].isin(changed_positions).eq(changed)
            subset = frame.loc[selected].reset_index(drop=True)
            if subset.empty:
                continue
            colour = ([235, 105, 20, 255] if changed else
                      [115, 125, 140, 115] if original_interventions is not None else component_colour)
            if hubs:
                layer = ScatterplotLayer.from_geopandas(
                    subset, get_fill_color=colour, get_line_color=[255, 255, 255, 255],
                    get_radius=11.0 if changed else 7.0, radius_units="pixels",
                    radius_min_pixels=6, radius_max_pixels=14, stroked=True,
                    pickable=True, auto_highlight=True,
                )
            else:
                layer = PathLayer.from_geopandas(
                    subset, get_color=colour,
                    get_width=6.0 if changed else 2.5 if original_interventions is not None else 4.0,
                    width_units="pixels", width_min_pixels=2, width_max_pixels=9,
                    pickable=True, auto_highlight=True,
                    highlight_color=[20, 20, 20, 230],
                )

            def select_intervention(change: dict[str, Any], rows: Any = subset) -> None:
                try:
                    position = int(change.get("new"))
                except (TypeError, ValueError):
                    return
                if 0 <= position < len(rows):
                    on_intervention_click(intervention_key, int(rows.iloc[position]["intervention_position"]))

            layer.observe(select_intervention, names="selected_index")
            layer.visible = bool(show_interventions)
            (foreground_layers if changed else background_interventions).append(layer)
            layer_groups["interventions"].append(layer)

    if intervention_key in ("railway_expansions", "bike_highways"):
        for position, intervention in enumerate(interventions):
            effects = "; ".join(
                f"{name.replace('_pct', '').replace('_', ' ')}: {float(value):.1f}%"
                for name, value in (intervention.get("effects", {}) or {}).items()
                if float(value) != 0.0
            ) or "No numerical effects"
            pairs = intervention.get("area_pairs", []) or []
            for pair in pairs:
                origin_selector = pair.get("origin", {})
                destination_selector = pair.get("destination", {})
                origin_point = selector_point(origin_selector)
                destination_point = selector_point(destination_selector)
                if origin_point is None or destination_point is None:
                    continue
                records.append({
                    "intervention_position": int(position),
                    "intervention": str(intervention.get("name", "Unnamed")),
                    "origin": selector_text(origin_selector),
                    "destination": selector_text(destination_selector),
                    "both_directions": bool(intervention.get("both_directions", True)),
                    "effects": effects,
                    "geometry": LineString([origin_point, destination_point]),
                })
            for pair in intervention.get("od_pairs", []) or []:
                if len(pair) != 2:
                    continue
                origin_point = selector_point(pair[0])
                destination_point = selector_point(pair[1])
                if origin_point is None or destination_point is None:
                    continue
                records.append({
                    "intervention_position": int(position),
                    "intervention": str(intervention.get("name", "Unnamed")),
                    "origin": selector_text(pair[0]),
                    "destination": selector_text(pair[1]),
                    "both_directions": bool(intervention.get("both_directions", True)),
                    "effects": effects,
                    "geometry": LineString([origin_point, destination_point]),
                })
        if records:
            lines = gpd.GeoDataFrame(records, geometry="geometry", crs=4326).reset_index(drop=True)
            add_intervention_layers(lines)

    else:
        for position, intervention in enumerate(interventions):
            effects = "; ".join(
                f"{name.replace('_pct', '').replace('_', ' ')}: {float(value):.1f}%"
                for name, value in (intervention.get("effects", {}) or {}).items()
                if float(value) != 0.0
            ) or "No numerical effects"
            for selector in intervention.get("zones", []) or []:
                point = selector_point(selector)
                if point is None:
                    continue
                records.append({
                    "intervention_position": int(position),
                    "intervention": str(intervention.get("name", "Unnamed")),
                    "area": selector_text(selector),
                    "effects": effects,
                    "geometry": point,
                })
        if records:
            hubs = gpd.GeoDataFrame(records, geometry="geometry", crs=4326).reset_index(drop=True)
            add_intervention_layers(hubs, hubs=True)

    if len(boundaries):
        area_points = boundaries[["area", "geometry"]].copy()
        area_points.geometry = area_points.geometry.representative_point()
        area_points = area_points.reset_index(drop=True)
        area_layer = ScatterplotLayer.from_geopandas(
            area_points,
            get_fill_color=[52, 58, 64, 205],
            get_line_color=[255, 255, 255, 240],
            get_radius=4.0,
            radius_units="pixels",
            radius_min_pixels=3,
            radius_max_pixels=7,
            stroked=True,
            pickable=True,
            auto_highlight=True,
        )

        def select_area(change: dict[str, Any]) -> None:
            selected_index = change.get("new")
            if selected_index is None:
                return
            try:
                position = int(selected_index)
            except (TypeError, ValueError):
                return
            if 0 <= position < len(area_points):
                on_area_click(str(area_points.iloc[position]["area"]))

        area_layer.observe(select_area, names="selected_index")
        area_layer.visible = bool(show_area_nodes)
        layers.append(area_layer)
        layer_groups["area_nodes"].append(area_layer)

    layers.extend(background_interventions)
    layers.extend(foreground_layers)
    return (
        _lonboard_map(layers, height=height, show_side_panel=False),
        layer_groups,
    )


def desire_line_stage_editor(
    context: tmi.TransportContext,
    corridor: tmi.CorridorContext,
    stage_specs: dict[int, dict[str, Any]],
    *,
    height: int = 560,
    stage_selector: Any | None = None,
    fixed_intervention_type: str | None = None,
    defer_map: bool = False,
    on_change: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Edit physical OD/hub inputs; draw the optional scope map on request."""
    import html
    import json

    import ipywidgets as widgets

    family_keys = {
        "Public transport": "railway_expansions",
        "Bicycle": "bike_highways",
        "PT access and hubs": "mobility_hubs",
    }
    editable = [value for value in sorted(stage_specs) if int(value) != 0]
    if not editable:
        raise ValueError("The intervention editor requires at least one stage after Stage 0.")
    if fixed_intervention_type not in (None, *family_keys):
        raise ValueError(f"Unknown intervention page: {fixed_intervention_type}")
    original = deepcopy(stage_specs)
    zones = context.zones.copy()
    zones["grid_id"] = zones["grid_id"].astype(str)
    zones = zones.loc[zones["grid_id"].isin({str(v) for v in corridor.zone_ids})].copy()
    municipalities = sorted(zones["municipality_name"].dropna().astype(str).unique())
    zone_options = [
        (f"{row.grid_id} — {row.municipality_name}", row.grid_id)
        for row in zones[["grid_id", "municipality_name"]].itertuples(index=False)
    ]
    owns_stage_selector = stage_selector is None
    stage = stage_selector if stage_selector is not None else widgets.Dropdown(
        options=[(_stage_editor_label(s, stage_specs[s]), s) for s in editable],
        value=editable[0], description="Edit stage:", layout=widgets.Layout(width="560px"),
    )
    if stage.value not in editable:
        raise ValueError("The shared stage selector must reference an editable stage.")
    intervention_type = widgets.Dropdown(
        options=[fixed_intervention_type] if fixed_intervention_type else list(family_keys),
        value=fixed_intervention_type or "Public transport", description="Intervention:",
        layout=widgets.Layout(width="360px"),
    )
    existing = widgets.Dropdown(description="Intervention:", layout=widgets.Layout(width="760px"))
    new = widgets.Button(description="New intervention", icon="plus")
    enabled = widgets.Checkbox(value=True, description="Enabled in working stages", indent=False)
    name = widgets.Text(description="Name:", layout=widgets.Layout(width="680px"))
    spatial_level = widgets.ToggleButtons(options=["Municipality", "FSM zone"], description="Select by:")
    origin = widgets.Dropdown(description="Origin/area:", layout=widgets.Layout(width="420px"))
    destination = widgets.Dropdown(description="Destination:", layout=widgets.Layout(width="420px"))
    both_directions = widgets.Checkbox(value=True, description="Both directions", indent=False)
    replace_scope = widgets.Checkbox(value=False, description="Replace selected intervention's scope", indent=False)
    scope_note = widgets.HTML()
    effect = widgets.Dropdown(description="Effect:", layout=widgets.Layout(width="420px"))
    magnitude = widgets.FloatText(value=10.0, description="Value (%):")
    add = widgets.Button(description="Add intervention", button_style="success", icon="plus")
    update = widgets.Button(description="Update selected", button_style="primary", icon="check")
    remove = widgets.Button(description="Remove selected", button_style="danger", icon="trash")
    reset = widgets.Button(description="Reset this family", button_style="warning", icon="undo")
    status = widgets.HTML()
    table_output = widgets.HTML(layout=widgets.Layout(width="100%"))
    stage_summary = widgets.HTML()
    preview = widgets.Button(description="Preview scope map", icon="map")
    map_status = widgets.HTML("<i>Optional: draw the current intervention scopes.</i>")
    map_output = widgets.VBox()
    show_boundaries = widgets.Checkbox(value=False, description="Area boundaries", indent=False)
    state: dict[str, Any] = {
        "stage_specs": stage_specs, "stage_selector": stage, "active_stage": int(stage.value),
        "intervention_type": intervention_type.value, "catalogue": {}, "published": {},
        "refreshing": False, "editing": False, "rendering": False, "closed": False,
        "map_active": not defer_map, "map": None, "map_layers": {}, "revision": 0,
    }

    def stage_id() -> int:
        return int(stage.value)

    def intervention_key() -> str:
        return family_keys[intervention_type.value]

    def intervention_list(value: Any) -> list[dict[str, Any]]:
        if not value:
            return []
        if isinstance(value, Mapping):
            return [value]
        return list(value)

    def entries() -> list[dict[str, Any]]:
        key = (stage_id(), intervention_key())
        current = intervention_list(stage_specs[stage_id()].get(key[1]))
        if key not in state["catalogue"] or current != state["published"].get(key):
            state["catalogue"][key] = [
                {"item": deepcopy(item), "enabled": True} for item in current
            ]
            state["published"][key] = deepcopy(current)
        return state["catalogue"][key]

    def selected_entry() -> dict[str, Any] | None:
        values = entries()
        position = existing.value
        return values[int(position)] if position is not None and int(position) < len(values) else None

    def scope_text(item: Mapping[str, Any]) -> str:
        def selector_text(value: Any) -> str:
            if isinstance(value, Mapping):
                return ", ".join(f"{key}: {requested}" for key, requested in value.items())
            return str(value)
        scopes = [
            f"{selector_text(pair['origin'])} → {selector_text(pair['destination'])}"
            for pair in item.get("area_pairs", []) or []
        ]
        scopes += [f"{pair[0]} → {pair[1]}" for pair in item.get("od_pairs", []) or []]
        scopes += [selector_text(value) for value in item.get("zones", []) or []]
        if not scopes and any(key in item for key in (
            "section_time_saving_min", "headway_reduction_min", "capacity_increase"
        )):
            return "Configured counting section / service corridor"
        return "; ".join(scopes) or "No scope specified"

    def service_only(item: Mapping[str, Any]) -> bool:
        return any(key in item for key in (
            "section_time_saving_min", "headway_reduction_min", "capacity_increase"
        ))

    def refresh_area_options(_: Any = None) -> None:
        choices = municipalities if spatial_level.value == "Municipality" else zone_options
        values = [v[1] if isinstance(v, tuple) else v for v in choices]
        for control, default_index in ((origin, 0), (destination, 1)):
            previous = control.value
            control.options = choices
            control.value = previous if previous in values else (values[min(default_index, len(values) - 1)] if values else None)

    def form_state(_: Any = None) -> None:
        item = selected_entry()
        read_only = bool(item and service_only(item["item"]))
        scope_locked = item is not None and not replace_scope.value
        for control in (spatial_level, origin, destination, both_directions):
            control.disabled = scope_locked or read_only
        is_hub = intervention_key() == "mobility_hubs"
        destination.layout.display = "none" if is_hub else ""
        both_directions.layout.display = "none" if is_hub else ""
        replace_scope.disabled = item is None or read_only
        enabled.disabled = item is None or read_only
        update.disabled = item is None or read_only
        remove.disabled = item is None or read_only
        add.disabled = item is not None
        effect.disabled = magnitude.disabled = name.disabled = read_only
        if item:
            note = html.escape(scope_text(item["item"]))
            if read_only:
                note += "<br>Section/service settings are read-only here. Edit them in stages.py."
            elif not replace_scope.value:
                note += "<br>Update preserves this complete scope and all other effects."
            else:
                note += "<br>Update will replace this complete scope with the selectors below."
            scope_note.value = f"<small>{note}</small>"
        else:
            scope_note.value = "<small>Select a scope and effect for the new intervention.</small>"

    def load_effect(_: Any = None) -> None:
        if state["refreshing"]:
            return
        item = selected_entry()
        if item:
            field = _DESIRE_LINE_EFFECTS[intervention_type.value][effect.value]
            magnitude.value = float(item["item"].get("effects", {}).get(field, 0.0))

    def load_selection(_: Any = None) -> None:
        if state["refreshing"]:
            return
        state["refreshing"] = True
        try:
            entry = selected_entry()
            replace_scope.value = False
            name.value = str(entry["item"].get("name", "")) if entry else ""
            enabled.value = bool(entry["enabled"]) if entry else True
            if entry:
                item = entry["item"]
                choices = _DESIRE_LINE_EFFECTS[intervention_type.value]
                present = [label for label, field in choices.items() if item.get("effects", {}).get(field, 0.0)]
                if present and effect.value not in present:
                    effect.value = present[0]
                magnitude.value = float(item.get("effects", {}).get(choices[effect.value], 0.0))
                simple_scope = None
                if intervention_key() == "mobility_hubs" and len(item.get("zones", [])) == 1:
                    value = item["zones"][0]
                    simple_scope = (value if isinstance(value, Mapping) else {"grid_id": str(value)}, None)
                elif len(item.get("area_pairs", [])) == 1 and not item.get("od_pairs"):
                    pair = item["area_pairs"][0]
                    simple_scope = (pair["origin"], pair["destination"])
                elif len(item.get("od_pairs", [])) == 1 and not item.get("area_pairs"):
                    pair = item["od_pairs"][0]
                    simple_scope = ({"grid_id": str(pair[0])}, {"grid_id": str(pair[1])})
                if simple_scope:
                    scope_key = next(iter(simple_scope[0])) if len(simple_scope[0]) == 1 else None
                    if scope_key in ("grid_id", "municipality_name") and all(
                        value is None or list(value) == [scope_key] for value in simple_scope
                    ):
                        spatial_level.value = "Municipality" if scope_key == "municipality_name" else "FSM zone"
                        refresh_area_options()
                        for control, value in zip((origin, destination), simple_scope):
                            values = [v[1] if isinstance(v, tuple) else v for v in control.options]
                            if value and value[scope_key] in values:
                                control.value = value[scope_key]
                both_directions.value = bool(item.get("both_directions", True))
            else:
                magnitude.value = 10.0
        finally:
            state["refreshing"] = False
        form_state()

    def refresh(_: Any = None) -> None:
        if state["refreshing"] or state["closed"]:
            return
        state["refreshing"] = True
        try:
            selected = existing.value
            values = entries()
            existing.options = [("New intervention", None)] + [
                (f"{'On' if entry['enabled'] else 'Off'} · {entry['item'].get('name', 'Unnamed')}", i)
                for i, entry in enumerate(values)
            ]
            existing.value = selected if selected is not None and selected < len(values) else None
            rows = []
            numeric_rows = []
            for entry in values:
                item = entry["item"]
                effects = [f"{field}: {value:g}%" for field, value in item.get("effects", {}).items() if float(value) != 0]
                effects += [f"{field}: {item[field]}" for field in (
                    "section_time_saving_min", "headway_reduction_min", "capacity_increase"
                ) if field in item]
                rows.append({"Enabled": "Yes" if entry["enabled"] else "No",
                             "Intervention": item.get("name", "Unnamed"), "Scope": scope_text(item),
                             "Effects": "; ".join(effects) or "No active effects"})
                numeric_rows.append({
                    "Enabled": bool(entry["enabled"]), "Intervention": item.get("name", "Unnamed"),
                    "Scope": scope_text(item),
                    **item.get("effects", {}),
                    **{field: item[field] for field in (
                        "section_time_saving_min", "headway_reduction_min", "capacity_increase"
                    ) if field in item},
                })
            table_output.value = (
                "<div style='max-height:280px;overflow:auto'>" + pd.DataFrame(rows).to_html(index=False, escape=True) + "</div>"
                if rows else "<i>No interventions in this family. Choose New intervention to add one.</i>"
            )
            state["active_stage"] = stage_id()
            exports.save_outputs(
                f"3_2_interventions_{stage_id()}_{intervention_key()}",
                tables={"": pd.DataFrame(numeric_rows) if numeric_rows else pd.DataFrame(
                    columns=["Enabled", "Intervention", "Scope"]),
                    "settings": pd.Series({"stage_or_package": stage_id(),
                        "name": _stage_editor_label(stage_id(), stage_specs[stage_id()]),
                        "family": intervention_key()}, name="value")},
            )
            stage_summary.value = f"<b>{html.escape(_stage_editor_label(stage_id(), stage_specs[stage_id()]))}</b> — {sum(v['enabled'] for v in values)} enabled {html.escape(intervention_type.value.lower())} intervention(s)."
        finally:
            state["refreshing"] = False
        load_selection()

    def mark_map_stale() -> None:
        map_status.value = "<i>Preview needs updating. Click Preview scope map when needed.</i>"
        map_output.layout.display = "none"

    def publish() -> None:
        key = (stage_id(), intervention_key())
        values = [deepcopy(v["item"]) for v in state["catalogue"][key] if v["enabled"]]
        stage_specs[stage_id()][key[1]] = values
        state["published"][key] = deepcopy(values)
        state["revision"] += 1
        mark_map_stale()
        if on_change is not None:
            on_change(stage_id(), key[1])

    def canonical(item: Mapping[str, Any]) -> str:
        payload = {key: value for key, value in item.items() if key != "name"}
        payload["effects"] = {k: float(v) for k, v in item.get("effects", {}).items() if float(v) != 0.0}
        if "area_pairs" in payload or "od_pairs" in payload:
            payload.setdefault("both_directions", True)
        return json.dumps(payload, sort_keys=True, default=str)

    def make_item(updating: bool) -> dict[str, Any]:
        values = entries()
        entry = selected_entry() if updating else None
        if updating and (entry is None or service_only(entry["item"])):
            raise ValueError("Select an OD or hub intervention to update.")
        item = deepcopy(entry["item"]) if entry else {}
        value = float(magnitude.value)
        if not np.isfinite(value):
            raise ValueError("Effect value must be finite.")
        field = _DESIRE_LINE_EFFECTS[intervention_type.value][effect.value]
        if field == "speed_increase_pct" and value <= -100:
            raise ValueError("Speed increase must be greater than -100%.")
        if field.endswith("reduction_pct") and value > 100:
            raise ValueError("A reduction cannot exceed 100%.")
        item.setdefault("effects", {})[field] = value
        item_name = name.value.strip()
        if not item_name:
            number = 1
            names = {str(v["item"].get("name", "")).casefold() for v in values}
            while f"{intervention_type.value} intervention {number}".casefold() in names:
                number += 1
            item_name = f"{intervention_type.value} intervention {number}"
        item["name"] = item_name
        if not updating or replace_scope.value:
            if origin.value is None or (intervention_key() != "mobility_hubs" and destination.value is None):
                raise ValueError("Choose a valid area for the intervention.")
            for key in ("area_pairs", "od_pairs", "zones"):
                item.pop(key, None)
            scope_key = "municipality_name" if spatial_level.value == "Municipality" else "grid_id"
            if intervention_key() == "mobility_hubs":
                item["zones"] = [{scope_key: str(origin.value)}]
            else:
                if origin.value == destination.value:
                    raise ValueError("Choose two different areas.")
                item["area_pairs"] = [{"origin": {scope_key: str(origin.value)}, "destination": {scope_key: str(destination.value)}}]
                item["both_directions"] = bool(both_directions.value)
        for i, other in enumerate(values):
            if updating and i == existing.value:
                continue
            if str(other["item"].get("name", "")).strip().casefold() == item_name.casefold():
                raise ValueError("This name already exists. Select that intervention to update or enable it.")
            if canonical(other["item"]) == canonical(item):
                raise ValueError("An identical intervention already exists. Select it to update or enable it.")
        return item

    def save_intervention(updating: bool) -> None:
        item = make_item(updating)
        values = entries()
        if updating:
            position = int(existing.value)
            values[position]["item"] = item
        else:
            values.append({"item": item, "enabled": True})
            position = len(values) - 1
        publish()
        refresh()
        existing.value = position
        status.value = f"<span style='color:#1b5e20'><b>{'Updated' if updating else 'Added'} {html.escape(item['name'])}.</b> Working stages and the table are updated.</span>"

    def remove_intervention() -> None:
        entry = selected_entry()
        if entry is None:
            return
        if service_only(entry["item"]):
            raise ValueError("Edit section/service settings in stages.py.")
        removed = str(entry["item"].get("name", "Unnamed"))
        entries().pop(int(existing.value))
        publish()
        existing.value = None
        refresh()
        status.value = f"Removed {html.escape(removed)} from the working stage."

    def reset_family() -> None:
        key = (stage_id(), intervention_key())
        state["catalogue"][key] = [
            {"item": deepcopy(item), "enabled": True}
            for item in intervention_list(original.get(stage_id(), {}).get(key[1]))
        ]
        publish()
        existing.value = None
        refresh()
        status.value = "This intervention family has been restored to its loaded definition."

    def set_enabled(_: Any = None) -> None:
        if state["refreshing"] or state["editing"]:
            return
        requested = bool(enabled.value)
        def apply_enabled() -> None:
            entry = selected_entry()
            if entry is not None:
                if service_only(entry["item"]):
                    raise ValueError("Edit section/service settings in stages.py.")
                entry["enabled"] = requested
                publish()
                refresh()
                status.value = "Enabled interventions are included in the working stages; disabled interventions are omitted."
        guarded(apply_enabled)()

    def close_map() -> None:
        children = map_output.children
        map_output.children = ()
        state["map"] = None
        state["map_layers"] = {}
        for child in children:
            _close_stage_widgets(child)

    def refresh_map(_: Any = None) -> None:
        if state["rendering"] or state["closed"]:
            return
        state["rendering"] = True
        preview.disabled = True
        map_status.value = "<i>Drawing intervention scopes…</i>"
        try:
            visible_entries = [i for i, entry in enumerate(entries()) if entry["enabled"]]
            def select_map_item(key: str, position: int) -> None:
                if key == intervention_key() and position < len(visible_entries):
                    existing.value = visible_entries[position]
            map_spec = dict(stage_specs[stage_id()])
            map_spec[intervention_key()] = intervention_list(map_spec.get(intervention_key()))
            figure, layers = _stage_intervention_editor_map(
                context, corridor, map_spec, intervention_key=intervention_key(),
                boundary_level=str(spatial_level.value), on_intervention_click=select_map_item,
                on_area_click=lambda _: None,
                height=height, show_interventions=True, show_zone_boundaries=show_boundaries.value,
                show_area_nodes=False, show_corridor=True,
                original_interventions=intervention_list(original[stage_id()].get(intervention_key())),
            )
            close_map()
            state["map"], state["map_layers"] = figure, layers
            map_output.children = (figure,)
            map_output.layout.display = ""
            map_status.value = "<small>Orange: added or changed interventions (on top). Grey: unchanged. Straight lines show OD selections, not routes.</small>"
            state.pop("last_error", None)
        except Exception as error:
            state["last_error"] = f"{type(error).__name__}: {error}"
            map_status.value = f"<span style='color:#b71c1c'>Map unavailable: {html.escape(state['last_error'])}. Your edits and table remain available.</span>"
        finally:
            state["rendering"] = False
            preview.disabled = False

    def guarded(action: Any) -> Any:
        def run(_: Any = None) -> None:
            if state["editing"] or state["closed"]:
                return
            state["editing"] = True
            snapshot = None
            try:
                key = (stage_id(), intervention_key())
                entries()
                snapshot = {
                    "family_present": key[1] in stage_specs[key[0]],
                    "family": deepcopy(stage_specs[key[0]].get(key[1])),
                    "catalogue": deepcopy(state["catalogue"][key]),
                    "published": deepcopy(state["published"][key]),
                    "revision": state["revision"], "selection": existing.value,
                    "map_status": map_status.value, "map_display": map_output.layout.display,
                }
                action()
                state.pop("last_action_error", None)
            except Exception as error:
                if snapshot is not None:
                    if snapshot["family_present"]:
                        stage_specs[key[0]][key[1]] = snapshot["family"]
                    else:
                        stage_specs[key[0]].pop(key[1], None)
                    state["catalogue"][key] = snapshot["catalogue"]
                    state["published"][key] = snapshot["published"]
                    state["revision"] = snapshot["revision"]
                    refresh()
                    existing.value = snapshot["selection"]
                    load_selection()
                    map_status.value = snapshot["map_status"]
                    map_output.layout.display = snapshot["map_display"]
                state["last_action_error"] = f"{type(error).__name__}: {error}"
                status.value = f"<span style='color:#b71c1c'><b>Edit not applied.</b> {html.escape(str(error))}</span>"
            finally:
                state["editing"] = False
                form_state()
        return run

    def change_stage(_: Any = None) -> None:
        existing.value = None
        status.value = ""
        mark_map_stale()
        refresh()

    def change_family(_: Any = None) -> None:
        state["intervention_type"] = intervention_type.value
        state["refreshing"] = True
        try:
            effect.options = list(_DESIRE_LINE_EFFECTS[intervention_type.value])
            effect.value = next(iter(_DESIRE_LINE_EFFECTS[intervention_type.value]))
        finally:
            state["refreshing"] = False
        change_stage()

    def sync_from_specs(selected_stage: int | None = None, family: str | None = None) -> None:
        for key in list(state["catalogue"]):
            if (selected_stage is None or key[0] == selected_stage) and (family is None or key[1] == family):
                state["catalogue"].pop(key, None)
                state["published"].pop(key, None)
        refresh()

    def initialize_stage(new_stage_id: int) -> None:
        original.setdefault(int(new_stage_id), deepcopy(stage_specs[int(new_stage_id)]))

    def activate() -> None:
        state["map_active"] = True
        refresh()

    def deactivate() -> None:
        state["map_active"] = False

    def close() -> None:
        state["closed"] = True
        stage.unobserve(change_stage, names="value")
        close_map()
        _close_stage_widgets(ui)

    refresh_area_options()
    effect.options = list(_DESIRE_LINE_EFFECTS[intervention_type.value])
    effect.value = next(iter(_DESIRE_LINE_EFFECTS[intervention_type.value]))
    stage.observe(change_stage, names="value")
    intervention_type.observe(change_family, names="value")
    existing.observe(load_selection, names="value")
    effect.observe(load_effect, names="value")
    spatial_level.observe(refresh_area_options, names="value")
    spatial_level.observe(lambda _: mark_map_stale(), names="value")
    show_boundaries.observe(lambda _: mark_map_stale(), names="value")
    replace_scope.observe(form_state, names="value")
    enabled.observe(set_enabled, names="value")
    new.on_click(lambda _: setattr(existing, "value", None))
    add.on_click(guarded(lambda: save_intervention(False)))
    update.on_click(guarded(lambda: save_intervention(True)))
    remove.on_click(guarded(remove_intervention))
    reset.on_click(guarded(reset_family))
    preview.on_click(refresh_map)
    refresh()
    ui = widgets.VBox((([stage] if owns_stage_selector else []) + [
        stage_summary,
        widgets.HBox([preview, show_boundaries]), map_status, map_output,
        *([intervention_type] if fixed_intervention_type is None else []),
        widgets.HBox([existing, new]), enabled, name, scope_note,
        replace_scope, spatial_level, widgets.HBox([origin, destination, both_directions]),
        widgets.HBox([effect, magnitude]), widgets.HBox([add, update, remove, reset]), status,
        table_output,
    ]))
    state.update(
        refresh=refresh, refresh_map=refresh_map, activate=activate, deactivate=deactivate,
        close=close, sync_from_specs=sync_from_specs, initialize_stage=initialize_stage,
        controls={"stage": stage, "intervention_type": intervention_type, "existing": existing,
                  "new": new, "enabled": enabled, "name": name, "replace_scope": replace_scope,
                  "spatial_level": spatial_level, "origin": origin, "destination": destination,
                  "both_directions": both_directions, "effect": effect, "magnitude": magnitude,
                  "add": add, "update": update, "remove": remove, "reset": reset,
                  "preview": preview, "status": status, "table": table_output,
                  "map_status": map_status, "scope_note": scope_note},
    )
    return ui, state


def stage_intervention_editor(
    context: tmi.TransportContext,
    corridor: tmi.CorridorContext,
    stage_specs: Mapping[int, Mapping[str, Any]],
    *,
    height: int = 560,
    packages: Mapping[str, Mapping[str, Any]] | None = None,
    combined_effects: Mapping[str, Any] | None = None,
    assemble_stages: Any | None = None,
    intervention_types: Sequence[str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Explore package edits in memory; prepare tables before optional maps.

    Package assembly is supplied by stages.py. Without it, arbitrary individual
    transport states can be edited. Pages follow the configured intervention keys.
    """
    import html
    import pprint
    import ipywidgets as widgets

    family_pages = {
        "railway_expansions": ("Public transport", "PT"),
        "bike_highways": ("Bicycle", "Bicycle"),
        "mobility_hubs": ("PT access and hubs", "Mobility hubs"),
        "road_capacity": (None, "Road links"),
    }
    effective = deepcopy(stage_specs)
    original = deepcopy(stage_specs)
    package_names = list(packages) if packages is not None else []
    if packages is not None:
        if not package_names or not callable(assemble_stages):
            raise ValueError("Package editing requires packages and an assemble_stages function.")
        edit_specs = {0: deepcopy(stage_specs[0])}
        edit_specs.update({i: deepcopy(dict(packages[name]))
                           for i, name in enumerate(package_names, 1)})
        combined_id = len(package_names) + 1
        edit_specs[combined_id] = {**deepcopy(dict(combined_effects or {})),
                                   "name": "Combined-only additions"}
        for i, name in enumerate(package_names, 1):
            edit_specs[i].setdefault("name", name)
    else:
        edit_specs = deepcopy(stage_specs)
    editable = [s for s in sorted(edit_specs) if int(s) != 0]
    if not editable:
        raise ValueError("Define at least one intervention package or non-baseline stage.")
    if intervention_types is None:
        selected_types = [key for key in family_pages
                          if any(spec.get(key) for s, spec in edit_specs.items() if s != 0)]
        selected_types = selected_types or list(family_pages)
    else:
        selected_types = list(dict.fromkeys(intervention_types))
        unknown = set(selected_types) - set(family_pages)
        if unknown or not selected_types:
            raise ValueError(f"intervention_types must select from {list(family_pages)}; unknown: {sorted(unknown)}")
    stage = widgets.Dropdown(
        options=[(_stage_editor_label(s, edit_specs[s]), s) for s in editable],
        value=editable[0], description="Edit package:" if package_names else "Edit stage:",
        layout=widgets.Layout(width="650px"), style={"description_width": "initial"},
    )
    status = widgets.HTML()
    export = widgets.Textarea(layout=widgets.Layout(width="100%", height="230px"), disabled=True)
    export_box = widgets.Accordion(children=[export], selected_index=None)
    export_box.set_title(0, "Current definitions to copy to stages.py")
    state = {"stage_specs": effective, "original_stage_specs": original,
             "editing_specs": edit_specs, "stage_selector": stage, "revision": 0,
             "listeners": [], "corridors": {}, "road_audits": {},
             "original_road_contexts": {}, "desire_line_states": {}, "page_states": [],
             "intervention_types": selected_types, "status": status}

    def assembled() -> dict:
        if not package_names:
            return deepcopy(edit_specs)
        working_packages = {name: deepcopy(edit_specs[i])
                            for i, name in enumerate(package_names, 1)}
        bonus = deepcopy(edit_specs[combined_id])
        bonus.pop("name", None)
        return assemble_stages(working_packages, bonus)

    def road_contexts(specifications: Mapping, target: dict, audits: dict | None = None) -> None:
        for key, spec in specifications.items():
            if spec.get("road_capacity"):
                target[key], audit = tmi.apply_road_capacity_stage(corridor, spec)
            else:
                target[key], audit = corridor, pd.DataFrame()
            if audits is not None:
                audits[key] = audit

    def export_definitions() -> None:
        if package_names:
            value = {name: edit_specs[i] for i, name in enumerate(package_names, 1)}
            bonus = {k: v for k, v in edit_specs[combined_id].items() if k != "name"}
            export.value = "PACKAGES = " + pprint.pformat(value, sort_dicts=False) + "\n\nCOMBINED_EFFECTS = " + pprint.pformat(bonus, sort_dicts=False)
        else:
            export.value = pprint.pformat(edit_specs, sort_dicts=False)

    def changed(stage_id: int, family: str) -> None:
        updated = assembled()
        # Publish only after assembly succeeds; keep references used in §3.3.
        if family == "road_capacity":
            new_roads, new_audits = {}, {}
            road_contexts(updated, new_roads, new_audits)
            state["corridors"].clear(); state["corridors"].update(new_roads)
            state["road_audits"].clear(); state["road_audits"].update(new_audits)
        affected = [spec.get("name", str(key)) for key, spec in updated.items()
                    if spec != effective.get(key)]
        effective.clear(); effective.update(updated)
        state["revision"] += 1
        export_definitions()
        status.value = ("<b>Working copy updated.</b> " + html.escape("; ".join(affected))
                        + "<br><small>In memory only. Compare values below; run Section 3.3 for model outcomes.</small>")
        for listener in tuple(state["listeners"]):
            try:
                listener()
            except Exception as error:
                status.value += ("<br><small>Comparison needs reopening: "
                                 + html.escape(str(error)) + "</small>")

    road_contexts(original, state["original_road_contexts"], state["road_audits"])
    state["corridors"].update(state["original_road_contexts"])
    tabs = widgets.Tab()

    def add_page(key: str) -> None:
        component, title = family_pages[key]
        if component is None:
            page, page_state = road_link_stage_editor(
                context, corridor, edit_specs, height=height, stage_selector=stage,
                defer_map=True, on_change=changed)
            state["road_state"] = page_state
        else:
            page, page_state = desire_line_stage_editor(
                context, corridor, edit_specs, height=height, stage_selector=stage,
                fixed_intervention_type=component, defer_map=True, on_change=changed)
            state["desire_line_states"][component] = page_state
            state.setdefault("desire_line_state", page_state)
        state["page_states"].append(page_state)
        tabs.children = (*tabs.children, page)
        tabs.set_title(len(tabs.children) - 1, title)

    for key in selected_types:
        add_page(key)
    extra_type = widgets.Dropdown(description="Add page:", layout=widgets.Layout(width="350px"))
    extra_button = widgets.Button(description="Enable editor", icon="plus")

    def refresh_extra() -> None:
        extra_type.options = [(title, key) for key, (_, title) in family_pages.items()
                              if key not in selected_types]
        extra_button.disabled = not extra_type.options
        extra_type.value = extra_type.options[0][1] if extra_type.options else None

    def enable_page(_: Any) -> None:
        key = extra_type.value
        if key and key not in selected_types:
            add_page(key)
            selected_types.append(key)
            tabs.selected_index = len(tabs.children) - 1
            refresh_extra()

    extra_button.on_click(enable_page)
    refresh_extra()
    active = {"index": 0}
    state["page_states"][0]["activate"]()

    def switch_page(change: dict) -> None:
        if change["new"] is None:
            return
        state["page_states"][active["index"]]["deactivate"]()
        active["index"] = int(change["new"])
        state["page_states"][active["index"]]["activate"]()

    tabs.observe(switch_page, names="selected_index")
    export_definitions()
    ui = widgets.VBox([
        widgets.HTML("<b>Temporary exploration.</b> Changes update the working stages, including combined states. "
                     "Save final definitions in <code>stages.py</code>." if package_names else
                     "<b>Temporary exploration.</b> Each stage is edited independently. Save final definitions in <code>stages.py</code>."),
        stage, status, tabs, widgets.HBox([extra_type, extra_button]), export_box,
    ])

    def close() -> None:
        tabs.unobserve(switch_page, names="selected_index")
        for page_state in state["page_states"]:
            page_state["close"]()
        state["listeners"].clear()
        _close_stage_widgets(ui)

    state.update({"tabs": tabs, "active_page": active, "close": close,
                  "publish": changed, "export": export, "add_page": extra_type,
                  "enable_page": extra_button})
    return ui, state


def baseline_relative_values(frame: pd.DataFrame, baseline: Any) -> pd.DataFrame:
    """Subtract the named baseline row from numeric indicators without changing inputs."""
    if not frame.columns.is_unique:
        raise ValueError("Indicator columns must have unique names.")
    baseline_rows = frame.index == baseline
    if int(np.count_nonzero(baseline_rows)) != 1:
        raise ValueError(f"Expected exactly one baseline row named {baseline!r}.")
    numeric = frame.apply(pd.to_numeric, errors="raise")
    return numeric.subtract(numeric.loc[baseline], axis="columns")


def plot_baseline_changes(
    frame: pd.DataFrame,
    baseline: Any,
    *,
    title: str,
    panels: Mapping[str, Sequence[str]] | None = None,
    labels: Mapping[str, str] | None = None,
    units: Mapping[str, str] | None = None,
    scales: Mapping[str, float] | None = None,
    directions: Mapping[str, str] | None = None,
    ncols: int = 2,
    fontsize: float = 12,
) -> Any:
    """Plot signed changes from the same-year baseline in original reporting units.

    Panels group source columns; by default Year 1/Year 40 columns with the same
    suffix share a panel. Scales apply to individual columns (e.g. 100 for
    fractions to percentage points). Labels, units and low/high/context
    directions use panel keys. Green/red mark favourable/unfavourable changes;
    grey is used for contextual indicators. Each panel has its own numeric axis.
    The narrow default layout keeps labels readable at notebook display width.
    """
    import math
    import textwrap

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    delta = baseline_relative_values(frame, baseline)
    # Display subtraction roundoff as zero, relative to the precision of the inputs.
    magnitude = np.maximum(1.0, np.maximum(
        np.abs(frame.to_numpy(dtype=float)), np.abs(frame.loc[baseline].to_numpy(dtype=float))))
    delta = delta.mask(np.isfinite(delta) & (delta.abs() <= 32 * np.finfo(float).eps * magnitude), 0.0)
    delta = delta.loc[delta.index != baseline]
    if delta.empty or not len(delta.columns):
        raise ValueError("Select at least one indicator and one alternative to the baseline.")
    labels, units, scales, directions = labels or {}, units or {}, scales or {}, directions or {}
    if panels is None:
        panels = {}
        for column in delta.columns:
            name = str(column)
            key = next((name[len(prefix):] for prefix in ("Year 1 ", "Year 40 ", "Year1 ", "Year40 ")
                        if name.startswith(prefix)), name)
            panels.setdefault(key, []).append(column)
    if not panels:
        raise ValueError("Select at least one indicator panel.")
    if not isinstance(ncols, int) or isinstance(ncols, bool) or not 1 <= ncols <= 4:
        raise ValueError("ncols must be an integer between 1 and 4.")
    if not np.isfinite(fontsize) or fontsize <= 0:
        raise ValueError("fontsize must be a positive number.")
    for key, columns in panels.items():
        if not columns or isinstance(columns, str):
            raise ValueError(f"Panel {key!r} needs a nonempty list of indicator columns.")
        missing = [column for column in columns if column not in delta.columns]
        if missing:
            raise ValueError(f"Unknown indicator columns in {key!r}: {missing}")
        if directions.get(key, "context") not in {"low", "high", "context"}:
            raise ValueError(f"Direction for {key!r} must be low, high or context.")
    for column, scale in scales.items():
        if column not in delta.columns or not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid indicator scale for {column!r}: {scale!r}")
        delta[column] *= float(scale)

    count = len(panels)
    ncols = min(ncols, count)
    nrows = math.ceil(count / ncols)
    row_height = 1.65 + 0.65 * len(delta)
    figure_height = nrows * row_height + 1.2
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.2 + 4.1 * ncols, figure_height),
        squeeze=False, constrained_layout=True,
    )
    y = np.arange(len(delta))
    colours = {"favourable": "#278360", "unfavourable": "#bd4b4b", "context": "#59738b"}
    hatches = ("", "///", "...", "xx")
    used_series: dict[str, str] = {}
    has_direction = any(directions.get(key, "context") != "context" for key in panels)

    def series_name(column: str) -> str:
        if str(column).startswith(("Year 1 ", "Year1 ")):
            return "Year 1"
        if str(column).startswith(("Year 40 ", "Year40 ")):
            return "Year 40"
        return str(column)

    def number(value: float) -> str:
        if value == 0:
            return "0"
        if abs(value) < 1:
            return f"{value:+.3g}"
        return f"{value:+,.2f}".rstrip("0").rstrip(".")

    for index, (key, columns) in enumerate(panels.items()):
        ax = axes.flat[index]
        width = 0.72 / len(columns)
        values_all = delta[list(columns)].to_numpy(dtype=float)
        finite = values_all[np.isfinite(values_all)]
        minimum, maximum = (min(0.0, float(finite.min())), max(0.0, float(finite.max()))) if finite.size else (0.0, 0.0)
        span = maximum - minimum
        if span == 0:
            span = 1.0
        ax.set_xlim(minimum - 0.25 * span, maximum + 0.25 * span)
        for series_index, column in enumerate(columns):
            values = delta[column].to_numpy(dtype=float)
            positions = y + (series_index - (len(columns) - 1) / 2) * width
            direction = directions.get(key, "context")
            score = values * (-1 if direction == "low" else 1)
            bar_colours = [
                colours["context"] if direction == "context" or value == 0 else
                colours["favourable"] if value > 0 else colours["unfavourable"]
                for value in score
            ]
            hatch = hatches[series_index % len(hatches)]
            ax.barh(positions, values, height=width * 0.9, color=bar_colours,
                    edgecolor="white", linewidth=0.5, hatch=hatch)
            if len(columns) > 1:
                used_series.setdefault(series_name(column), hatch)
            for position, value in zip(positions, values):
                if np.isfinite(value):
                    ax.annotate(number(float(value)), (value, position),
                                xytext=(3 if value >= 0 else -3, 0), textcoords="offset points",
                                va="center", ha="left" if value >= 0 else "right", fontsize=fontsize - 1)
                else:
                    ax.annotate("n/a", (0, position), xytext=(3, 0), textcoords="offset points",
                                va="center", fontsize=fontsize - 1, color="#666666")
        ax.axvline(0, color="#555555", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([textwrap.fill(str(value), width=30, break_long_words=False)
                            for value in delta.index] if index % ncols == 0 else [])
        ax.invert_yaxis()
        ax.set_title("\n".join(textwrap.fill(line, width=34) for line in labels.get(key, key).splitlines()),
                     fontsize=fontsize + 1, pad=10)
        ax.set_xlabel(textwrap.fill(units.get(key, "Change from baseline"), width=34), fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=fontsize - 1)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda value, _: f"{value:,.2f}".rstrip("0").rstrip(".") if abs(value) >= 1 else f"{value:.3g}"))
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    legend = [Patch(facecolor="#8b97a1", edgecolor="white", hatch=hatch, label=name)
              for name, hatch in used_series.items()]
    if has_direction:
        legend.extend([Patch(facecolor=colours["favourable"], label="Favourable change"),
                       Patch(facecolor=colours["unfavourable"], label="Unfavourable change")])
    if legend:
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0),
                   ncol=min(2, len(legend)), fontsize=fontsize - 1, frameon=False)
        fig.get_layout_engine().set(rect=(0, 0.7 / figure_height, 1, 1 - 0.7 / figure_height))
    fig.suptitle(textwrap.fill(title, width=85 if ncols > 1 else 48), fontsize=fontsize + 2)
    return fig
