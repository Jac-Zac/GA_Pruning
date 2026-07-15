# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.14",
#     "numpy>=2.0",
#     "plotly>=6.0",
# ]
# ///

# NOTE: AI generated conversion notebook from my original report be fully self-substained

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", app_title="GAMO pruning report")


@app.cell(hide_code=True)
def _():
    import base64
    import json
    import mimetypes
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    RANDOM = "Random"
    MAGNITUDE = "Magnitude"
    MAGNITUDE_PER_LAYER = "Magnitude (per-layer)"
    RANDOM_SEARCH = "Random search (equal budget)"
    HILL_CLIMBING = "Hill climbing (equal budget)"
    STRUCTURED_REPORT_METHODS = (
        "GA-uniform",
        "GA-two_point",
        HILL_CLIMBING,
        RANDOM_SEARCH,
        MAGNITUDE,
        MAGNITUDE_PER_LAYER,
        RANDOM,
    )
    STRUCTURED_METHOD_LABELS = {
        "GA-uniform": "GA · uniform",
        "GA-two_point": "GA · two-point",
        HILL_CLIMBING: "Hill climbing",
        RANDOM_SEARCH: "Random search",
        MAGNITUDE: "Magnitude · global",
        MAGNITUDE_PER_LAYER: "Magnitude · per layer",
        RANDOM: "Random mask",
    }

    _notebook_location = mo.notebook_location()
    if _notebook_location is None:
        raise RuntimeError("Unable to determine the notebook location.")
    _public = _notebook_location / "public"

    def _read_json(filename):
        source = _public / "results" / filename
        if sys.platform == "emscripten":
            from pyodide.http import open_url

            return json.load(open_url(str(source)))
        with Path(source).open(encoding="utf-8") as handle:
            return json.load(handle)

    def _asset_uri(filename):
        source = _public / "images" / filename
        if sys.platform == "emscripten":
            return str(source)
        path = Path(source)
        if not path.exists():
            return ""
        mime = mimetypes.guess_type(path)[0] or "image/webp"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _structured_artifact_error(sweep):
        required = {
            "sparsities",
            "architecture",
            "methods",
            "dense_acc",
            "dense_val_acc",
            "config",
        }
        if not required.issubset(sweep):
            return "The structured result is incomplete; missing: " + ", ".join(
                sorted(required - set(sweep))
            )
        sparsities = sweep["sparsities"]
        architecture = sweep["architecture"]
        hidden_sizes = architecture.get("hidden_sizes", [])
        if (
            not sparsities
            or architecture.get("input_size", 0) <= 0
            or architecture.get("output_size", 0) <= 0
            or not hidden_sizes
        ):
            return "The structured result has an empty sparsity sweep or invalid architecture."
        missing = set(STRUCTURED_REPORT_METHODS) - set(sweep["methods"])
        if missing:
            return "The structured result is missing methods: " + ", ".join(
                sorted(missing)
            )
        genome_length = sum(hidden_sizes)
        for method in STRUCTURED_REPORT_METHODS:
            result = sweep["methods"][method]
            if len(result.get("accuracy", [])) != len(sparsities) or len(
                result.get("accuracy_std", [])
            ) != len(sparsities):
                return (
                    f"The structured result has the wrong series length for {method}."
                )
            for sparsity in sparsities:
                runs = result.get("runs", {}).get(f"{sparsity:.4f}", [])
                if not runs or any(
                    len(run.get("mask", [])) != genome_length for run in runs
                ):
                    return f"The structured result has missing or invalid runs for {method}."
        return None

    def load_report_artifacts():
        try:
            sweep = _read_json("structured.json")
            unstructured = _read_json("unstructured.json")
            ablation = _read_json("ablation.json")
        except (FileNotFoundError, OSError, ValueError) as exc:
            return None, None, None, f"Unable to load the public report data: {exc}"
        error = _structured_artifact_error(sweep)
        if error is None and not {
            "sparsities",
            "dense_acc",
            "parameter_counts",
            "config",
        }.issubset(unstructured):
            error = "The unstructured result is missing or incomplete."
        if error is None and not {"sensitivity", "config"}.issubset(ablation):
            error = "The sensitivity result is missing or incomplete."
        if error is None and unstructured["dense_acc"] != sweep["dense_acc"]:
            error = "Structured and unstructured artifacts use different checkpoints."
        return sweep, unstructured, ablation, error

    def mean_runs(runs):
        values = list(runs)
        if values and any(len(value) != len(values[0]) for value in values[1:]):
            raise ValueError("artifact runs must have equal lengths")
        return [sum(point) / len(point) for point in zip(*values)] if values else []

    def _runs(sweep, method, sparsity):
        try:
            return sweep["methods"][method]["runs"][f"{sparsity:.4f}"]
        except KeyError as exc:
            raise ValueError(f"missing runs for {method} at {sparsity:.0%}") from exc

    def effective_sparsity_rows(sweep, primary_method):
        rows = []
        weight_sparsities = []
        for index, sparsity in enumerate(sweep["sparsities"]):
            method = sweep["methods"][primary_method]
            stats = [
                run["sparsity_stats"] for run in _runs(sweep, primary_method, sparsity)
            ]
            weight_mean = sum(item["weight_sparsity"] for item in stats) / len(stats)
            parameter_mean = sum(item["parameter_sparsity"] for item in stats) / len(
                stats
            )
            weight_sparsities.append(weight_mean)
            rows.append(
                {
                    "Neuron sparsity": f"{sparsity:.0%}",
                    "Weight sparsity": f"{weight_mean:.1%}",
                    "Parameter sparsity": f"{parameter_mean:.1%}",
                    "Test accuracy": f"{method['accuracy'][index]:.1f}%",
                }
            )
        return rows, weight_sparsities

    def _split_mask(mask, hidden_sizes):
        flat = np.asarray(mask, dtype=bool).reshape(-1)
        if flat.size != sum(hidden_sizes):
            raise ValueError("neuron mask length does not match hidden_sizes")
        return tuple(np.split(flat, np.cumsum(hidden_sizes)[:-1]))

    def neuron_to_weight_masks(
        neuron_mask, hidden_sizes, input_size=784, output_size=10
    ):
        layers = _split_mask(neuron_mask, hidden_sizes)
        weight_masks = {
            "Input → L0": np.broadcast_to(
                layers[0][:, None], (hidden_sizes[0], input_size)
            )
        }
        for index in range(1, len(layers)):
            weight_masks[f"L{index - 1} → L{index}"] = (
                layers[index][:, None] & layers[index - 1][None, :]
            )
        weight_masks[f"L{len(layers) - 1} → Output"] = np.broadcast_to(
            layers[-1][None, :], (output_size, hidden_sizes[-1])
        )
        return weight_masks

    def weight_layer_rows(weight_masks):
        rows = []
        for layer, mask in weight_masks.items():
            total = mask.size
            kept = int(mask.sum())
            rows.append(
                {
                    "Weight matrix": layer,
                    "Shape": f"{mask.shape[0]} × {mask.shape[1]}",
                    "Weights removed": f"{total - kept:,}",
                    "Weight sparsity": f"{1 - kept / total:.1%}",
                }
            )
        return rows

    def neuron_layer_rows(neuron_mask, hidden_sizes):
        rows = []
        for index, layer_mask in enumerate(
            _split_mask(neuron_mask, hidden_sizes), start=1
        ):
            kept = int(layer_mask.sum())
            total = layer_mask.size
            rows.append(
                {
                    "Hidden layer": f"Layer {index}",
                    "Neurons kept": f"{kept:,} / {total:,}",
                    "Neurons removed": f"{total - kept:,}",
                    "Neuron sparsity": f"{1 - kept / total:.1%}",
                }
            )
        return rows

    def structured_explorer_view(sweep, method, sparsity):
        if method not in sweep["methods"]:
            raise ValueError(f"unknown structured method: {method}")
        try:
            index = sweep["sparsities"].index(sparsity)
        except ValueError as exc:
            raise ValueError(f"unknown structured sparsity: {sparsity}") from exc
        result = sweep["methods"][method]
        runs = _runs(sweep, method, sparsity)
        representative = next(
            (run for run in runs if run["seed"] == sweep["config"]["seed"]), runs[0]
        )
        raw_mask = np.asarray(representative["mask"], dtype=bool)
        architecture = sweep["architecture"]
        return {
            "accuracy": result["accuracy"][index],
            "std": result["accuracy_std"][index],
            "weight_sparsity": sum(
                float(run["sparsity_stats"]["weight_sparsity"]) for run in runs
            )
            / len(runs),
            "curve": mean_runs(run["curve"] for run in runs if run["curve"]),
            "distribution": mean_runs(run["layer_sparsity"] for run in runs),
            "neuron_mask": raw_mask,
            "representative_seed": representative["seed"],
        }

    pruning_img = _asset_uri("pruning.webp")
    steps_img = _asset_uri("steps.webp")
    training_protocol = {
        "epochs": 10,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "minimum_learning_rate": 1e-5,
        "weight_decay": 1e-3,
        "validation_split": 0.3,
        "seed": 1337,
        "normalization_mean": 0.286,
        "normalization_std": 0.353,
    }
    _source_url = "https://github.com/Jac-Zac/GA_Pruning/blob/main"
    ga_step_sources = {
        "Complete evolution loop": f"# Full implementation: {_source_url}/src/gamo/ga/search.py",
        "Initialization": f"# Full implementation: {_source_url}/src/gamo/ga/operators.py#L12-L30",
        "Selection": f"# Full implementation: {_source_url}/src/gamo/ga/operators.py#L33-L59",
        "Crossover": f"# Full implementation: {_source_url}/src/gamo/ga/operators.py#L62-L98",
        "Mutation": f"# Full implementation: {_source_url}/src/gamo/ga/operators.py#L117-L124",
        "Mask repair": f"# Full implementation: {_source_url}/src/gamo/ga/operators.py#L101-L114",
    }

    # ---------------------------------------------------------------------------
    # Palette — baselines in warm neutrals, GA configs in a qualitative palette.
    # ---------------------------------------------------------------------------
    INK = "#1c1b19"

    COL: dict[str, str] = {
        "Random": "#9ca3af",
        "Random search (equal budget)": "#6b7280",
        "Magnitude": "#b08968",
        "Magnitude (per-layer)": "#dcab6b",
        "Dense": "#8a8a8a",
    }

    GA_COLORS: list[str] = [
        "#3a8a94",
        "#7b4d91",
        "#d97c3d",
        "#4d8f5c",
        "#c04851",
        "#5b8db8",
    ]

    def _ga_color(label: str) -> str:
        # Use a stable label-derived color instead of Python's randomized hash.
        idx = sum(ord(c) for c in label) % len(GA_COLORS)
        return GA_COLORS[idx]

    # ---------------------------------------------------------------------------
    # Layout helper
    # ---------------------------------------------------------------------------
    def _plotly_layout(**kw: float | str | dict) -> dict:
        base = dict(
            font=dict(
                family="Inter, system-ui, -apple-system, sans-serif",
                color=INK,
                size=13,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=52, r=12, t=42, b=40),
            legend=dict(
                bgcolor="rgba(255,255,255,0.65)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0,
            ),
            hoverlabel=dict(font_size=12),
            xaxis=dict(
                gridcolor="#ece6db", zerolinecolor="#ddd4c6", linecolor="#ddd4c6"
            ),
            yaxis=dict(
                gridcolor="#ece6db", zerolinecolor="#ddd4c6", linecolor="#ddd4c6"
            ),
        )
        base.update(kw)
        return base

    # ---------------------------------------------------------------------------
    # Accuracy versus sparsity sweep
    # ---------------------------------------------------------------------------
    def fig_accuracy_vs_sparsity(
        sparsities: list[float],
        series: dict[str, list[float]],
        series_std: dict[str, list[float]],
        dense_acc: float,
        method_order: list[str] | None = None,
        title: str = "Test accuracy vs neuron sparsity",
        sparsity_label: str = "Neurons pruned (%)",
    ) -> go.Figure:
        """Line-plot of test accuracy across the sparsity sweep."""
        sp = [s * 100 for s in sparsities]
        fig = go.Figure()
        fig.add_hline(
            y=dense_acc,
            line=dict(color=COL["Dense"], dash="dot", width=1.5),
            annotation_text=f"dense {dense_acc:.1f}%",
            annotation_position="top left",
        )
        order = method_order or list(series)
        for name in order:
            if name not in series:
                continue
            is_ga = name.startswith("GA-")
            color = COL.get(name, _ga_color(name))
            fig.add_trace(
                go.Scatter(
                    x=sp,
                    y=series[name],
                    mode="lines+markers",
                    name=name,
                    error_y=(
                        dict(type="data", array=series_std[name], visible=True)
                        if name in series_std
                        else None
                    ),
                    line=dict(
                        color=color,
                        width=3 if is_ga else 2,
                        dash="dash" if not is_ga else None,
                    ),
                    marker=dict(
                        size=8 if is_ga else 6,
                        symbol="square" if not is_ga else None,
                    ),
                )
            )
        fig.update_layout(
            **_plotly_layout(
                title=title,
                xaxis_title=sparsity_label,
                yaxis_title="Test accuracy (%)",
                hovermode="x unified",
                height=500,
            )
        )
        return fig

    def fig_accuracy_snapshot(
        accuracies: dict[str, float],
        stds: dict[str, float],
        dense_acc: float,
        sparsity: float,
        display_names: dict[str, str] | None = None,
    ) -> go.Figure:
        """Horizontal comparison of every structured method at one sparsity."""
        names = list(accuracies)
        labels = [
            display_names.get(name, name) if display_names else name for name in names
        ]
        values = [accuracies[name] for name in names]
        fig = go.Figure(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color=[COL.get(name, _ga_color(name)) for name in names],
                error_x=dict(
                    type="data",
                    array=[stds.get(name, 0.0) for name in names],
                    visible=True,
                ),
                text=[f"{value:.1f}%" for value in values],
                textposition="inside",
                insidetextanchor="middle",
                cliponaxis=False,
                hovertemplate="%{y}<br>test accuracy: %{x:.2f}%<extra></extra>",
            )
        )
        fig.add_vline(
            x=dense_acc,
            line=dict(color=COL["Dense"], dash="dot", width=1.5),
            annotation_text=f"dense {dense_acc:.1f}%",
            annotation_position="top right",
        )
        fig.update_layout(
            **_plotly_layout(
                title=f"All structured methods at {sparsity:.0%} neuron sparsity",
                xaxis_title="Test accuracy (%)",
                yaxis_title="",
                height=max(410, 42 * len(names) + 120),
                margin=dict(l=175, r=75, t=50, b=42),
            )
        )
        ceiling = max([dense_acc, *values]) * 1.08
        fig.update_xaxes(range=[0, ceiling])
        fig.update_yaxes(autorange="reversed")
        return fig

    # ---------------------------------------------------------------------------
    # Per-layer pruning distribution
    # ---------------------------------------------------------------------------
    def fig_distribution(
        dist: dict[str, list[float]],
        n_layers: int,
        display_names: dict[str, str] | None = None,
    ) -> go.Figure:
        """Show per-layer pruning as bars for one method or a readable comparison grid."""
        layers = [f"Layer {i + 1}" for i in range(n_layers)]
        rows = []
        for key, fractions in dist.items():
            method = key.split("@")[0]
            label = display_names.get(method, method) if display_names else method
            rows.append((label, [fraction * 100 for fraction in fractions]))

        if len(rows) > 4:
            fig = go.Figure(
                go.Heatmap(
                    x=layers,
                    y=[label for label, _ in rows],
                    z=[values for _, values in rows],
                    zmin=0,
                    zmax=100,
                    colorscale=[[0, "#efe3d3"], [1, "#8bb7bb"]],
                    texttemplate="%{z:.0f}%",
                    textfont=dict(color=INK),
                    hovertemplate=(
                        "%{y}<br>%{x}<br>neurons pruned: %{z:.1f}%<extra></extra>"
                    ),
                    colorbar=dict(title="Pruned (%)"),
                )
            )
            fig.update_layout(
                **_plotly_layout(
                    title="Where each method prunes",
                    xaxis_title="Hidden layer",
                    yaxis_title="",
                    height=max(390, 38 * len(rows) + 140),
                    margin=dict(l=175, r=75, t=50, b=42),
                )
            )
            fig.update_yaxes(autorange="reversed")
            return fig

        fig = go.Figure()
        for display_name, values in rows:
            color = COL.get(display_name, _ga_color(display_name))
            fig.add_trace(
                go.Bar(
                    x=layers,
                    y=values,
                    name=display_name,
                    marker_color=color,
                    text=[f"{value:.0f}%" for value in values],
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate=("%{x}<br>neurons pruned: %{y:.1f}%<extra></extra>"),
                )
            )
        fig.update_layout(
            **_plotly_layout(
                title="Where each method prunes — fraction of neurons removed per layer",
                xaxis_title="Hidden layer",
                yaxis_title="Neurons pruned (%)",
                barmode="group",
                height=420,
            )
        )
        fig.update_yaxes(range=[0, 105])
        return fig

    # ---------------------------------------------------------------------------
    # Search convergence curves
    # ---------------------------------------------------------------------------
    def fig_progress(curves: dict[str, list[float]], dense_val_acc: float) -> go.Figure:
        """Plot best validation accuracy per generation for search methods."""
        fig = go.Figure()
        fig.add_hline(
            y=dense_val_acc,
            line=dict(color=COL["Dense"], dash="dot", width=1.5),
            annotation_text=f"dense validation {dense_val_acc:.1f}%",
            annotation_position="top left",
        )
        for key, curve in curves.items():
            if not (
                key.startswith("GA-")
                or key.startswith("Random search (equal budget)")
                or key.startswith("Hill climbing (equal budget)")
            ):
                continue
            display_name = key.split("@")[0]
            is_random_search = display_name == "Random search (equal budget)"
            fig.add_trace(
                go.Scatter(
                    y=curve,
                    mode="lines",
                    name=display_name,
                    line=dict(
                        color=COL.get(display_name, _ga_color(display_name)),
                        width=2.5,
                        dash="dash" if is_random_search else None,
                    ),
                )
            )
        fig.update_layout(
            **_plotly_layout(
                title="Search convergence — best validation accuracy per generation",
                xaxis_title="Generation",
                yaxis_title="Best validation accuracy (%)",
                hovermode="x unified",
                height=420,
            )
        )
        return fig

    # ---------------------------------------------------------------------------
    # Figure — crossover convergence
    # ---------------------------------------------------------------------------
    def fig_crossover_convergence(
        curves: dict[str, list[float]], sp: float, dense_val_acc: float
    ) -> go.Figure:
        """Plot the convergence curves for each crossover strategy."""
        fig = go.Figure()
        fig.add_hline(
            y=dense_val_acc,
            line=dict(color=COL["Dense"], dash="dot", width=1.5),
            annotation_text=f"dense validation {dense_val_acc:.1f}%",
            annotation_position="top left",
        )
        for key in sorted(curves):
            if not key.startswith("GA-"):
                continue
            crossover = key.split("@")[0].removeprefix("GA-")
            curve = curves[key]
            fig.add_trace(
                go.Scatter(
                    y=curve,
                    mode="lines",
                    name=crossover.replace("_", " "),
                    line=dict(
                        color="#3a8a94" if crossover == "uniform" else "#d97c3d",
                        width=2.5,
                    ),
                )
            )
        fig.update_layout(
            **_plotly_layout(
                title=f"Crossover convergence at {sp:.0%} sparsity",
                xaxis_title="Generation",
                yaxis_title="Best validation accuracy (%)",
                hovermode="x unified",
                height=420,
            )
        )
        return fig

    def fig_ablation_accuracy(
        study: dict, dense_acc: float, sparsity: float
    ) -> go.Figure:
        """Compare final test accuracy for every sensitivity-study variant."""
        names = list(study["values"])
        display_names = [
            "Baseline (canonical GA)" if name == "Canonical" else name for name in names
        ]
        accuracies = list(study["accuracies"])
        stds = list(study["stds"])
        group_colors = {
            "Operators": "#c04851",
            "Tournament size": "#5b8db8",
            "Population size": "#b08968",
        }
        colors = [
            INK if name == "Canonical" else group_colors[group]
            for name, group in zip(names, study["groups"])
        ]
        fig = go.Figure(
            go.Bar(
                x=accuracies,
                y=display_names,
                orientation="h",
                marker_color=colors,
                error_x=dict(type="data", array=stds, visible=True),
                text=[f"{accuracy:.1f}%" for accuracy in accuracies],
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate=(
                    "%{y}<br>mean test accuracy: %{x:.2f}%"
                    "<br>standard deviation: %{error_x.array:.2f} pp<extra></extra>"
                ),
            )
        )
        fig.add_vline(
            x=dense_acc,
            line=dict(color=COL["Dense"], dash="dot", width=1.5),
            annotation_text=f"dense model {dense_acc:.1f}%",
            annotation_position="top right",
        )
        fig.update_layout(
            **_plotly_layout(
                title=f"Sensitivity study at {sparsity:.0%} neuron sparsity",
                xaxis_title="Test accuracy (%)",
                yaxis_title="",
                height=455,
                margin=dict(l=175, r=70, t=55, b=42),
                showlegend=False,
            )
        )
        fig.update_xaxes(range=[0, max(dense_acc, *accuracies) * 1.08])
        fig.update_yaxes(autorange="reversed")
        return fig

    # ---------------------------------------------------------------------------
    # Effective sparsity comparison (structured vs unstructured)
    # ---------------------------------------------------------------------------
    def fig_effective_sparsity_comparison(
        method_eff_sparsities: dict[str, list[float | None]],
        series: dict[str, list[float]],
        dense_acc: float,
        method_order: list[str] | None = None,
    ) -> go.Figure:
        """Compare methods on the common x-axis of actual removed weights."""
        fig = go.Figure()
        fig.add_hline(
            y=dense_acc,
            line=dict(color=COL["Dense"], dash="dot", width=1.5),
            annotation_text=f"dense {dense_acc:.1f}%",
            annotation_position="top left",
        )
        order = method_order or list(series)
        for name in order:
            if name not in series or name not in method_eff_sparsities:
                continue
            pairs = [
                (effective * 100, accuracy)
                for effective, accuracy in zip(
                    method_eff_sparsities[name], series[name]
                )
                if effective is not None
            ]
            if not pairs:
                continue
            x_vals, vals = zip(*pairs)
            is_ga = name.startswith("GA-") or name == "Best structured GA"
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=vals,
                    mode="lines+markers",
                    name=name,
                    line=dict(
                        color=COL.get(name, _ga_color(name)),
                        width=2.5,
                        dash=None if is_ga else "dash",
                    ),
                    marker=dict(size=8),
                )
            )
        fig.update_layout(
            **_plotly_layout(
                title="Accuracy at actual weight sparsity",
                xaxis_title="Effective weight sparsity (%)",
                yaxis_title="Test accuracy (%)",
                hovermode="x unified",
                height=460,
            )
        )
        return fig

    _BINARY_MASK_COLORSCALE = [
        [0.0, "#c0492f"],
        [0.5, "#efe3d3"],
        [1.0, "#2f8f5b"],
    ]

    def fig_neuron_mask_heatmap(
        neuron_masks: dict[str, np.ndarray],
        hidden_sizes: list[int],
        title: str = "Neuron masks — green = kept, red = removed",
    ) -> go.Figure:
        """Show structured chromosomes in one true-width subplot per hidden layer.

        Each plotted row represents one hidden layer, preserving the fact that the
        chromosome is a neuron keep-mask rather than an arbitrary weight matrix.
        """
        if not neuron_masks:
            raise ValueError("neuron_masks must not be empty")
        if not hidden_sizes:
            raise ValueError("hidden_sizes must not be empty")

        methods = list(neuron_masks)
        split_masks: dict[str, tuple[np.ndarray, ...]] = {}
        boundaries = np.cumsum(hidden_sizes)[:-1]
        for method, raw_mask in neuron_masks.items():
            mask = np.asarray(raw_mask, dtype=bool).reshape(-1)
            if mask.size != sum(hidden_sizes):
                raise ValueError("neuron mask length does not match hidden_sizes")
            split_masks[method] = tuple(np.split(mask, boundaries))

        fig = make_subplots(
            rows=1,
            cols=len(hidden_sizes),
            subplot_titles=[
                f"Layer {index} · {width} neurons"
                for index, width in enumerate(hidden_sizes, start=1)
            ],
            column_widths=hidden_sizes,
            horizontal_spacing=0.045,
        )
        for layer_index, width in enumerate(hidden_sizes, start=1):
            values = [
                [int(value) for value in split_masks[method][layer_index - 1]]
                for method in methods
            ]
            customdata = [
                [
                    [method, layer_index, neuron_index + 1]
                    for neuron_index in range(width)
                ]
                for method in methods
            ]
            fig.add_trace(
                go.Heatmap(
                    z=values,
                    x=list(range(1, width + 1)),
                    y=methods,
                    customdata=customdata,
                    zmin=0,
                    zmax=1,
                    colorscale=_BINARY_MASK_COLORSCALE,
                    showscale=layer_index == len(hidden_sizes),
                    colorbar=dict(
                        title="Mask bit",
                        tickmode="array",
                        tickvals=[0, 1],
                        ticktext=["removed", "kept"],
                    ),
                    hovertemplate=(
                        "%{customdata[0]}<br>hidden layer %{customdata[1]}"
                        "<br>neuron %{customdata[2]}<br>mask bit: %{z}"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=layer_index,
            )
            fig.update_xaxes(
                title_text="Neuron index",
                range=[0.5, width + 0.5],
                row=1,
                col=layer_index,
            )
            fig.update_yaxes(
                showticklabels=layer_index == 1,
                autorange="reversed",
                row=1,
                col=layer_index,
            )

        fig.update_layout(
            **_plotly_layout(
                title=title,
                height=max(270, 34 * len(methods) + 170),
                margin=dict(l=175, r=88, t=62, b=55),
            )
        )
        return fig

    def fig_weight_matrix_heatmap(
        weight_masks: dict[str, np.ndarray],
        max_dim: int = 256,
        title: str = "Full weight matrices — green = active, red = pruned",
    ) -> go.Figure:
        """Grid showing kept (green) vs pruned (red) regions in each weight matrix.

        Parameters
        ----------
        weight_masks : dict
            Keys like ``"Input → L0"``, ``"L0 → L1"``, etc. Values are 2D boolean
            tensors matching the weight matrix shape: True = kept, False = pruned.
        max_dim : int
            Downsample the full matrix to at most this many rows/columns. Each
            displayed cell is the active fraction in its source block.
        """
        keys = list(weight_masks)
        if not keys:
            raise ValueError("weight_masks must not be empty")
        columns = min(2, len(keys))
        rows = (len(keys) + columns - 1) // columns
        fig = make_subplots(
            rows=rows,
            cols=columns,
            subplot_titles=keys,
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
        )

        for idx, key in enumerate(keys):
            row = idx // columns + 1
            col = idx % columns + 1
            wmask = np.asarray(weight_masks[key], dtype=float)
            r = min(wmask.shape[0], max_dim)
            c = min(wmask.shape[1], max_dim)
            if (r, c) == wmask.shape:
                region = wmask
            else:
                # Equal-width bins retain the active fraction without a native tensor runtime.
                row_bins = np.linspace(0, wmask.shape[0], r + 1, dtype=int)
                col_bins = np.linspace(0, wmask.shape[1], c + 1, dtype=int)
                row_sums = np.add.reduceat(wmask, row_bins[:-1], axis=0)
                block_sums = np.add.reduceat(row_sums, col_bins[:-1], axis=1)
                block_sizes = np.diff(row_bins)[:, None] * np.diff(col_bins)[None, :]
                region = block_sums / block_sizes

            fig.add_trace(
                go.Heatmap(
                    z=region,
                    zmin=0,
                    zmax=1,
                    colorscale=_BINARY_MASK_COLORSCALE,
                    showscale=False,
                    hovertemplate=(
                        "display row %{y}<br>display col %{x}"
                        "<br>active fraction: %{z:.2f}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            fig.update_xaxes(visible=False, row=row, col=col)
            fig.update_yaxes(
                visible=False,
                autorange="reversed",
                row=row,
                col=col,
            )

        fig.update_layout(
            **_plotly_layout(
                title=title,
                height=250 * rows,
                margin=dict(l=16, r=16, t=54, b=16),
            )
        )
        return fig

    return (
        STRUCTURED_METHOD_LABELS,
        STRUCTURED_REPORT_METHODS,
        effective_sparsity_rows,
        fig_ablation_accuracy,
        fig_accuracy_snapshot,
        fig_accuracy_vs_sparsity,
        fig_crossover_convergence,
        fig_distribution,
        fig_effective_sparsity_comparison,
        fig_neuron_mask_heatmap,
        fig_progress,
        fig_weight_matrix_heatmap,
        ga_step_sources,
        load_report_artifacts,
        mo,
        neuron_to_weight_masks,
        neuron_layer_rows,
        pruning_img,
        steps_img,
        structured_explorer_view,
        training_protocol,
        weight_layer_rows,
    )


@app.cell(hide_code=True)
def _(load_report_artifacts):
    sweep, unstructured, ablation, report_error = load_report_artifacts()
    if report_error:
        raise RuntimeError(report_error)
    return ablation, sweep, unstructured


@app.cell(hide_code=True)
def _(mo, pruning_img, sweep):
    _architecture = " → ".join(
        [
            str(sweep["architecture"]["input_size"]),
            *(str(size) for size in sweep["architecture"]["hidden_sizes"]),
            str(sweep["architecture"]["output_size"]),
        ]
    )
    _title = mo.md(f"""
    # Genetic algorithms for neural-network pruning

    **How the mask representation changes the usefulness of genetic search**

    FashionMNIST · `{_architecture}` ReLU MLP · {sweep["dense_acc"]:.2f}% dense test
    accuracy · frozen weights after training
    """)
    _image = (
        mo.Html(
            f'<img src="{pruning_img}" alt="Dense and pruned neural networks" '
            'style="width:100%; border-radius:8px; margin-top:0.8rem;" />'
        )
        if pruning_img
        else mo.md("")
    )
    mo.vstack([_title, _image])
    return


@app.cell(hide_code=True)
def _(mo, steps_img):
    _image = (
        mo.Html(
            '<div style="max-width:100%; margin-top:0.75rem;">'
            f'<img src="{steps_img}" alt="Detailed genetic pruning workflow" '
            'style="display:block; max-width:100%; height:auto; margin:0 auto;" />'
            "</div>"
        )
        if steps_img
        else mo.md("The workflow asset is unavailable.")
    )
    mo.vstack(
        [
            mo.md("""
            ## 1. Experimental workflow
            The project compares weight-level and neuron-level pruning while keeping the
            trained weights, data split, fitness budget, and test protocol fixed.
            """),
            _image,
        ]
    )
    return


@app.cell(hide_code=True)
def _(ga_step_sources, mo):
    _summaries = {
        "Complete evolution loop": """
        1. Evaluate every mask using validation accuracy and retain the best seen mask.
        2. Preserve the generation's elite, then select and cross parents.
        3. Mutate and repair the children before evaluating the next generation.
        """,
        "Initialization": """
        1. Draw one random score for every gene in every mask.
        2. Keep the `target_ones` highest-scoring genes in each row.
        3. Return a Boolean population with the exact requested sparsity.
        """,
        "Selection": """
        1. Randomly draw `tournament_size` candidates for every parent slot.
        2. Compare their validation-fitness scores.
        3. Copy the winner of each tournament into the parent pool.
        """,
        "Crossover": """
        1. Split selected parents into pairs.
        2. Build a uniform or two-point Boolean crossover mask.
        3. Exchange the selected genes to create two complementary children.
        """,
        "Mutation": """
        1. Give each gene a flip probability of `1 / genome_length`.
        2. Sample all flips in one vectorized operation.
        3. Apply them with Boolean exclusive-or (`^`).
        """,
        "Mask repair": """
        1. Give existing kept genes priority over dropped genes.
        2. Randomly break ties within those two groups.
        3. Keep exactly `target_ones` genes in every child.
        """,
    }
    _step_tabs = {
        _name: mo.ui.tabs(
            {
                "In a few lines": mo.md(_summaries[_name]),
                "Actual project code": mo.md(
                    f"```python\n{ga_step_sources[_name]}\n```"
                ),
            }
        )
        for _name in _summaries
    }
    mo.vstack(
        [
            mo.md("### Inspect the main GA steps"),
            mo.md(
                "Choose the complete loop or one operator, then switch between a short "
                "explanation and the exact implementation used by the experiments."
            ),
            mo.ui.tabs(_step_tabs),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, sweep, training_protocol):
    _config = sweep["config"]
    _budget = _config["pop_size"] * _config["num_iterations"]
    mo.md(f"""
    ## 2. Training and experimental contract

    The MLP was trained for **{training_protocol["epochs"]} epochs** with AdamW,
    learning rate {training_protocol["learning_rate"]:.0e}, weight decay
    {training_protocol["weight_decay"]:.0e}, cosine decay, and a
    {training_protocol["validation_split"]:.0%} validation split. The checkpoint with the
    lowest validation loss was frozen before pruning.

    Each pruning chromosome is a binary keep-mask. Search fitness is full validation
    accuracy; the test set is used only after the best mask in a run has been selected.
    Every method uses the same exact integer keep count. GA, random search, and hill
    climbing each receive **{_budget:,} mask evaluations per seed and sparsity**.

    Results use {_config["num_seeds"]} mask-search seeds. The checkpoint and data split stay
    fixed, so error bars measure search randomness.
    """)
    return


@app.cell(hide_code=True)
def _(ablation, fig_accuracy_vs_sparsity, mo, unstructured):
    _entries = unstructured["sparsities"]
    _sparsities = [float(value) for value in _entries]
    _series = {
        "Magnitude": [_entries[str(value)]["mag_accuracy"] for value in _sparsities],
        "GA-unstructured": [
            _entries[str(value)]["ga_accuracy"] for value in _sparsities
        ],
        "Random": [_entries[str(value)]["random_accuracy"] for value in _sparsities],
    }
    _stds = {
        "GA-unstructured": [
            _entries[str(value)]["ga_accuracy_std"] for value in _sparsities
        ],
        "Random": [
            _entries[str(value)]["random_accuracy_std"] for value in _sparsities
        ],
    }
    _figure = fig_accuracy_vs_sparsity(
        _sparsities,
        _series,
        _stds,
        unstructured["dense_acc"],
        title="Unstructured pruning: magnitude remains the strongest baseline",
        sparsity_label="Weights pruned (%)",
    )
    _focus_sparsity = ablation["config"]["focus"]
    _focus = _entries[str(_focus_sparsity)]
    mo.vstack(
        [
            mo.md(f"""
            ## 3. Unstructured baseline: one bit per weight

            The unstructured chromosome contains one bit for each of
            **{unstructured["parameter_counts"]["weights"]:,} weights**. This is the most
            direct binary-mask formulation and matches the weight-level view common in
            pruning and lottery-ticket literature. Keeping weights frozen isolates mask
            selection from retraining effects. The GA improves
            greatly over random masks, but magnitude pruning is stronger at every tested
            sparsity. At {_focus_sparsity:.0%}, magnitude retains {_focus["mag_accuracy"]:.1f}% accuracy,
            while the GA retains {_focus["ga_accuracy"]:.1f}%.

            This is a useful negative result. The genome is extremely large, bit-flip
            mutation changes only a tiny fraction of it, crossover has little semantic
            locality at the weight level, and magnitude already provides a strong local
            importance signal. It does **not** prove that GAs can never perform
            unstructured pruning; it shows that this representation is poorly matched to
            this model and evaluation budget.
            """),
            mo.hstack(
                [
                    mo.stat(
                        f"{_focus['mag_accuracy']:.2f}%",
                        label=f"Magnitude · {_focus_sparsity:.0%}",
                    ),
                    mo.stat(
                        f"{_focus['ga_accuracy']:.2f}%",
                        label=f"GA · {_focus_sparsity:.0%}",
                    ),
                    mo.stat(
                        f"{_focus['mag_accuracy'] - _focus['ga_accuracy']:+.2f} pp",
                        label="Magnitude advantage",
                    ),
                ],
                widths="equal",
            ),
            mo.as_html(_figure),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, sweep):
    _config = sweep["config"]
    _genome = sum(sweep["architecture"]["hidden_sizes"])
    mo.md(f"""
    ## 4. Structured approach: one bit per hidden neuron
    > This is the regime where a GA has a real chance to help and pruning whole neurons can be materialized as a smaller dense network.

    The structured chromosome contains **{_genome} bits**, with one gene per hidden
    neuron. A zero disables that neuron's output;
    its incoming row, outgoing column, and hidden bias can then be removed from a compact
    network. This gives genes a functional meaning and makes the result implementable as
    smaller dense layers rather than scattered zeros.

    The GA starts from exact-cardinality random masks, uses tournament selection of size
    {_config["tournament_size"]}, preserves one elite, applies uniform or two-point
    crossover, mutates each bit with probability `1/n`, and repairs children to the exact
    keep count. Frozen weights make accuracy the only fitness objective needed.
    """)
    return


@app.cell(hide_code=True)
def _(
    STRUCTURED_REPORT_METHODS,
    ablation,
    fig_accuracy_vs_sparsity,
    mo,
    sweep,
):
    _methods = list(STRUCTURED_REPORT_METHODS)
    _focus = ablation["config"]["focus"]
    _figure = fig_accuracy_vs_sparsity(
        sweep["sparsities"],
        {name: sweep["methods"][name]["accuracy"] for name in _methods},
        {name: sweep["methods"][name]["accuracy_std"] for name in _methods},
        sweep["dense_acc"],
        method_order=_methods,
        title="Structured pruning: test accuracy vs hidden-neuron sparsity",
    )
    _focus_index = sweep["sparsities"].index(_focus)
    _ga = sweep["methods"]["GA-uniform"]["accuracy"][_focus_index]
    _hill = sweep["methods"]["Hill climbing (equal budget)"]["accuracy"][_focus_index]
    _random = sweep["methods"]["Random search (equal budget)"]["accuracy"][_focus_index]
    mo.vstack(
        [
            mo.md(f"""
            ## 5. Main result

            At {_focus:.0%} neuron sparsity, uniform-crossover GA reaches **{_ga:.1f}%** test
            accuracy: {_ga - _hill:.1f} percentage points above equal-budget hill climbing
            and {_ga - _random:.1f} points above equal-budget random search. The advantage
            grows as the constraint becomes harder, supporting the value of evolutionary
            operators in the smaller, structured search space.
            """),
            mo.hstack(
                [
                    mo.stat(f"{_ga:.2f}%", label=f"GA-uniform · {_focus:.0%}"),
                    mo.stat(f"{_hill:.2f}%", label=f"Hill climbing · {_focus:.0%}"),
                    mo.stat(f"{_random:.2f}%", label=f"Random search · {_focus:.0%}"),
                ],
                widths="equal",
            ),
            mo.as_html(_figure),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    STRUCTURED_METHOD_LABELS,
    STRUCTURED_REPORT_METHODS,
    ablation,
    mo,
    sweep,
):
    _focus = ablation["config"]["focus"]
    explorer_sparsity = mo.ui.slider(
        steps=[round(value * 100) for value in sweep["sparsities"]],
        value=round(_focus * 100),
        show_value=True,
        debounce=True,
        label="**Neuron sparsity (%)**",
        full_width=True,
    )
    explorer_method = mo.ui.dropdown(
        options={
            "All methods": "__all__",
            **{
                STRUCTURED_METHOD_LABELS[method]: method
                for method in STRUCTURED_REPORT_METHODS
            },
        },
        value="All methods",
        allow_select_none=False,
        label="**Method**",
        full_width=True,
    )
    explorer_panel = mo.ui.dropdown(
        options=[
            "Accuracy comparison",
            "Convergence",
            "Layer allocation",
            "Mask anatomy",
        ],
        value="Accuracy comparison",
        allow_select_none=False,
        label="**View**",
        full_width=True,
    )
    explorer_weight_masks = mo.ui.switch(
        value=False,
        label="Show derived weight masks",
    )
    mo.vstack(
        [
            mo.md(f"""
            ## 6. Explore the structured result

            The default compares every method at {_focus:.0%}. Select one method to inspect its
            convergence, layer allocation, and the saved structured masks. No experiment
            is rerun.
            """),
            mo.hstack(
                [explorer_sparsity, explorer_method, explorer_panel], widths="equal"
            ),
        ]
    )
    return (
        explorer_method,
        explorer_panel,
        explorer_sparsity,
        explorer_weight_masks,
    )


@app.cell(hide_code=True)
def _(
    STRUCTURED_METHOD_LABELS,
    STRUCTURED_REPORT_METHODS,
    explorer_method,
    explorer_panel,
    explorer_sparsity,
    explorer_weight_masks,
    fig_accuracy_snapshot,
    fig_distribution,
    fig_neuron_mask_heatmap,
    fig_progress,
    fig_weight_matrix_heatmap,
    mo,
    neuron_to_weight_masks,
    neuron_layer_rows,
    structured_explorer_view,
    sweep,
    weight_layer_rows,
):
    _method = explorer_method.value
    _sparsity = explorer_sparsity.value / 100
    _methods = list(STRUCTURED_REPORT_METHODS)
    _display_names = STRUCTURED_METHOD_LABELS
    _views = {
        method: structured_explorer_view(sweep, method, _sparsity)
        for method in _methods
    }
    _accuracy_plot = fig_accuracy_snapshot(
        {method: _views[method]["accuracy"] for method in _methods},
        {method: _views[method]["std"] for method in _methods},
        sweep["dense_acc"],
        _sparsity,
        display_names=_display_names,
    )

    if _method == "__all__":
        _best_method = max(_methods, key=lambda method: _views[method]["accuracy"])
        _best = _views[_best_method]
        _curves = {
            f"{method}@{_sparsity:.4f}": _views[method]["curve"]
            for method in _methods
            if _views[method]["curve"]
        }
        _allocations = {
            f"{method}@{_sparsity:.4f}": _views[method]["distribution"]
            for method in _methods
        }
        _weight_detail = (
            mo.md(
                "Select one method to inspect the full weight masks derived from its "
                "neuron mask."
            ).callout(kind="info")
            if explorer_weight_masks.value
            else mo.md("")
        )
        _mask = mo.vstack(
            [
                mo.md(
                    "Each subplot is one hidden layer with its own true neuron count. "
                    "Green bits keep neurons; red bits remove their incoming row, "
                    "outgoing column, and hidden bias. Stochastic methods show the "
                    f"saved seed {sweep['config']['seed']}; "
                    "accuracy and variability summarize all seeds."
                ),
                mo.as_html(
                    fig_neuron_mask_heatmap(
                        {
                            _display_names[method]: _views[method]["neuron_mask"]
                            for method in _methods
                        },
                        sweep["architecture"]["hidden_sizes"],
                        title=f"Representative structured masks at {_sparsity:.0%} neuron sparsity",
                    )
                ),
                explorer_weight_masks,
                _weight_detail,
            ]
        )
        _cards = mo.hstack(
            [
                mo.stat(f"{_sparsity:.0%}", label="Neuron sparsity"),
                mo.stat(_display_names[_best_method], label="Best method"),
                mo.stat(f"{_best['accuracy']:.1f}%", label="Best test accuracy"),
                mo.stat(
                    f"{_best['accuracy'] - sweep['dense_acc']:+.1f} pp",
                    label="Best gap to dense",
                ),
            ],
            widths="equal",
        )
        _convergence = mo.as_html(fig_progress(_curves, sweep["dense_val_acc"]))
        _allocation = mo.as_html(
            fig_distribution(
                _allocations,
                len(sweep["architecture"]["hidden_sizes"]),
                display_names=_display_names,
            )
        )
    else:
        _view = _views[_method]
        _key = f"{_method}@{_sparsity:.4f}"
        _architecture = sweep["architecture"]
        _weight_masks = (
            neuron_to_weight_masks(
                _view["neuron_mask"],
                _architecture["hidden_sizes"],
                _architecture["input_size"],
                _architecture["output_size"],
            )
            if explorer_weight_masks.value
            else None
        )
        _seed_note = (
            "This is a deterministic mask."
            if _view["representative_seed"] is None
            else (
                f"This mask is from seed {_view['representative_seed']}; accuracy, "
                "variability, convergence, and allocation summarize all seeds."
            )
        )
        _convergence = (
            mo.as_html(fig_progress({_key: _view["curve"]}, sweep["dense_val_acc"]))
            if _view["curve"]
            else mo.md(
                "This deterministic baseline ranks neurons once and has no search curve."
            ).callout(kind="info")
        )
        _allocation = mo.as_html(
            fig_distribution(
                {_key: _view["distribution"]},
                len(sweep["architecture"]["hidden_sizes"]),
                display_names=_display_names,
            )
        )
        _weight_detail = (
            mo.vstack(
                [
                    mo.md(
                        "Derived weight connectivity: green entries survive and red "
                        "entries are removed. Large matrices are block-averaged only "
                        "for display."
                    ),
                    mo.as_html(
                        fig_weight_matrix_heatmap(
                            _weight_masks,
                            title=(
                                f"{_method} derived weight masks at "
                                f"{_sparsity:.0%} neuron sparsity"
                            ),
                        )
                    ),
                    mo.ui.table(
                        weight_layer_rows(_weight_masks),
                        selection=None,
                        pagination=False,
                        show_search=False,
                        show_download=False,
                        show_data_types=False,
                    ),
                ]
            )
            if explorer_weight_masks.value
            else mo.md("")
        )
        _mask = mo.vstack(
            [
                mo.md(
                    "The chromosome is shown first. Its removed neurons induce the "
                    "regular removed rows and columns in the weight matrices below. "
                    + _seed_note
                ),
                mo.as_html(
                    fig_neuron_mask_heatmap(
                        {_display_names[_method]: _view["neuron_mask"]},
                        sweep["architecture"]["hidden_sizes"],
                        title=f"{_method} neuron mask at {_sparsity:.0%} neuron sparsity",
                    )
                ),
                mo.ui.table(
                    neuron_layer_rows(
                        _view["neuron_mask"],
                        sweep["architecture"]["hidden_sizes"],
                    ),
                    selection=None,
                    pagination=False,
                    show_search=False,
                    show_download=False,
                    show_data_types=False,
                ),
                explorer_weight_masks,
                _weight_detail,
            ]
        )
        _cards = mo.hstack(
            [
                mo.stat(f"{_view['accuracy']:.1f}%", label="Test accuracy"),
                mo.stat(f"±{_view['std']:.1f} pp", label="Search variability"),
                mo.stat(
                    f"{_view['weight_sparsity']:.1%}",
                    label="Weights removed · mean",
                ),
                mo.stat(
                    f"{_view['accuracy'] - sweep['dense_acc']:+.1f} pp",
                    label="Gap to dense",
                ),
            ],
            widths="equal",
        )

    _panels = {
        "Accuracy comparison": mo.as_html(_accuracy_plot),
        "Convergence": _convergence,
        "Layer allocation": _allocation,
        "Mask anatomy": _mask,
    }

    mo.vstack(
        [
            _cards,
            _panels[explorer_panel.value],
            mo.md(
                "*The live Marimo app recomputes this panel from saved JSON only. The "
                f"static HTML backup shows the default all-methods {_sparsity:.0%} view.*"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(ablation, fig_ablation_accuracy, mo, sweep):
    _study = ablation["sensitivity"]
    _baseline = _study["settings"]["Canonical"]
    _focus = ablation["config"]["focus"]
    _num_seeds = ablation["config"]["num_seeds"]
    _budget = ablation["config"]["canonical_fitness_evaluations"]
    _first_seed = ablation["config"]["seed"]
    _last_seed = _first_seed + _num_seeds - 1
    _accuracy_figure = fig_ablation_accuracy(_study, sweep["dense_acc"], _focus)
    _baseline_row = {
        "Sparsity": f"{_focus:.0%}",
        "Population": _baseline["pop_size"],
        "Generations": _baseline["n_gen"],
        "Evaluations": f"{_baseline['fitness_evaluations']:,}",
        "Tournament": _baseline["t_size"],
        "Crossover": _baseline["crossover"].replace("_", " ").title(),
        "Elitism": "One elite" if _baseline["elitism"] else "Disabled",
        "Mutation": ablation["config"]["mutation_rule"],
        "Eval. batch": ablation["config"]["batch_size"],
        "Seeds": f"{_first_seed}–{_last_seed}",
        "Fitness": "Validation accuracy",
        "Weights": "Frozen",
    }
    mo.vstack(
        [
            mo.md(f"""
            ## 7. Sensitivity study

            Mean test accuracy across {_num_seeds} search seeds; error bars show one
            standard deviation. Every variant receives the same
            **{_budget:,}-evaluation budget**.
            """),
            mo.md("### Canonical GA configuration"),
            mo.ui.table(
                [_baseline_row],
                selection=None,
                pagination=False,
                show_search=False,
                show_download=False,
                show_data_types=False,
            ),
            mo.as_html(_accuracy_figure),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 8. Conclusions

    - The weight-level chromosome is *too large* and weakly structured for this GA and
      evaluation budget; magnitude pruning remains the strongest unstructured method
      tested.
    - The neuron-level chromosome produces *stronger high-sparsity* masks and a compact
      dense network that can deliver practical memory and computational savings.

    **Next steps**

    - Repeat the structured comparison across independently trained checkpoints.
    - Benchmark compact-model latency and memory use on real hardware, including edge
      devices.
    - Fine-tune the network after pruning to recover accuracy.
    - Explore multi-objective optimization of accuracy, model size, and latency.
    """)
    return


@app.cell(hide_code=True)
def _(
    ablation,
    effective_sparsity_rows,
    fig_crossover_convergence,
    fig_effective_sparsity_comparison,
    mo,
    structured_explorer_view,
    sweep,
    training_protocol,
    unstructured,
):
    _rows, _structured_weight_sparsity = effective_sparsity_rows(sweep, "GA-uniform")
    _entries = unstructured["sparsities"]
    _focus = ablation["config"]["focus"]
    _crossover_curves = {
        f"{method}@{_focus:.4f}": structured_explorer_view(sweep, method, _focus)[
            "curve"
        ]
        for method in ("GA-uniform", "GA-two_point")
    }
    _effective_figure = fig_effective_sparsity_comparison(
        {
            "GA-uniform": _structured_weight_sparsity,
            "Unstructured magnitude": [
                _entries[str(value)]["mag_sparsity"] for value in sweep["sparsities"]
            ],
        },
        {
            "GA-uniform": sweep["methods"]["GA-uniform"]["accuracy"],
            "Unstructured magnitude": [
                _entries[str(value)]["mag_accuracy"] for value in sweep["sparsities"]
            ],
        },
        sweep["dense_acc"],
    )
    _appendix = mo.accordion(
        {
            "Actual parameter sparsity": mo.vstack(
                [
                    mo.as_html(_effective_figure),
                    mo.ui.table(
                        _rows,
                        selection=None,
                        show_search=False,
                        show_download=False,
                        show_data_types=False,
                    ),
                ]
            ),
            "Crossover comparison": mo.as_html(
                fig_crossover_convergence(
                    _crossover_curves, _focus, sweep["dense_val_acc"]
                )
            ),
            "Training details": mo.md(f"""
                | Setting | Value |
                |---|---|
                | Epochs / batch | {training_protocol["epochs"]} / {training_protocol["batch_size"]} |
                | AdamW learning rate | {training_protocol["learning_rate"]:.0e} |
                | Weight decay | {training_protocol["weight_decay"]:.0e} |
                | Cosine minimum LR | {training_protocol["minimum_learning_rate"]:.0e} |
                | Validation split | {training_protocol["validation_split"]:.0%} |
                | Normalization | μ={training_protocol["normalization_mean"]:.3f}, σ={training_protocol["normalization_std"]:.3f} |
                | Seed | {training_protocol["seed"]} |
            """),
            "Limitations and references": mo.md("""
                **Limitations:** one dataset, architecture, checkpoint, and split;
                repeated seeds vary only mask search; validation selects both the checkpoint and
                pruning masks; there is no pruning-time fine-tuning or latency benchmark.

                **Selected references:** LeCun et al. (1990), *Optimal Brain Damage*;
                Frankle & Carbin (2019), *The Lottery Ticket Hypothesis*; Gale et al.
                (2019), *The State of Sparsity*; He & Xiao (2023), *Structured Pruning for
                Deep CNNs*; Eiben & Smit (2011)
            """),
        }
    )
    mo.vstack([mo.md("## Appendix"), _appendix])
    return


if __name__ == "__main__":
    app.run()
