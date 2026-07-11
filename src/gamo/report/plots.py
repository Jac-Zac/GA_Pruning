"""Reusable Plotly figures for the GAMO report.

Used by ``notebooks/report.py`` so the report figures stay separate from the
experiment-running code.
"""

import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

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
        xaxis=dict(gridcolor="#ece6db", zerolinecolor="#ddd4c6", linecolor="#ddd4c6"),
        yaxis=dict(gridcolor="#ece6db", zerolinecolor="#ddd4c6", linecolor="#ddd4c6"),
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
        annotation_position="top left",
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
def fig_progress(curves: dict[str, list[float]]) -> go.Figure:
    """Plot best validation accuracy per generation for search methods."""
    fig = go.Figure()
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
            yaxis_title="Best val accuracy (%)",
            hovermode="x unified",
            height=420,
        )
    )
    return fig


# ---------------------------------------------------------------------------
# Figure — crossover convergence
# ---------------------------------------------------------------------------
def fig_crossover_convergence(curves: dict[str, list[float]], sp: float) -> go.Figure:
    """Plot the convergence curves for each crossover strategy."""
    fig = go.Figure()
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
            yaxis_title="Best val accuracy (%)",
            hovermode="x unified",
            height=420,
        )
    )
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
            for effective, accuracy in zip(method_eff_sparsities[name], series[name])
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
    neuron_masks: dict[str, torch.Tensor],
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
    split_masks: dict[str, tuple[torch.Tensor, ...]] = {}
    for method, raw_mask in neuron_masks.items():
        mask = raw_mask.detach().cpu().bool().flatten()
        if mask.numel() != sum(hidden_sizes):
            raise ValueError("neuron mask length does not match hidden_sizes")
        split_masks[method] = mask.split(hidden_sizes)

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
            [[method, layer_index, neuron_index + 1] for neuron_index in range(width)]
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
    weight_masks: dict[str, torch.Tensor],
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
        wmask = weight_masks[key]
        r = min(wmask.shape[0], max_dim)
        c = min(wmask.shape[1], max_dim)
        # Average pooling preserves the active fraction when a matrix is downsampled.
        region = torch.nn.functional.adaptive_avg_pool2d(
            wmask.float().unsqueeze(0).unsqueeze(0), (r, c)
        )[0, 0].numpy()

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
