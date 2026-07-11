import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", app_title="GAMO pruning report")


@app.cell(hide_code=True)
def _():
    import base64
    import inspect
    import mimetypes
    import os

    import marimo as mo

    from gamo.ga.operators import (
        bit_flip_mutation,
        crossover_population,
        random_population,
        repair_population,
        tournament_selection,
    )
    from gamo.train.train import (
        TRAIN_BATCH_SIZE,
        TRAIN_EPOCHS,
        TRAIN_LEARNING_RATE,
        TRAIN_MIN_LEARNING_RATE,
        TRAIN_SEED,
        TRAIN_VALIDATION_SPLIT,
        TRAIN_WEIGHT_DECAY,
    )
    from gamo.utils.data import FASHION_MNIST_NORMALIZATION
    from gamo.report.data import (
        STRUCTURED_METHOD_LABELS,
        STRUCTURED_REPORT_METHODS,
        effective_sparsity_rows,
        load_report_artifacts,
        neuron_layer_rows,
        structured_explorer_view,
        weight_layer_rows,
    )
    from gamo.report.plots import (
        fig_accuracy_snapshot,
        fig_accuracy_vs_sparsity,
        fig_crossover_convergence,
        fig_distribution,
        fig_effective_sparsity_comparison,
        fig_neuron_mask_heatmap,
        fig_progress,
        fig_weight_matrix_heatmap,
    )

    def load_image(path):
        if not os.path.exists(path):
            return ""
        mime = mimetypes.guess_type(path)[0] or "image/png"
        with open(path, "rb") as asset_file:
            encoded = base64.b64encode(asset_file.read()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    pruning_img = load_image("assets/pruning.png")
    steps_img = load_image("assets/steps.png")

    training_protocol = {
        "epochs": TRAIN_EPOCHS,
        "batch_size": TRAIN_BATCH_SIZE,
        "learning_rate": TRAIN_LEARNING_RATE,
        "minimum_learning_rate": TRAIN_MIN_LEARNING_RATE,
        "weight_decay": TRAIN_WEIGHT_DECAY,
        "validation_split": TRAIN_VALIDATION_SPLIT,
        "seed": TRAIN_SEED,
        "normalization_mean": FASHION_MNIST_NORMALIZATION[0][0],
        "normalization_std": FASHION_MNIST_NORMALIZATION[1][0],
    }
    ga_step_sources = {
        "Initialization": inspect.getsource(random_population),
        "Selection": inspect.getsource(tournament_selection),
        "Crossover": inspect.getsource(crossover_population),
        "Mutation": inspect.getsource(bit_flip_mutation),
        "Mask repair": inspect.getsource(repair_population),
    }
    return (
        STRUCTURED_METHOD_LABELS,
        STRUCTURED_REPORT_METHODS,
        effective_sparsity_rows,
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
                "Choose an operator, then switch between a short explanation and the "
                "exact implementation used by the experiments."
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
    fixed, so error bars measure search randomness—not training uncertainty.
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
    > This is the regime where a GA has a real chance to help and pruning whole neurons can be materialized as a smaller dense network, unlike zeros scattered through a matrix.

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
def _(STRUCTURED_REPORT_METHODS, ablation, fig_accuracy_vs_sparsity, mo, sweep):
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
def _(STRUCTURED_METHOD_LABELS, STRUCTURED_REPORT_METHODS, ablation, mo, sweep):
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
    return explorer_method, explorer_panel, explorer_sparsity, explorer_weight_masks


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
        _convergence = mo.as_html(fig_progress(_curves))
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
        _seed_note = (
            "This is a deterministic mask."
            if _view["representative_seed"] is None
            else (
                f"This mask is from seed {_view['representative_seed']}; accuracy, "
                "variability, convergence, and allocation summarize all seeds."
            )
        )
        _convergence = (
            mo.as_html(fig_progress({_key: _view["curve"]}))
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
                            _view["weight_masks"],
                            title=(
                                f"{_method} derived weight masks at "
                                f"{_sparsity:.0%} neuron sparsity"
                            ),
                        )
                    ),
                    mo.ui.table(
                        weight_layer_rows(_view["weight_masks"]),
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
def _(ablation, mo, sweep):
    _study = ablation["sensitivity"]
    _canonical = _study["accuracies"][_study["values"].index("Canonical")]
    _focus = ablation["config"]["focus"]
    _focus_index = sweep["sparsities"].index(_focus)
    _main = sweep["methods"]["GA-uniform"]["accuracy"][_focus_index]
    _seed = ablation["config"]["seed"]
    _num_seeds = ablation["config"]["num_seeds"]
    _budget = ablation["config"]["canonical_fitness_evaluations"]
    _num_variants = len(_study["values"])
    _raw_seed_text = f"{_seed}–{_seed + _num_seeds - 1}"
    _seed_note = (
        f"The main sweep and sensitivity study both use `{_raw_seed_text}`. "
        "Reusing each repeat seed across sparsities pairs the comparisons without "
        "making the masks identical: the sparsity constraint still changes the "
        "search problem."
    )
    _rows = []
    for _name, _group, _accuracy, _std in zip(
        _study["values"],
        _study["groups"],
        _study["accuracies"],
        _study["stds"],
    ):
        _settings = _study["settings"][_name]
        _rows.append(
            {
                "Study": _group,
                "Variant": (
                    "Reference configuration" if _name == "Canonical" else _name
                ),
                "Population": _settings["pop_size"],
                "Generations": _settings["n_gen"],
                "Test accuracy": f"{_accuracy:.1f}%",
                "Δ reference": f"{_accuracy - _canonical:+.1f} pp",
                "Std": f"{_std:.1f} pp",
            }
        )
    mo.vstack(
        [
            mo.md(f"""
            ## 7. Sensitivity study

            Each variant changes one GA choice, and population-size variants adjust
            generations so every row keeps the same **{_budget:,}-evaluation budget**. All
            {_num_variants} variants use the same {_num_seeds} raw streams
            (`{_raw_seed_text}`), so the
            comparisons are paired and fair within this study.

            {_seed_note}
            """),
            mo.hstack(
                [
                    mo.stat(f"{_main:.1f}%", label=f"Main GA · {_focus:.0%}"),
                    mo.stat(f"{_canonical:.1f}%", label="Sensitivity reference"),
                    mo.stat(f"{_num_seeds}", label="Seeds per variant"),
                ],
                widths="equal",
            ),
            mo.ui.table(
                _rows,
                selection=None,
                pagination=False,
                show_search=False,
                show_download=False,
                show_data_types=False,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 8. Conclusions

    1. The weight-level chromosome is too large and weakly structured for this GA and
       budget; magnitude pruning remains the appropriate unstructured winner.
    2. A neuron-level chromosome makes the search operators useful and produces much
       stronger high-sparsity masks than the included heuristics and equal-budget controls.
    3. The central result is about **representation and search**—not a claim that genetic
       algorithms universally beat magnitude pruning.

    The next step is to repeat the focused structured comparison across independently
    trained checkpoints and benchmark compact-model latency on real hardware.
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
        f"{method}@{_focus:.4f}": structured_explorer_view(
            sweep, method, _focus
        )["curve"]
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
                fig_crossover_convergence(_crossover_curves, _focus)
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
                Deep CNNs*; Eiben & Smit (2011), *Parameter Tuning for Evolutionary
                Algorithms*.
            """),
        }
    )
    mo.vstack([mo.md("## Appendix"), _appendix])
    return


if __name__ == "__main__":
    app.run()
