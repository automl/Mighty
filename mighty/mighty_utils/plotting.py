from typing import  Tuple, Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import io 
from PIL import Image


def set_rc_params():
    # Figure
    mpl.rcParams['figure.figsize'] = (6, 3)

    # Fontsizes
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.titlesize'] = 12

    # Colors
    # - Seaborn Color Palette: colorblind
    # - default context always plotted in black

    sns.set_style("whitegrid")
    sns.set_palette("colorblind")

def plotly_fig2array(fig):
    #convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="png", scale=5)
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)

def fig2img(fig, figsize=None, dpi=300):
    """Convert matplotlib figure to image as numpy array.

    :param fig: Plot to get image for.
    :type fig: matplotlib figure

    :param figsize: Optional figure size in inches, e.g. ``(10, 7)``.
    :type figsize: None or tuple of int

    :param dpi: Optional dpi.
    :type dpi: None or int

    :return: RGB image of plot
    :rtype: np.array
    """
    if dpi is not None:
        fig.set_dpi(dpi)
    if figsize is not None:
        fig.set_size_inches(figsize)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
    image = np.reshape(image, (int(height), int(width), 3))

    # s, (width, height) = canvas.print_to_buffer()
    # image = np.fromstring(s, dtype=np.uint8).reshape((height, width, 3))

    return image

def plot_performance_over_time(data: pd.DataFrame, x: str, y: str, hue: str = None, marker: str = None, col: str = None, row: str = None, logx: bool = False, logy: bool = False, xlim: Tuple = None, ylim: Tuple = None, errorbar: str = "ci", xlabel: str = None, ylabel: str = None, aggregation: str = np.mean, save_path: str = None):
    set_rc_params()
    if aggregation == "iqm":
        aggregation = metrics.aggregate_iqm
        agg_name = "IQM"
    elif aggregation == "mean":
        aggregation = np.mean
        agg_name = "Mean"
    elif aggregation == "median":
        aggregation = np.median
        agg_name = "Median"
    elif aggregation == "rank":
        aggregation = np.mean
        groups = data.columns.values.tolist()
        groups.remove(y)
        data["rank"] = data.groupby(groups)[y].rank(method="first")
        y = "rank"
        agg_name = "Rank"

    fig = _plot_performance_over_time(data, x, y, hue, marker, col, row, logx, logy, xlim, ylim, errorbar, xlabel, ylabel, aggregation, agg_name)
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=600)
    return fig2img(fig)

def plot_configuration_footprint(run_path:str = None, run_object = None, budget_id: int = None, objective_id: int = None, save_path: str = None, details: bool = True, show_supports: bool = True, show_borders: bool = True):
    from deepcave.plugins.summary.footprint import FootPrint
    plugin = FootPrint()
    kwargs = {"details": details, "budget_id": budget_id, "show_supports": show_supports, "show_borders": show_borders}
    return plot_deepcave(plugin=plugin, run_path=run_path, run_object=run_object, budget_id=budget_id, objective_id=objective_id, save_path=save_path, kwargs=kwargs)

def plot_hp_importance(hyperparameter_names, run_path:str = None, run_object = None, budget_id: int = None, objective_id: int = None, save_path: str = None, n_trees: int = 10, n_hps: int = 10, method: str = "global"):
    from deepcave.plugins.hyperparameter.importances import Importances

    plugin = Importances()
    kwargs = {"n_trees": n_trees, "n_hps": n_hps, "method": method, "hyperparameter_names": hyperparameter_names, "budget_ids": None}
    return plot_deepcave(plugin=plugin, run_path=run_path, run_object=run_object, budget_id=budget_id, objective_id=objective_id, save_path=save_path, kwargs=kwargs)
    

def plot_improvement_probability(data: pd.DataFrame, x: str, y: str, save_path: str = None):
    set_rc_params()
    algorithm_pairs = {}
    for m in data[x].unique():
        for m2 in data[x].unique():
            if m2 != m:
                values = []
                values2 = []
                for s in data["seed"].unique():
                    values.append(data[(data[x] == m) & (data["seed"] == s)][y].values)
                    values2.append(data[(data[x] == m2) & (data["seed"] == s)][y].values)
                min_len = min([len(v) for v in values + values2])
                values = [v[:min_len] for v in values]
                values2 = [v[:min_len] for v in values2]
                algorithm_pairs[f"{m},{m2}"] = (np.array(values), np.array(values2))

    average_probabilities, average_prob_cis = rly.get_interval_estimates(algorithm_pairs, metrics.probability_of_improvement, reps=2000)
    fig = plot_utils.plot_probability_of_improvement(average_probabilities, average_prob_cis)
    fig = fig.get_figure()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=600)
    return fig2img(fig)

def plot_final_performance_comparison(data: pd.DataFrame, x: str, y: str, aggregation: str = "improvement_prob", save_path: str = None, xlabel: str = None):
    set_rc_params()

    if type(aggregation) == str:
        aggregation = [aggregation]

    aggregation_funcs = []
    metric_names = []
    if "iqm" in aggregation and "mean" in aggregation and "median" in aggregation:
        aggregation_funcs = lambda x: np.array([ metrics.aggregate_median(x), metrics.aggregate_iqm(x), metrics.aggregate_mean(x)])
        metric_names = ["Median", "IQM", "Mean"]
    elif "iqm" in aggregation and "mean" in aggregation:
        aggregation_funcs = lambda x: np.array([ metrics.aggregate_iqm(x), metrics.aggregate_mean(x)])
        metric_names = ["IQM", "Mean"]
    elif "iqm" in aggregation and "median" in aggregation:
        aggregation_funcs = lambda x: np.array([ metrics.aggregate_median(x), metrics.aggregate_iqm(x)])
        metric_names = ["Median", "IQM"]
    elif "mean" in aggregation and "median" in aggregation:
        aggregation_funcs = lambda x: np.array([ metrics.aggregate_median(x), metrics.aggregate_mean(x)])
        metric_names = ["Median", "Mean"]
    elif "iqm" in aggregation:
        aggregation_funcs = lambda x: np.array([ metrics.aggregate_iqm(x)])
        metric_names = ["IQM"]
    elif "mean" in aggregation:
        aggregation_funcs = lambda x: np.array([metrics.aggregate_mean(x)])
        metric_names = ["Mean"]
    elif "median" in aggregation:   
        aggregation_funcs = lambda x: np.array([metrics.aggregate_median(x)])
        metric_names = ["Median"]

    score_dict = {}
    for m in data[x].unique():
        values = []
        for s in data["seed"].unique():
            values.append(data[(data[x] == m) & (data["seed"] == s)][y].values)
        min_len = min([len(v) for v in values])
        values = [v[:min_len] for v in values]
        score_dict[m] = np.array(values)

    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(score_dict, aggregation_funcs, reps=50000)
    fig, _ = plot_utils.plot_interval_estimates(aggregate_scores, aggregate_score_cis, metric_names=metric_names, algorithms=np.unique(data[x].values), xlabel=None)
    fig.text(0.5, -0.3, xlabel, ha='center')
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=600)
    return fig2img(fig)

def _plot_performance_over_time(data: pd.DataFrame, x: str, y: str, hue: str = None, marker: str = None, col: str = None, row: str = None, logx: bool = False, logy: bool = False, xlim: Tuple = None, ylim: Tuple = None, errorbar: str = "ci", xlabel: str = None, ylabel: str = None, aggregation: Callable = np.mean, agg_name= "Performance", agg_name_short="perf"):
    fig = plt.figure(dpi = 300, figsize = (4, 4))
    nseeds = len(data["seed"].unique())
    if ylim is None:
        ylim = (min(data[y]), max(data[y]))
    if xlim is None:
        xlim = (min(data[x]), max(data[x]))

    if col is not None or row is not None:
        grid = sns.FacetGrid(data=data, col=col, row=row, hue=hue, sharex=True, sharey=True)
        sets = {"ylim": ylim, "xlim": xlim}
        if logx:
            sets["xscale"] = "log"
        if logy:
            sets["yscale"] = "log"
        grid.map_dataframe(sns.lineplot, x=x, y=y, marker=marker, errorbar=errorbar, estimator=aggregation).set(**sets)
        grid.fig.subplots_adjust(top=0.92)
        grid.fig.suptitle(f"{agg_name} over Time (num_seeds={nseeds})")
        grid.set_axis_labels(xlabel, ylabel)
        grid.add_legend()
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax = sns.lineplot(data=data, x=x, y=y, ax=ax, marker=marker, hue=hue, errorbar=errorbar, estimator=aggregation,palette=sns.color_palette('colorblind', as_cmap = True))
        if logy:
            ax.set_yscale("log")
        if logx:
            ax.set_xscale("log")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"{agg_name} over Time (num_seeds={nseeds})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1.1), ncol=5, title=None, frameon=False)
        fig.set_tight_layout(True)
    return fig

def plot_deepcave(plugin, run_path=None, run_object=None, kwargs={}, budget_id=None, objective_id=None, save_path=None):
    from deepcave.runs.converters.deepcave import DeepCAVERun

    set_rc_params()
    if run_object is not None:
        run = run_object
    elif run_path is not None:
        from pathlib import Path
        run = DeepCAVERun.from_path(Path(run_path))
    else:
        raise ValueError("Either run_path or run_object must be provided.")
    
    if objective_id is None:
        objective_id = run.get_objective_ids()[0]
    if budget_id is None and "budget_id" in kwargs.keys():
        kwargs["budget_id"] = run.get_budget_ids()[-1]
    if budget_id is None and "budget_ids" in kwargs.keys():
        kwargs["budget_ids"] = run.get_budget_ids()
    
    inputs = plugin.generate_inputs(
        objective_id=objective_id,
        **kwargs
    )
    print(inputs)
    outputs = plugin.generate_outputs(run, inputs)
    fig = plugin.load_outputs(run, inputs, outputs)
    if type(fig) == list:
        imgs = []
        for i, f in enumerate(fig):
            if save_path is not None:
                f.savefig(f"{i}_{save_path}", bbox_inches="tight", dpi=600)
            imgs.append(plotly_fig2array(f))
        return imgs
    else:
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=600)
        return plotly_fig2array(fig)