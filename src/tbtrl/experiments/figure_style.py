"""Publication styling/export scoped to a call; no global state or RNG changes."""

from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from textwrap import fill

COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#4D4D4D")
MODES = {
    "scratch": ("From scratch", COLORS[0], "o", "-"),
    "sequential": ("Sequential transfer", COLORS[1], "s", "--"),
    "recovery": ("Independent recovery", COLORS[2], "^", "-."),
}
BRANCHES = {
    "q": "DQN critic",
    "critic1": "SAC critic 1",
    "critic2": "SAC critic 2",
    "actor_mu": "Actor mean",
    "actor_log_std": "Actor log standard deviation",
}


@dataclass(frozen=True)
class FigureStyle:
    """Physical inches/points; customise to a venue without changing notebooks."""

    width: float = 7.2
    font_size: float = 9.0
    dpi: int = 300
    display_dpi: int = 120
    formats: tuple[str, ...] = ("png", "pdf", "svg")

    def __post_init__(self):
        if min(self.width, self.font_size, self.dpi, self.display_dpi) <= 0:
            raise ValueError("Figure dimensions, font size and resolutions must be positive.")
        if not self.formats or set(self.formats) - {"png", "pdf", "svg"}:
            raise ValueError("Choose export formats from png, pdf, svg.")

    def rc_params(self):
        from cycler import cycler

        return {
            "font.family": "DejaVu Sans",
            "font.size": self.font_size,
            "axes.titlesize": self.font_size + 0.5,
            "axes.labelsize": self.font_size,
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "axes.prop_cycle": cycler(color=COLORS),
            "axes.axisbelow": True,
            "xtick.labelsize": self.font_size - 1,
            "ytick.labelsize": self.font_size - 1,
            "legend.fontsize": self.font_size - 1,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "figure.dpi": self.display_dpi,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": self.dpi,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.usetex": False,
        }


DEFAULT_STYLE = FigureStyle()


def styled(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        import matplotlib as mpl

        style = (
            kwargs.get("style")
            or (getattr(args[0], "style", None) if args else None)
            or DEFAULT_STYLE
        )
        with mpl.rc_context(style.rc_params()):
            return function(*args, **kwargs)

    return wrapped


def subplots(rows=1, cols=1, *, style=DEFAULT_STYLE, height=None, **kwargs):
    import matplotlib.pyplot as plt

    return plt.subplots(
        rows, cols, figsize=(style.width, height or 2.8 * rows), layout="constrained", **kwargs
    )


def numeric_axis(ax, *, x=False, y=False, percent=False):
    from matplotlib.ticker import EngFormatter, MaxNLocator, PercentFormatter

    ax.grid(axis="both" if x else "y", color="#D9DDE3", linewidth=0.5, alpha=0.65)
    if x:
        ax.xaxis.set_major_formatter(EngFormatter(sep=""))
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    if y:
        ax.yaxis.set_major_formatter(EngFormatter(sep=""))
    if percent:
        ax.set_ylim(-0.03, 1.08)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))


def legend(ax, *, columns=2):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=min(columns, len(handles)),
        )


def goal_ticks(ax, goals, *, axis="x", matrix=False):
    """Keep goal IDs readable when task sequences exceed a dozen goals."""
    from math import ceil

    positions = list(range(len(goals))) if matrix else goals
    stride = max(1, ceil(len(goals) / 12)) if axis == "y" else 1
    indices = range(0, len(goals), stride)
    ticks, labels = [positions[i] for i in indices], [goals[i] for i in indices]
    if axis == "x":
        ax.set_xticks(ticks, labels=labels, rotation=90 if len(goals) > 12 else 0)
    else:
        ax.set_yticks(ticks, labels=labels)


def finish(fig, *, title=None, note=None):
    import matplotlib as mpl

    size = mpl.rcParams["font.size"]
    if title:
        fig.suptitle(title, fontsize=size + 2, fontweight="semibold")
    if note:
        fig.supxlabel(
            fill(note, max(35, int(fig.get_figwidth() * 15))), fontsize=size - 1.5, color="#444444"
        )


def save_figure(fig, destination, *, style=DEFAULT_STYLE):
    """Export to all configured formats; return absolute paths."""
    import matplotlib as mpl

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    with mpl.rc_context(style.rc_params()):
        for extension in style.formats:
            path = destination.with_suffix(f".{extension}")
            fig.savefig(path, dpi=style.dpi, bbox_inches="tight", pad_inches=0.06)
            paths.append(path.resolve())
    return paths


def empty_panel(ax, message):
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="#666666",
        wrap=True,
    )


def annotate_matrix(ax, matrix, *, percent=False):
    import numpy as np

    if max(matrix.shape) > 12:
        return
    for (row, col), value in np.ndenumerate(matrix):
        if np.isfinite(value):
            rgba = ax.images[-1].cmap(ax.images[-1].norm(value))
            luminance = sum(c * w for c, w in zip(rgba[:3], (0.2126, 0.7152, 0.0722)))
            ax.text(
                col,
                row,
                f"{value:.1%}".replace(".0%", "%") if percent else f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if luminance > 0.5 else "white",
            )
