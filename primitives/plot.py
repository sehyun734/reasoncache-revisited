import json
from pathlib import Path

from huggingface_hub import snapshot_download
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from primitives.hub import get_repo_name

METHOD_COLORS = {"prefix": "tab:blue", "lora": "tab:orange", "base": "tab:gray"}
GRID_COLOR = "#e5e4df"


def load_results(
    entry_name: str,
    model_name: str,
    file_name: str,
) -> dict[str, dict]:
    result_dir = Path(snapshot_download(get_repo_name(), allow_patterns=[f"{entry_name}/*/{model_name}/*{file_name}.json"])) / entry_name
    return {str(file.parent.relative_to(result_dir)): json.loads(file.read_text()) for file in sorted(result_dir.glob(f"*/{model_name}/**/{file_name}.json"))}


def load_runs(
    entry_name: str,
    model_name: str,
    file_name: str,
    gsm8k_file_name: str,
) -> list[dict]:
    allow_patterns = [f"gsm8k/*/{model_name}/*{gsm8k_file_name}.json", f"{entry_name}/*/{model_name}/*{file_name}.json"]
    result_dir = Path(snapshot_download(get_repo_name(), allow_patterns=allow_patterns))
    runs = []
    for file in sorted(result_dir.glob(f"{entry_name}/*/{model_name}/**/{file_name}.json")):
        target = str(file.parent.relative_to(result_dir / entry_name))
        gsm8k_file = result_dir / "gsm8k" / target / f"{gsm8k_file_name}.json"
        if not gsm8k_file.exists():
            continue
        method, _, *settings = target.split("/")
        run_config = dict(setting.split("=") for setting in settings[0].split(",")) if settings else {}
        run = {"target": target, "method": method, "size": int(run_config.get("prefix_len", run_config.get("rank", 0))), "learning_rate": float(run_config.get("learning_rate", 0))}
        run |= json.loads(gsm8k_file.read_text())
        run["stats"] = json.loads(file.read_text())
        runs.append(run)

    base_accuracy = next((run["accuracy"] for run in runs if run["method"] == "base"), 0.0)
    for run in runs:
        if run["method"] != "base" and run["accuracy"] <= base_accuracy:
            print(f"excluded {run['target']} accuracy {run['accuracy']:.2%}")
    runs = [run for run in runs if run["method"] == "base" or run["accuracy"] > base_accuracy]

    for run in runs:
        if run["method"] == "base":
            run["marker_size"] = 90
            run["alpha"] = 1.0
            continue
        others = [other for other in runs if other["method"] == run["method"]]
        sizes = sorted({other["size"] for other in others})
        learning_rates = sorted({other["learning_rate"] for other in others})
        run["marker_size"] = 20 + 70 * sizes.index(run["size"]) / max(len(sizes) - 1, 1)
        run["alpha"] = 0.25 + 0.75 * learning_rates.index(run["learning_rate"]) / max(len(learning_rates) - 1, 1)
    return runs


def style_axis(
    ax: Axes,
    xlabel: str,
) -> None:
    ax.set_xlabel(xlabel)
    ax.margins(x=0.15, y=0.12)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(
    fig: Figure,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    print(f"saved {path}")


def plot_correlations(
    runs: list[dict],
    stat_names: list[str],
    title: str,
) -> Figure:
    fig, axes = plt.subplots(1, len(stat_names), figsize=(5 * len(stat_names), 5), sharey=True, layout="constrained")
    for ax, stat_name in zip(axes, stat_names):
        stat_runs = [run for run in runs if stat_name in run]
        for run in stat_runs:
            ax.scatter(run[stat_name], run["accuracy"], s=run["marker_size"], color=METHOD_COLORS[run["method"]], alpha=run["alpha"], linewidth=0, zorder=3)

        # 두 방법을 합치면 방법 간 차이가 상관처럼 보이므로 방법별로 구함.
        # run이 적고 튀는 값이 있어 spearman을 씀.
        rho_texts = []
        for method in ["prefix", "lora"]:
            method_runs = [run for run in stat_runs if run["method"] == method]
            if len(method_runs) > 2:
                rho = spearmanr([run[stat_name] for run in method_runs], [run["accuracy"] for run in method_runs]).statistic
                rho_texts.append(f"{method} spearman {rho:+.2f}")
        ax.set_title(", ".join(rho_texts), fontsize=11)
        style_axis(ax, stat_name)

    axes[0].set_ylabel("gsm8k accuracy")
    for method, color in METHOD_COLORS.items():
        axes[0].scatter([], [], s=36, color=color, label=method)
    prefix_sizes = sorted({run["size"] for run in runs if run["method"] == "prefix"})
    lora_sizes = sorted({run["size"] for run in runs if run["method"] == "lora"})
    for index, (prefix_size, lora_size) in enumerate(zip(prefix_sizes, lora_sizes)):
        axes[0].scatter([], [], s=20 + 70 * index / max(len(prefix_sizes) - 1, 1), color=METHOD_COLORS["base"], label=f"prefix_len={prefix_size} / rank={lora_size}")
    fig.legend(loc="outside upper center", ncol=len(METHOD_COLORS) + len(prefix_sizes), frameon=False, title=f"{title} (size = prefix_len/rank, transparent = learning_rate)")
    return fig


def plot_accuracies(
    results: dict[str, dict],
    base_accuracy: float,
    title: str,
) -> Figure:
    runs = []
    for target, result in results.items():
        method, _, settings = target.split("/")
        run_config = dict(setting.split("=") for setting in settings.split(","))
        size_name = "prefix_len" if method == "prefix" else "rank"
        runs.append({"method": method, "size_name": size_name, "size": int(run_config[size_name]), "learning_rate": float(run_config["learning_rate"]), "accuracy": result["accuracy"]})
    # 두 방법에 같은 진하기와 위치를 쓰도록 learning rate는 합쳐서 정렬.
    learning_rates = sorted({run["learning_rate"] for run in runs})
    width = 0.8 / len(learning_rates)

    methods = ["prefix", "lora"]
    fig, axes = plt.subplots(1, len(methods), figsize=(14, 5), sharey=True, layout="constrained")
    for ax, method in zip(axes, methods):
        method_runs = [run for run in runs if run["method"] == method]
        sizes = sorted({run["size"] for run in method_runs})
        for run in method_runs:
            index = learning_rates.index(run["learning_rate"])
            x = sizes.index(run["size"]) + (index - (len(learning_rates) - 1) / 2) * width
            ax.bar(x, run["accuracy"], width=width, color=METHOD_COLORS[method], alpha=0.25 + 0.75 * index / max(len(learning_rates) - 1, 1), zorder=3)
            ax.annotate(f"{run['accuracy']:.1%}", (x, run["accuracy"]), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        ax.axhline(base_accuracy, color=METHOD_COLORS["base"], linestyle="--", linewidth=1, zorder=2)
        ax.set_xticks(range(len(sizes)), [f"{method_runs[0]['size_name']}={size}" for size in sizes])
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0, 1.15 * max(run["accuracy"] for run in runs))
    axes[0].set_ylabel("gsm8k accuracy")

    handles = [Patch(color=METHOD_COLORS[method], label=method) for method in methods]
    handles += [Line2D([], [], color=METHOD_COLORS["base"], linestyle="--", label=f"base {base_accuracy:.1%}")]
    handles += [Patch(color=METHOD_COLORS["base"], alpha=0.25 + 0.75 * index / max(len(learning_rates) - 1, 1), label=f"learning_rate={learning_rate:g}") for index, learning_rate in enumerate(learning_rates)]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, title=title)
    return fig
