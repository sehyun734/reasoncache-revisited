from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from simple_parsing import parse

from primitives.plot import load_runs, plot_correlations, save_figure


@dataclass
class Config:
    model_name: str = "Llama-3.2-1B"
    gsm8k_num_samples: int = 1319
    num_samples: int = 256


def main() -> None:
    config = parse(Config)
    runs = load_runs("vrank", config.model_name, f"num_samples={config.num_samples}", f"num_samples={config.gsm8k_num_samples}")
    # layer별 지표는 마지막 layer만 head 평균으로 봄.
    for run in runs:
        run |= {name: mean(stat[-1]) if isinstance(stat, list) else stat for name, stat in run.pop("stats").items()}
    fig = plot_correlations(runs, ["vrank", "new_ratio", "prefix_vrank", "prefix_new_ratio", "hidden_rank"], f"{config.model_name} vrank")
    save_figure(fig, Path(f"figures/{config.model_name}/vrank.png"))


if __name__ == "__main__":
    main()
