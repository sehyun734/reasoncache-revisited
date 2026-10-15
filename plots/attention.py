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
    runs = load_runs("attention", config.model_name, f"num_samples={config.num_samples}", f"num_samples={config.gsm8k_num_samples}")
    for run in runs:
        run |= {name: mean(value for layer_stat in stat for value in layer_stat) for name, stat in run.pop("stats").items()}
    fig = plot_correlations(runs, ["sink_ratio", "prefix_ratio", "entropy", "jsd"], f"{config.model_name} attention")
    save_figure(fig, Path(f"figures/{config.model_name}/attention.png"))


if __name__ == "__main__":
    main()
