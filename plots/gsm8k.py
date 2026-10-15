from dataclasses import dataclass
from pathlib import Path

from simple_parsing import parse

from primitives.plot import load_results, plot_accuracies, save_figure


@dataclass
class Config:
    model_name: str = "Llama-3.2-1B"
    num_samples: int = 1319


def main() -> None:
    config = parse(Config)
    results = load_results("gsm8k", config.model_name, f"num_samples={config.num_samples}")
    base_accuracy = results.pop(f"base/{config.model_name}")["accuracy"]
    fig = plot_accuracies(results, base_accuracy, f"{config.model_name} gsm8k")
    save_figure(fig, Path(f"figures/{config.model_name}/gsm8k.png"))


if __name__ == "__main__":
    main()
