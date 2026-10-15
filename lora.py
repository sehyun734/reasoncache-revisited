from dataclasses import dataclass
import math

from simple_parsing import parse
import torch
from torch.nn import Parameter
from transformers import PreTrainedModel, set_seed

from primitives.dataset import load_metamathqa
from primitives.hub import make_lora_path, save_tensors
from primitives.model import hook_lora, load_model
from primitives.train import train


@dataclass
class Config:
    model_name: str = "Llama-3.2-1B"
    rank: int = 4
    scale: float = 2.0
    num_samples: int = 32768
    max_sequence_len: int = 2048
    num_epochs: int = 3
    batch_size: int = 128
    micro_size: int = 4
    learning_rate: float = 1e-3
    warmup_ratio: float = 0.05
    seed: int = 0


def initialize_lora(
    model: PreTrainedModel,
    rank: int,
) -> dict[str, Parameter]:
    lora = {}
    for module_name, module in model.named_modules():
        if module_name.endswith(("q_proj", "k_proj", "v_proj")):
            a = torch.randn(rank, module.in_features, device=model.device) / math.sqrt(module.in_features)
            # b를 0으로 둬 학습을 base 모델 그대로에서 시작함.
            b = torch.zeros(module.out_features, rank, device=model.device)
            lora[f"{module_name}.a"] = Parameter(a)
            lora[f"{module_name}.b"] = Parameter(b)
    return lora


def main() -> None:
    config = parse(Config)
    set_seed(config.seed)
    model, tokenizer = load_model(config.model_name)
    samples = load_metamathqa(tokenizer, config.num_samples, config.max_sequence_len)

    lora = initialize_lora(model, config.rank)
    hook_lora(model, lora, scale=config.scale)
    train(model, samples, list(lora.values()), config.num_epochs, config.batch_size, config.micro_size, config.learning_rate, config.warmup_ratio)

    # 평가와 측정에서 scale 없이 hook을 걸도록 b에 scale을 곱해 저장.
    lora = {name: tensor * config.scale if name.endswith(".b") else tensor for name, tensor in lora.items()}
    save_tensors(make_lora_path(config.model_name, config.rank, config.learning_rate, config.seed), "lora", lora)


if __name__ == "__main__":
    main()
