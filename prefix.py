from dataclasses import dataclass

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Linear, Module, Parameter, Sequential, Tanh
from transformers import DynamicCache, PreTrainedModel, set_seed

from primitives.dataset import load_metamathqa
from primitives.hub import make_prefix_path, save_tensors
from primitives.model import load_model
from primitives.train import train


@dataclass
class Config:
    model_name: str = "Llama-3.2-1B"
    prefix_len: int = 16
    mlp_dim: int = 512
    num_samples: int = 32768
    max_sequence_len: int = 2048
    num_epochs: int = 3
    batch_size: int = 128
    micro_size: int = 4
    learning_rate: float = 1e-3
    warmup_ratio: float = 0.05
    seed: int = 0


class PrefixMLP(Module):
    def __init__(
        self,
        prefix: Tensor,
        mlp_dim: int,
    ) -> None:
        # prefix를 직접 학습하면 위치마다 따로 update되어 불안정함.
        # 따라서 모든 위치를 같은 mlp에서 만듦.
        super().__init__()
        num_layers, _, _, num_kv_heads, prefix_len, head_dim = prefix.shape
        self.prefix = prefix
        self.embedding = Parameter(torch.randn(prefix_len, mlp_dim))
        self.mlp = Sequential(
            Linear(mlp_dim, mlp_dim),
            Tanh(),
            Linear(mlp_dim, num_layers * 2 * 1 * num_kv_heads * head_dim, bias=False),
        )
        # 마지막 층을 0으로 둬 학습을 초기 prefix 그대로에서 시작함.
        torch.nn.init.zeros_(self.mlp[-1].weight)

    def forward(
        self,
    ) -> Tensor:
        num_layers, _, _, num_kv_heads, prefix_len, head_dim = self.prefix.shape
        delta = self.mlp(self.embedding).view(prefix_len, num_layers, 2, 1, num_kv_heads, head_dim)
        return self.prefix + delta.movedim(0, 4)


@torch.no_grad()
def initialize_prefix(
    model: PreTrainedModel,
    prefix_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    # 맨 앞 bos는 sink로 고정하고 나머지만 prefix로 학습.
    cache = DynamicCache()
    model(input_ids=prefix_ids.unsqueeze(0).to(model.device), past_key_values=cache)
    key_value = torch.stack([torch.stack([layer.keys, layer.values]) for layer in cache.layers])
    sink = key_value[..., :1, :].clone()
    prefix = key_value[..., 1:, :].float()
    return sink, prefix


def main() -> None:
    config = parse(Config)
    set_seed(config.seed)
    model, tokenizer = load_model(config.model_name)

    # prefix 초기화에 쓸 question을 prefix_len개 더 불러옴.
    # 이어 붙일 때 bos는 첫 question의 것만 남김.
    samples = load_metamathqa(tokenizer, config.prefix_len + config.num_samples, config.max_sequence_len)
    question_ids = [sample_question_ids if index == 0 else sample_question_ids[1:] for index, (sample_question_ids, _) in enumerate(samples[: config.prefix_len])]
    prefix_ids = torch.cat(question_ids)[: config.prefix_len + 1]
    samples = samples[config.prefix_len :]

    sink, prefix = initialize_prefix(model, prefix_ids)
    prefix_mlp = PrefixMLP(prefix, config.mlp_dim).to(model.device)
    train(model, samples, list(prefix_mlp.parameters()), config.num_epochs, config.batch_size, config.micro_size, config.learning_rate, config.warmup_ratio, sink=sink, prefix_mlp=prefix_mlp)

    save_tensors(make_prefix_path(config.model_name, config.prefix_len, config.learning_rate, config.seed), "prefix", {"sink": sink, "prefix": prefix_mlp()})


if __name__ == "__main__":
    main()
