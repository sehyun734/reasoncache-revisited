from collections import defaultdict
from dataclasses import dataclass

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn.functional import pad
from transformers import DynamicCache, PreTrainedModel

from primitives.dataset import load_metamathqa
from primitives.hub import get_pending_targets, load_tensors, save_result
from primitives.model import hook_lora, load_model, make_prefix_cache


@dataclass
class Config:
    model_name: str = "Llama-3.2-1B"
    num_samples: int = 256
    max_sequence_len: int = 768
    mass_ratio: float = 0.9
    energy_ratio: float = 0.9


def capture_values(
    model: PreTrainedModel,
    samples: list[tuple[Tensor, Tensor]],
    sink: Tensor | None = None,
    prefix: Tensor | None = None,
) -> tuple[list[Tensor], list[Tensor]]:
    # sink, prefix, 입력 bos를 뺀 입력 토큰의 value와 마지막 hidden만 모음.
    values = []
    hiddens = []
    for question_ids, answer_ids in samples:
        cache = make_prefix_cache(sink, prefix, 1) if prefix is not None else DynamicCache()
        cache_len = cache.get_seq_length()
        input_ids = torch.cat([question_ids, answer_ids]).unsqueeze(0).to(model.device)
        output = model(input_ids=input_ids, past_key_values=cache, output_hidden_states=True)
        values.append(torch.stack([layer.values[0, :, cache_len + 1 :] for layer in cache.layers]))
        hiddens.append(output.hidden_states[-1][0, 1:].float().cpu())
    return values, hiddens


def compute_basis(
    value: Tensor,
    energy_ratio: float,
) -> Tensor:
    # value는 noise 때문에 거의 항상 full rank임.
    # 따라서 특이값 제곱의 누적 합이 energy_ratio를 채우는 방향만 씀.
    _, singular, basis = torch.linalg.svd(value.float(), full_matrices=False)
    energy = singular.square()
    return basis[: int((energy.cumsum(0) / energy.sum() < energy_ratio).sum()) + 1]


def compute_new_ratio(
    value: Tensor,
    basis: Tensor,
) -> float:
    value = value.float()
    return ((value - value @ basis.T @ basis).norm() / value.norm()).item()


def compute_hidden_rank(
    hiddens: list[Tensor],
    mass_ratio: float,
) -> int:
    # sample마다 특이값을 합이 1이 되게 정규화해 평균 냄.
    # sample마다 길이가 달라 특이값 개수가 다르므로 0으로 채움.
    spectra = []
    for hidden in hiddens:
        singular = torch.linalg.svdvals(hidden.float())
        spectra.append(pad(singular / singular.sum(), (0, hidden.shape[-1] - len(singular))))
    spectrum = torch.stack(spectra).mean(0)
    return int((spectrum.cumsum(0) < mass_ratio).sum()) + 1


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    file_name = f"num_samples={config.num_samples}"
    targets = get_pending_targets("vrank", config.model_name, file_name)
    # attention 구현에 따라 hidden 값이 조금 달라짐.
    # 따라서 output_attentions 때문에 eager를 쓰는 attention.py와 같은 기준으로 맞춤.
    model, tokenizer = load_model(config.model_name, attn_implementation="eager")
    samples = load_metamathqa(tokenizer, config.num_samples, config.max_sequence_len, held_out=True)

    num_samples = len(samples)
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    base_values, base_hiddens = capture_values(model, samples)
    for target in targets:
        method = target.split("/")[0]
        prefix_value = None
        if method == "prefix":
            tensors = load_tensors(target, "prefix")
            sink = tensors["sink"].to(model.device)
            prefix = tensors["prefix"].to(model.device)
            values, hiddens = capture_values(model, samples, sink=sink, prefix=prefix)
            prefix_value = prefix[:, 1, 0]
        elif method == "lora":
            lora = {name: tensor.to(model.device) for name, tensor in load_tensors(target, "lora").items()}
            handles = hook_lora(model, lora)
            values, hiddens = capture_values(model, samples)
            for handle in handles:
                handle.remove()
        elif method == "base":
            values, hiddens = base_values, base_hiddens

        stats = defaultdict(lambda: torch.zeros(num_layers, num_kv_heads))
        for layer in range(num_layers):
            for head in range(num_kv_heads):
                # value를 이어 붙여 한 번에 구하면 sample마다 다른 주요 방향이 섞여 vrank가 부풀려짐.
                # 따라서 basis와 vrank는 sample별로 구해 평균 냄.
                for value, base_value in zip(values, base_values):
                    basis = compute_basis(base_value[layer, head], config.energy_ratio)
                    stats["vrank"][layer, head] += len(compute_basis(value[layer, head], config.energy_ratio)) / num_samples
                    stats["new_ratio"][layer, head] += compute_new_ratio(value[layer, head], basis) / num_samples
                    if prefix_value is not None:
                        stats["prefix_new_ratio"][layer, head] += compute_new_ratio(prefix_value[layer, head], basis) / num_samples
                if prefix_value is not None:
                    stats["prefix_vrank"][layer, head] = len(compute_basis(prefix_value[layer, head], config.energy_ratio))

        hidden_rank = compute_hidden_rank(hiddens, config.mass_ratio)
        result = {"hidden_rank": hidden_rank} | {name: stat.tolist() for name, stat in stats.items()}
        save_result(f"vrank/{target}", file_name, result)


if __name__ == "__main__":
    main()
