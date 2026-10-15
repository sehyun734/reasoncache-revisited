from collections import defaultdict
from dataclasses import dataclass

from simple_parsing import parse
import torch
from torch import Tensor
from transformers import DynamicCache, PreTrainedModel

from primitives.dataset import load_metamathqa
from primitives.hub import get_pending_targets, load_tensors, save_result
from primitives.model import hook_lora, load_model, make_prefix_cache


@dataclass
class Config:
    model_name: str = "Llama-3.2-1B"
    num_samples: int = 256
    max_sequence_len: int = 768


def capture_attns(
    model: PreTrainedModel,
    samples: list[tuple[Tensor, Tensor]],
    sink: Tensor | None = None,
    prefix: Tensor | None = None,
) -> list[Tensor]:
    attns = []
    for question_ids, answer_ids in samples:
        cache = make_prefix_cache(sink, prefix, 1) if prefix is not None else DynamicCache()
        input_ids = torch.cat([question_ids, answer_ids]).unsqueeze(0).to(model.device)
        output = model(input_ids=input_ids, past_key_values=cache, output_attentions=True)
        attns.append(torch.stack(output.attentions)[:, 0].cpu())
    return attns


def compute_entropy(
    prob: Tensor,
) -> Tensor:
    # sink나 prefix를 잘라낸 attention은 합이 1보다 작으므로 다시 정규화.
    prob = prob.float() / prob.float().sum(-1, keepdim=True)
    return -(prob * prob.clamp_min(1e-12).log()).sum(-1)


def compute_jsd(
    prob_a: Tensor,
    prob_b: Tensor,
) -> Tensor:
    prob_a = prob_a.float() / prob_a.float().sum(-1, keepdim=True)
    prob_b = prob_b.float() / prob_b.float().sum(-1, keepdim=True)
    return compute_entropy((prob_a + prob_b) / 2) - (compute_entropy(prob_a) + compute_entropy(prob_b)) / 2


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    file_name = f"num_samples={config.num_samples}"
    targets = get_pending_targets("attention", config.model_name, file_name)
    # sdpa는 attention 가중치를 반환하지 않으므로 eager를 씀.
    model, tokenizer = load_model(config.model_name, attn_implementation="eager")
    samples = load_metamathqa(tokenizer, config.num_samples, config.max_sequence_len, held_out=True)

    num_query_tokens = sum(len(question_ids) + len(answer_ids) - 1 for question_ids, answer_ids in samples)
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    base_attns = capture_attns(model, samples)
    for target in targets:
        method = target.split("/")[0]
        if method == "prefix":
            tensors = load_tensors(target, "prefix")
            sink = tensors["sink"].to(model.device)
            prefix = tensors["prefix"].to(model.device)
            attns = capture_attns(model, samples, sink=sink, prefix=prefix)
            prefix_len = prefix.shape[-2]
        elif method == "lora":
            lora = {name: tensor.to(model.device) for name, tensor in load_tensors(target, "lora").items()}
            handles = hook_lora(model, lora)
            attns = capture_attns(model, samples)
            for handle in handles:
                handle.remove()
        elif method == "base":
            attns = base_attns

        stats = defaultdict(lambda: torch.zeros(num_layers, num_heads))
        for layer in range(num_layers):
            for head in range(num_heads):
                for attn, base_attn in zip(attns, base_attns):
                    if method == "prefix":
                        # cache 맨 앞의 sink가 입력 bos의 sink 역할을 가져갈 수 있음.
                        # 따라서 base와 같은 기준으로 보려고 cache sink와 입력 bos를 합쳐 sink로 봄.
                        sink_attn = attn[layer, head, 1:, 0]
                        bos_attn = attn[layer, head, 1:, prefix_len + 1]
                        prefix_attn = attn[layer, head, 1:, 1 : prefix_len + 1]
                        input_attn = attn[layer, head, 1:, prefix_len + 2 :]
                        stats["sink_ratio"][layer, head] += (sink_attn + bos_attn).sum() / num_query_tokens
                        stats["prefix_ratio"][layer, head] += prefix_attn.sum() / num_query_tokens
                    else:
                        bos_attn = attn[layer, head, 1:, 0]
                        input_attn = attn[layer, head, 1:, 1:]
                        stats["sink_ratio"][layer, head] += bos_attn.sum() / num_query_tokens
                    base_input_attn = base_attn[layer, head, 1:, 1:]
                    stats["entropy"][layer, head] += compute_entropy(input_attn).sum() / num_query_tokens
                    stats["jsd"][layer, head] += compute_jsd(input_attn, base_input_attn).sum() / num_query_tokens

        result = {name: stat.tolist() for name, stat in stats.items()}
        save_result(f"attention/{target}", file_name, result)


if __name__ == "__main__":
    main()
