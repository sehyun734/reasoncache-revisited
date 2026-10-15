from dataclasses import dataclass
import math
import time

from simple_parsing import parse

from primitives.dataset import load_gsm8k
from primitives.evaluation import parse_gsm8k_answer
from primitives.hub import get_pending_targets, load_tensors, save_result
from primitives.model import generate, hook_lora, load_model
from primitives.utils import format_duration


@dataclass
class Config:
    model_name: str = "Llama-3.2-1B"
    num_samples: int = 1319
    batch_size: int = 64
    max_generation_len: int = 512


def main() -> None:
    config = parse(Config)
    file_name = f"num_samples={config.num_samples}"
    targets = get_pending_targets("gsm8k", config.model_name, file_name)
    model, tokenizer = load_model(config.model_name)
    samples = load_gsm8k(tokenizer, config.num_samples)

    num_batches = math.ceil(len(samples) / config.batch_size)
    for target in targets:
        method = target.split("/")[0]
        sink = None
        prefix = None
        handles = []
        if method == "prefix":
            tensors = load_tensors(target, "prefix")
            sink = tensors["sink"].to(model.device)
            prefix = tensors["prefix"].to(model.device)
        elif method == "lora":
            lora = {name: tensor.to(model.device) for name, tensor in load_tensors(target, "lora").items()}
            handles = hook_lora(model, lora)

        corrects = []
        log_time = time.perf_counter()
        for step, start in enumerate(range(0, len(samples), config.batch_size)):
            batch = samples[start : start + config.batch_size]
            output_ids = generate(model, tokenizer, [question_ids for question_ids, _ in batch], config.max_generation_len, sink=sink, prefix=prefix)
            texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            corrects.extend(parse_gsm8k_answer(text) == answer for text, (_, answer) in zip(texts, batch))
            print(f"batch {step + 1}/{num_batches} accuracy {sum(corrects) / len(corrects):.2%} time {format_duration(time.perf_counter() - log_time)}")
            log_time = time.perf_counter()

        for handle in handles:
            handle.remove()
        accuracy = sum(corrects) / len(samples)
        print(f"{target} accuracy {accuracy:.2%}")
        save_result(f"gsm8k/{target}", file_name, {"accuracy": accuracy})


if __name__ == "__main__":
    main()
