import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file, save
from torch import Tensor


def get_repo_name() -> str:
    return f"{HfApi().whoami()['name']}/reasoncache-revisited"


def make_lora_path(
    model_name: str,
    rank: int,
    learning_rate: float,
    seed: int,
) -> str:
    return f"lora/{model_name}/rank={rank},learning_rate={learning_rate:g},seed={seed}"


def make_prefix_path(
    model_name: str,
    prefix_len: int,
    learning_rate: float,
    seed: int,
) -> str:
    return f"prefix/{model_name}/prefix_len={prefix_len},learning_rate={learning_rate:g},seed={seed}"


def get_pending_targets(
    entry_name: str,
    model_name: str,
    file_name: str,
) -> list[str]:
    files = set(HfApi().list_repo_files(get_repo_name()))
    targets = [f"base/{model_name}"]
    for file in sorted(files):
        if file.startswith((f"lora/{model_name}/", f"prefix/{model_name}/")) and file.endswith(("/lora.safetensors", "/prefix.safetensors")):
            targets.append(str(Path(file).parent))
    targets = [target for target in targets if f"{entry_name}/{target}/{file_name}.json" not in files]
    print(f"{entry_name} pending {len(targets)}", *targets, sep="\n  ")
    return targets


def save_result(
    path: str,
    file_name: str,
    result: dict,
) -> None:
    api = HfApi()
    repo_name = get_repo_name()
    api.create_repo(repo_name, exist_ok=True)
    api.upload_file(path_or_fileobj=json.dumps(result, indent=2).encode(), path_in_repo=f"{path}/{file_name}.json", repo_id=repo_name, commit_message=f"Save {path}/{file_name}.json")
    print(f"saved {path}/{file_name}.json")


def save_tensors(
    path: str,
    file_name: str,
    tensors: dict[str, Tensor],
) -> None:
    api = HfApi()
    repo_name = get_repo_name()
    api.create_repo(repo_name, exist_ok=True)
    data = save({name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()})
    api.upload_file(path_or_fileobj=data, path_in_repo=f"{path}/{file_name}.safetensors", repo_id=repo_name, commit_message=f"Save {path}/{file_name}.safetensors")
    print(f"saved {path}/{file_name}.safetensors")


def load_tensors(
    path: str,
    file_name: str,
) -> dict[str, Tensor]:
    return load_file(hf_hub_download(get_repo_name(), f"{path}/{file_name}.safetensors"))
