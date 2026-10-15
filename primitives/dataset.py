from binascii import crc32

from datasets import load_dataset
from torch import Tensor
from transformers import PreTrainedTokenizerBase

QUESTION_PROMPT = """\
Question: {question}
Answer:"""


def load_metamathqa(
    tokenizer: PreTrainedTokenizerBase,
    num_samples: int,
    max_sequence_len: int,
    held_out: bool = False,
) -> list[tuple[Tensor, Tensor]]:
    dataset = load_dataset("meta-math/MetaMathQA", split="train").shuffle(seed=0)
    seen_questions = set()
    samples = []
    for row in dataset:
        # gsm8k로 평가하므로 gsm 유형만 씀.
        if not row["type"].startswith("GSM"):
            continue

        # metamathqa는 원본 질문 하나를 여러 방식으로 변형해 담음.
        # 같은 질문이 학습과 측정에 함께 들어가지 않도록 원본 질문의 crc32로 나눔.
        original_question = row["original_question"]
        if (crc32(original_question.encode()) % 10 == 0) != held_out:
            continue
        if held_out and original_question in seen_questions:
            continue

        # llama tokenizer는 앞에 bos를 자동으로 붙임.
        # answer는 question 바로 뒤에 이어지므로 bos 없이 토큰화.
        question_ids = tokenizer(QUESTION_PROMPT.format(question=row["query"]), return_tensors="pt")["input_ids"][0]
        answer_ids = tokenizer(f" {row['response']}{tokenizer.eos_token}", add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        if len(question_ids) + len(answer_ids) > max_sequence_len:
            continue

        seen_questions.add(original_question)
        samples.append((question_ids, answer_ids))
        if len(samples) == num_samples:
            break
    return samples


def load_gsm8k(
    tokenizer: PreTrainedTokenizerBase,
    num_samples: int,
) -> list[tuple[Tensor, float]]:
    dataset = load_dataset("openai/gsm8k", "main", split="test").select(range(num_samples))
    samples = []
    for row in dataset:
        question_ids = tokenizer(QUESTION_PROMPT.format(question=row["question"]), return_tensors="pt")["input_ids"][0]
        answer = float(row["answer"].split("####")[-1].replace(",", ""))
        samples.append((question_ids, answer))
    return samples
