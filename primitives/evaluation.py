import re


def parse_gsm8k_answer(
    text: str,
) -> float | None:
    # "The answer is" 뒤의 첫 숫자, 없으면 마지막 숫자를 답으로 봄.
    before, _, after = text.replace(",", "").partition("The answer is")
    numbers = re.findall(r"-?\d*\.?\d+", after)[:1] if after else re.findall(r"-?\d*\.?\d+", before)[-1:]
    return float(numbers[0]) if numbers else None
