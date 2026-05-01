import json
import random
from pathlib import Path

INPUT_PATH = Path("data/creative_writing_sentences.txt")
OUTPUT_PATH = Path("data/creative_writing_prompts.json")

GROUP_SIZE = 4
SHUFFLE = True


def load_sentences(path):
    with open(path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    return sentences


def build_test_cases(sentences, group_size=4, shuffle=True):
    if shuffle:
        random.shuffle(sentences)

    if len(sentences) % group_size != 0:
        raise ValueError(
            f"Number of sentences ({len(sentences)}) must be divisible by {group_size}"
        )

    test_cases = []
    for i in range(0, len(sentences), group_size):
        case = {
            "id": (i // group_size) + 1,
            "sentences": sentences[i:i + group_size]
        }
        test_cases.append(case)

    return test_cases


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    sentences = load_sentences(INPUT_PATH)
    test_cases = build_test_cases(sentences, GROUP_SIZE, SHUFFLE)
    save_json(test_cases, OUTPUT_PATH)
    print(f"Saved {len(test_cases)} test cases to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()