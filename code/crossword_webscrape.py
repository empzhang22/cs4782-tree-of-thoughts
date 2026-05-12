import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.goobix.com/crosswords/0505/{puzzle_id}/"

TEST_PUZZLES = list(range(1, 97, 5))          # 1, 6, 11, ..., 96
PROMPT_PUZZLES = [136, 141, 146, 151, 156]

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "crossword"
OUTPUT_PATH = OUTPUT_DIR / "crosswords_5x5.json"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def scrape_puzzle(puzzle_id: int, split: str) -> dict:
    url = BASE_URL.format(puzzle_id=puzzle_id)
    response = requests.get(url, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    text_lines = [
        clean_text(line)
        for line in soup.get_text("\n").splitlines()
        if clean_text(line)
    ]

    horizontal_clues = []
    vertical_clues = []

    section = None
    for line in text_lines:
        if line == "Horizontal":
            section = "horizontal"
            continue
        if line == "Vertical":
            section = "vertical"
            continue
        if line.startswith("Another random puzzle"):
            section = None
            continue

        if section == "horizontal":
            horizontal_clues.append(line)
        elif section == "vertical":
            vertical_clues.append(line)

    if len(horizontal_clues) != 5 or len(vertical_clues) != 5:
        raise ValueError(
            f"Puzzle {puzzle_id} did not return 5 horizontal and 5 vertical clues. "
            f"Found {len(horizontal_clues)} horizontal, {len(vertical_clues)} vertical."
        )

    queries = {
        "horizontal": {
            f"h{i + 1}": clue for i, clue in enumerate(horizontal_clues)
        },
        "vertical": {
            f"v{i + 1}": clue for i, clue in enumerate(vertical_clues)
        },
    }

    answers = {
        "horizontal": {
            f"h{i + 1}": "" for i in range(5)
        },
        "vertical": {
            f"v{i + 1}": "" for i in range(5)
        },
    }

    return {
        "id": puzzle_id,
        "split": split,
        "url": url,
        "queries": queries,
        "answers": answers,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    puzzles = []

    for puzzle_id in TEST_PUZZLES:
        print(f"Scraping test puzzle {puzzle_id}...")
        puzzles.append(scrape_puzzle(puzzle_id, split="test"))
        time.sleep(0.5)

    for puzzle_id in PROMPT_PUZZLES:
        print(f"Scraping prompting puzzle {puzzle_id}...")
        puzzles.append(scrape_puzzle(puzzle_id, split="prompting"))
        time.sleep(0.5)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(puzzles, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(puzzles)} puzzles to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()