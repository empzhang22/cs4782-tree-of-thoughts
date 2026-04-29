from pathlib import Path
import json
from selenium import webdriver
from selenium.webdriver.common.by import By

URL = "https://www.4nums.com/game/difficulties/"

def scrape_puzzles(url, start_rank=901, end_rank=1000):
    driver = webdriver.Chrome()
    driver.get(url)

    rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")

    puzzles = []

    for row in rows:
        cells = row.find_elements(By.TAG_NAME, "td")

        if len(cells) < 2:
            continue

        try:
            rank = int(cells[0].text.strip())
        except ValueError:
            continue

        if start_rank <= rank <= end_rank:
            puzzle_text = cells[1].text.strip()

            try:
                numbers = list(map(int, puzzle_text.split()))
            except ValueError:
                numbers = puzzle_text

            puzzles.append({
                "rank": rank,
                "puzzle": puzzle_text,
                "numbers": numbers
            })

    driver.quit()
    return puzzles


def save_puzzles(puzzles, filename="puzzles_901_1000.json"):
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    data_dir.mkdir(exist_ok=True)

    output_path = data_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(puzzles, f, indent=2)

    print(f"Saved {len(puzzles)} puzzles to {output_path}")


def main():
    puzzles = scrape_puzzles(URL, start_rank=901, end_rank=1000)
    save_puzzles(puzzles)


if __name__ == "__main__":
    main()