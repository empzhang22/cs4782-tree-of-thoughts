from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass

from tasks.base_task import BaseTask
from tot.node import ThoughtNode
from .prompts import cot_prompt, finish_prompt, propose_prompt, standard_prompt, value_prompt


CLUE_KEYS = [f"h{i}" for i in range(1, 6)] + [f"v{i}" for i in range(1, 6)]
CONFIDENCE_TO_VALUE = {"certain": 1.0, "high": 0.5, "medium": 0.2, "low": 0.1}
VALUE_TO_SCORE = {"sure": 1.0, "maybe": 0.2, "impossible": 0.0}


@dataclass
class CrosswordState:
    board: list[list[str]]
    status: dict[str, int]
    steps: list[str]


class CrosswordTask(BaseTask):
    def __init__(self, dataset_path: str, problem_offset: int = 0):
        with open(dataset_path, encoding="utf-8") as f:
            items = json.load(f)
        self.prompt_items = [item for item in items if item.get("split") == "prompting"]
        self.test_items = [item for item in items if item.get("split") == "test"]
        self.offset = problem_offset
        self.examples = self._format_examples()

    def __len__(self) -> int:
        return len(self.test_items)

    def get_item(self, idx: int) -> dict:
        return self.test_items[self.offset + idx]

    def get_input(self, idx: int) -> str:
        return self.format_clues(self.get_item(idx))

    def make_root(self, idx: int) -> ThoughtNode:
        item = self.get_item(idx)
        return ThoughtNode(
            thought=self.get_input(idx),
            step_outputs={
                "item": item,
                "state": CrosswordState(
                    board=[["_"] * 5 for _ in range(5)],
                    status={key: 0 for key in CLUE_KEYS},
                    steps=[],
                ),
            },
        )

    def is_success(self, node: ThoughtNode) -> bool:
        item = node.step_outputs.get("item") or node.get_path()[0].step_outputs["item"]
        grid = node.step_outputs.get("grid") or self.state_to_grid(node.step_outputs["state"])
        return self.score_grid(grid, item)["game_correct"]

    def standard_prompt_fn(self, clue_text: str) -> str:
        return standard_prompt.format(examples=self.examples["standard"], input=clue_text)

    def cot_prompt_fn(self, clue_text: str) -> str:
        return cot_prompt.format(examples=self.examples["cot"], input=clue_text)

    def propose_prompt_fn(self, node: ThoughtNode) -> str:
        item = node.get_path()[0].step_outputs["item"]
        state = node.step_outputs["state"]
        return propose_prompt.format(input=self.render_state(item, state))

    def value_prompt_fn(self, clue: str, pattern: str) -> str:
        return value_prompt.format(input=f"{clue}: {' '.join(pattern.lower())}")

    def finish_prompt_fn(self, item: dict, state: CrosswordState) -> str:
        return finish_prompt.format(input=self.render_state(item, state))

    @staticmethod
    def format_clues(item: dict) -> str:
        lines = []
        for key, clue in item["queries"]["horizontal"].items():
            lines.append(f"{key}. {clue}")
        for key, clue in item["queries"]["vertical"].items():
            lines.append(f"{key}. {clue}")
        return "\n".join(lines)

    @staticmethod
    def format_grid(grid: list[list[str]]) -> str:
        return "\n".join(" ".join(row) for row in grid)

    @staticmethod
    def state_to_grid(state: CrosswordState) -> list[list[str]]:
        return [[letter if letter != "_" else "_" for letter in row] for row in state.board]

    @staticmethod
    def is_filled(state: CrosswordState) -> bool:
        return all(letter != "_" for row in state.board for letter in row)

    def _format_examples(self) -> dict[str, str]:
        standard_blocks = []
        cot_blocks = []
        for item in self.prompt_items:
            clues = self.format_clues(item)
            grid = self.format_grid(item["grid"])
            thoughts = self.format_thoughts(item)
            standard_blocks.append(f"Input:\n{clues}\n\nOutput:\n{grid}")
            cot_blocks.append(f"Input:\n{clues}\n\nThoughts:\n{thoughts}\n\nOutput:\n{grid}")
        return {"standard": "\n\n".join(standard_blocks), "cot": "\n\n".join(cot_blocks)}

    @staticmethod
    def format_thoughts(item: dict) -> str:
        lines = []
        for key, clue in item["queries"]["horizontal"].items():
            lines.append(f"{key}. {clue}: {item['answers']['horizontal'][key]}")
        for key, clue in item["queries"]["vertical"].items():
            lines.append(f"{key}. {clue}: {item['answers']['vertical'][key]}")
        return "\n".join(lines)

    @staticmethod
    def parse_grid(text: str) -> list[list[str]] | None:
        text = text.split("Output:")[-1]
        rows = []
        for line in text.strip().splitlines():
            letters = re.findall(r"[A-Za-z]", line.upper())
            if len(letters) >= 5:
                rows.append(letters[:5])
            if len(rows) == 5:
                return rows
        return None

    @staticmethod
    def parse_proposals(text: str) -> list[tuple[str, str, float]]:
        proposals = {}
        pattern = re.compile(r"\b([hv][1-5])\.\s*([A-Za-z]{5})\s*\((certain|high|medium|low)\)", re.I)
        for key, word, confidence in pattern.findall(text):
            proposal = (key.lower(), word.upper())
            proposals[proposal] = proposals.get(proposal, 0.0) + CONFIDENCE_TO_VALUE[confidence.lower()]
        return [(key, word, score) for (key, word), score in sorted(proposals.items(), key=lambda x: x[1], reverse=True)]

    @staticmethod
    def answer_words_from_grid(grid: list[list[str]]) -> dict[str, str]:
        words = {}
        for r in range(5):
            words[f"h{r + 1}"] = "".join(grid[r])
        for c in range(5):
            words[f"v{c + 1}"] = "".join(grid[r][c] for r in range(5))
        return words

    @staticmethod
    def apply_word(state: CrosswordState, key: str, word: str) -> CrosswordState:
        new_state = copy.deepcopy(state)
        old_words = CrosswordTask.answer_words_from_grid(new_state.board)
        if key.startswith("h"):
            row = int(key[1]) - 1
            new_state.board[row] = list(word.upper())
        else:
            col = int(key[1]) - 1
            for row in range(5):
                new_state.board[row][col] = word.upper()[row]

        new_words = CrosswordTask.answer_words_from_grid(new_state.board)
        for clue_key in CLUE_KEYS:
            if new_state.status[clue_key] == 1 and old_words[clue_key] != new_words[clue_key]:
                new_state.status[clue_key] = 2
        new_state.status[key] = 1
        new_state.steps.append(f"{key}. {word.upper()}")
        return new_state

    @staticmethod
    def score_grid(grid: list[list[str]] | None, item: dict) -> dict:
        if grid is None:
            grid = [["_"] * 5 for _ in range(5)]
        target = item["grid"]
        letters_correct = sum(grid[r][c] == target[r][c] for r in range(5) for c in range(5))
        pred_words = CrosswordTask.answer_words_from_grid(grid)
        target_words = {
            **item["answers"]["horizontal"],
            **item["answers"]["vertical"],
        }
        words_correct = sum(pred_words[key] == target_words[key] for key in CLUE_KEYS)
        return {
            "letter_accuracy": letters_correct / 25,
            "word_accuracy": words_correct / 10,
            "game_correct": letters_correct == 25,
            "letters_correct": letters_correct,
            "words_correct": words_correct,
            "pred_words": pred_words,
        }

    def render_state(self, item: dict, state: CrosswordState) -> str:
        return "\n".join([
            self.render_board(state),
            "",
            "Unfilled:",
            self.render_answers(item, state, status=0),
            "",
            "Filled:",
            self.render_answers(item, state, status=1),
            "",
            "Changed:",
            self.render_answers(item, state, status=2),
        ])

    @staticmethod
    def render_board(state: CrosswordState) -> str:
        return "Current Board:\n" + "\n".join("".join(row) for row in state.board)

    @staticmethod
    def render_answers(item: dict, state: CrosswordState, status: int | None = None) -> str:
        words = CrosswordTask.answer_words_from_grid(state.board)
        lines = []
        for key in CLUE_KEYS:
            if status is not None and state.status[key] != status:
                continue
            section = "horizontal" if key.startswith("h") else "vertical"
            lines.append(f"{key}. {item['queries'][section][key]}: {words[key]}")
        return "\n".join(lines)

    @staticmethod
    def constrained_entries(item: dict, state: CrosswordState) -> list[tuple[str, str, str]]:
        words = CrosswordTask.answer_words_from_grid(state.board)
        entries = []
        for key in CLUE_KEYS:
            pattern = words[key]
            if pattern.count("_") >= 4:
                continue
            section = "horizontal" if key.startswith("h") else "vertical"
            entries.append((key, item["queries"][section][key], pattern))
        return entries

    @staticmethod
    def parse_value(text: str) -> float:
        tail = text.strip().splitlines()[-1].strip().lower() if text.strip() else ""
        for label, score in VALUE_TO_SCORE.items():
            if label in tail:
                return score
        return 0.0
