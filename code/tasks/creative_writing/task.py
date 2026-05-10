from __future__ import annotations

import json
import re

from tasks.base_task import BaseTask
from tot.node import ThoughtNode
from .prompts import cot_prompt, passage_prompt, plan_prompt, score_prompt, standard_prompt, vote_prompt


class CreativeWritingTask(BaseTask):
    def __init__(self, dataset_path: str, problem_offset: int = 0):
        with open(dataset_path, encoding="utf-8") as f:
            self._items = json.load(f)
        self.offset = problem_offset

    def __len__(self) -> int:
        return len(self._items)

    def get_input(self, idx: int) -> str:
        item = self._items[self.offset + idx]
        return self.format_sentences(item["sentences"])

    def get_sentences(self, idx: int) -> list[str]:
        return list(self._items[self.offset + idx]["sentences"])

    def get_id(self, idx: int) -> int:
        return int(self._items[self.offset + idx].get("id", self.offset + idx + 1))

    def make_root(self, idx: int) -> ThoughtNode:
        return ThoughtNode(
            thought=self.get_input(idx),
            step_outputs={"sentences": self.get_sentences(idx), "stage": "root"},
        )

    @staticmethod
    def format_sentences(sentences: list[str]) -> str:
        return "\n".join(f"{i}. {sentence}" for i, sentence in enumerate(sentences, 1))

    def standard_prompt_fn(self, sentences_text: str) -> str:
        return standard_prompt.format(input=sentences_text)

    def cot_prompt_fn(self, sentences_text: str) -> str:
        return cot_prompt.format(input=sentences_text)

    def sample_prompt_fn(self, node: ThoughtNode) -> str:
        sentences_text = node.get_path()[0].thought
        stage = node.step_outputs.get("stage")
        if stage == "root":
            return plan_prompt.format(input=sentences_text)
        if stage == "plan":
            return passage_prompt.format(
                input=sentences_text,
                plan=node.step_outputs.get("plan", node.thought),
            )
        raise ValueError(f"Cannot sample from creative writing stage: {stage!r}")

    def vote_prompt_fn(self, nodes: list[ThoughtNode]) -> str:
        if not nodes:
            raise ValueError("Cannot vote over an empty candidate list.")
        sentences_text = nodes[0].get_path()[0].thought
        prompt = vote_prompt.format(input=sentences_text)
        for i, node in enumerate(nodes, 1):
            prompt += f"Choice {i}:\n{node.thought.strip()}\n\n"
        return prompt

    def score_prompt_fn(self, passage: str) -> str:
        return score_prompt.format(passage=passage.strip())

    def parse_sample(self, text: str, node: ThoughtNode) -> tuple[str, dict]:
        stage = node.step_outputs.get("stage")
        clean = text.strip()
        if stage == "root":
            return clean, {"stage": "plan", "plan": clean}
        if stage == "plan":
            passage = self.extract_passage(clean)
            return passage, {
                "stage": "passage",
                "plan": node.step_outputs.get("plan", node.thought),
                "passage": passage,
            }
        return clean, {"stage": "unknown"}

    @staticmethod
    def extract_passage(text: str) -> str:
        parts = re.split(r"(?im)^\s*Passage:\s*$", text.strip(), maxsplit=1)
        return parts[-1].strip()

    @staticmethod
    def extract_plan(text: str) -> str:
        plan_match = re.search(r"(?is)^\s*Plan:\s*(.*?)(?:^\s*Passage:\s*|\Z)", text.strip(), re.MULTILINE)
        return plan_match.group(1).strip() if plan_match else ""

    @staticmethod
    def parse_vote(text: str, n_candidates: int) -> int:
        match = re.search(r"best choice is\s*\{?(\d+)\}?", text, re.IGNORECASE | re.DOTALL)
        if not match:
            numbers = re.findall(r"\b([1-9]\d*)\b", text)
            vote = int(numbers[-1]) if numbers else 1
        else:
            vote = int(match.group(1))
        return max(0, min(vote - 1, n_candidates - 1))

    @staticmethod
    def parse_score(text: str) -> int | None:
        match = re.search(r"coherency score is\s*\{?(\d+)\}?", text, re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        score = int(match.group(1))
        return max(1, min(score, 10))

    @staticmethod
    def constraints_satisfied(output: str, sentences: list[str]) -> bool:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", output.strip()) if p.strip()]
        if len(paragraphs) != 4:
            return False
        return all(paragraph.endswith(sentence) for paragraph, sentence in zip(paragraphs, sentences))

    def is_terminal(self, node: ThoughtNode) -> bool:
        return node.step_outputs.get("stage") == "passage"

    def is_success(self, node: ThoughtNode) -> bool:
        passage = node.step_outputs.get("passage", node.thought)
        sentences = node.get_path()[0].step_outputs.get("sentences", [])
        return self.constraints_satisfied(passage, sentences)
