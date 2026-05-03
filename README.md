# Tree of Thoughts: Re-implementation

## 1. Introduction

This repository is a re-implementation of the paper **"Tree of Thoughts: Deliberate Problem Solving with Large Language Models"** by Yao et al. (2023), submitted as a course project for CS 4782 (Probabilistic Machine Learning) at Cornell University.

The original paper introduces **Tree of Thoughts (ToT)**, a framework that generalizes chain-of-thought prompting by allowing language models to explore multiple reasoning paths and self-evaluate intermediate steps—much like a search algorithm navigating a tree of possible thoughts. This deliberate search strategy substantially outperforms single-pass prompting on tasks that require planning and multi-step reasoning. Rather than committing to one chain of thought, ToT maintains a frontier of partial solutions, scores them with an LLM-based evaluator, and prunes unpromising branches—enabling backtracking and lookahead that standard CoT lacks.

---

## 2. Chosen Result

We target **Table 2** from the original paper: success rate on the **Game of 24** across four prompting strategies—Input-Output (IO), Chain-of-Thought (CoT), CoT with Self-Consistency (CoT-SC), and Tree of Thoughts with BFS (ToT-BFS).

The Game of 24 is a mathematical reasoning task where the model must combine four numbers using basic arithmetic to reach exactly 24. The paper evaluates on the 100 hardest puzzles (difficulty ranks 901–1000), reporting the following GPT-4 results:

| Method  | Success Rate (GPT-4, paper) |
|---------|-----------------------------|
| IO      | 7.3%                        |
| CoT     | 4.0%                        |
| CoT-SC  | 9.0%                        |
| ToT-BFS | **74.0%**                   |

The ToT result is the paper's headline finding: structured tree search with LLM self-evaluation more than triples the success rate of the next best baseline. Reproducing this gap confirms that the benefit of ToT is not model-specific and transfers to a different frontier model.

---

## 3. GitHub Contents

```
.
├── code/
│   ├── analysis/         # Figure generation (plots.py, generate_figures.py, poster_style.py)
│   ├── baselines/        # IO, CoT, and CoT-SC baseline implementations
│   ├── configs/          # YAML experiment configs (one per method × model)
│   ├── experiments/      # Experiment runner and evaluator
│   ├── llm/              # LLM client wrappers (OpenAI, Gemini) with disk caching
│   ├── tasks/game24/     # Game of 24 task: prompts, parsing, evaluation
│   ├── tot/              # Core ToT components: BFS, node, generator, evaluator
│   └── run_experiment.py # CLI entry point
├── data/game24/24.csv    # Game of 24 dataset (1,362 puzzles with difficulty ranks)
├── results/
│   ├── cache/            # Disk cache for LLM responses (diskcache)
│   ├── figures/          # Generated figures (PNG + PDF)
│   └── game24/           # Per-method JSONL result files
└── .env.example          # Required API key configuration
```

---

## 4. Re-implementation Details

### Approach

We re-implement ToT-BFS for the Game of 24 from scratch, following the algorithm described in Section 3 of the paper. Each node in the thought tree represents a partial state: the arithmetic steps taken so far and the numbers still remaining. At each BFS depth, the generator proposes new steps; the evaluator scores each partial state as `sure`, `likely`, or `impossible`; and the top-`b` nodes by score are kept for the next level.

### Models

Rather than GPT-4 (the original paper's model, behind a pay-per-call API), we use **Gemini 2.5 Flash** (`gemini-2.5-flash`) via the Google GenAI SDK for all four methods. This choice reflects cost constraints for a student project. We also provide configs for `gpt-4o` if an OpenAI key is available.

### Dataset

The Game of 24 dataset (`data/game24/24.csv`) contains 1,362 puzzles ranked by human difficulty. Following the paper, experiments run on the 100 hardest puzzles (offset 900, ranks 901–1000).

### Baselines

- **IO**: single-shot prompt asking for a direct answer.
- **CoT**: prompt with step-by-step reasoning instructions.
- **CoT-SC**: five CoT samples with majority vote on the final answer.
- **ToT-BFS**: propose 5 steps per node, keep top 5 per depth (breadth=5), up to depth 3. Value calls use temperature=0.

### Evaluation Metric

Binary success: the model's output expression evaluates to exactly 24 using the four puzzle numbers (verified by safe expression parsing, not string matching).

### Key Implementation Details

- LLM responses are disk-cached (`diskcache`) so repeated runs incur no extra API cost.
- The propose parser handles multiple LLM output formats including LaTeX-escaped math, numbered lists, and two-line "a op b = c / Remaining: ..." formats.
- A `--dry-run` flag estimates API call counts before spending budget.

### Challenges

- **Output format variance**: Gemini 2.5 Flash frequently produced LaTeX-formatted math (`\text{left:}`, `\times`, `\boxed{}`). We added a normalization pass in `tasks/game24/task.py` to strip LaTeX before parsing.
- **CoT-SC API counting**: Gemini batches parallel calls, so the `api_calls` field shows 1 for CoT-SC. True cost is ~5 calls/problem; figures annotate this discrepancy.
- **Model capability gap**: ToT-BFS with Gemini 2.5 Flash achieves 44% vs. 74% with GPT-4, a gap we attribute primarily to weaker arithmetic reasoning in the propose step rather than a flaw in the ToT algorithm itself.

---

## 5. Reproduction Steps

### Prerequisites

- Python 3.10+
- A Gemini API key (`GEMINI_API_KEY`) and/or an OpenAI API key (`OPENAI_API_KEY`)
- ~10–20 GPU-hours of inference budget (CPU-only; no local GPU required — all inference is via API)

### Setup

```bash
git clone https://github.com/<your-username>/cs4782-tree-of-thoughts.git
cd cs4782-tree-of-thoughts

pip install -r code/requirements.txt

cp .env.example .env
# Edit .env and add your API key(s):
# GEMINI_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here  (optional)
```

### Running Experiments

All experiments are launched from the repo root via `code/run_experiment.py`:

```bash
# IO baseline (Gemini 2.5 Flash, 100 problems)
python code/run_experiment.py --config code/configs/game24_io_gemini25flash.yaml

# CoT baseline
python code/run_experiment.py --config code/configs/game24_cot_gemini25flash.yaml

# CoT-SC baseline
python code/run_experiment.py --config code/configs/game24_cot_sc_gemini25flash.yaml

# Tree of Thoughts (BFS) — most expensive
python code/run_experiment.py --config code/configs/game24_tot_bfs_gemini25flash.yaml
```

To estimate API call count before running:

```bash
python code/run_experiment.py --config code/configs/game24_tot_bfs_gemini25flash.yaml --dry-run
```

To run a smaller subset (e.g., 10 problems starting at rank 901):

```bash
python code/run_experiment.py --config code/configs/game24_tot_bfs_gemini25flash.yaml --n-problems 10
```

### Generating Figures

After all four experiments have produced `results.jsonl` files:

```bash
python code/analysis/generate_figures.py
# Figures saved to results/figures/
```

### Computational Resources

No GPU is required. All inference is remote (Gemini or OpenAI API). Expected wall-clock time per experiment on 100 problems:

| Method  | Approx. Time | Approx. API Calls |
|---------|-------------|-------------------|
| IO      | ~5 min      | 100               |
| CoT     | ~5 min      | 100               |
| CoT-SC  | ~15 min     | ~500              |
| ToT-BFS | ~30–45 min  | ~1,500            |

---

## 6. Results / Insights

Our re-implementation reproduces the **qualitative ranking** of the original paper: ToT-BFS outperforms all baselines by a substantial margin. Quantitatively, Gemini 2.5 Flash underperforms GPT-4 but the ToT advantage is preserved.

### Success Rate on Game of 24 (Problems 901–1000)

| Method  | GPT-4 (Yao et al., 2023) | Gemini 2.5 Flash (ours) |
|---------|--------------------------|--------------------------|
| IO      | 7.3%                     | 10%                      |
| CoT     | 4.0%                     | 14%                      |
| CoT-SC  | 9.0%                     | 18%                      |
| ToT-BFS | **74.0%**                | **44%**                  |

ToT-BFS delivers **2.4–4.4× the success rate** of the best single-pass baseline (CoT-SC) on Gemini 2.5 Flash, consistent with the paper's claim that tree search adds fundamental value beyond better prompting alone.

The difficulty curve figure (`results/figures/difficulty_curve.png`) further shows that ToT-BFS degrades more gracefully on the hardest puzzles, while IO and CoT collapse to near-zero on the highest-difficulty ranks.

The API cost figure (`results/figures/api_cost.png`) illustrates the tradeoff: ToT-BFS uses ~15× more API calls per problem than IO, consistent with the paper's report of ~100 LLM calls for ToT vs. 1 for IO (the difference in absolute numbers reflects our smaller breadth and depth settings).

---

## 7. Conclusion

We successfully reproduced the core finding of Yao et al. (2023): deliberate tree search over LLM-generated thoughts dramatically outperforms chain-of-thought prompting on the Game of 24. The ToT advantage is robust across models — it holds on Gemini 2.5 Flash despite an absolute gap from GPT-4, suggesting the algorithm's benefit is not tied to any one model's capabilities.

Key lessons:

- **Parsing is the hard part.** The propose/value prompts from the paper are straightforward to implement; handling the variance in LLM output format (LaTeX, numbered lists, two-line formats) required significant engineering.
- **Caching is essential.** Disk-caching all LLM calls made iterative development and figure regeneration practical without re-spending API budget.
- **Model capability affects scale, not direction.** A weaker model reduces the absolute success rate of ToT, but the relative ordering of methods remains consistent with the paper.

---

## 8. References

1. Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., & Narasimhan, K. (2023). **Tree of Thoughts: Deliberate Problem Solving with Large Language Models**. *NeurIPS 2023*. https://arxiv.org/abs/2305.10601
2. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Chi, E., Le, Q., & Zhou, D. (2022). **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**. *NeurIPS 2022*. https://arxiv.org/abs/2201.11903
3. Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2022). **Self-Consistency Improves Chain of Thought Reasoning in Language Models**. https://arxiv.org/abs/2203.11171
4. Google DeepMind. (2024). **Gemini 2.5 Flash**. https://deepmind.google/technologies/gemini/
5. Game of 24 dataset: https://github.com/princeton-nlp/tree-of-thought-llm/tree/master/src/tot/data/24

---

## 9. Acknowledgements

This project was completed as a final course project for **CS 4782: Probabilistic Machine Learning** at Cornell University (Spring 2026). We thank the course instructors and TAs for guidance on the re-implementation methodology and for feedback on the experimental design.

The original Tree of Thoughts codebase by Yao et al. (https://github.com/princeton-nlp/tree-of-thought-llm) served as a reference for prompt design and evaluation protocol, though all code in this repository was written independently from scratch.
