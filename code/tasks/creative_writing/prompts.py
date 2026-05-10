# Prompts adapted from the Creative Writing/Text task in:
# https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/src/tot/prompts/text.py

standard_prompt = """Write a coherent passage of 4 short paragraphs.
The end sentence of each paragraph must be:
{input}
"""

cot_prompt = """Write a coherent passage of 4 short paragraphs.
The end sentence of each paragraph must be:
{input}

Make a plan then write. Your output should be of the following format:

Plan:
Your plan here.

Passage:
Your passage here.
"""

plan_prompt = """Write a concise plan for a coherent passage of 4 short paragraphs.
The end sentence of each paragraph must be:
{input}

Only write the plan.
"""

passage_prompt = """Write a coherent passage of 4 short paragraphs following the plan.
The end sentence of each paragraph must be:
{input}

Plan:
{plan}

Passage:
"""

vote_prompt = """Given an instruction and several choices, decide which choice is most promising.
Analyze each choice in detail, then conclude in the last line "The best choice is {{s}}", where s is the integer id of the choice.

Instruction:
Write a coherent passage of 4 short paragraphs.
The end sentence of each paragraph must be:
{input}

"""

score_prompt = """Analyze the following passage, then at the last line conclude "Thus the coherency score is {{s}}", where s is an integer from 1 to 10.

Passage:
{passage}
"""
