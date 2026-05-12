standard_prompt = """Solve 5x5 mini crosswords. Given an input of 5 horizontal clues and 5 vertical clues, generate an output of 5 rows, where each row is 5 letters separated by spaces.

{examples}
Input:
{input}

Output:
"""

cot_prompt = """Solve 5x5 mini crosswords. Given an input of 5 horizontal clues and 5 vertical clues, generate thoughts about which 5-letter word fits each clue, then an output of 5 rows, where each row is 5 letters separated by spaces.

{examples}
Input:
{input}

Thoughts:
"""

propose_prompt = """Let's play a 5 x 5 mini crossword, where each word should have exactly 5 letters.

{input}

Given the current status, list possible answers for unfilled or changed words, and your confidence levels (certain/high/medium/low), using exactly this format:
h1. apple (medium)

Use "certain" cautiously and only when you are 100% sure this is the correct word.
You can list more than one possible answer for each word.
"""

value_prompt = """Evaluate if there exists a five letter word of some meaning that fits some letter constraints (sure/maybe/impossible).

Incorrect; to injure: w _ o _ g
The letter constraint is: 5 letters, letter 1 is w, letter 3 is o, letter 5 is g.
Some possible words that mean "Incorrect; to injure": wrong (w r o n g): 5 letters, letter 1 is w, letter 3 is o, letter 5 is g.
fit!
sure

A person with an all-consuming enthusiasm, such as for computers or anime: _ _ _ _ u
The letter constraint is: 5 letters, letter 5 is u.
Some possible words that mean "A person with an all-consuming enthusiasm, such as for computers or anime": geek (g e e k): 4 letters, not 5; otaku (o t a k u): 5 letters, letter 5 is u.
sure

Dewy; roscid: r _ _ _ l
The letter constraint is: 5 letters, letter 1 is r, letter 5 is l.
Some possible words that mean "Dewy; roscid": moist (m o i s t): 5 letters, letter 1 is m, not r; humid (h u m i d): 5 letters, letter 1 is h, not r.
I cannot think of any words now. Only 2 letters are constrained, it is still likely.
maybe

A woodland: _ l _ d e
The letter constraint is: 5 letters, letter 2 is l, letter 4 is d, letter 5 is e.
Some possible words that mean "A woodland": forest (f o r e s t): 6 letters, not 5; woods (w o o d s): 5 letters, letter 2 is o, not l; grove (g r o v e): 5 letters, letter 2 is r, not l.
I cannot think of any words now. 3 letters are constrained, and _ l _ d e seems a common pattern.
maybe

An inn: _ d _ w f
The letter constraint is: 5 letters, letter 2 is d, letter 4 is w, letter 5 is f.
Some possible words that mean "An inn": hotel (h o t e l): 5 letters, letter 2 is o, not d; lodge (l o d g e): 5 letters, letter 2 is o, not d.
I cannot think of any words now. 3 letters are constrained, and it is extremely unlikely to have a word with pattern _ d _ w f to mean "An inn".
impossible

{input}
"""

finish_prompt = """Complete this 5x5 mini crossword from the current partial board.
Each row must contain exactly 5 letters. Respect all already-filled letters in the board.
Return only the final grid as 5 rows of 5 uppercase letters separated by spaces. Do not include explanations.

{input}

Final grid:
"""
