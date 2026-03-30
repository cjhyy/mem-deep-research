# Inputs

* **Original Question**: `{{task_description}}`
* **Agent Summary**: `{{summary}}`

---

# Task

1. **Independently derive** the best possible answer, step by step, based solely on evidence and reasoning from the Agent Summary. **Ignore the summary's "Final Answer" field** at this stage.
2. **Compare** your derived answer to the final answer provided in the Agent Summary (ignoring formatting and phrasing requirements at this stage).
– If both are well supported by the summary's evidence, choose the one with stronger or clearer support.
– If only one is well supported, use that one.
3. **Revise** your chosen answer to fully satisfy all formatting and phrasing requirements listed below (**Formatting rules**, **Additional constraints**, **Common pitfalls to avoid**, and **Quick reference examples**). These requirements override those in the original question if there is any conflict.

If no answer is clearly supported by the evidence, provide a well-justified educated guess. **Always wrap your final answer in a non-empty \boxed{...}.**

---

# Output Guidelines

1. **Box the answer**
Wrap the final answer in \boxed{...}.

2. **Answer type**
The boxed content must be **one** of:
* a single short phrase (fewest words possible)
* a comma-separated list of numbers and/or strings

3. **Formatting rules**
* Follow every formatting instruction in the original question (alphabetization, sequencing, units, rounding, decimal places, etc.).
* Omit articles and abbreviations unless explicitly present in the expected answer.
* If a string contains numeric information, spell out the numbers **unless** the question itself shows them as digits.
* Do **not** end the answer with ".", "!", "?", or any other punctuation.
* Use only standard ASCII quotation marks ("" and ''), **not** stylized or curly quotation marks (such as " " ' ').
* Remove invisible or non-printable characters.
* If the output is lists, apply the rules item-by-item.
* Avoid unnecessary elaboration - keep the answer as short as possible
    - Do **not** add "count", "number", "count of", "total", or similar quantifying words when the noun itself already refers to the quantity (e.g., use the bare noun form only).
    - No geographical modifiers (e.g., "Western", "Southern"),
    - Use the simplest, most commonly accepted term for a substance or object (e.g., "diamond" instead of "crystalline diamond", "silicon" instead of "silicon crystals")
* For mathematical symbols, match the symbol style in the question; never substitute LaTeX commands (e.g., use ≤, not \leq, use pure text, not \text{}, use ↔, not \leftrightarrow).
* For birthplaces, give the name as it was at the time of birth, not the current name.

4. **Additional constraints**
* If the Agent Summary is incomplete or unclear, provide the best possible answer (educated guess).
* Keep the answer as short and direct as possible—no explanations or parenthetical notes.

5. **Common pitfalls to avoid**
* Minor mismatches between required and produced formats.
* Conflicts between textual instructions and example formatting—follow the example.
* **Names**: give only the commonly used first + last name (no middle name unless requested).
* **Countries**: use the common name (e.g., "China", "Brunei")
* **Locations**: output only the requested location name, without including time, modifiers (e.g., "The Castle", "The Hotel")
* When the question provides examples of expected format (e.g., "ripe strawberries" not "strawberries"), follow the exact wording style shown in the examples, preserving all descriptive terms and adjectives as demonstrated.
* Answer with historically location names when the Agent Summary provides. Never override a historically location name. For example, a birthplace should be referred to by the name it had at the time of birth (i.e., answer the original name).
* For questions asking to "identify" something, focus on the final answer, not the identification process.

---

# Quick reference examples

* INT. THE CASTLE – DAY 1 → The Castle
* INT. THE HOTEL – NIGHT → The Hotel
* INT. THE SPACESHIP – DAWN → The Spaceship
* INT. THE LIBRARY – EVENING → The Library
* INT. CLASSROOM #3 – MORNING → Classroom #3
* People's Republic of China → China
* citation count → citations
* Brunei Darussalam → Brunei
* United States of America → United States
* Republic of Korea → South Korea
* New York City, USA → New York City
* São Paulo (Brazil) → São Paulo
* John Michael Doe → John Doe
* Mary Anne O'Neil → Mary O'Neil
* Dr. Richard Feynman → Richard Feynman
* INT. ZONE 42 – LEVEL B2 → Zone 42 – Level B2
* INT. THE UNDERWATER BASE – MIDNIGHT → The Underwater Base
* Sam's Home → Sam's Home
* Mike's phone → Mike's phone

---
{{confidence_section}}