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
Wrap the answer in `\boxed{}`.

2. **Answer type**
The boxed content must be a time.

3. **Formatting rules**
* Follow every formatting instruction in the original question (units, rounding, decimal places, etc.).
* Do **not** add any units (e.g., "s", "second", "seconds"), unless required.
* Ensure the correct unit (e.g., hours versus thousand hours); if the question specifies "thousand hours" or "1000 hours", treat it as the required unit — output a number like 13 (thousand hours) instead of 13000 (hours).
* If the question's written instructions for precision or rounding differ from the examples, treat the examples as authoritative — match their number of decimal places and rounding style.

4. **Additional constraints**
* If the **Agent Summary** is incomplete or unclear, provide the best possible answer (educated guess).

5. **Common pitfalls to avoid**
* Minor mismatches in the required format.
* Unit-conversion errors, especially with uncommon units.
* Incorrect precision, rounding or scale (e.g., 0.01 vs 0.001), **double-check the required level**.
* Conflicts between textual instructions and example formatting, just follow the example: if the question says to "retain the percentile" but the example shows 0.001, use 0.001 rather than 0.01.

---

# Quick reference examples

* If the question says to "rounding the seconds to the nearest hundredth", but the example shows "0.001", 1:23.4567 → 1:23.457
* If the question says to "rounding the seconds to the nearest hundredth", but the example shows "0.001", 10:08.47445 → 10:08.474
* If the question says to "round to one decimal place", but the example shows "0.01", 2:17.456 → 2:17.46
* If the question says to "round to the nearest minute", but the example keeps seconds ("0:45"), 3:44.8 → 3:45
* If the question says "keep three decimal places", but the example shows "0.1", 1:03.987 → 1:03.1
* If the question asks for "thousand hours", 13000 -> 13

---
{{confidence_section}}