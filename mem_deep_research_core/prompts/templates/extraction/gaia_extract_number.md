# Inputs

* **Original Question**: `{{task_description}}`
* **Agent Summary**: `{{summary}}`

---

# Task

1. **Independently derive** the best possible answer, step by step, based solely on evidence and reasoning from the Agent Summary. **Ignore the summary's "Final Answer" field** at this stage.
2. **Compare** your derived answer to the final answer provided in the Agent Summary (ignoring formatting and phrasing requirements at this stage).
– If both are well supported by the summary's evidence, choose the one with stronger or clearer support.
– If only one is well supported, use that one.
– For questions involving calculations, if your answer and the Agent Summary's final answer are numerically similar, prefer the summary's answer.
3. **Revise** your chosen answer to fully satisfy all formatting and phrasing requirements listed below (**Formatting rules**, **Additional constraints**, **Common pitfalls to avoid**, and **Quick reference examples**). These requirements override those in the original question if there is any conflict.

If no answer is clearly supported by the evidence, provide a well-justified educated guess. **Always wrap your final answer in a non-empty \boxed{...}.**

---

# Output Guidelines

1. **Box the answer**
Wrap the answer in `\boxed{}`.

2. **Answer type**
The boxed content must be a single number.

3. **Formatting rules**
* Follow every formatting instruction in the original question (units, rounding, decimal places, etc.).
* Use digits only; do **not** use words, commas or symbols (e.g., "$", "!", "?", "/").
* Do **not** add any units (e.g., "%", "$", "USD", "Å", "m", "m^2", "m^3"), unless required.
* Ensure the correct unit (e.g., grams versus kilograms, meters versus kilometers, hours versus thousand hours); if the question specifies "thousand hours" or "1000 hours", treat it as the required unit — output a number like 13 (thousand hours) instead of 13000 (hours).

4. **Additional constraints**
* If the **Agent Summary** is incomplete or unclear, provide the best possible answer (educated guess).

5. **Common pitfalls to avoid**
* Minor mismatches in the required format.
* Unit-conversion errors, especially with uncommon units.
* Incorrect precision, rounding or scale (e.g., 0.01 vs 0.001), **double-check the required level**.
* Conflicts between textual instructions and example formatting, just follow the example: if the question says to "retain the percentile" but the example shows 0.001, use 0.001 rather than 0.01.
* Do not partially convert text-based numbers—ensure full and accurate conversion (e.g., "one hundred million" → 100000000, not 100).

---

# Quick reference examples

* $100 → 100
* 100 USD → 100
* €50 → 50
* £75 → 75
* ¥1,000 → 1000
* 1,234 m → 1234
* 3,456,789 kg → 3456789
* 70% → 70
* 12.5% → 12.5
* 0.045 m³ → 0.045
* 0.045 m^3 → 0.045
* −40 °C → -40
* 100 km/h → 100
* 5000 m^2 → 5000
* 2.54 cm → 2.54
* 50 kg → 50
* 4.0 L → 4
* 13 thousand hours → 13
* Page 123/456 → 123/456
* 100 million → 100000000
* 200 Ω → 200
* 200 Å → 200
* 9.81 m/s² → 9.81
* 0 dB → 0

---
{{confidence_section}}