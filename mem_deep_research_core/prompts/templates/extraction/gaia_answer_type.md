Input:
`{{task_description}}`

Question:
Determine the expected data type of the answer. For questions asking to "identify" something, focus on the final answer type, not the identification process. Format requirements in the question often hint at the expected data type. If the question asks you to write a specific word, return string. Choose only one of the four types below:
- number — a pure number (may include decimals or signs), e.g., price, distance, length
- date   — a specific calendar date (e.g., 2025-08-05 or August 5, 2025)
- time   — a specific time of day or formated time cost (e.g., 14:30 or 1:30.12)
- string — any other textual answer

Output:
Return exactly one of the [number, date, time, string], nothing else.