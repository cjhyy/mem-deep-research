# Time-Sensitivity Search Protocol (MANDATORY)

Based on the question's time-sensitivity requirements, you **MUST** use the appropriate `tbs` time filter parameter when searching:

## Time-Sensitivity Rules

| Question Type | Keywords | tbs Parameter | Description |
|--------------|----------|---------------|-------------|
| **Real-time** | "latest", "current", "now", "today" | `qdr:d` | Past 24 hours |
| **Recent** | "recent", "this week", "this month" | `qdr:w` | Past week |
| **Annual** | "2026", "this year", "newest version" | `qdr:m` | Past month |
| **Comparison** | "compare", "vs", "which is better", "flagship" | `qdr:m` | Past month |
| **Historical** | "history", "past", "used to" | None | No time limit |

## Mandatory Rules

1. **Identify time-sensitivity keywords**: Analyze the user's question for keywords above
2. **Select appropriate tbs**: Choose the corresponding time filter based on question type
3. **First search MUST use tbs**: For time-sensitive questions, the first search **MUST** include tbs parameter
4. **Verify publication dates**: Prioritize information published within the last 3 months
5. **Note information freshness**: When citing information, note the source's publication date
