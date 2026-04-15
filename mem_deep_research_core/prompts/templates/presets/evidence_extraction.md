# Evidence Extraction Protocol

After receiving tool results, you MUST extract key evidence before continuing your reasoning. This ensures important findings survive context compression.

## Format

After each tool call returns results, output an `<evidence>` block. Each fact should include a source URL (if available) and confidence level:

<evidence>
- [Fact with specific data points] (source: URL_HERE) (confidence: high)
- [Another fact, including numbers/names/dates] (source: URL_HERE) (confidence: medium)
- [Contradicts earlier finding about X] (confidence: low)
</evidence>

## Confidence Levels

- **high**: Directly stated in a primary/authoritative source
- **medium**: Inferred from multiple consistent sources, or from a single non-authoritative source
- **low**: Single unverified source, indirect inference, or conflicting with other evidence

## Rules

- Extract ONLY verified facts from the tool result — no speculation
- Include specific data: numbers, names, dates, URLs, IDs
- Keep each fact concise (one line) but information-dense
- Always include the source URL when the tool result provides one
- Note conflicts with previous findings explicitly
- Skip if the tool result contains no useful information (e.g., errors, empty results)
- Do NOT repeat evidence already extracted in previous turns
