# Confidence Assessment

Provide a confidence score (0-100) based on objective criteria for how likely this answer is to be judged correct by an automated verifier:

**Calibration Guidelines (use these as objective anchors):**
- **85-100**: Direct factual evidence found, no contradictions, formatting requirements clearly satisfied
- **70-84**: Strong supporting evidence with minor gaps or slight formatting uncertainty
- **55-69**: Moderate evidence but requires interpretation, or some conflicting information exists
- **40-54**: Limited evidence, significant uncertainty, multiple plausible answers possible
- **25-39**: Weak evidence, mostly reasoning-based, likely incomplete information
- **0-24**: No supporting evidence found, pure speculation, or contradicts known facts

**Objective Calibration Checks:**
1. **Evidence Verifiability**: Can the key facts be directly verified from the agent summary?
2. **Contradiction Test**: Does anything in the summary contradict this answer?
3. **Completeness Test**: Does the summary contain sufficient information to answer confidently?
4. **Formatting Clarity**: Are the format requirements unambiguous and correctly followed?

Rate conservatively - if unsure between two ranges, choose the lower one.

---

# Output Format

Return your analysis in this exact format:

**Step-by-step Analysis:**
[Your detailed reasoning process]

**Final Answer:** \boxed{...}

**Confidence:** [0-100 integer]

**Supporting Evidence:** [Brief summary of evidence that supports this answer]

**Potential Weaknesses:** [Any limitations, uncertainties, or factors that might make this answer incorrect - be objective and thorough]