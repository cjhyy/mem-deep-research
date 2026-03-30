{{#if task_failed}}
**Important: You have either exhausted the context token limit or reached the maximum number of interaction turns without arriving at a conclusive answer. Therefore, you failed to complete the task. You Must explicitly state that you failed to complete the task in your response.**

{{/if}}
This is a direct instruction to you (the assistant), not the result of a tool call.

We are now ending this session. You must NOT initiate any further tool use. Summarize the above conversation and output the FINAL ANSWER to the original question.

If a clear answer has already been provided earlier in the conversation, do not rethink or recalculate it — simply extract that answer and reformat it.
If a definitive answer could not be determined, make a well-informed educated guess based on the conversation.

The original question is:
---
{{task_description}}
---

## Requirements
- **Language**: Write the entire response in **{{target_language}}**.
- **Focus**: Directly answer the original question. Do not just summarize gathered information — provide a clear, actionable answer.
- **Response Length**: Match the complexity of your response to the question.
- Use clear and structured Markdown formatting when appropriate.
- **Currency Format**: Use `\$` instead of `$` for currency amounts.
- **Citation Format**: Use `[ID]` format for in-text citations. Add a References section at the end if sources were used.
- Do NOT mention tools, tool calls, or internal reasoning steps.

{{#if chinese_context}}
## 中文输出要求
- 使用中文进行总结和回答
- 保持中文信息的原始格式和表达方式
- 确保最终答案符合中文表达习惯
{{/if}}
