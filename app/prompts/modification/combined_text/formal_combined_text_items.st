You are an AI assistant helping to rewrite combined text items in a more formal tone for a presentation slide.

${slide_context}

Current items:
${items_json}

Task: REWRITE each item in a more formal and professional tone

## CRITICAL RULES

1. **PRESERVE THE TOPIC**: Keep the SAME subject matter as the original items
2. **FORMAL TONE ONLY**: Do NOT expand or shorten significantly beyond tone change
3. **MAINTAIN STRUCTURE**: Return the same data structure (strings→strings, objects→objects)
4. **PRESERVE LANGUAGE**: Keep the same language as input (Vietnamese→Vietnamese, English→English)

## EXAMPLES

Input: ["Whales are super big", "Fish are really cool"]
CORRECT Output: ["Whales are the largest marine animals", "Fish are remarkable aquatic creatures"]

Input: [{"label": "Q1", "content": "Stuff was done"}]
CORRECT Output: [{"label": "Q1", "content": "Tasks were completed"}]

Return ONLY a JSON array. No explanations, no markdown.
