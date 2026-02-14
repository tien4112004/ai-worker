You are an AI assistant helping to expand combined text items in a presentation slide.

${slide_context}

Current items:
${items_json}

Task: EXPAND each item with more detail

## CRITICAL RULES

1. **PRESERVE THE TOPIC**: Keep the SAME subject matter as the original items
2. **EXPAND BY 20-40%**: Add moderate detail, not excessive elaboration
3. **MAINTAIN STRUCTURE**: Return the same data structure (strings→strings, objects→objects)
4. **PRESERVE LANGUAGE**: Keep the same language as input (Vietnamese→Vietnamese, English→English)

## EXAMPLES

Input: ["Trái Đất đang nóng lên", "Động vật gặp nguy hiểm"]
CORRECT Output: ["Trái Đất đang nóng lên do khí nhà kính tăng cao", "Động vật gặp nguy hiểm khi môi trường sống bị phá hủy"]

Input: [{"label": "Q1", "content": "Revenue"}]
CORRECT Output: [{"label": "Q1", "content": "Revenue increased by 15% due to strong sales"}]

Return ONLY a JSON array. No explanations, no markdown.
