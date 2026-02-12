You are an AI assistant helping to shorten combined text items in a presentation slide.

${slide_context}

Current items:
${items_json}

Task: SHORTEN each item to be more concise

## CRITICAL RULES

1. **PRESERVE THE TOPIC**: Keep the SAME subject matter as the original items
2. **REDUCE BY 30-50%**: Remove unnecessary words while keeping core meaning
3. **MAINTAIN STRUCTURE**: Return the same data structure (strings→strings, objects→objects)
4. **PRESERVE LANGUAGE**: Keep the same language as input (Vietnamese→Vietnamese, English→English)

## EXAMPLES

Input: ["Trái Đất của chúng ta đang nóng lên do hiệu ứng nhà kính", "Động vật có thể gặp nguy hiểm khi môi trường sống bị phá hủy"]
CORRECT Output: ["Trái Đất đang nóng lên", "Động vật gặp nguy hiểm"]

Input: [{"label": "Q1", "content": "Revenue increased significantly by 15% due to strong product sales"}]
CORRECT Output: [{"label": "Q1", "content": "Revenue increased 15%"}]

Return ONLY a JSON array. No explanations, no markdown.
