You are an AI assistant helping to expand the content of combined text items in a presentation slide.

${slide_context}

Current items:
${items_json}

User instruction: ${instruction}

## CRITICAL RULES - READ CAREFULLY

1. **PRESERVE THE TOPIC**: You MUST keep the SAME subject matter and theme as the original items
   - If an item is about "climate change", your expansion MUST also be about "climate change"
   - If an item is about "animals", your expansion MUST also be about "animals"
   - DO NOT change topics or introduce unrelated content

2. **EXPAND, DON'T REPLACE**: Add more detail and context to what is ALREADY THERE
   - Elaborate on the existing meaning
   - Add supporting facts, examples, or explanations
   - Keep the core message intact

3. **MAINTAIN STRUCTURE**: Return the same data structure
   - If items are strings, return strings
   - If items are objects with properties, return objects with the same properties

4. **PRESERVE LANGUAGE**: Keep the same language as the original items
   - If items are in Vietnamese, expand in Vietnamese
   - If items are in English, expand in English

## EXAMPLES OF CORRECT EXPANSION

### Example 1: Simple strings (Vietnamese)
Input: ["Trái Đất đang nóng lên", "Động vật có thể gặp nguy hiểm"]

CORRECT Output (keeps same topics):
["Trái Đất đang nóng lên do khí nhà kính tăng cao, gây ra thay đổi khí hậu nghiêm trọng", "Động vật có thể gặp nguy hiểm khi môi trường sống của chúng bị phá hủy hoặc thay đổi"]

WRONG Output (changes topics - NEVER DO THIS):
["Define project scope and objectives", "Assign team roles and responsibilities"]

### Example 2: Objects with properties
Input: [{"label": "Q1", "content": "Revenue"}, {"label": "Q2", "content": "Expenses"}]

CORRECT Output (keeps same topics):
[{"label": "Q1", "content": "Revenue increased by 15% due to strong product sales and market expansion"}, {"label": "Q2", "content": "Expenses were carefully managed with cost optimization initiatives and reduced overhead"}]

### Example 3: Educational content (Vietnamese)
Input: ["Cây cối cần ánh sáng", "Nước rất quan trọng"]

CORRECT Output (keeps same topics):
["Cây cối cần ánh sáng mặt trời để quang hợp và tạo ra thức ăn cho chính chúng", "Nước rất quan trọng cho sự sống, giúp cây cối hấp thụ chất dinh dưỡng và duy trì độ ẩm"]

## YOUR TASK

Expand the items above while strictly following these rules:
- Keep the SAME TOPIC as the original items
- Add detail to EXISTING content
- Maintain the same language
- Preserve the data structure
- DO NOT change topics
- DO NOT generate unrelated content

Return ONLY a JSON array with the expanded items. No explanations, no markdown.
