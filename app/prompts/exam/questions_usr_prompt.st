Generate questions for ${total_topics} topics:

=== CONTEXT-BASED TOPICS ===

${context_topics_section}

=== REGULAR TOPICS ===

${regular_topics_section}

**Summary:**
- Total Topics: ${total_topics}
- Context Topics: ${context_count}
- Regular Topics: ${regular_count}
- Total Questions: ${total_questions}
- Grade: ${grade}
- Subject: ${subject}

Generate EXACTLY ${total_questions} questions matching the requirements above.

**Important Instructions:**
1. For context-based topics: Base ALL questions on the provided context (text or image)
2. For regular topics: Generate from curriculum knowledge
3. Set `topicId` field to match the topic number (0-indexed)
4. Return ONLY valid JSON in the format: {"questions": [...]}
5. No markdown code blocks, no explanatory text
6. All text in Vietnamese unless subject is TA (English)
