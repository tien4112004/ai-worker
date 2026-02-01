Generate exam questions based on the provided context with the following specifications:

**Context Type**: ${context_type}

**Objectives**:
${objectives}

**Grade Level**: ${grade_level}
**Subject**: ${subject}

**Generate ${total_questions} questions distributed as follows**:
${difficulty_distribution}

**Question Types to use**: ${question_types}
${additional_requirements}

**Important Instructions**:
- Analyze the provided context carefully.
- Create EXACTLY ${total_questions} questions derived directly from the context.
- Ensure the questions align with the specified objectives.
- Follow the difficulty distribution specified above.
- Use only the question types listed (${question_types}).
- Ensure each question follows the schema in the system prompt.
- For grade level ${grade_level}, use age-appropriate language and concepts.
- Return ONLY a valid JSON array, no code blocks or extra text.

Generate the questions now.
