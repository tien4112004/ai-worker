Generate an exam matrix with the following specifications:

**Exam Details:**
- Name: ${name}
- Grade Level: ${grade}
- Total Questions: ${total_questions}
- Total Points: ${total_points}

**Topics to Cover:**
${topics}

**Difficulty Levels:**
${difficulties}

**Question Types to Use:**
${question_types}

**Additional Requirements:**
${additional_requirements}

Generate a comprehensive 3D exam matrix that:
1. Distributes questions across all topics fairly
2. Balances difficulty levels appropriately for the grade level
3. Uses a variety of question types suitable for each topic
4. Ensures total questions = ${total_questions}
5. Ensures total points = ${total_points}

Remember:
- Return ONLY a valid JSON object matching the specified structure
- No explanatory text before or after the JSON
- Generate unique UUIDs for topic IDs
