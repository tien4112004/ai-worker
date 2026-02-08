Generate an exam matrix with the following specifications:

**Exam Details:**
- Name: ${name}
- Grade Level: ${grade}
- Subject: ${subject}
- Total Questions: ${total_questions}
- Total Points: ${total_points}

**Difficulty Levels:**
${difficulties}

**Question Types to Use:**
${question_types}

**Additional Requirements:**
${additional_requirements}

**Instructions:**
Using the curriculum chapters provided in the system prompt:
1. Organize chapters into logical topics (high-level categories)
2. Group related chapters as subtopics under each topic
3. Distribute questions across all subtopics fairly (subtopics are the matrix rows)
4. Balance difficulty levels appropriately for the grade level
5. Use a variety of question types suitable for each subtopic
6. Ensure total questions = ${total_questions}
7. Ensure total points = ${total_points}
8. Generate unique subtopic IDs (NOT topic IDs - topics are organizational only)
9. Consider the additional requirements when planning the distribution

Remember:
- Return ONLY a valid JSON object matching the specified structure
- No explanatory text before or after the JSON
- Generate unique UUIDs for subtopic IDs
- Topics serve as containers; subtopics are the actual dimension
- Matrix has [num_subtopics][num_difficulties][num_question_types] structure
