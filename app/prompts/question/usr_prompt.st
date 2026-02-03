Generate exam questions with the following specifications:

**Topic**: ${topic}
**Grade Level**: ${grade}
**Subject**: ${subject}

**Generate ${total_questions} questions distributed as follows**:
${difficulty_distribution}

**Question Types to use**: ${question_types}
${additional_requirements}

**Important Instructions**:
- Create EXACTLY ${total_questions} questions
- Follow the difficulty distribution specified above
- Use only the question types listed (${question_types})
- Ensure each question follows the schema in the system prompt
- For grade level ${grade}, use age-appropriate language and concepts
- For subject ${subject}, use appropriate language (Vietnamese or English)
- Return ONLY a valid JSON array, no code blocks or extra text

Example output structure:
[
  {
    "context": null,
    "question_number": null,
    "topic": "${topic}",
    "grade": "${grade}",
    "difficulty": "easy",
    "question": {
      "question_type": "multiple_choice",
      "content": "Your question text here?",
      "answers": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Option B",
      "explanation": "Explanation of why this is correct"
    },
    "default_points": 2
  }
]

Generate the questions now.
