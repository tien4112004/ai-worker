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

**User prompt:**
${prompt}

**Instructions:**
Using the curriculum chapters provided in the system prompt:
1. **Evaluate question-to-chapter ratio**:
   - If totalQuestions (${total_questions}) < 10: Select ONLY 1-2 chapters maximum
   - If totalQuestions is 10-20: Select 2-4 chapters maximum  
   - If totalQuestions > 20: Can use more chapters as appropriate
2. **Select relevant chapters** - You do NOT need to use all chapters, only the most important ones
3. Organize selected chapters into logical topics (high-level categories that make sense together)
4. Group related chapters as subtopics under each topic (for organization)
5. Do NOT force unrelated chapters into topics just to include everything
6. Distribute questions strategically - not every cell needs questions
7. Use "0:0" for empty cells (no questions for that combination)
8. Balance difficulty levels appropriately for the grade level
9. Use a variety of question types suitable for each topic
10. **CRITICAL**: Ensure the sum of ALL question counts across ALL cells EXACTLY equals ${total_questions}
11. **CRITICAL**: Ensure the sum of ALL points across ALL cells EXACTLY equals ${total_points}
12. Consider the user prompt to give the answer that match user requirements.

**Before submitting, verify:**
- Number of selected chapters is appropriate for ${total_questions} questions
- Topics and subtopics are logically coherent (not forced together)
- Count all "count" values in all cells (including 0s): MUST = ${total_questions}
- Count all "points" values in all cells (including 0s): MUST = ${total_points}
- If totals don't match, redistribute until they do (add/remove questions or use "0:0")

Remember:
- Return ONLY a valid JSON object matching the specified structure
- No explanatory text before or after the JSON
- Topics are the first dimension; subtopics are organizational subdivisions
- Matrix has [num_topics][num_difficulties][num_question_types] structure
- Questions from any subtopic within a topic count toward that topic's cell
