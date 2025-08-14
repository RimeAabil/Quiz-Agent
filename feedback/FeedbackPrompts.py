# Prompt templates for Groq generation

FEEDBACK_PROMPT_TEMPLATE = """
Generate comprehensive, personalized feedback for a learner based on these quiz results:

{results_json}

Structure the feedback in markdown format with the following sections:
- 📊 **Quiz Results Summary**: Include score, total, and percentage.
- Performance Level Assessment: Provide an encouraging assessment based on the percentage.
- 🎯 **Performance by Difficulty Level**: Summarize performance for easy, medium, hard with emojis indicating strength.
- 📋 **Topic Performance**: If topic_performance is available, summarize with emojis.
- ❌ **Questions to Review**: For each incorrect answer, list question number, difficulty, user's answer, correct answer, and explanation if available.
- 🧠 **Learning Insights**: If learning_insights are provided, use them; otherwise, generate 3-5 actionable insights based on the results.
- 🎯 **Recommendations for Improvement**: Provide personalized recommendations based on difficulty analysis, topic performance, and response patterns.

Make the language engaging, encouraging, and personalized. Use emojis appropriately. End with a motivational note.
"""

INSIGHTS_PROMPT_TEMPLATE = """
Based on the following quiz results, generate 3-5 actionable, personalized learning insights:

{results_json}

Each insight should be a single bullet-point ready string, encouraging, and specific to:
- Overall performance
- Difficulty-specific patterns
- Topic strengths/weaknesses if available
- Response patterns like consistency and improvement trend

Output only the list of insights, one per line.
"""