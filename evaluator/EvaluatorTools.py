"""
Quiz Agent Tools - Helper functions for quiz generation, evaluation, and feedback
Includes detailed NLP functions for content analysis and processing
"""
import os
import json
import re
import numpy as np
from typing import Dict, List, Any, Tuple, Type
from collections import Counter, defaultdict
import groq
from textstat import flesch_reading_ease, flesch_kincaid_grade
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()  


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')


class GradeAnswersInput(BaseModel):
    """Input schema for GradeAnswersTool"""
    user_answers: str = Field(..., description="JSON string of user's answers")
    correct_answers: str = Field(..., description="JSON string of correct answers")
    questions_metadata: str = Field(..., description="JSON string of question metadata")


class GradeAnswersTool(BaseTool):
    name: str = "Grade Answers Tool"
    description: str = "Grades quiz answers and provides detailed performance analysis including score, topic performance, difficulty analysis, and learning insights."
    args_schema: Type[BaseModel] = GradeAnswersInput

    def _run(self, user_answers: str, correct_answers: str, questions_metadata: str) -> str:
        """
        Grade quiz answers with detailed performance analysis
        
        Args:
            user_answers: JSON string of user's answers
            correct_answers: JSON string of correct answers  
            questions_metadata: JSON string of question metadata
        
        Returns:
            JSON string with grading results
        """
        try:
            # Parse input arguments
            user_ans_list = json.loads(user_answers) if isinstance(user_answers, str) else user_answers
            correct_ans_list = json.loads(correct_answers) if isinstance(correct_answers, str) else correct_answers
            metadata_list = json.loads(questions_metadata) if isinstance(questions_metadata, str) else questions_metadata
            
            # Extract difficulties from metadata
            difficulties = [q.get('difficulty', 'medium') for q in metadata_list]
            
            # Grade the answers
            results = grade_answers(user_ans_list, correct_ans_list, difficulties, metadata_list)
            
            return json.dumps(results, indent=2)
            
        except Exception as e:
            error_result = {
                'error': f"Error grading answers: {str(e)}",
                'score': 0,
                'total': 0,
                'percentage': 0.0
            }
            return json.dumps(error_result, indent=2)


def grade_answers(user_answers: List[str], correct_answers: List[str], 
                 difficulties: List[str], questions: List[Dict] = None) -> Dict[str, Any]:
    """
    Grade quiz answers with detailed performance analysis
    """
    if len(user_answers) != len(correct_answers):
        raise ValueError("Mismatch between user answers and correct answers length")
    
    # Initialize results structure
    results = {
        'score': 0,
        'total': len(correct_answers),
        'percentage': 0.0,
        'correct_answers': [],
        'incorrect_answers': [],
        'difficulty_analysis': {
            'easy': {'correct': 0, 'total': 0, 'percentage': 0.0},
            'medium': {'correct': 0, 'total': 0, 'percentage': 0.0},
            'hard': {'correct': 0, 'total': 0, 'percentage': 0.0}
        },
        'topic_performance': {},
        'response_patterns': {
            'consistency': 0.0,
            'improvement_trend': 0.0,
            'difficulty_progression': []
        },
        'learning_insights': []
    }
    
    # Grade each answer
    for i, (user_ans, correct_ans, difficulty) in enumerate(zip(user_answers, correct_answers, difficulties)):
        is_correct = user_ans.upper() == correct_ans.upper()
        
        if is_correct:
            results['score'] += 1
            results['correct_answers'].append({
                'question_number': i + 1,
                'difficulty': difficulty,
                'topic': questions[i].get('topic', 'General') if questions else 'General'
            })
        else:
            results['incorrect_answers'].append({
                'question_number': i + 1,
                'user_answer': user_ans,
                'correct_answer': correct_ans,
                'difficulty': difficulty,
                'topic': questions[i].get('topic', 'General') if questions else 'General',
                'explanation': questions[i].get('explanation', '') if questions else ''
            })
        
        # Update difficulty analysis
        results['difficulty_analysis'][difficulty]['total'] += 1
        if is_correct:
            results['difficulty_analysis'][difficulty]['correct'] += 1
        
        # Update topic performance
        topic = questions[i].get('topic', 'General') if questions else 'General'
        if topic not in results['topic_performance']:
            results['topic_performance'][topic] = {'correct': 0, 'total': 0, 'percentage': 0.0}
        
        results['topic_performance'][topic]['total'] += 1
        if is_correct:
            results['topic_performance'][topic]['correct'] += 1
    
    # Calculate percentages
    results['percentage'] = (results['score'] / results['total']) * 100
    
    for difficulty in results['difficulty_analysis']:
        total = results['difficulty_analysis'][difficulty]['total']
        if total > 0:
            correct = results['difficulty_analysis'][difficulty]['correct']
            results['difficulty_analysis'][difficulty]['percentage'] = (correct / total) * 100
    
    for topic in results['topic_performance']:
        total = results['topic_performance'][topic]['total']
        if total > 0:
            correct = results['topic_performance'][topic]['correct']
            results['topic_performance'][topic]['percentage'] = (correct / total) * 100
    
    # Analyze response patterns
    results['response_patterns'] = analyze_response_patterns(user_answers, correct_answers, difficulties)
    
    # Generate learning insights
    results['learning_insights'] = generate_learning_insights(results)
    
    return results

def analyze_response_patterns(user_answers: List[str], correct_answers: List[str], 
                            difficulties: List[str]) -> Dict[str, float]:
    """Analyze patterns in quiz responses"""
    patterns = {}
    
    # Calculate consistency (how often user gets similar difficulty questions right)
    difficulty_performance = {'easy': [], 'medium': [], 'hard': []}
    for user_ans, correct_ans, difficulty in zip(user_answers, correct_answers, difficulties):
        is_correct = user_ans.upper() == correct_ans.upper()
        difficulty_performance[difficulty].append(is_correct)
    
    # Consistency score (variance in performance within difficulty levels)
    consistency_scores = []
    for difficulty, performance in difficulty_performance.items():
        if performance:
            consistency_scores.append(1.0 - np.var(performance))
    
    patterns['consistency'] = np.mean(consistency_scores) if consistency_scores else 0.0
    
    # Improvement trend (performance over time during quiz)
    correct_sequence = [user_ans.upper() == correct_ans.upper() 
                       for user_ans, correct_ans in zip(user_answers, correct_answers)]
    
    if len(correct_sequence) > 1:
        # Calculate trend using linear regression
        x = np.arange(len(correct_sequence))
        y = np.array(correct_sequence, dtype=float)
        slope = np.polyfit(x, y, 1)[0]
        patterns['improvement_trend'] = slope
    else:
        patterns['improvement_trend'] = 0.0
    
    # Difficulty progression analysis
    patterns['difficulty_progression'] = [
        difficulty_performance['easy'].count(True) / max(len(difficulty_performance['easy']), 1),
        difficulty_performance['medium'].count(True) / max(len(difficulty_performance['medium']), 1),
        difficulty_performance['hard'].count(True) / max(len(difficulty_performance['hard']), 1)
    ]
    
    return patterns

def generate_learning_insights(results: Dict[str, Any]) -> List[str]:
    """Generate learning insights based on quiz performance"""
    insights = []
    
    # Overall performance insights
    percentage = results['percentage']
    if percentage >= 90:
        insights.append("Excellent performance! You have a strong grasp of the material.")
    elif percentage >= 70:
        insights.append("Good performance overall with room for targeted improvement.")
    elif percentage >= 50:
        insights.append("Moderate performance. Focus on reviewing key concepts.")
    else:
        insights.append("Needs improvement. Consider reviewing the material thoroughly.")
    
    # Difficulty-based insights
    diff_analysis = results['difficulty_analysis']
    if diff_analysis['easy']['total'] > 0 and diff_analysis['easy']['percentage'] < 80:
        insights.append("Focus on mastering basic concepts before moving to advanced topics.")
    
    if diff_analysis['hard']['total'] > 0 and diff_analysis['hard']['percentage'] > 70:
        insights.append("Strong performance on challenging questions shows good conceptual understanding.")
    
    # Topic performance insights
    topic_perf = results['topic_performance']
    weak_topics = [topic for topic, perf in topic_perf.items() if perf['percentage'] < 60]
    strong_topics = [topic for topic, perf in topic_perf.items() if perf['percentage'] >= 80]
    
    if weak_topics:
        insights.append(f"Consider additional practice in: {', '.join(weak_topics)}")
    
    if strong_topics:
        insights.append(f"Strong performance in: {', '.join(strong_topics)}")
    
    # Response pattern insights
    patterns = results['response_patterns']
    if patterns['improvement_trend'] > 0.1:
        insights.append("Positive trend: Performance improved during the quiz.")
    elif patterns['improvement_trend'] < -0.1:
        insights.append("Performance declined during the quiz. Consider taking breaks between questions.")
    
    if patterns['consistency'] < 0.5:
        insights.append("Inconsistent performance suggests need for more focused study.")
    
    return insights