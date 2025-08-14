import os
import json
import re
import time
import numpy as np
from typing import List, Dict, Any
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from textstat import flesch_reading_ease, flesch_kincaid_grade
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()  

from crewai.tools import BaseTool

# Import quiz prompts
from generator.quiz_prompts import (
    get_quiz_generation_prompt, 
    get_simplified_retry_prompt, 
    get_fallback_question_templates,
    get_content_analysis_prompt
)


# Constants
MIN_QUESTIONS = 8
DEFAULT_QUESTIONS = 10
MAX_QUESTIONS = 50

class ContentAnalyzer(BaseTool):
    """Advanced NLP content analysis for educational materials"""
    
    name: str = "Content Analyzer"
    description: str = "Analyzes educational text for key concepts, complexity, and learning objectives."
    
    def _run(self, text: str) -> Dict[str, Any]:
        """Run the content analysis tool"""
        return {
            "key_concepts": self.extract_key_concepts(text),
            "complexity": self.analyze_text_complexity(text),
            "objectives": self.extract_learning_objectives(text)
        }
    
    @property
    def stop_words(self):
        """Get stop words, initializing if needed"""
        if not hasattr(self, '_stop_words'):
            try:
                self._stop_words = set(stopwords.words('english'))
            except:
                self._stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        return self._stop_words
    
    @property
    def lemmatizer(self):
        """Get lemmatizer, initializing if needed"""
        if not hasattr(self, '_lemmatizer'):
            try:
                self._lemmatizer = WordNetLemmatizer()
            except:
                self._lemmatizer = None
        return self._lemmatizer
    
    def extract_key_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Extract key concepts from educational content using NLP techniques"""
        try:
            sentences = sent_tokenize(text)
        except:
            # Fallback sentence splitting
            sentences = text.split('. ')
            
        concepts = []
        
        for sentence in sentences:
            try:
                # Tokenize and POS tag
                tokens = word_tokenize(sentence.lower())
                pos_tags = pos_tag(tokens)
            except:
                # Fallback tokenization
                tokens = sentence.lower().split()
                pos_tags = [(token, 'NN') for token in tokens]  # Assume all are nouns as fallback
            
            # Extract nouns and noun phrases (likely concepts)
            concept_candidates = []
            current_phrase = []
            
            for word, tag in pos_tags:
                if tag.startswith(('NN', 'NNP')):  # Nouns and proper nouns
                    current_phrase.append(word)
                elif tag.startswith('JJ') and current_phrase:  # Adjectives modifying nouns
                    current_phrase.append(word)
                else:
                    if current_phrase:
                        phrase = ' '.join(current_phrase)
                        if len(phrase) > 2 and phrase not in self.stop_words:
                            concept_candidates.append({
                                'concept': phrase,
                                'context': sentence,
                                'importance': len(current_phrase)
                            })
                        current_phrase = []
            
            # Handle phrase at end of sentence
            if current_phrase:
                phrase = ' '.join(current_phrase)
                if len(phrase) > 2 and phrase not in self.stop_words:
                    concept_candidates.append({
                        'concept': phrase,
                        'context': sentence,
                        'importance': len(current_phrase)
                    })
            
            concepts.extend(concept_candidates)
        
        # Sort by importance and remove duplicates
        unique_concepts = {}
        for concept in concepts:
            key = concept['concept']
            if key not in unique_concepts or concept['importance'] > unique_concepts[key]['importance']:
                unique_concepts[key] = concept
        
        return sorted(unique_concepts.values(), key=lambda x: x['importance'], reverse=True)[:15]
    



    def analyze_text_complexity(self, text: str) -> Dict[str, float]:
        """Analyze text complexity using various readability metrics"""
        try:
            return {
                'flesch_reading_ease': flesch_reading_ease(text),
                'flesch_kincaid_grade': flesch_kincaid_grade(text),
                'avg_sentence_length': np.mean([len(word_tokenize(sent)) for sent in sent_tokenize(text)]),
                'vocabulary_diversity': len(set(word_tokenize(text.lower()))) / len(word_tokenize(text.lower()))
            }
        except:
            # Fallback complexity analysis
            words = text.split()
            sentences = text.split('.')
            return {
                'flesch_reading_ease': 50.0,  # Default moderate difficulty
                'flesch_kincaid_grade': 8.0,
                'avg_sentence_length': len(words) / max(len(sentences), 1),
                'vocabulary_diversity': len(set(words)) / max(len(words), 1)
            }
    
    def extract_learning_objectives(self, text: str) -> List[str]:
        """Extract potential learning objectives from content"""
        # Look for objective indicators
        objective_patterns = [
            r"(?:students? (?:will|should|can|must))\s+(.+?)(?:\.|;|$)",
            r"(?:learn(?:ing)? (?:to|how to))\s+(.+?)(?:\.|;|$)",
            r"(?:understand(?:ing)?)\s+(.+?)(?:\.|;|$)",
            r"(?:be able to)\s+(.+?)(?:\.|;|$)",
            r"(?:objective(?:s)?:?)\s+(.+?)(?:\.|;|$)"
        ]
        
        objectives = []
        for pattern in objective_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            objectives.extend([match.strip() for match in matches])
        
        return objectives[:5]  # Return top 5 objectives



class QuestionDifficultyAnalyzer(BaseTool):
    """Analyze and categorize question difficulty levels"""
    
    name: str = "Question Difficulty Analyzer"
    description: str = "Assesses the difficulty level (easy, medium, hard) of a quiz question."
    
    def _run(self, question: str) -> str:
        """Run the difficulty analysis tool"""
        return self.assess_difficulty(question)
    
    @property
    def difficulty_keywords(self):
        """Get difficulty keywords"""
        if not hasattr(self, '_difficulty_keywords'):
            self._difficulty_keywords = {
                'easy': ['what', 'who', 'when', 'where', 'is', 'are', 'define', 'list', 'name'],
                'medium': ['how', 'why', 'explain', 'describe', 'compare', 'contrast', 'analyze'],
                'hard': ['evaluate', 'synthesize', 'create', 'design', 'justify', 'critique', 'apply']
            }
        return self._difficulty_keywords
    
    def assess_difficulty(self, question: str) -> str:
        """Assess question difficulty based on cognitive complexity"""
        question_lower = question.lower()
        
        # Count difficulty indicators
        difficulty_scores = {
            'easy': sum(1 for word in self.difficulty_keywords['easy'] if word in question_lower),
            'medium': sum(1 for word in self.difficulty_keywords['medium'] if word in question_lower),
            'hard': sum(1 for word in self.difficulty_keywords['hard'] if word in question_lower)
        }
        
        # Determine difficulty based on highest score
        max_score = max(difficulty_scores.values())
        if max_score == 0:
            return 'medium'  # Default
        
        return max(difficulty_scores, key=difficulty_scores.get)


class GenerateQuestionsTool(BaseTool):
    """Generate quiz questions using Groq API"""

    name: str = "Generate Questions Tool"
    description: str = "Generates customizable number of quiz questions (minimum 8) based on educational content and learner profile using Groq API."

    def __init__(self, llm=None, model: str = "llama3-8b-8192", **kwargs):
        super().__init__(**kwargs)
        self._llm = llm
        # Store raw model name without groq/ prefix
        self._model = model.replace("groq/", "") if model.startswith("groq/") else model

    def _run(self, profile_str: str, module_content: str, num_questions: int = DEFAULT_QUESTIONS) -> str:
        try:
            # Validate number of questions
            num_questions = validate_question_count(num_questions)
            
            if isinstance(profile_str, str):
                profile = json.loads(profile_str)
            else:
                profile = profile_str

            # Pass the stored llm instance and raw model name
            questions = generate_questions_with_groq(
                profile=profile,
                module_content=module_content,
                num_questions=num_questions,
                api_key=os.environ.get("GROQ_API_KEY"),
                llm=self._llm,
                model=self._model  # Pass raw model name (no groq/ prefix)
            )
            return json.dumps(questions, indent=2)
        except Exception as e:
            print(f"Error in GenerateQuestionsTool: {str(e)}")
            return json.dumps({"error": f"Failed to generate questions: {str(e)}"})



def validate_crescendo_pattern(questions: List[Dict], easy_count: int, 
                              medium_count: int, hard_count: int) -> List[Dict]:
    """Validate and enforce crescendo difficulty pattern"""
    
    # Assign proper difficulty levels based on position for crescendo pattern
    for i, question in enumerate(questions):
        if i < easy_count:
            question['difficulty'] = 'easy'
        elif i < easy_count + medium_count:
            question['difficulty'] = 'medium'
        else:
            question['difficulty'] = 'hard'
        
        # Ensure all required fields are present
        if 'explanation' not in question:
            question['explanation'] = f"This question tests {question.get('topic', 'knowledge')} at {question['difficulty']} level."
        
        if 'topic' not in question:
            question['topic'] = 'General Knowledge'
    
    return questions



def get_question_count_info() -> Dict[str, Any]:
    """
    Get information about supported question counts
    
    Returns:
        Dictionary with min, max, default question counts and recommendations
    """
    return {
        "min_questions": MIN_QUESTIONS,
        "max_questions": MAX_QUESTIONS,
        "default_questions": DEFAULT_QUESTIONS,
        "recommended_questions": DEFAULT_QUESTIONS
    }

def clean_json_response(response_text: str) -> str:
    """Clean and extract JSON from API response with multiple strategies"""
    if not response_text or response_text.strip() == "":
        raise ValueError("Empty response from API")
    
    # Remove leading/trailing whitespace
    response_text = response_text.strip()
    
    # Remove markdown code blocks
    if response_text.startswith('```json'):
        response_text = response_text[7:]
    elif response_text.startswith('```'):
        response_text = response_text[3:]
    
    if response_text.endswith('```'):
        response_text = response_text[:-3]
    
    response_text = response_text.strip()
    
    # Strategy 1: Try to find JSON array pattern
    array_pattern = r'\[.*?\]'
    array_match = re.search(array_pattern, response_text, re.DOTALL)
    if array_match:
        potential_json = array_match.group(0)
        try:
            json.loads(potential_json)  # Test if it's valid JSON
            return potential_json
        except:
            pass
    
    # Strategy 2: Try to find multiple objects and wrap them in array
    object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    objects = re.findall(object_pattern, response_text, re.DOTALL)
    if objects:
        try:
            # Test if first object is valid
            json.loads(objects[0])
            return '[' + ','.join(objects) + ']'
        except:
            pass
    
    # Strategy 3: Return original if it looks like JSON
    if response_text.startswith('[') and response_text.endswith(']'):
        return response_text
    
    # Strategy 4: Return original if it starts with {
    if response_text.startswith('{'):
        return f'[{response_text}]'
    
    raise ValueError(f"Could not extract valid JSON from response: {response_text[:100]}...")


def calculate_difficulty_distribution(num_questions: int) -> tuple:
    """
    Calculate optimal distribution of easy, medium, and hard questions
    Maintains consistent proportions regardless of total number
    
    Args:
        num_questions: Total number of questions
        
    Returns:
        Tuple of (easy_count, medium_count, hard_count)
    """
    # Standard distribution: ~30% easy, ~40% medium, ~30% hard
    # But ensure at least 1 of each difficulty for smaller quizzes
    
    if num_questions <= MIN_QUESTIONS:
        # For minimum questions, ensure representation of all difficulties
        easy_count = max(2, num_questions // 4)
        hard_count = max(2, num_questions // 4) 
        medium_count = num_questions - easy_count - hard_count
    else:
        # For larger quizzes, use proportional distribution
        easy_count = max(2, int(num_questions * 0.3))
        hard_count = max(2, int(num_questions * 0.3))
        medium_count = num_questions - easy_count - hard_count
    
    # Ensure medium gets any remainder
    if easy_count + medium_count + hard_count < num_questions:
        medium_count += num_questions - (easy_count + medium_count + hard_count)
    
    print(f"Question distribution for {num_questions} questions: {easy_count} easy, {medium_count} medium, {hard_count} hard")
    return easy_count, medium_count, hard_count



def validate_question_count(num_questions: int) -> int:
    """
    Validate and adjust the number of questions requested
    
    Args:
        num_questions: Requested number of questions
        
    Returns:
        Validated number of questions (minimum MIN_QUESTIONS, maximum MAX_QUESTIONS)
    """
    if num_questions < MIN_QUESTIONS:
        print(f"Warning: Minimum {MIN_QUESTIONS} questions required. Adjusting from {num_questions} to {MIN_QUESTIONS}.")
        return MIN_QUESTIONS
    elif num_questions > MAX_QUESTIONS:
        print(f"Warning: Maximum {MAX_QUESTIONS} questions allowed. Adjusting from {num_questions} to {MAX_QUESTIONS}.")
        return MAX_QUESTIONS
    return num_questions



def generate_questions_with_groq(profile: Dict[str, Any], module_content: str, 
                                num_questions: int = DEFAULT_QUESTIONS, api_key: str = None, 
                                llm=None, model: str = "llama3-8b-8192", 
                                max_retries: int = 3) -> List[Dict[str, Any]]:
    """
    Generate customizable number of quiz questions using Groq API with improved error handling and retries
    
    Args:
        profile: Student profile information
        module_content: Educational content to base questions on
        num_questions: Number of questions to generate (minimum MIN_QUESTIONS)
        api_key: Groq API key
        llm: Pre-configured language model instance
        model: Model name to use
        max_retries: Maximum number of retry attempts
        
    Returns:
        List of question dictionaries
    """
    # Validate and adjust question count
    num_questions = validate_question_count(num_questions)
    
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("Warning: GROQ_API_KEY not found, using fallback questions")
            return generate_fallback_questions(module_content, num_questions)
    
    # Initialize Groq client if not provided
    if llm is None:
        # Remove groq/ prefix if present
        clean_model = model.replace("groq/", "") if model.startswith("groq/") else model
        llm = ChatGroq(
            api_key=api_key,
            model=clean_model,  
            temperature=0.3,  # Lower temperature for more consistent JSON
            max_tokens=4000,  # Increased for more questions
            timeout=90  # Increased timeout for larger requests
        )
    
    # Calculate difficulty distribution with new algorithm
    easy_count, medium_count, hard_count = calculate_difficulty_distribution(num_questions)
    
    # Generate main prompt using external prompt file
    prompt = get_quiz_generation_prompt(
        module_content=module_content,
        profile=profile,
        num_questions=num_questions,
        easy_count=easy_count,
        medium_count=medium_count,
        hard_count=hard_count
    )
    
    # Retry logic with progressive simplification
    for attempt in range(max_retries):
        try:
            print(f"Generating {num_questions} questions (attempt {attempt + 1}/{max_retries})...")
            
            # Make API call with timeout
            response = llm.invoke(prompt)
            
            # Validate response object
            if not response or not hasattr(response, 'content'):
                raise ValueError("Invalid response object from API")
            
            response_text = response.content
            
            # Check for empty response
            if not response_text or response_text.strip() == "":
                raise ValueError("Empty response content from API")
            
            print(f"Got response with {len(response_text)} characters")
            
            # Clean and parse JSON
            try:
                cleaned_response = clean_json_response(response_text)
                questions = json.loads(cleaned_response)
                
                # Validate structure
                if not isinstance(questions, list) or len(questions) == 0:
                    raise ValueError("Invalid question format received")
                
                # Validate and enhance questions
                validated_questions = validate_crescendo_pattern(questions, easy_count, medium_count, hard_count)
                
                # Ensure we have the right number of questions
                if len(validated_questions) < num_questions:
                    print(f"Warning: Only got {len(validated_questions)} questions, filling with fallback")
                    fallback_needed = num_questions - len(validated_questions)
                    fallback_questions = generate_fallback_questions(module_content, fallback_needed)
                    validated_questions.extend(fallback_questions)
                
                print(f"Successfully generated {len(validated_questions)} questions")
                return validated_questions[:num_questions]
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    print("Retrying with simplified prompt...")
                    # Use simplified prompt for retry
                    prompt = get_simplified_retry_prompt(module_content, num_questions)
                    time.sleep(1)
                else:
                    print("All JSON parsing attempts failed")
                    raise
                    
        except Exception as e:
            print(f"API error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print("Retrying after delay...")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print("All API attempts failed, using fallback")
                break
    
    # If all retries failed, use enhanced fallback
    print(f"Using fallback question generation for {num_questions} questions")
    return generate_fallback_questions(module_content, num_questions)



def generate_fallback_questions(content: str, num_questions: int) -> List[Dict[str, Any]]:
    """
    Generate enhanced fallback questions when API is unavailable
    Now supports customizable number of questions with proper distribution
    """
    # Validate question count
    num_questions = validate_question_count(num_questions)
    
    # Simple content analysis for fallback
    try:
        sentences = sent_tokenize(content)
        words = word_tokenize(content.lower())
        # Extract potential key terms (nouns)
        key_terms = [word for word, pos in pos_tag(words) if pos.startswith('NN')]
    except:
        # Fallback splitting
        sentences = content.split('. ')
        words = content.lower().split()
        key_terms = [word for word in words if len(word) > 3]  # Simple term extraction
    
    key_terms = list(set(key_terms))[:max(15, num_questions)]  # Scale with question count
    
    # Get question templates from external file
    templates = get_fallback_question_templates()
    
    fallback_questions = []
    
    # Calculate difficulty distribution
    easy_count, medium_count, hard_count = calculate_difficulty_distribution(num_questions)
    
    # Generate questions with crescendo pattern
    difficulties = ['easy'] * easy_count + ['medium'] * medium_count + ['hard'] * hard_count
    
    for i in range(num_questions):
        difficulty = difficulties[i] if i < len(difficulties) else 'medium'
        
        # Select term and template
        if key_terms:
            term = key_terms[i % len(key_terms)]
        else:
            term = f"concept_{i+1}"
        
        # Get templates for this difficulty level
        difficulty_templates = templates[difficulty]
        template = difficulty_templates[i % len(difficulty_templates)]
        
        question = {
            "question": template["template"].format(term=term),
            "options": template["options"],
            "answer": template["answer"],
            "difficulty": difficulty,
            "topic": "Content Analysis",
            "explanation": template["explanation"].format(term=term) if '{term}' in template["explanation"] else template["explanation"]
        }
        
        fallback_questions.append(question)
    
    print(f"Generated {len(fallback_questions)} fallback questions")
    return fallback_questions



# Legacy function wrapper for backward compatibility
def ContentAnalyzerTool():
    """Legacy wrapper - use ContentAnalyzer directly"""
    return ContentAnalyzer()

def QuestionDifficultyTool():
    """Legacy wrapper - use QuestionDifficultyAnalyzer directly"""
    return QuestionDifficultyAnalyzer()


__all__ = [
    'ContentAnalyzer', 'QuestionDifficultyAnalyzer', 'GenerateQuestionsTool',
    'ContentAnalyzerTool', 'QuestionDifficultyTool', 'generate_questions_with_groq',
    'generate_fallback_questions', 'validate_question_count', 'get_question_count_info'
]