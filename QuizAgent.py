"""
Quiz Agent - Main agent implementation for quiz generation, evaluation, and feedback
Implements the QuizAgent class with routing to specialized tasks
Now supports customizable number of questions (minimum 8)
"""
import os
import yaml
from typing import Dict, List, Any, Optional
from generator.GeneratorTools import (
    generate_questions_with_groq, 
    ContentAnalyzer, 
    QuestionDifficultyAnalyzer,
    validate_question_count,
    get_question_count_info,
    MIN_QUESTIONS,
    DEFAULT_QUESTIONS
)
from evaluator.EvaluatorTools import grade_answers
from feedback.FeedbackTools import create_feedback, generate_learning_insights
from langchain_groq import ChatGroq

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class QuizAgent:
    """
    Quiz Orchestrator Agent that manages quiz generation, evaluation, and feedback
    Implements a modular architecture with task-based routing
    Now supports customizable question numbers with minimum of 8 questions
    """
    
    def __init__(self, config_path: str = None, model: str = "llama3-8b-8192"):
        self.config_path = config_path or os.path.dirname(__file__)
        # Store raw model name (NO groq/ prefix for ChatGroq)
        self.model = model
        self.role_config = self._load_role_config()
        self.tasks_config = self._load_tasks_config()
        self.content_analyzer = ContentAnalyzer()
        self.difficulty_analyzer = QuestionDifficultyAnalyzer()

        # Initialize ChatGroq with raw model name
        self.llm = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model=self.model,  # Use raw model name like "llama3-8b-8192"
            temperature=0.7,
            max_tokens=4000  # Increased for larger quizzes
        )

        # Create task router
        self.task_router = {
            'generate_quiz': self._generate_quiz_task,
            'evaluate_quiz': self._evaluate_quiz_task,
            'give_feedback': self._give_feedback_task,
            'get_question_limits': self._get_question_limits_task
        }
    
    def _load_role_config(self) -> Dict[str, Any]:
        """Load agent role configuration from YAML"""
        try:
            role_path = os.path.join(self.config_path, 'role.yaml')
            with open(role_path, 'r') as file:
                config = yaml.safe_load(file) or {}
                # Ensure a valid model is specified
                valid_models = ['llama3-8b-8192', 'llama3-70b-8192', 'llama-3.1-8b-instant', 'llama-3.1-70b-versatile']
                if config.get('model') not in valid_models:
                    print(f"Warning: Invalid or decommissioned model in role.yaml. Using default: {self.model}")
                    config['model'] = self.model
                
                # Update default question count if not specified
                if 'default_num_questions' not in config:
                    config['default_num_questions'] = DEFAULT_QUESTIONS
                elif config['default_num_questions'] < MIN_QUESTIONS:
                    print(f"Warning: default_num_questions in config is below minimum. Setting to {MIN_QUESTIONS}")
                    config['default_num_questions'] = MIN_QUESTIONS
                    
                return config
        except FileNotFoundError:
            # Default configuration if file not found
            return {
                'agent_name': 'QuizAgent',
                'api_key': os.environ.get('GROQ_API_KEY', ''),
                'model': self.model,
                'default_num_questions': DEFAULT_QUESTIONS
            }
        except Exception as e:
            print(f"Error loading role config: {e}")
            return {
                'agent_name': 'QuizAgent',
                'api_key': os.environ.get('GROQ_API_KEY', ''),
                'model': self.model,
                'default_num_questions': DEFAULT_QUESTIONS
            }
    
    def _load_tasks_config(self) -> Dict[str, Any]:
        """Load task-specific configuration from YAML"""
        try:
            tasks_path = os.path.join(self.config_path, 'tasks.yaml')
            with open(tasks_path, 'r') as file:
                config = yaml.safe_load(file) or {'tasks': []}
                # Convert list of tasks to dictionary for easier lookup
                task_dict = {task['name']: task for task in config.get('tasks', [])}
                
                # Ensure generate_quiz task has proper num_questions setting
                if 'generate_quiz' in task_dict:
                    if 'num_questions' not in task_dict['generate_quiz']:
                        task_dict['generate_quiz']['num_questions'] = DEFAULT_QUESTIONS
                    elif task_dict['generate_quiz']['num_questions'] < MIN_QUESTIONS:
                        print(f"Warning: num_questions in tasks.yaml is below minimum. Setting to {MIN_QUESTIONS}")
                        task_dict['generate_quiz']['num_questions'] = MIN_QUESTIONS
                
                return task_dict
        except FileNotFoundError:
            # Default task configuration
            return {
                'generate_quiz': {
                    'num_questions': DEFAULT_QUESTIONS,
                    'difficulty_distribution': 'crescendo',
                    'allow_custom_count': True
                },
                'evaluate_quiz': {
                    'require_explanations': True
                },
                'give_feedback': {
                    'include_insights': True,
                    'include_recommendations': False  # No recommendations per tasks.yaml
                }
            }
        except Exception as e:
            print(f"Error loading tasks config: {e}")
            return {
                'generate_quiz': {
                    'num_questions': DEFAULT_QUESTIONS,
                    'difficulty_distribution': 'crescendo',
                    'allow_custom_count': True
                },
                'evaluate_quiz': {
                    'require_explanations': True
                },
                'give_feedback': {
                    'include_insights': True,
                    'include_recommendations': False
                }
            }
    
    def _get_question_limits_task(self) -> Dict[str, Any]:
        """
        Get information about question count limits and recommendations
        
        Returns:
            Dictionary with min, max, default, and recommended question counts
        """
        return get_question_count_info()
    
    def _generate_quiz_task(self, profile: Dict[str, Any], module_content: str, 
                           milestone: str, num_questions: Optional[int] = None, 
                           api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate quiz questions using the Quiz Generator Agent tools
        Now supports customizable number of questions with minimum validation
        
        Args:
            profile: Student profile (name, learning_style, initial_level)
            module_content: Educational content to base questions on
            milestone: Course milestone triggering the quiz
            num_questions: Number of questions to generate (minimum 8, overrides config)
            api_key: Groq API key (overrides config)
        
        Returns:
            Dictionary with 'questions' (list of question dicts with question, options, correct_answer, difficulty)
        """
        task_config = self.tasks_config.get('generate_quiz', {})
        
        # Determine number of questions to generate
        if num_questions is not None:
            # User specified a custom number - validate it
            num_questions = validate_question_count(num_questions)
            print(f"Using custom question count: {num_questions}")
        else:
            # Use config or default
            num_questions = task_config.get('num_questions', 
                                          self.role_config.get('default_num_questions', DEFAULT_QUESTIONS))
            num_questions = validate_question_count(num_questions)
            print(f"Using configured question count: {num_questions}")
        
        api_key = api_key or self.role_config.get('api_key')
        
        # Validate profile
        required_profile_fields = ['name', 'learning_style', 'initial_level']
        if not all(field in profile for field in required_profile_fields):
            raise ValueError(f"Profile must include: {', '.join(required_profile_fields)}")
        
        if not api_key:
            raise ValueError("GROQ_API_KEY is not provided or invalid")
        
        try:
            print(f"Generating {num_questions} questions for milestone: {milestone}")
            
            # Generate questions using tools.py with explicit LLM and model
            questions = generate_questions_with_groq(
                profile=profile,
                module_content=module_content,
                num_questions=num_questions,
                api_key=api_key,
                llm=self.llm,  # Pass ChatGroq instance
                model=self.model  # Pass raw model name
            )
            
            # Transform to match expected output
            formatted_questions = [
                {
                    'question': q['question'],
                    'options': q['options'],
                    'correct_answer': q['answer'],
                    'difficulty': q['difficulty'],
                    'topic': q.get('topic', 'General'),
                    'explanation': q.get('explanation', '')
                } for q in questions
            ]
            
            print(f"Successfully generated {len(formatted_questions)} questions")
            return {
                'questions': formatted_questions,
                'metadata': {
                    'total_questions': len(formatted_questions),
                    'difficulties': {
                        'easy': sum(1 for q in formatted_questions if q['difficulty'] == 'easy'),
                        'medium': sum(1 for q in formatted_questions if q['difficulty'] == 'medium'),
                        'hard': sum(1 for q in formatted_questions if q['difficulty'] == 'hard')
                    },
                    'milestone': milestone
                }
            }
        except Exception as e:
            print(f"Error in quiz generation for milestone {milestone}: {e}")
            return {
                'questions': [], 
                'error': str(e),
                'metadata': {
                    'total_questions': 0,
                    'difficulties': {'easy': 0, 'medium': 0, 'hard': 0},
                    'milestone': milestone
                }
            }
    
    def _evaluate_quiz_task(self, user_answers: List[str], correct_answers: List[str], 
                           questions_metadata: List[Dict[str, Any]], milestone: str) -> Dict[str, Any]:
        """
        Evaluate quiz answers using the Evaluator Agent tools
        Now handles variable number of questions
        
        Args:
            user_answers: List of student-provided answers
            correct_answers: List of correct answers
            questions_metadata: List of dicts with question, topic, difficulty
            milestone: Course milestone at which the quiz was taken
        
        Returns:
            Dictionary with score, total, correct, incorrect, topic_performance
        """
        task_config = self.tasks_config.get('evaluate_quiz', {})
        
        # Validate inputs
        if len(user_answers) != len(correct_answers) or len(user_answers) != len(questions_metadata):
            raise ValueError("Mismatch in lengths of user_answers, correct_answers, and questions_metadata")
        
        num_questions = len(user_answers)
        print(f"Evaluating quiz with {num_questions} questions for milestone: {milestone}")
        
        # Extract difficulties from questions_metadata
        difficulties = [q.get('difficulty', 'medium') for q in questions_metadata]
        
        try:
            # Use grade_answers from tools.py
            results = grade_answers(
                user_answers=user_answers,
                correct_answers=correct_answers,
                difficulties=difficulties,
                questions=questions_metadata
            )
            
            # Transform to match expected output
            output = {
                'score': results['score'],
                'total': results['total'],
                'percentage': round((results['score'] / results['total']) * 100, 1) if results['total'] > 0 else 0,
                'correct': [
                    {
                        'question': questions_metadata[i-1]['question'],
                        'topic': questions_metadata[i-1].get('topic', 'General')
                    } for i in [x['question_number'] for x in results['correct_answers']]
                ],
                'incorrect': [
                    {
                        'question': questions_metadata[i-1]['question'],
                        'topic': questions_metadata[i-1].get('topic', 'General'),
                        'user_answer': x['user_answer'],
                        'correct_answer': x['correct_answer']
                    } for i, x in [(m['question_number'], m) for m in results['incorrect_answers']]
                ],
                'topic_performance': {
                    topic: {
                        'percentage': perf['percentage'],
                        'correct': perf['correct'],
                        'total': perf['total']
                    } for topic, perf in results['topic_performance'].items()
                },
                'difficulty_performance': {
                    'easy': {'correct': 0, 'total': 0},
                    'medium': {'correct': 0, 'total': 0},
                    'hard': {'correct': 0, 'total': 0}
                }
            }
            
            # Calculate performance by difficulty
            for i, difficulty in enumerate(difficulties):
                output['difficulty_performance'][difficulty]['total'] += 1
                if i < len(results['correct_answers']) and any(ca['question_number'] == i+1 for ca in results['correct_answers']):
                    output['difficulty_performance'][difficulty]['correct'] += 1
            
            # Add diagnostic feedback for first attempts
            if 'first_attempt' in milestone.lower():
                output['diagnostic_feedback'] = results['learning_insights']
            
            print(f"Quiz evaluation completed - Score: {results['score']}/{results['total']} ({output['percentage']}%)")
            return output
        except Exception as e:
            print(f"Error in quiz evaluation for milestone {milestone}: {e}")
            return {
                'score': 0,
                'total': len(correct_answers),
                'percentage': 0,
                'correct': [],
                'incorrect': [],
                'topic_performance': {},
                'difficulty_performance': {'easy': {'correct': 0, 'total': 0}, 'medium': {'correct': 0, 'total': 0}, 'hard': {'correct': 0, 'total': 0}},
                'error': str(e)
            }
    
    def _give_feedback_task(self, evaluation_results: Dict[str, Any], profile: Dict[str, Any], 
                           milestone: str) -> Dict[str, Any]:
        """
        Generate feedback based on quiz results using the Feedback Agent tools
        Enhanced to handle variable question counts
        
        Args:
            evaluation_results: Results from evaluate_quiz_task
            profile: Student profile (includes name)
            milestone: Course milestone at which the quiz was taken
        
        Returns:
            Dictionary with feedback_summary, explanations, strengths, weaknesses
        """
        task_config = self.tasks_config.get('give_feedback', {})
        
        # Validate inputs
        if 'name' not in profile:
            raise ValueError("Profile must include 'name'")
        
        try:
            print(f"Generating feedback for {evaluation_results.get('total', 0)} question quiz")
            
            # Use create_feedback from tools.py
            feedback_text = create_feedback(evaluation_results)
            
            # Parse feedback text to extract required components
            feedback_lines = feedback_text.split('\n')
            feedback_summary = []
            explanations = {}
            strengths = []
            weaknesses = []
            
            current_section = None
            current_question = None
            
            for line in feedback_lines:
                if line.startswith('📊 **Quiz Results Summary**'):
                    current_section = 'summary'
                elif line.startswith('❌ **Questions to Review**'):
                    current_section = 'explanations'
                elif line.startswith('📋 **Topic Performance**'):
                    current_section = 'topic_performance'
                elif line.startswith('🧠 **Learning Insights**'):
                    current_section = 'insights'
                
                if current_section == 'summary' and line and not line.startswith('📊'):
                    feedback_summary.append(line)
                elif current_section == 'explanations' and line.startswith('Question'):
                    try:
                        q_num = int(line.split()[1].rstrip(':'))  # Extract question number
                        current_question = q_num
                        explanations[q_num] = []
                    except (ValueError, IndexError):
                        pass
                elif current_section == 'explanations' and line.startswith('  💡') and current_question:
                    explanations[current_question].append(line.lstrip('  💡 '))
                elif current_section == 'topic_performance' and '🟢' in line:
                    # Extract topic name from strong performance lines
                    topic = line.split(':')[0].replace('🟢 ', '').strip()
                    strengths.append(topic)
                elif current_section == 'topic_performance' and ('🟡' in line or '🔴' in line):
                    # Extract topic name from weak performance lines
                    topic = line.split(':')[0].replace('🟡 ', '').replace('🔴 ', '').strip()
                    weaknesses.append(topic)
            
            # Personalize summary with student's name and quiz size info
            quiz_size = evaluation_results.get('total', 0)
            score = evaluation_results.get('score', 0)
            percentage = evaluation_results.get('percentage', 0)
            
            personalized_intro = f"Hello, {profile['name']}! Here's your feedback for the {quiz_size}-question quiz at {milestone}:"
            performance_summary = f"You scored {score}/{quiz_size} ({percentage}%) on this assessment."
            
            feedback_summary.insert(0, personalized_intro)
            feedback_summary.insert(1, performance_summary)
            
            return {
                'feedback_summary': '\n'.join(feedback_summary),
                'explanations': explanations,
                'strengths': strengths,
                'weaknesses': weaknesses,
                'quiz_metadata': {
                    'total_questions': quiz_size,
                    'score': score,
                    'percentage': percentage,
                    'milestone': milestone
                }
            }
        except Exception as e:
            print(f"Error in feedback generation for milestone {milestone}: {e}")
            return {
                'feedback_summary': f"Hello, {profile['name']}! Unable to generate detailed feedback due to an error.",
                'explanations': {},
                'strengths': [],
                'weaknesses': [],
                'quiz_metadata': {'total_questions': 0, 'score': 0, 'percentage': 0, 'milestone': milestone},
                'error': str(e)
            }
    
    def run_task(self, task_name: str, **kwargs) -> Any:
        """
        Route and execute a specific task
        
        Args:
            task_name: Name of the task to execute (generate_quiz, evaluate_quiz, give_feedback, get_question_limits)
            **kwargs: Task-specific arguments
        
        Returns:
            Task-specific output (questions, results, feedback, or limits info)
        """
        task_func = self.task_router.get(task_name)
        if not task_func:
            available_tasks = list(self.task_router.keys())
            raise ValueError(f"Unknown task: {task_name}. Available tasks: {available_tasks}")
        
        try:
            return task_func(**kwargs)
        except Exception as e:
            print(f"Error executing task {task_name}: {e}")
            return None
    
    def get_question_count_limits(self) -> Dict[str, Any]:
        """
        Get information about supported question counts
        
        Returns:
            Dictionary with min, max, default question counts and recommendations
        """
        return self.run_task('get_question_limits')


# Test function to verify the QuizAgent works with customizable question counts
def test_quiz_agent():
    """Test the QuizAgent with a sample workflow including custom question counts"""
    try:
        # Initialize the agent
        agent = QuizAgent()
        
        # Test getting question limits
        print("Testing question limits...")
        limits = agent.get_question_count_limits()
        print(f"✓ Question limits: Min={limits['min_questions']}, Max={limits['max_questions']}, Default={limits['default_questions']}")
        
        # Sample data
        sample_profile = {
            'name': 'John Doe',
            'learning_style': 'visual',
            'initial_level': 'beginner',
            'strengths': ['problem-solving'],
            'weaknesses': ['syntax']
        }
        
        sample_content = """
        Python is a versatile programming language used for web development, data analysis, and more.
        Variables in Python are used to store data values. Python has different data types including
        integers, floats, strings, and booleans. Functions in Python are defined using the 'def' keyword.
        Lists and dictionaries are important data structures in Python. Control flow includes if statements,
        for loops, and while loops. Object-oriented programming in Python uses classes and objects.
        Exception handling uses try-except blocks to manage errors gracefully.
        """
        
        # Test quiz generation with custom question count
        test_question_counts = [8, 12, 15]  # Test minimum and some larger counts
        
        for num_questions in test_question_counts:
            print(f"\nTesting quiz generation with {num_questions} questions...")
            quiz_result = agent.run_task(
                'generate_quiz',
                profile=sample_profile,
                module_content=sample_content,
                milestone=f"Chapter 1 end - {num_questions} questions test",
                num_questions=num_questions
            )
            
            if quiz_result and quiz_result.get('questions'):
                actual_count = len(quiz_result['questions'])
                metadata = quiz_result.get('metadata', {})
                print(f"✓ Quiz generated successfully with {actual_count} questions")
                print(f"  Difficulty distribution: {metadata.get('difficulties', {})}")
                
                if actual_count == num_questions:
                    print(f"✓ Correct number of questions generated ({num_questions})")
                else:
                    print(f"⚠ Expected {num_questions}, got {actual_count}")
                
                # Test evaluation with sample answers
                questions = quiz_result['questions']
                # Generate sample answers (mix of correct and incorrect)
                user_answers = []
                correct_answers = [q['correct_answer'] for q in questions]
                
                # Make about 70% correct for testing
                for i, correct in enumerate(correct_answers):
                    if i % 10 < 7:  # 70% correct
                        user_answers.append(correct)
                    else:
                        # Pick a different option
                        options = ['A', 'B', 'C', 'D']
                        wrong_options = [opt for opt in options if opt != correct]
                        user_answers.append(wrong_options[i % len(wrong_options)])
                
                questions_metadata = [
                    {
                        'question': q['question'],
                        'difficulty': q['difficulty'],
                        'topic': q['topic'],
                        'explanation': q['explanation']
                    } for q in questions
                ]
                
                print(f"Testing quiz evaluation for {num_questions} questions...")
                eval_result = agent.run_task(
                    'evaluate_quiz',
                    user_answers=user_answers,
                    correct_answers=correct_answers,
                    questions_metadata=questions_metadata,
                    milestone=f"Chapter 1 end - {num_questions} questions test"
                )
                
                if eval_result:
                    score = eval_result.get('score', 0)
                    total = eval_result.get('total', 0)
                    percentage = eval_result.get('percentage', 0)
                    print(f"✓ Quiz evaluated successfully - Score: {score}/{total} ({percentage}%)")
                    
                    # Test feedback generation
                    print(f"Testing feedback generation for {num_questions} questions...")
                    feedback_result = agent.run_task(
                        'give_feedback',
                        evaluation_results=eval_result,
                        profile=sample_profile,
                        milestone=f"Chapter 1 end - {num_questions} questions test"
                    )
                    
                    if feedback_result:
                        quiz_metadata = feedback_result.get('quiz_metadata', {})
                        print(f"✓ Feedback generated successfully for {quiz_metadata.get('total_questions', 0)} questions")
                    else:
                        print("✗ Feedback generation failed")
                        return False
                else:
                    print("✗ Quiz evaluation failed")
                    return False
            else:
                print(f"✗ Quiz generation failed for {num_questions} questions")
                return False
        
        print("\n✓ All QuizAgent tasks working correctly with customizable question counts!")
        return True
        
    except Exception as e:
        print(f"✗ QuizAgent test failed: {e}")
        return False


if __name__ == "__main__":
    print("=== Testing QuizAgent with Customizable Question Counts ===")
    success = test_quiz_agent()
    if success:
        print("🎉 QuizAgent is working correctly with all features!")
    else:
        print("⚠ QuizAgent has issues that need to be resolved.")