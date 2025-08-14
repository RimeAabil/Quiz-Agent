'''
import os
import json
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from typing import Dict, List, Any
from generator.GeneratorTools import generate_questions_with_groq, ContentAnalyzer, QuestionDifficultyAnalyzer
from evaluator.EvaluatorTools import grade_answers
from feedback.FeedbackTools import create_feedback_tool, generate_learning_insights_tool
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from dotenv import load_dotenv

load_dotenv()

# Initialize rich console for colorful output
console = Console()

# Global variables for dynamic setup
groq_llm = None
quiz_generator_agent = None
quiz_evaluator_agent = None
quiz_feedback_agent = None

def get_available_groq_models():
    """Try different Groq model names to find available ones"""
    models_to_try = [
        "llama3-8b-8192",
        "llama3-70b-8192", 
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile",
        "gemma-7b-it",
        "gemma2-9b-it"
    ]
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        console.print("[red]❌ GROQ_API_KEY environment variable not set[/red]")
        return None
    
    for model in models_to_try:
        try:
            test_llm = ChatGroq(
                api_key=api_key,
                model=model,
                temperature=0.7
            )
            response = test_llm.invoke("Hello")
            console.print(f"[green]✓ Model {model} is available[/green]")
            return model
        except Exception as e:
            console.print(f"[yellow]✗ Model {model} failed: {str(e)}[/yellow]")
            continue
    
    console.print("[red]❌ No Groq models are available. Please check your API key.[/red]")
    return None

def initialize_groq_agents():
    """Initialize agents with a working Groq model"""
    global groq_llm, quiz_generator_agent, quiz_evaluator_agent, quiz_feedback_agent
    
    working_model = get_available_groq_models()
    if not working_model:
        raise Exception("No working Groq model found")
    
    groq_llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model=working_model,
        temperature=0.7,
        max_tokens=2048
    )
    
    quiz_generator_agent = Agent(
        role='Quiz Generator',
        goal='Generate personalized quizzes based on student profiles and course content',
        backstory='You are an expert in educational content analysis and question generation, skilled at creating adaptive quizzes tailored to individual learning styles and skill levels.',
        verbose=True,
        allow_delegation=False,
        llm=groq_llm
    )
    
    quiz_evaluator_agent = Agent(
        role='Quiz Evaluator',
        goal='Evaluate quiz answers and provide detailed performance analysis',
        backstory='You are an expert in assessing student performance, providing detailed insights into strengths, weaknesses, and learning patterns.',
        verbose=True,
        allow_delegation=False,
        llm=groq_llm
    )
    
    quiz_feedback_agent = Agent(
        role='Quiz Feedback Provider',
        goal='Provide personalized and constructive feedback based on quiz performance',
        backstory='You are skilled at delivering actionable, encouraging feedback to help students improve their learning outcomes.',
        verbose=True,
        allow_delegation=False,
        llm=groq_llm
    )

def create_generate_quiz_task(profile: Dict[str, Any], module_content: str, milestone: str, num_questions: int):
    """Create a quiz generation task that calls functions directly"""
    return Task(
        description=f"""
        Generate a personalized quiz for student {profile.get('name', 'Student')} based on their profile and the provided module content.
        
        Student Profile:
        - Name: {profile.get('name', 'Student')}
        - Learning Style: {profile.get('learning_style', 'visual')}
        - Initial Level: {profile.get('initial_level', 'beginner')}
        - Strengths: {profile.get('strengths', [])}
        - Weaknesses: {profile.get('weaknesses', [])}
        
        Module Content: {module_content[:1000]}...
        Milestone: {milestone}
        
        The quiz should:
        1. Adapt to the student's learning style and skill level
        2. Follow a crescendo difficulty pattern (easy → medium → hard)
        3. Include {num_questions} questions total
        4. Cover key concepts from the module content
        
        Generate the questions and return them in the following JSON format:
        {{
            "questions": [
                {{
                    "question": "Question text here?",
                    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
                    "answer": "A",
                    "difficulty": "easy",
                    "topic": "relevant topic",
                    "explanation": "Brief explanation of the correct answer"
                }}
            ]
        }}
        """,
        expected_output="""
        A well-structured quiz in JSON format containing:
        - questions: list of question objects with question, options, answer, difficulty, topic, explanation
        """,
        agent=quiz_generator_agent
    )

def create_evaluate_quiz_task(user_answers: List[str], correct_answers: List[str], 
                            questions_metadata: List[Dict], milestone: str):
    """Create a quiz evaluation task"""
    return Task(
        description=f"""
        Grade the student's quiz answers and provide detailed performance analysis.
        
        User Answers: {user_answers}
        Correct Answers: {correct_answers}
        Questions Metadata: {questions_metadata}
        Milestone: {milestone}
        
        Calculate:
        1. Overall score and percentage
        2. Performance breakdown by topic
        3. Performance analysis by difficulty level
        4. Response patterns and learning insights
        
        Return the results in JSON format with score, total, percentage, correct_answers, incorrect_answers, 
        topic_performance, difficulty_analysis, and learning_insights.
        """,
        expected_output="""
        JSON object containing:
        - score: numeric (total correct answers)
        - total: numeric (total questions)
        - percentage: numeric (score percentage)
        - correct_answers: list of correctly answered questions
        - incorrect_answers: list of incorrectly answered questions
        - topic_performance: dictionary with performance per topic
        - difficulty_analysis: dictionary with performance per difficulty level
        - learning_insights: list of diagnostic insights
        """,
        agent=quiz_evaluator_agent
    )

def create_feedback_task(profile: Dict[str, Any], evaluation_results: Dict[str, Any], milestone: str):
    """Create a feedback task"""
    return Task(
        description=f"""
        Provide detailed, personalized feedback for {profile.get('name', 'Student')} based on their quiz performance.
        
        Student Profile: {profile}
        Evaluation Results: {evaluation_results}
        Milestone: {milestone}
        
        Create feedback that:
        1. Addresses the student by name
        2. Provides explanations for incorrect answers
        3. Highlights strengths and areas for improvement
        4. Offers encouragement and constructive guidance
        
        Return feedback in JSON format with feedback_summary, explanations, strengths, and weaknesses.
        """,
        expected_output="""
        Dictionary containing:
        - feedback_summary: personalized performance summary
        - explanations: explanations for incorrect answers
        - strengths: areas of strong performance
        - weaknesses: areas needing improvement
        """,
        agent=quiz_feedback_agent
    )

def generate_fallback_questions(content: str, num_questions: int) -> List[Dict[str, Any]]:
    """Generate basic fallback questions when API is unavailable"""
    try:
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.tag import pos_tag
        sentences = sent_tokenize(content)
        words = word_tokenize(content.lower())
        key_terms = [word for word, pos in pos_tag(words) if pos.startswith('NN')]
    except:
        sentences = content.split('. ')
        words = content.lower().split()
        key_terms = [word for word in words if len(word) > 3]
    
    key_terms = list(set(key_terms))[:10]
    
    fallback_questions = []
    for i in range(num_questions):
        if i < len(key_terms):
            term = key_terms[i]
        else:
            term = f"concept_{i+1}"
            
        if i < num_questions // 3:
            difficulty = 'easy'
        elif i < 2 * num_questions // 3:
            difficulty = 'medium'
        else:
            difficulty = 'hard'
        
        question = {
            "question": f"What is the significance of '{term}' in the given context?",
            "options": [
                "A) It is a fundamental concept",
                "B) It is a minor detail", 
                "C) It is irrelevant to the topic",
                "D) It is mentioned briefly"
            ],
            "answer": "A",
            "difficulty": difficulty,
            "topic": "Content Analysis",
            "explanation": f"This question focuses on understanding the role of '{term}' in the educational material."
        }
        fallback_questions.append(question)
    
    return fallback_questions

def run_quiz_workflow_direct(
    profile: Dict[str, Any],
    module_content: str,
    milestone: str,
    num_questions: int,
    user_answers: List[str] = None,
    correct_answers: List[str] = None,
    questions_metadata: List[Dict] = None
) -> Dict[str, Any]:
    try:
        if not os.environ.get("GROQ_API_KEY"):
            return {
                "error": "GROQ_API_KEY environment variable not set",
                "quiz": {"questions": generate_fallback_questions(module_content, num_questions)},
                "evaluation": {},
                "feedback": {}
            }
        
        initialize_groq_agents()
        
        console.print("[cyan]Generating quiz using direct function...[/cyan]")
        try:
            working_model = get_available_groq_models()
            if not working_model:
                raise Exception("No working model found")
                
            questions = generate_questions_with_groq(
                profile=profile,
                module_content=module_content,
                num_questions=num_questions,
                api_key=os.environ.get("GROQ_API_KEY"),
                llm=groq_llm,
                model=working_model
            )
            
            quiz_data = {"questions": questions}
            
        except Exception as gen_error:
            console.print(f"[red]Direct generation failed: {gen_error}[/red]")
            quiz_data = {"questions": generate_fallback_questions(module_content, num_questions)}
        
        result = {"quiz": quiz_data}
        
        if user_answers and correct_answers and questions_metadata:
            console.print("[cyan]Evaluating quiz answers...[/cyan]")
            try:
                # Validate lengths
                if len(user_answers) != len(correct_answers) or len(user_answers) != len(questions_metadata):
                    raise ValueError(f"Mismatch in lengths: user_answers ({len(user_answers)}), correct_answers ({len(correct_answers)}), questions ({len(questions_metadata)})")
                
                difficulties = [q.get('difficulty', 'medium') for q in questions_metadata]
                evaluation_data = grade_answers(
                    user_answers=user_answers,
                    correct_answers=correct_answers,
                    difficulties=difficulties,
                    questions=questions_metadata
                )
                
                # Generate learning insights using Groq API
                console.print("[cyan]Generating learning insights...[/cyan]")
                insights_result = json.loads(generate_learning_insights_tool._run(json.dumps(evaluation_data)))
                if insights_result.get("success"):
                    evaluation_data["learning_insights"] = insights_result["learning_insights"]
                else:
                    console.print(f"[red]Failed to generate insights: {insights_result.get('error', 'Unknown error')}[/red]")
                    evaluation_data["learning_insights"] = []
                
                result["evaluation"] = evaluation_data
                
                console.print("[cyan]Generating feedback...[/cyan]")
                feedback_result = json.loads(create_feedback_tool._run(json.dumps(evaluation_data)))
                if feedback_result.get("success"):
                    feedback_text = feedback_result["feedback_summary"]
                    feedback_data = {
                        "feedback_summary": f"Hello, {profile['name']}! " + feedback_text,
                        "explanations": {ans["question_number"]: ans["explanation"] for ans in evaluation_data.get("incorrect_answers", [])},
                        "strengths": [],
                        "weaknesses": []
                    }
                    
                    if evaluation_data.get('topic_performance'):
                        for topic, perf in evaluation_data['topic_performance'].items():
                            if perf['percentage'] >= 80:
                                feedback_data['strengths'].append(topic)
                            elif perf['percentage'] < 60:
                                feedback_data['weaknesses'].append(topic)
                    
                    result["feedback"] = feedback_data
                else:
                    console.print(f"[red]Failed to generate feedback: {feedback_result.get('error', 'Unknown error')}[/red]")
                    result["feedback"] = {"error": feedback_result.get("error", "Feedback generation failed")}
                
            except Exception as eval_error:
                console.print(f"[red]Evaluation/Feedback failed: {eval_error}[/red]")
                result["evaluation"] = {"error": str(eval_error)}
                result["feedback"] = {"error": str(eval_error)}
        
        return result
        
    except Exception as e:
        console.print(f"[red]Error in quiz workflow: {str(e)}[/red]")
        return {
            "error": str(e),
            "quiz": {"questions": generate_fallback_questions(module_content, num_questions)},
            "evaluation": {},
            "feedback": {}
        }

def run_quiz_workflow(
    profile: Dict[str, Any],
    module_content: str,
    milestone: str,
    num_questions: int,
    user_answers: List[str] = None,
    correct_answers: List[str] = None,
    questions_metadata: List[Dict] = None
) -> Dict[str, Any]:
    try:
        return run_quiz_workflow_direct(profile, module_content, milestone, num_questions, user_answers, correct_answers, questions_metadata)
    except Exception as e:
        console.print(f"[red]Direct workflow failed: {e}, falling back to basic generation[/red]")
        return {
            "error": str(e),
            "quiz": {"questions": generate_fallback_questions(module_content, num_questions)},
            "evaluation": {},
            "feedback": {}
        }

def test_nltk_offline():
    try:
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk.corpus import stopwords
        
        test_text = "This is a test sentence. This is another sentence."
        try:
            sentences = sent_tokenize(test_text)
            console.print(f"[green]✓ NLTK sentence tokenization works: {len(sentences)} sentences[/green]")
        except:
            sentences = test_text.split('. ')
            console.print(f"[green]✓ Fallback sentence tokenization works: {len(sentences)} sentences[/green]")
        
        try:
            words = word_tokenize(test_text)
            console.print(f"[green]✓ NLTK word tokenization works: {len(words)} words[/green]")
        except:
            words = test_text.split()
            console.print(f"[green]✓ Fallback word tokenization works: {len(words)} words[/green]")
        
        try:
            stop_words = set(stopwords.words('english'))
            console.print(f"[green]✓ NLTK stopwords available: {len(stop_words)} words[/green]")
        except:
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            console.print(f"[green]✓ Fallback stopwords available: {len(stop_words)} words[/green]")
        
        return True
    except Exception as e:
        console.print(f"[red]NLTK test failed: {e}[/red]")
        return False

def print_quiz_results(result: Dict[str, Any]):
    """Print quiz results in a colorful and organized way using rich"""
    console.print(Panel("[bold cyan]Quiz Generation Results[/bold cyan]", border_style="bright_cyan"))
    
    # Quiz Questions Table
    quiz_table = Table(title="Generated Questions", show_header=True, header_style="bold cyan")
    quiz_table.add_column("No.", style="cyan", width=5)
    quiz_table.add_column("Question", style="white")
    quiz_table.add_column("Options", style="bright_cyan")
    quiz_table.add_column("Answer", style="green")
    quiz_table.add_column("Difficulty", style="cyan")
    quiz_table.add_column("Topic", style="bright_cyan")
    quiz_table.add_column("Explanation", style="white")
    
    for i, q in enumerate(result['quiz']['questions'], 1):
        options = "\n".join(q['options'])
        quiz_table.add_row(
            str(i),
            q['question'],
            options,
            q['answer'],
            q['difficulty'].title(),
            q['topic'],
            q['explanation']
        )
    
    console.print(quiz_table)

def print_evaluation_results(evaluation: Dict[str, Any]):
    """Print evaluation results in a colorful and organized way"""
    console.print(Panel("[bold cyan]Evaluation Results[/bold cyan]", border_style="bright_cyan"))
    
    if "error" in evaluation:
        console.print(f"[red]Evaluation failed: {evaluation['error']}[/red]")
        return
    
    # Score Summary
    console.print(f"[bold green]Score: {evaluation['score']}/{evaluation['total']} ({evaluation['percentage']:.1f}%)[/bold green]")
    
    # Difficulty Analysis Table
    diff_table = Table(title="Performance by Difficulty", show_header=True, header_style="bold cyan")
    diff_table.add_column("Difficulty", style="cyan")
    diff_table.add_column("Correct", style="green")
    diff_table.add_column("Total", style="bright_cyan")
    diff_table.add_column("Percentage", style="white")
    
    for diff, stats in evaluation['difficulty_analysis'].items():
        diff_table.add_row(
            diff.title(),
            str(stats['correct']),
            str(stats['total']),
            f"{stats['percentage']:.1f}%"
        )
    
    console.print(diff_table)
    
    # Topic Performance Table
    topic_table = Table(title="Performance by Topic", show_header=True, header_style="bold cyan")
    topic_table.add_column("Topic", style="cyan")
    topic_table.add_column("Correct", style="green")
    topic_table.add_column("Total", style="bright_cyan")
    topic_table.add_column("Percentage", style="white")
    
    for topic, stats in evaluation['topic_performance'].items():
        topic_table.add_row(
            topic,
            str(stats['correct']),
            str(stats['total']),
            f"{stats['percentage']:.1f}%"
        )
    
    console.print(topic_table)
    
    # Incorrect Answers
    if evaluation['incorrect_answers']:
        console.print("[bold red]Incorrect Answers:[/bold red]")
        for ans in evaluation['incorrect_answers']:
            console.print(f"[cyan]Question {ans['question_number']} ({ans['difficulty']}):[/cyan]")
            console.print(f"  Your Answer: [red]{ans['user_answer']}[/red]")
            console.print(f"  Correct Answer: [green]{ans['correct_answer']}[/green]")
            console.print(f"  Explanation: [white]{ans['explanation']}[/white]\n")

def print_feedback_results(feedback: Dict[str, Any]):
    """Print feedback results in a colorful and organized way"""
    console.print(Panel("[bold cyan]Feedback for Student[/bold cyan]", border_style="bright_cyan"))
    
    if "error" in feedback:
        console.print(f"[red]Feedback failed: {feedback['error']}[/red]")
        return
    
    # Feedback Summary
    console.print(Text(feedback['feedback_summary'], style="white"))
    
    # Strengths and Weaknesses
    if feedback['strengths']:
        console.print("[bold green]Strengths:[/bold green]")
        for strength in feedback['strengths']:
            console.print(f"  • {strength}", style="bright_cyan")
    
    if feedback['weaknesses']:
        console.print("[bold red]Areas for Improvement:[/bold red]")
        for weakness in feedback['weaknesses']:
            console.print(f"  • {weakness}", style="cyan")

def interactive_quiz_workflow():
    """Run an interactive quiz workflow in the terminal"""
    console.print(Panel("[bold cyan]=== Welcome to the Interactive Quiz System ===[/bold cyan]", border_style="bright_cyan"))
    
    # Collect student profile
    name = Prompt.ask("[cyan]Enter your name[/cyan]", default="John Doe")
    learning_style = Prompt.ask(
        "[cyan]Enter your learning style (visual/auditory/kinesthetic)[/cyan]",
        default="visual",
        choices=["visual", "auditory", "kinesthetic"]
    )
    initial_level = Prompt.ask(
        "[cyan]Enter your skill level (beginner/intermediate/advanced)[/cyan]",
        default="beginner",
        choices=["beginner", "intermediate", "advanced"]
    )
    
    profile = {
        "name": name,
        "learning_style": learning_style,
        "initial_level": initial_level,
        "strengths": ["problem-solving"],
        "weaknesses": ["syntax"]
    }
    
    # Select topic
    available_topics = [
        "Python Programming",
        "Cybersecurity Fundamentals",
        "Statistics & Data Analysis",
        "Deep Learning",
        "SQL Database Management",
        "Data Science"
    ]
    console.print("[cyan]Available topics:[/cyan]")
    for i, topic in enumerate(available_topics, 1):
        console.print(f"[bright_cyan]{i}. {topic}[/bright_cyan]")
    
    topic_choice = IntPrompt.ask(
        "[cyan]Select a topic by number[/cyan]",
        choices=[str(i) for i in range(1, len(available_topics) + 1)],
        default=1
    )
    topic = available_topics[topic_choice - 1]
    
    # Sample content for each topic
    topic_content = {
        "Python Programming": """
            Python is a versatile programming language used for web development, data analysis, and more.
            Variables in Python are used to store data values. Python has different data types including
            integers, floats, strings, and booleans. Functions in Python are defined using the 'def' keyword.
        """,
        "Cybersecurity Fundamentals": """
            Cybersecurity involves protecting systems, networks, and data from unauthorized access.
            Key concepts include encryption, authentication, and firewalls. Common threats are malware,
            phishing, and denial-of-service attacks.
        """,
        "Statistics & Data Analysis": """
            Statistics involves collecting, analyzing, and interpreting data. Key concepts include mean,
            median, mode, standard deviation, and hypothesis testing. Data analysis often uses tools like Python or R.
        """,
        "Deep Learning": """
            Deep Learning is a subset of machine learning using neural networks with many layers.
            It is used for tasks like image recognition and natural language processing. Key concepts include
            neural networks, backpropagation, and activation functions.
        """,
        "SQL Database Management": """
            SQL is a language for managing relational databases. Key concepts include tables, queries,
            joins, and indexes. Common commands are SELECT, INSERT, UPDATE, and DELETE.
        """,
        "Data Science": """
            Data Science combines statistics, programming, and domain knowledge to extract insights from data.
            Key techniques include data cleaning, visualization, and machine learning. Tools include Python, R, and SQL.
        """
    }
    
    module_content = topic_content.get(topic, topic_content["Python Programming"])
    milestone = f"{topic} Assessment"
    
    # Specify number of questions
    num_questions = IntPrompt.ask(
        "[cyan]How many questions would you like in the quiz? (1-10)[/cyan]",
        default=6,
        choices=[str(i) for i in range(1, 11)]
    )
    
    # Confirm quiz settings
    console.print(Panel(
        f"[cyan]Quiz Settings:[/cyan]\n"
        f"[bright_cyan]Student: {name}\nTopic: {topic}\nNumber of Questions: {num_questions}\n"
        f"Learning Style: {learning_style}\nSkill Level: {initial_level}[/bright_cyan]",
        border_style="green"
    ))
    if not Confirm.ask("[cyan]Would you like to start the quiz?[/cyan]", default=True):
        console.print("[red]Quiz cancelled.[/red]")
        return
    
    # Generate quiz
    console.print(f"\n[bold cyan]=== Generating {num_questions} questions for {topic} ===[/bold cyan]")
    result = run_quiz_workflow(
        profile=profile,
        module_content=module_content,
        milestone=milestone,
        num_questions=num_questions
    )
    
    if "error" in result:
        console.print(f"[red]Failed to generate quiz: {result['error']}[/red]")
        return
    
    console.print("\n[bold cyan]Quiz Questions:[/bold cyan]")
    print_quiz_results(result)
    
    # Collect answers interactively
    user_answers = []
    questions = result['quiz']['questions']
    for i, q in enumerate(questions, 1):
        console.print(Panel(f"[bold cyan]Question {i}: {q['question']}[/bold cyan]", border_style="bright_cyan"))
        console.print("[bright_cyan]Options:[/bright_cyan]")
        for opt in q['options']:
            console.print(opt, style="bright_cyan")
        answer = Prompt.ask(
            "[cyan]Enter your answer (A/B/C/D)[/cyan]",
            choices=["A", "B", "C", "D"],
            default="A"
        ).upper()
        user_answers.append(answer)
    
    # Prepare evaluation data
    correct_answers = [q['answer'] for q in questions]
    questions_metadata = [
        {
            'question': q['question'],
            'difficulty': q['difficulty'],
            'topic': q['topic'],
            'explanation': q['explanation']
        } for q in questions
    ]
    
    # Evaluate and generate feedback
    console.print("[cyan]Processing your answers...[/cyan]")
    result_with_eval = run_quiz_workflow(
        profile=profile,
        module_content=module_content,
        milestone=milestone,
        num_questions=num_questions,
        user_answers=user_answers,
        correct_answers=correct_answers,
        questions_metadata=questions_metadata
    )
    
    console.print("\n[bold cyan]Full Quiz Workflow Result (with Evaluation and Feedback):[/bold cyan]")
    print_quiz_results(result_with_eval)
    print_evaluation_results(result_with_eval['evaluation'])
    print_feedback_results(result_with_eval['feedback'])

if __name__ == "__main__":
    console.print(Panel("[bold cyan]=== Testing NLTK offline functionality ===[/bold cyan]", border_style="bright_cyan"))
    nltk_works = test_nltk_offline()
    
    console.print(Panel("[bold cyan]=== Testing Groq connection ===[/bold cyan]", border_style="bright_cyan"))
    working_model = get_available_groq_models()
    if not working_model:
        console.print("[red]❌ No working Groq model found. Please check your GROQ_API_KEY and internet connection.[/red]")
        exit(1)
    
    console.print(Panel(f"[bold cyan]=== Initializing agents with model: {working_model} ===[/bold cyan]", border_style="green"))
    try:
        initialize_groq_agents()
        console.print("[green]✓ Agents initialized successfully![/green]")
    except Exception as e:
        console.print(f"[red]❌ Failed to initialize agents: {e}[/red]")
        exit(1)
    
    interactive_quiz_workflow()
'''

import os
import json
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from typing import Dict, List, Any
from generator.GeneratorTools import generate_questions_with_groq, ContentAnalyzer, QuestionDifficultyAnalyzer
from evaluator.EvaluatorTools import grade_answers
from feedback.FeedbackTools import create_feedback_tool, generate_learning_insights_tool
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from dotenv import load_dotenv

load_dotenv()

# Initialize rich console for colorful output
console = Console()

# Global variables for dynamic setup
groq_llm = None
quiz_generator_agent = None
quiz_evaluator_agent = None
quiz_feedback_agent = None

# Enhanced topic definitions with detailed content and skill levels
TOPICS_CATALOG = {
    "Python Programming": {
        "description": "Learn Python programming from basics to advanced concepts",
        "content": {
            "beginner": """
            Python is a high-level, interpreted programming language known for its simplicity and readability.
            Variables are containers for storing data values. In Python, you don't need to declare variables explicitly.
            Python has several built-in data types: integers (int), floating-point numbers (float), strings (str), and booleans (bool).
            Lists are ordered collections that can store multiple items: [1, 2, 3, 'hello'].
            Dictionaries store key-value pairs: {'name': 'John', 'age': 30}.
            The print() function displays output to the screen.
            Basic operators include +, -, *, / for arithmetic and ==, !=, <, > for comparison.
            """,
            "intermediate": """
            Functions in Python are defined using the 'def' keyword and allow code reuse and organization.
            Control flow statements include if-elif-else for conditionals, for and while loops for iteration.
            List comprehensions provide a concise way to create lists: [x**2 for x in range(10)].
            Exception handling uses try-except blocks to manage errors gracefully.
            Modules and packages help organize code into reusable components using import statements.
            Object-oriented programming uses classes to create custom data types with attributes and methods.
            File handling operations allow reading from and writing to files using open(), read(), write().
            Lambda functions are anonymous functions useful for short, simple operations.
            """,
            "advanced": """
            Advanced Python concepts include decorators that modify function behavior without changing code.
            Generators yield values on-demand, providing memory-efficient iteration using yield keyword.
            Context managers handle resource management with 'with' statements for proper cleanup.
            Metaclasses control class creation and behavior, allowing dynamic class modification.
            Multithreading and multiprocessing enable concurrent execution for performance optimization.
            Regular expressions (regex) provide powerful pattern matching and text processing capabilities.
            Design patterns like Singleton, Factory, and Observer solve common programming problems.
            Python's data model includes special methods (__init__, __str__, __len__) for custom behavior.
            """
        },
        "level_descriptions": {
            "beginner": "Basic syntax, variables, data types, simple operations",
            "intermediate": "Functions, control flow, OOP basics, file handling",
            "advanced": "Decorators, generators, design patterns, advanced concepts"
        }
    },
    "Data Science": {
        "description": "Master data science concepts from statistics to machine learning",
        "content": {
            "beginner": """
            Data Science combines statistics, programming, and domain knowledge to extract insights from data.
            Data comes in various forms: structured (tables), semi-structured (JSON), and unstructured (text, images).
            Basic statistics include measures of central tendency (mean, median, mode) and variability (standard deviation).
            Data visualization helps understand patterns using charts: bar charts, histograms, scatter plots, line graphs.
            CSV (Comma Separated Values) files are common for storing tabular data.
            Pandas library in Python provides DataFrames for data manipulation and analysis.
            Data cleaning involves handling missing values, removing duplicates, and correcting errors.
            Exploratory Data Analysis (EDA) is the initial investigation to understand data characteristics.
            """,
            "intermediate": """
            Statistical hypothesis testing helps validate assumptions and make data-driven decisions.
            Correlation measures relationships between variables, while regression predicts outcomes.
            Data preprocessing includes normalization, standardization, and feature engineering.
            Machine learning algorithms fall into supervised (labeled data) and unsupervised (pattern finding) categories.
            Classification predicts categories (spam/not spam), while regression predicts continuous values.
            Cross-validation techniques assess model performance and prevent overfitting.
            Feature selection identifies the most relevant variables for prediction.
            Data visualization libraries like Matplotlib and Seaborn create publication-quality plots.
            """,
            "advanced": """
            Advanced machine learning includes ensemble methods, deep learning, and neural networks.
            Model evaluation metrics: precision, recall, F1-score, ROC curves, and confusion matrices.
            Dimensionality reduction techniques like PCA and t-SNE handle high-dimensional data.
            Time series analysis forecasts future values based on historical patterns.
            Natural Language Processing (NLP) extracts insights from text data using techniques like tokenization and sentiment analysis.
            Big data technologies like Spark and Hadoop handle massive datasets.
            MLOps practices deploy and monitor machine learning models in production.
            Ethical considerations include bias detection, fairness, and interpretable AI.
            """
        },
        "level_descriptions": {
            "beginner": "Basic statistics, data visualization, pandas basics",
            "intermediate": "Machine learning fundamentals, preprocessing, model evaluation",
            "advanced": "Deep learning, NLP, MLOps, advanced algorithms"
        }
    },
    "Web Development": {
        "description": "Build modern web applications from frontend to backend",
        "content": {
            "beginner": """
            Web development involves creating websites and web applications accessible through browsers.
            HTML (HyperText Markup Language) structures web content using elements and tags.
            CSS (Cascading Style Sheets) controls the visual presentation: colors, fonts, layouts, spacing.
            JavaScript adds interactivity to web pages: user interactions, dynamic content updates.
            The DOM (Document Object Model) represents the page structure that JavaScript can manipulate.
            Basic HTML elements include headings (h1-h6), paragraphs (p), links (a), images (img).
            CSS selectors target HTML elements for styling: classes (.class), IDs (#id), and elements (div).
            JavaScript variables, functions, and event handling enable responsive user interfaces.
            """,
            "intermediate": """
            Frontend frameworks like React, Vue, or Angular organize code into reusable components.
            Backend development handles server-side logic, databases, and API creation.
            HTTP protocol defines communication between browsers and servers using GET, POST, PUT, DELETE methods.
            Databases store application data: SQL databases (MySQL, PostgreSQL) and NoSQL (MongoDB).
            RESTful APIs provide standardized ways for applications to communicate.
            Version control with Git tracks code changes and enables collaboration.
            Responsive design ensures websites work on different screen sizes using CSS media queries.
            Web security includes HTTPS, input validation, and protection against common attacks.
            """,
            "advanced": """
            Full-stack development combines frontend and backend skills for complete applications.
            Modern JavaScript features include async/await, promises, modules, and ES6+ syntax.
            State management in complex applications using Redux, Vuex, or context providers.
            Server-side rendering (SSR) and static site generation (SSG) optimize performance and SEO.
            Microservices architecture breaks applications into smaller, independent services.
            Cloud deployment using platforms like AWS, Google Cloud, or Azure.
            Performance optimization: code splitting, lazy loading, caching strategies.
            DevOps practices: CI/CD pipelines, containerization with Docker, monitoring and logging.
            """
        },
        "level_descriptions": {
            "beginner": "HTML, CSS, basic JavaScript, DOM manipulation",
            "intermediate": "Frameworks, backend basics, APIs, databases",
            "advanced": "Full-stack development, cloud deployment, performance optimization"
        }
    },
    "Cybersecurity": {
        "description": "Understand cybersecurity principles and practices",
        "content": {
            "beginner": """
            Cybersecurity protects digital systems, networks, and data from unauthorized access and attacks.
            Common threats include malware (viruses, worms, trojans), phishing emails, and social engineering.
            Authentication verifies user identity using passwords, biometrics, or multi-factor authentication.
            Encryption scrambles data to protect it from unauthorized access during transmission and storage.
            Firewalls act as barriers between trusted internal networks and untrusted external networks.
            Regular software updates patch security vulnerabilities discovered by developers.
            Strong passwords use a combination of letters, numbers, and special characters.
            Antivirus software detects and removes malicious programs from computer systems.
            """,
            "intermediate": """
            Network security involves monitoring and controlling access to network resources.
            Intrusion Detection Systems (IDS) identify suspicious activities and potential attacks.
            Virtual Private Networks (VPNs) create secure connections over public networks.
            Access control mechanisms ensure users have appropriate permissions for resources.
            Risk assessment evaluates potential threats and their impact on organizations.
            Security policies define rules and procedures for protecting information assets.
            Incident response plans outline steps to take when security breaches occur.
            Penetration testing simulates attacks to identify system vulnerabilities.
            """,
            "advanced": """
            Advanced persistent threats (APTs) are sophisticated, long-term cyber attacks.
            Digital forensics investigates cyber crimes and recovers evidence from digital devices.
            Security architecture designs comprehensive protection systems for organizations.
            Threat intelligence gathers and analyzes information about current and emerging threats.
            Cryptographic protocols secure communications using advanced mathematical techniques.
            Zero-trust security models verify every transaction and user, regardless of location.
            Security automation uses AI and machine learning to detect and respond to threats.
            Compliance frameworks like ISO 27001, NIST ensure adherence to security standards.
            """
        },
        "level_descriptions": {
            "beginner": "Basic security concepts, threats, authentication",
            "intermediate": "Network security, risk assessment, incident response",
            "advanced": "Advanced threats, forensics, security architecture"
        }
    },
    "Machine Learning": {
        "description": "Explore machine learning algorithms and applications",
        "content": {
            "beginner": """
            Machine Learning enables computers to learn patterns from data without explicit programming.
            Supervised learning uses labeled examples to predict outcomes for new, unseen data.
            Unsupervised learning finds hidden patterns in data without predetermined labels.
            Training data teaches the algorithm, while test data evaluates its performance.
            Features are individual measurable properties of observed phenomena.
            Common algorithms include linear regression, decision trees, and k-means clustering.
            Overfitting occurs when models perform well on training data but poorly on new data.
            Data preprocessing prepares raw data for machine learning algorithms.
            """,
            "intermediate": """
            Classification algorithms predict categories: logistic regression, random forests, support vector machines.
            Regression algorithms predict continuous values: linear regression, polynomial regression.
            Cross-validation techniques assess model performance and select optimal hyperparameters.
            Feature engineering creates new variables from existing data to improve model performance.
            Ensemble methods combine multiple models for better predictions: bagging, boosting.
            Model evaluation metrics include accuracy, precision, recall, and F1-score.
            Bias-variance tradeoff balances model complexity with generalization ability.
            Dimensionality reduction techniques like PCA reduce the number of features.
            """,
            "advanced": """
            Deep learning uses neural networks with multiple layers to model complex patterns.
            Convolutional Neural Networks (CNNs) excel at image recognition and computer vision tasks.
            Recurrent Neural Networks (RNNs) handle sequential data like text and time series.
            Transfer learning adapts pre-trained models to new, related tasks with limited data.
            Reinforcement learning trains agents to make decisions through interaction with environments.
            Generative models create new data similar to training examples: GANs, VAEs.
            AutoML automates machine learning pipeline design and hyperparameter optimization.
            Explainable AI techniques make model decisions interpretable and transparent.
            """
        },
        "level_descriptions": {
            "beginner": "ML basics, supervised/unsupervised learning, simple algorithms",
            "intermediate": "Advanced algorithms, feature engineering, model evaluation",
            "advanced": "Deep learning, neural networks, advanced architectures"
        }
    }
}

def get_available_groq_models():
    """Try different Groq model names to find available ones"""
    models_to_try = [
        "llama3-8b-8192",
        "llama3-70b-8192", 
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile",
        "gemma-7b-it",
        "gemma2-9b-it"
    ]
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        console.print("[red]⚠ GROQ_API_KEY environment variable not set[/red]")
        return None
    
    for model in models_to_try:
        try:
            test_llm = ChatGroq(
                api_key=api_key,
                model=model,
                temperature=0.7
            )
            response = test_llm.invoke("Hello")
            console.print(f"[green]✓ Model {model} is available[/green]")
            return model
        except Exception as e:
            console.print(f"[yellow]✗ Model {model} failed: {str(e)}[/yellow]")
            continue
    
    console.print("[red]⚠ No Groq models are available. Please check your API key.[/red]")
    return None

def initialize_groq_agents():
    """Initialize agents with a working Groq model"""
    global groq_llm, quiz_generator_agent, quiz_evaluator_agent, quiz_feedback_agent
    
    working_model = get_available_groq_models()
    if not working_model:
        raise Exception("No working Groq model found")
    
    groq_llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model=working_model,
        temperature=0.7,
        max_tokens=2048
    )
    
    quiz_generator_agent = Agent(
        role='Quiz Generator',
        goal='Generate personalized quizzes based on student profiles and course content',
        backstory='You are an expert in educational content analysis and question generation, skilled at creating adaptive quizzes tailored to individual learning styles and skill levels.',
        verbose=True,
        allow_delegation=False,
        llm=groq_llm
    )
    
    quiz_evaluator_agent = Agent(
        role='Quiz Evaluator',
        goal='Evaluate quiz answers and provide detailed performance analysis',
        backstory='You are an expert in assessing student performance, providing detailed insights into strengths, weaknesses, and learning patterns.',
        verbose=True,
        allow_delegation=False,
        llm=groq_llm
    )
    
    quiz_feedback_agent = Agent(
        role='Quiz Feedback Provider',
        goal='Provide personalized and constructive feedback based on quiz performance',
        backstory='You are skilled at delivering actionable, encouraging feedback to help students improve their learning outcomes.',
        verbose=True,
        allow_delegation=False,
        llm=groq_llm
    )

def display_topics_catalog():
    """Display available topics with descriptions and level information"""
    console.print(Panel("[bold cyan]=== Available Topics ===[/bold cyan]", border_style="bright_cyan"))
    
    # Create topics table
    topics_table = Table(title="Choose Your Learning Topic", show_header=True, header_style="bold cyan")
    topics_table.add_column("No.", style="cyan", width=4)
    topics_table.add_column("Topic", style="bright_cyan", width=20)
    topics_table.add_column("Description", style="white", width=40)
    topics_table.add_column("Available Levels", style="green", width=30)
    
    for i, (topic_name, topic_info) in enumerate(TOPICS_CATALOG.items(), 1):
        levels = ", ".join([f"{level.title()}" for level in topic_info["level_descriptions"].keys()])
        topics_table.add_row(
            str(i),
            topic_name,
            topic_info["description"],
            levels
        )
    
    console.print(topics_table)

def display_topic_levels(topic_name: str):
    """Display available levels for a specific topic"""
    topic_info = TOPICS_CATALOG[topic_name]
    
    console.print(Panel(f"[bold cyan]=== {topic_name} - Skill Levels ===[/bold cyan]", border_style="bright_cyan"))
    
    # Create levels table
    levels_table = Table(title=f"Select Your Level in {topic_name}", show_header=True, header_style="bold cyan")
    levels_table.add_column("No.", style="cyan", width=4)
    levels_table.add_column("Level", style="bright_cyan", width=15)
    levels_table.add_column("What You'll Learn", style="white", width=50)
    
    for i, (level, description) in enumerate(topic_info["level_descriptions"].items(), 1):
        levels_table.add_row(
            str(i),
            level.title(),
            description
        )
    
    console.print(levels_table)

def select_topic_and_level():
    """Interactive topic and level selection"""
    # Step 1: Choose topic
    display_topics_catalog()
    
    topic_names = list(TOPICS_CATALOG.keys())
    topic_choice = IntPrompt.ask(
        "[cyan]Select a topic by number[/cyan]",
        choices=[str(i) for i in range(1, len(topic_names) + 1)],
        default=1
    )
    
    selected_topic = topic_names[topic_choice - 1]
    console.print(f"\n[green]✓ Selected Topic: {selected_topic}[/green]")
    
    # Step 2: Choose level within that topic
    display_topic_levels(selected_topic)
    
    available_levels = list(TOPICS_CATALOG[selected_topic]["level_descriptions"].keys())
    level_choice = IntPrompt.ask(
        f"[cyan]Select your skill level in {selected_topic}[/cyan]",
        choices=[str(i) for i in range(1, len(available_levels) + 1)],
        default=1
    )
    
    selected_level = available_levels[level_choice - 1]
    console.print(f"[green]✓ Selected Level: {selected_level.title()}[/green]")
    
    # Step 3: Show what will be covered
    topic_content = TOPICS_CATALOG[selected_topic]["content"][selected_level]
    level_description = TOPICS_CATALOG[selected_topic]["level_descriptions"][selected_level]
    
    console.print(Panel(
        f"[cyan]Topic:[/cyan] {selected_topic}\n"
        f"[cyan]Level:[/cyan] {selected_level.title()}\n"
        f"[cyan]Focus Areas:[/cyan] {level_description}\n\n"
        f"[cyan]Content Preview:[/cyan] {topic_content[:200]}...",
        title="Quiz Configuration",
        border_style="green"
    ))
    
    return selected_topic, selected_level, topic_content

def create_enhanced_profile(name: str, topic: str, level: str, learning_style: str):
    """Create an enhanced profile with topic-specific information"""
    # Define topic-specific strengths and weaknesses based on level
    topic_profiles = {
        "Python Programming": {
            "beginner": {
                "strengths": ["basic syntax understanding", "simple problem solving"],
                "weaknesses": ["advanced concepts", "debugging", "object-oriented programming"]
            },
            "intermediate": {
                "strengths": ["functions", "basic OOP", "control structures"],
                "weaknesses": ["advanced data structures", "decorators", "metaclasses"]
            },
            "advanced": {
                "strengths": ["complex algorithms", "design patterns", "performance optimization"],
                "weaknesses": ["very specialized libraries", "systems programming"]
            }
        },
        "Data Science": {
            "beginner": {
                "strengths": ["basic statistics", "data visualization", "simple analysis"],
                "weaknesses": ["advanced statistics", "machine learning", "big data tools"]
            },
            "intermediate": {
                "strengths": ["statistical analysis", "basic ML", "data preprocessing"],
                "weaknesses": ["deep learning", "advanced algorithms", "model optimization"]
            },
            "advanced": {
                "strengths": ["advanced ML", "neural networks", "model deployment"],
                "weaknesses": ["cutting-edge research", "specialized domains"]
            }
        },
        "Web Development": {
            "beginner": {
                "strengths": ["HTML structure", "basic CSS", "simple JavaScript"],
                "weaknesses": ["frameworks", "backend development", "databases"]
            },
            "intermediate": {
                "strengths": ["frontend frameworks", "API integration", "basic backend"],
                "weaknesses": ["advanced backend", "DevOps", "performance optimization"]
            },
            "advanced": {
                "strengths": ["full-stack development", "architecture design", "deployment"],
                "weaknesses": ["microservices at scale", "advanced security"]
            }
        },
        "Cybersecurity": {
            "beginner": {
                "strengths": ["basic security concepts", "password management", "awareness"],
                "weaknesses": ["technical implementation", "network security", "forensics"]
            },
            "intermediate": {
                "strengths": ["network security", "risk assessment", "basic tools"],
                "weaknesses": ["advanced threats", "forensics", "compliance"]
            },
            "advanced": {
                "strengths": ["threat analysis", "security architecture", "incident response"],
                "weaknesses": ["emerging threats", "advanced forensics"]
            }
        },
        "Machine Learning": {
            "beginner": {
                "strengths": ["basic concepts", "simple algorithms", "data preparation"],
                "weaknesses": ["advanced algorithms", "deep learning", "model optimization"]
            },
            "intermediate": {
                "strengths": ["supervised learning", "model evaluation", "feature engineering"],
                "weaknesses": ["deep learning", "advanced optimization", "MLOps"]
            },
            "advanced": {
                "strengths": ["deep learning", "advanced algorithms", "research concepts"],
                "weaknesses": ["cutting-edge research", "specialized applications"]
            }
        }
    }
    
    profile_data = topic_profiles.get(topic, {}).get(level, {
        "strengths": ["general knowledge"],
        "weaknesses": ["advanced concepts"]
    })
    
    return {
        "name": name,
        "learning_style": learning_style,
        "initial_level": level,
        "topic": topic,
        "topic_level": level,
        "strengths": profile_data["strengths"],
        "weaknesses": profile_data["weaknesses"]
    }


def print_quiz_results(result: Dict[str, Any]):
    """Print quiz results in a colorful and organized way using rich"""
    console.print(Panel("[bold cyan]Quiz Generation Results[/bold cyan]", border_style="bright_cyan"))
    
    # Quiz Questions Table
    quiz_table = Table(title="Generated Questions", show_header=True, header_style="bold cyan")
    quiz_table.add_column("No.", style="cyan", width=5)
    quiz_table.add_column("Question", style="white")
    quiz_table.add_column("Options", style="bright_cyan")
    quiz_table.add_column("Answer", style="green")
    quiz_table.add_column("Difficulty", style="cyan")
    quiz_table.add_column("Topic", style="bright_cyan")
    quiz_table.add_column("Explanation", style="white")
    
    for i, q in enumerate(result['quiz']['questions'], 1):
        options = "\n".join(q['options'])
        quiz_table.add_row(
            str(i),
            q['question'],
            options,
            q['answer'],
            q['difficulty'].title(),
            q['topic'],
            q['explanation']
        )
    
    console.print(quiz_table)

# Existing functions remain the same...
def create_generate_quiz_task(profile: Dict[str, Any], module_content: str, milestone: str, num_questions: int):
    """Create a quiz generation task that calls functions directly"""
    return Task(
        description=f"""
        Generate a personalized quiz for student {profile.get('name', 'Student')} based on their profile and the provided module content.
        
        Student Profile:
        - Name: {profile.get('name', 'Student')}
        - Topic: {profile.get('topic', 'General')}
        - Topic Level: {profile.get('topic_level', 'beginner')}
        - Learning Style: {profile.get('learning_style', 'visual')}
        - Initial Level: {profile.get('initial_level', 'beginner')}
        - Topic-Specific Strengths: {profile.get('strengths', [])}
        - Topic-Specific Weaknesses: {profile.get('weaknesses', [])}
        
        Module Content: {module_content[:1000]}...
        Milestone: {milestone}
        
        The quiz should:
        1. Adapt to the student's specific topic knowledge level
        2. Focus on {profile.get('topic', 'General')} concepts at {profile.get('topic_level', 'beginner')} level
        3. Follow a crescendo difficulty pattern (easy → medium → hard)
        4. Include {num_questions} questions total
        5. Cover key concepts from the module content relevant to their level
        
        Generate the questions and return them in the following JSON format:
        {{
            "questions": [
                {{
                    "question": "Question text here?",
                    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
                    "answer": "A",
                    "difficulty": "easy",
                    "topic": "relevant topic",
                    "explanation": "Brief explanation of the correct answer"
                }}
            ]
        }}
        """,
        expected_output="""
        A well-structured quiz in JSON format containing:
        - questions: list of question objects with question, options, answer, difficulty, topic, explanation
        """,
        agent=quiz_generator_agent
    )

def generate_fallback_questions(content: str, num_questions: int) -> List[Dict[str, Any]]:
    """Generate basic fallback questions when API is unavailable"""
    try:
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.tag import pos_tag
        sentences = sent_tokenize(content)
        words = word_tokenize(content.lower())
        key_terms = [word for word, pos in pos_tag(words) if pos.startswith('NN')]
    except:
        sentences = content.split('. ')
        words = content.lower().split()
        key_terms = [word for word in words if len(word) > 3]
    
    key_terms = list(set(key_terms))[:10]
    
    fallback_questions = []
    for i in range(num_questions):
        if i < len(key_terms):
            term = key_terms[i]
        else:
            term = f"concept_{i+1}"
            
        if i < num_questions // 3:
            difficulty = 'easy'
        elif i < 2 * num_questions // 3:
            difficulty = 'medium'
        else:
            difficulty = 'hard'
        
        question = {
            "question": f"What is the significance of '{term}' in the given context?",
            "options": [
                "A) It is a fundamental concept",
                "B) It is a minor detail", 
                "C) It is irrelevant to the topic",
                "D) It is mentioned briefly"
            ],
            "answer": "A",
            "difficulty": difficulty,
            "topic": "Content Analysis",
            "explanation": f"This question focuses on understanding the role of '{term}' in the educational material."
        }
        fallback_questions.append(question)
    
    return fallback_questions

def run_quiz_workflow_direct(
    profile: Dict[str, Any],
    module_content: str,
    milestone: str,
    num_questions: int,
    user_answers: List[str] = None,
    correct_answers: List[str] = None,
    questions_metadata: List[Dict] = None
) -> Dict[str, Any]:
    try:
        if not os.environ.get("GROQ_API_KEY"):
            return {
                "error": "GROQ_API_KEY environment variable not set",
                "quiz": {"questions": generate_fallback_questions(module_content, num_questions)},
                "evaluation": {},
                "feedback": {}
            }
        
        initialize_groq_agents()
        
        console.print("[cyan]Generating quiz using direct function...[/cyan]")
        try:
            working_model = get_available_groq_models()
            if not working_model:
                raise Exception("No working model found")
                
            questions = generate_questions_with_groq(
                profile=profile,
                module_content=module_content,
                num_questions=num_questions,
                api_key=os.environ.get("GROQ_API_KEY"),
                llm=groq_llm,
                model=working_model
            )
            
            quiz_data = {"questions": questions}
            
        except Exception as gen_error:
            console.print(f"[red]Direct generation failed: {gen_error}[/red]")
            quiz_data = {"questions": generate_fallback_questions(module_content, num_questions)}
        
        result = {"quiz": quiz_data}
        
        if user_answers and correct_answers and questions_metadata:
            console.print("[cyan]Evaluating quiz answers...[/cyan]")
            try:
                # Validate lengths
                if len(user_answers) != len(correct_answers) or len(user_answers) != len(questions_metadata):
                    raise ValueError(f"Mismatch in lengths: user_answers ({len(user_answers)}), correct_answers ({len(correct_answers)}), questions ({len(questions_metadata)})")
                
                difficulties = [q.get('difficulty', 'medium') for q in questions_metadata]
                evaluation_data = grade_answers(
                    user_answers=user_answers,
                    correct_answers=correct_answers,
                    difficulties=difficulties,
                    questions=questions_metadata
                )
                
                # Generate learning insights using Groq API
                console.print("[cyan]Generating learning insights...[/cyan]")
                insights_result = json.loads(generate_learning_insights_tool._run(json.dumps(evaluation_data)))
                if insights_result.get("success"):
                    evaluation_data["learning_insights"] = insights_result["learning_insights"]
                else:
                    console.print(f"[red]Failed to generate insights: {insights_result.get('error', 'Unknown error')}[/red]")
                    evaluation_data["learning_insights"] = []
                
                result["evaluation"] = evaluation_data
                
                console.print("[cyan]Generating feedback...[/cyan]")
                feedback_result = json.loads(create_feedback_tool._run(json.dumps(evaluation_data)))
                if feedback_result.get("success"):
                    feedback_text = feedback_result["feedback_summary"]
                    feedback_data = {
                        "feedback_summary": f"Hello, {profile['name']}! Based on your {profile.get('topic', 'topic')} quiz at {profile.get('topic_level', 'beginner')} level: " + feedback_text,
                        "explanations": {ans["question_number"]: ans["explanation"] for ans in evaluation_data.get("incorrect_answers", [])},
                        "strengths": [],
                        "weaknesses": []
                    }
                    
                    if evaluation_data.get('topic_performance'):
                        for topic, perf in evaluation_data['topic_performance'].items():
                            if perf['percentage'] >= 80:
                                feedback_data['strengths'].append(topic)
                            elif perf['percentage'] < 60:
                                feedback_data['weaknesses'].append(topic)
                    
                    result["feedback"] = feedback_data
                else:
                    console.print(f"[red]Failed to generate feedback: {feedback_result.get('error', 'Unknown error')}[/red]")
                    result["feedback"] = {"error": feedback_result.get("error", "Feedback generation failed")}
                
            except Exception as eval_error:
                console.print(f"[red]Evaluation/Feedback failed: {eval_error}[/red]")
                result["evaluation"] = {"error": str(eval_error)}
                result["feedback"] = {"error": str(eval_error)}
        
        return result
        
    except Exception as e:
        console.print(f"[red]Error in quiz workflow: {str(e)}[/red]")
        return {
            "error": str(e),
            "quiz": {"questions": generate_fallback_questions(module_content, num_questions)},
            "evaluation": {},
            "feedback": {}
        }

def run_quiz_workflow(
    profile: Dict[str, Any],
    module_content: str,
    milestone: str,
    num_questions: int,
    user_answers: List[str] = None,
    correct_answers: List[str] = None,
    questions_metadata: List[Dict] = None
) -> Dict[str, Any]:
    try:
        return run_quiz_workflow_direct(profile, module_content, milestone, num_questions, user_answers, correct_answers, questions_metadata)
    except Exception as e:
        console.print(f"[red]Direct workflow failed: {e}, falling back to basic generation[/red]")
        return {
            "error": str(e),
            "quiz": {"questions": generate_fallback_questions(module_content, num_questions)},
            "evaluation": {},
            "feedback": {}
        }

def test_nltk_offline():
    try:
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk.corpus import stopwords
        
        test_text = "This is a test sentence. This is another sentence."
        try:
            sentences = sent_tokenize(test_text)
            console.print(f"[green]✓ NLTK sentence tokenization works: {len(sentences)} sentences[/green]")
        except:
            sentences = test_text.split('. ')
            console.print(f"[green]✓ Fallback sentence tokenization works: {len(sentences)} sentences[/green]")
        
        try:
            words = word_tokenize(test_text)
            console.print(f"[green]✓ NLTK word tokenization works: {len(words)} words[/green]")
        except:
            words = test_text.split()
            console.print(f"[green]✓ Fallback word tokenization works: {len(words)} words[/green]")
        
        try:
            stop_words = set(stopwords.words('english'))
            console.print(f"[green]✓ NLTK stopwords available: {len(stop_words)} words[/green]")
        except:
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            console.print(f"[green]✓ Fallback stopwords available: {len(stop_words)} words[/green]")
        
        return True
    except Exception as e:
        console.print(f"[red]NLTK test failed: {e}[/red]")
        return False

    """Print quiz results in a colorful and organized way using rich"""
    console.print(Panel("[bold cyan]Quiz Generation Results[/bold cyan]", border_style="bright_cyan"))
    
    # Quiz Questions Table
    quiz_table = Table(title="Generated Questions", show_header=True, header_style="bold cyan")
    quiz_table.add_column("No.", style="cyan", width=5)
    quiz_table.add_column("Question", style="white")
    quiz_table.add_column("Options", style="bright_cyan")
    quiz_table.add_column("Answer", style="green")
    quiz_table.add_column("Difficulty", style="cyan")
    quiz_table.add_column("Topic", style="bright_cyan")
    quiz_table.add_column("Explanation", style="white")
    
    for i, q in enumerate(result['quiz']['questions'], 1):
        options = "\n".join(q['options'])
        quiz_table.add_row(
            str(i),
            q['question'],
            options,
            q['answer'],
            q['difficulty'].title(),
            q['topic'],
            q['explanation']
        )
    
    console.print(quiz_table)

def print_evaluation_results(evaluation: Dict[str, Any]):
    """Print evaluation results in a colorful and organized way"""
    console.print(Panel("[bold cyan]Evaluation Results[/bold cyan]", border_style="bright_cyan"))
    
    if "error" in evaluation:
        console.print(f"[red]Evaluation failed: {evaluation['error']}[/red]")
        return
    
    # Score Summary
    console.print(f"[bold green]Score: {evaluation['score']}/{evaluation['total']} ({evaluation['percentage']:.1f}%)[/bold green]")
    
    # Difficulty Analysis Table
    diff_table = Table(title="Performance by Difficulty", show_header=True, header_style="bold cyan")
    diff_table.add_column("Difficulty", style="cyan")
    diff_table.add_column("Correct", style="green")
    diff_table.add_column("Total", style="bright_cyan")
    diff_table.add_column("Percentage", style="white")
    
    for diff, stats in evaluation['difficulty_analysis'].items():
        diff_table.add_row(
            diff.title(),
            str(stats['correct']),
            str(stats['total']),
            f"{stats['percentage']:.1f}%"
        )
    
    console.print(diff_table)
    
    # Topic Performance Table
    topic_table = Table(title="Performance by Topic", show_header=True, header_style="bold cyan")
    topic_table.add_column("Topic", style="cyan")
    topic_table.add_column("Correct", style="green")
    topic_table.add_column("Total", style="bright_cyan")
    topic_table.add_column("Percentage", style="white")
    
    for topic, stats in evaluation['topic_performance'].items():
        topic_table.add_row(
            topic,
            str(stats['correct']),
            str(stats['total']),
            f"{stats['percentage']:.1f}%"
        )
    
    console.print(topic_table)
    
    # Incorrect Answers
    if evaluation['incorrect_answers']:
        console.print("[bold red]Incorrect Answers:[/bold red]")
        for ans in evaluation['incorrect_answers']:
            console.print(f"[cyan]Question {ans['question_number']} ({ans['difficulty']}):[/cyan]")
            console.print(f"  Your Answer: [red]{ans['user_answer']}[/red]")
            console.print(f"  Correct Answer: [green]{ans['correct_answer']}[/green]")
            console.print(f"  Explanation: [white]{ans['explanation']}[/white]\n")

def print_feedback_results(feedback: Dict[str, Any]):
    """Print feedback results in a colorful and organized way"""
    console.print(Panel("[bold cyan]Feedback for Student[/bold cyan]", border_style="bright_cyan"))
    
    if "error" in feedback:
        console.print(f"[red]Feedback failed: {feedback['error']}[/red]")
        return
    
    # Feedback Summary
    console.print(Text(feedback['feedback_summary'], style="white"))
    
    # Strengths and Weaknesses
    if feedback['strengths']:
        console.print("[bold green]Strengths:[/bold green]")
        for strength in feedback['strengths']:
            console.print(f"  • {strength}", style="bright_cyan")
    
    if feedback['weaknesses']:
        console.print("[bold red]Areas for Improvement:[/bold red]")
        for weakness in feedback['weaknesses']:
            console.print(f"  • {weakness}", style="cyan")

def interactive_quiz_workflow():
    """Run an enhanced interactive quiz workflow with topic-specific level selection"""
    console.print(Panel("[bold cyan]=== Enhanced Interactive Quiz System ===[/bold cyan]", border_style="bright_cyan"))
    
    # Step 1: Collect basic student information
    name = Prompt.ask("[cyan]Enter your name[/cyan]", default="John Doe")
    learning_style = Prompt.ask(
        "[cyan]Enter your learning style (visual/auditory/kinesthetic)[/cyan]",
        default="visual",
        choices=["visual", "auditory", "kinesthetic"]
    )
    
    # Step 2: Select topic and level
    console.print(f"\n[bold cyan]Welcome {name}! Let's select your learning topic and assess your level.[/bold cyan]")
    
    selected_topic, selected_level, topic_content = select_topic_and_level()
    
    # Step 3: Create enhanced profile
    profile = create_enhanced_profile(name, selected_topic, selected_level, learning_style)
    
    # Step 4: Quiz configuration
    milestone = f"{selected_topic} - {selected_level.title()} Level Assessment"
    
    # Specify number of questions
    num_questions = IntPrompt.ask(
        "[cyan]How many questions would you like in the quiz? (1-20)[/cyan]",
        default=8,
        choices=[str(i) for i in range(1, 21)]
    )
    
    # Step 5: Final confirmation with enhanced details
    console.print(Panel(
        f"[cyan]Enhanced Quiz Settings:[/cyan]\n"
        f"[bright_cyan]Student: {name}\n"
        f"Topic: {selected_topic}\n"
        f"Your Level in {selected_topic}: {selected_level.title()}\n"
        f"Number of Questions: {num_questions}\n"
        f"Learning Style: {learning_style}\n"
        f"Your Strengths: {', '.join(profile['strengths'])}\n"
        f"Areas to Focus On: {', '.join(profile['weaknesses'])}\n"
        f"Quiz Focus: Level-appropriate {selected_topic} concepts[/bright_cyan]",
        title="📋 Personalized Quiz Configuration",
        border_style="green"
    ))
    
    if not Confirm.ask("[cyan]Ready to start your personalized quiz?[/cyan]", default=True):
        console.print("[red]Quiz cancelled.[/red]")
        return
    
    # Step 6: Generate quiz
    console.print(f"\n[bold cyan]=== Generating {num_questions} {selected_level.title()}-Level Questions for {selected_topic} ===[/bold cyan]")
    result = run_quiz_workflow(
        profile=profile,
        module_content=topic_content,
        milestone=milestone,
        num_questions=num_questions
    )
    
    if "error" in result:
        console.print(f"[red]Failed to generate quiz: {result['error']}[/red]")
        return
    
    console.print(f"\n[bold green]✓ Successfully generated {len(result['quiz']['questions'])} questions tailored to your {selected_level} level in {selected_topic}![/bold green]")
    print_quiz_results(result)
    
    # Step 7: Take the quiz interactively
    user_answers = []
    questions = result['quiz']['questions']
    
    console.print(f"\n[bold cyan]=== Taking Your {selected_topic} Quiz ===[/bold cyan]")
    console.print(f"[yellow]Instructions: Answer each question by selecting A, B, C, or D[/yellow]\n")
    
    for i, q in enumerate(questions, 1):
        console.print(Panel(f"[bold cyan]Question {i}/{len(questions)}: {q['question']}[/bold cyan]", border_style="bright_cyan"))
        console.print(f"[yellow]Difficulty: {q['difficulty'].title()}[/yellow]")
        console.print("[bright_cyan]Options:[/bright_cyan]")
        for opt in q['options']:
            console.print(f"  {opt}", style="bright_cyan")
        
        answer = Prompt.ask(
            "[cyan]Enter your answer (A/B/C/D)[/cyan]",
            choices=["A", "B", "C", "D"],
            default="A"
        ).upper()
        user_answers.append(answer)
        
        # Show progress
        if i < len(questions):
            console.print(f"[green]✓ Answer recorded! ({i}/{len(questions)} completed)[/green]\n")
    
    # Step 8: Evaluate and provide feedback
    correct_answers = [q['answer'] for q in questions]
    questions_metadata = [
        {
            'question': q['question'],
            'difficulty': q['difficulty'],
            'topic': q['topic'],
            'explanation': q['explanation']
        } for q in questions
    ]
    
    console.print("[cyan]🔍 Analyzing your performance and generating personalized feedback...[/cyan]")
    result_with_eval = run_quiz_workflow(
        profile=profile,
        module_content=topic_content,
        milestone=milestone,
        num_questions=num_questions,
        user_answers=user_answers,
        correct_answers=correct_answers,
        questions_metadata=questions_metadata
    )
    
    # Step 9: Display comprehensive results
    console.print(f"\n[bold cyan]=== Your {selected_topic} Quiz Results ===[/bold cyan]")
    
    if 'evaluation' in result_with_eval and result_with_eval['evaluation']:
        eval_data = result_with_eval['evaluation']
        score = eval_data.get('score', 0)
        total = eval_data.get('total', len(questions))
        percentage = eval_data.get('percentage', 0)
        
        # Performance summary with topic context
        if percentage >= 80:
            performance_emoji = "🌟"
            performance_text = f"Excellent! You have strong {selected_level}-level knowledge in {selected_topic}!"
        elif percentage >= 70:
            performance_emoji = "👍"
            performance_text = f"Good work! You're on track with {selected_level}-level {selected_topic} concepts."
        elif percentage >= 60:
            performance_emoji = "📚"
            performance_text = f"You're making progress in {selected_level}-level {selected_topic}. Keep studying!"
        else:
            performance_emoji = "💪"
            performance_text = f"This shows areas to focus on in {selected_level}-level {selected_topic}. You can do it!"
        
        console.print(Panel(
            f"{performance_emoji} [bold green]{performance_text}[/bold green]\n"
            f"[cyan]Final Score: {score}/{total} ({percentage:.1f}%)[/cyan]",
            title=f"🎯 {selected_topic} Performance Summary",
            border_style="green"
        ))
    
    print_evaluation_results(result_with_eval['evaluation'])
    print_feedback_results(result_with_eval['feedback'])
    
    # Step 10: Next steps recommendations
    console.print(Panel(
        f"[cyan]🚀 Next Steps for {selected_topic}:[/cyan]\n"
        f"[white]• Review areas where you scored below 70%\n"
        f"• Practice more {selected_level}-level problems\n"
        f"• Consider advancing to the next level if you scored above 85%\n"
        f"• Retake the quiz after studying to track improvement[/white]",
        title="📈 Learning Path Recommendations",
        border_style="blue"
    ))

if __name__ == "__main__":
    console.print(Panel("[bold cyan]=== Testing Enhanced Quiz System ===[/bold cyan]", border_style="bright_cyan"))
    
    # Test NLTK functionality
    nltk_works = test_nltk_offline()
    
    # Test Groq connection
    console.print(Panel("[bold cyan]=== Testing Groq connection ===[/bold cyan]", border_style="bright_cyan"))
    working_model = get_available_groq_models()
    if not working_model:
        console.print("[red]⚠ No working Groq model found. Please check your GROQ_API_KEY and internet connection.[/red]")
        exit(1)
    
    console.print(Panel(f"[bold cyan]=== Initializing agents with model: {working_model} ===[/bold cyan]", border_style="green"))
    try:
        initialize_groq_agents()
        console.print("[green]✓ Agents initialized successfully![/green]")
    except Exception as e:
        console.print(f"[red]⚠ Failed to initialize agents: {e}[/red]")
        exit(1)
    
    # Run the enhanced interactive quiz workflow
    interactive_quiz_workflow()