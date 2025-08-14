"""
Quiz Agent API - Enhanced FastAPI implementation for CrewAI integration
Provides endpoints for quiz generation, evaluation, feedback, and content analysis
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import os
import json
import logging

# Import existing modules
from QuizAgent import QuizAgent
from generator.GeneratorTools import generate_questions_with_groq, ContentAnalyzer, QuestionDifficultyAnalyzer
from evaluator.EvaluatorTools import grade_answers
from feedback.FeedbackTools import create_feedback, generate_learning_insights

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Quiz Agent API",
    description="Educational Quiz System with AI-powered generation, evaluation, and feedback",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global quiz agent - initialize on first use
quiz_agent = None

def get_quiz_agent():
    """Get or initialize the quiz agent"""
    global quiz_agent
    if quiz_agent is None:
        try:
            quiz_agent = QuizAgent()
            logger.info("Quiz Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Quiz Agent: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Quiz Agent initialization failed: {str(e)}"
            )
    return quiz_agent

# Pydantic models for request/response validation
class StudentProfile(BaseModel):
    name: str
    learning_style: str = "visual"
    initial_level: str = "beginner"
    strengths: List[str] = []
    weaknesses: List[str] = []

    @validator('learning_style')
    def validate_learning_style(cls, v):
        valid_styles = ['visual', 'auditory', 'kinesthetic', 'reading']
        if v.lower() not in valid_styles:
            return 'visual'  # Default fallback
        return v.lower()

    @validator('initial_level')
    def validate_initial_level(cls, v):
        valid_levels = ['beginner', 'intermediate', 'advanced']
        if v.lower() not in valid_levels:
            return 'beginner'  # Default fallback
        return v.lower()

class QuizGenerationRequest(BaseModel):
    profile: StudentProfile
    module_content: str
    milestone: str
    num_questions: Optional[int] = 8  # Align with MIN_QUESTIONS from QuizAgent.py

    @validator('num_questions')
    def validate_num_questions(cls, v):
        if v < 8 or v > 50:  # Align with MIN_QUESTIONS and MAX_QUESTIONS
            raise ValueError("Number of questions must be between 8 and 50")
        return v

class QuizEvaluationRequest(BaseModel):
    user_answers: List[str]
    correct_answers: List[str]
    questions_metadata: List[Dict[str, Any]]
    milestone: str

class FeedbackRequest(BaseModel):
    evaluation_results: Dict[str, Any]
    profile: StudentProfile
    milestone: str

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Quiz Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        agent = get_quiz_agent()
        return {
            "status": "healthy",
            "service": "Quiz Agent API",
            "agent_initialized": True
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "agent_initialized": False
        }

@app.post("/quiz/generate")
async def generate_quiz(request: QuizGenerationRequest):
    """Generate a personalized quiz"""
    try:
        agent = get_quiz_agent()
        profile_dict = request.profile.dict()
        result = agent.run_task(
            'generate_quiz',
            profile=profile_dict,
            module_content=request.module_content,
            milestone=request.milestone,
            num_questions=request.num_questions
        )
        if not result or 'questions' not in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate quiz questions"
            )
        return {
            "success": True,
            "questions": result['questions'],
            "metadata": {
                "student_name": request.profile.name,
                "milestone": request.milestone,
                "total_questions": len(result['questions']),
                "difficulties": result.get('metadata', {}).get('difficulties', {})
            }
        }
    except Exception as e:
        logger.error(f"Error generating quiz: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quiz generation failed: {str(e)}"
        )

@app.post("/quiz/evaluate")
async def evaluate_quiz(request: QuizEvaluationRequest):
    """Evaluate quiz answers"""
    try:
        agent = get_quiz_agent()
        if len(request.user_answers) != len(request.correct_answers):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Mismatch between user answers and correct answers"
            )
        result = agent.run_task(
            'evaluate_quiz',
            user_answers=request.user_answers,
            correct_answers=request.correct_answers,
            questions_metadata=request.questions_metadata,
            milestone=request.milestone
        )
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to evaluate quiz"
            )
        return {
            "success": True,
            "evaluation": result
        }
    except Exception as e:
        logger.error(f"Error evaluating quiz: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quiz evaluation failed: {str(e)}"
        )

@app.post("/feedback/generate")
async def generate_feedback(request: FeedbackRequest):
    """Generate personalized feedback"""
    try:
        agent = get_quiz_agent()
        profile_dict = request.profile.dict()
        result = agent.run_task(
            'give_feedback',
            evaluation_results=request.evaluation_results,
            profile=profile_dict,
            milestone=request.milestone
        )
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate feedback"
            )
        return {
            "success": True,
            "feedback": result
        }
    except Exception as e:
        logger.error(f"Error generating feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feedback generation failed: {str(e)}"
        )

class ContentAnalysisRequest(BaseModel):
    content: str = Field(..., description="Educational content to analyze")

@app.post("/content/analyze")
async def analyze_content(request: ContentAnalysisRequest):
    """Analyze educational content for key concepts, complexity, and objectives"""
    try:
        analyzer = ContentAnalyzer()
        result = analyzer._run(request.content)
        return {
            "success": True,
            "analysis": result
        }
    except Exception as e:
        logger.error(f"Error analyzing content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Content analysis failed: {str(e)}"
        )

@app.post("/quiz/complete-workflow")
async def complete_quiz_workflow(request: QuizGenerationRequest):
    """Complete workflow: Generate quiz, return it ready for student interaction"""
    try:
        agent = get_quiz_agent()
        profile_dict = request.profile.dict()
        result = agent.run_task(
            'generate_quiz',
            profile=profile_dict,
            module_content=request.module_content,
            milestone=request.milestone,
            num_questions=request.num_questions
        )
        if not result or 'questions' not in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate quiz"
            )
        return {
            "success": True,
            "quiz": {
                "questions": result['questions'],
                "instructions": f"Hello {request.profile.name}! Your quiz is ready.",
                "metadata": {
                    "student_name": request.profile.name,
                    "milestone": request.milestone,
                    "total_questions": len(result['questions']),
                    "estimated_time_minutes": len(result['questions']) * 2
                }
            }
        }
    except Exception as e:
        logger.error(f"Error in complete workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Complete workflow failed: {str(e)}"
        )

@app.post("/quiz/submit-and-get-results")
async def submit_quiz_and_get_results(
    profile: StudentProfile,
    milestone: str,
    user_answers: List[str],
    quiz_questions: List[Dict[str, Any]]
):
    """Submit quiz answers and get complete results (evaluation + feedback)"""
    try:
        agent = get_quiz_agent()
        correct_answers = [q.get('correct_answer', q.get('answer', '')) for q in quiz_questions]
        questions_metadata = [
            {
                'question': q['question'],
                'difficulty': q.get('difficulty', 'medium'),
                'topic': q.get('topic', 'General'),
                'explanation': q.get('explanation', '')
            } for q in quiz_questions
        ]
        evaluation = agent.run_task(
            'evaluate_quiz',
            user_answers=user_answers,
            correct_answers=correct_answers,
            questions_metadata=questions_metadata,
            milestone=milestone
        )
        feedback = agent.run_task(
            'give_feedback',
            evaluation_results=evaluation,
            profile=profile.dict(),
            milestone=milestone
        )
        return {
            "success": True,
            "results": {
                "evaluation": evaluation,
                "feedback": feedback,
                "summary": {
                    "student_name": profile.name,
                    "milestone": milestone,
                    "score": f"{evaluation.get('score', 0)}/{evaluation.get('total', 0)}",
                    "percentage": f"{evaluation.get('percentage', 0):.1f}%"
                }
            }
        }
    except Exception as e:
        logger.error(f"Error in submit and evaluate: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Submit and evaluate failed: {str(e)}"
        )

# Backward-compatible endpoints for original crew_setup.py tools
@app.post("/generate-quiz")
async def generate_quiz_compat(request: QuizGenerationRequest):
    """Backward-compatible endpoint for /quiz/generate"""
    return await generate_quiz(request)

@app.post("/evaluate-quiz")
async def evaluate_quiz_compat(request: QuizEvaluationRequest):
    """Backward-compatible endpoint for /quiz/evaluate"""
    return await evaluate_quiz(request)

@app.post("/give-feedback")
async def give_feedback_compat(request: FeedbackRequest):
    """Backward-compatible endpoint for /feedback/generate"""
    return await generate_feedback(request)

# Test endpoints
@app.get("/test/sample-profile")
async def get_sample_profile():
    """Get a sample student profile for testing"""
    return {
        "name": "Test Student",
        "learning_style": "visual",
        "initial_level": "beginner",
        "strengths": ["problem-solving"],
        "weaknesses": ["syntax"]
    }

@app.get("/test/sample-content")
async def get_sample_content():
    """Get sample educational content for testing"""
    return {
        "content": """
        Python is a high-level programming language known for its simplicity and readability.
        Variables in Python store data values and don't need explicit type declaration.
        Python supports various data types: integers, floats, strings, and booleans.
        Functions are defined using the 'def' keyword followed by the function name.
        Control structures like if-else statements and loops control program flow.
        """,
        "milestone": "Python Basics - Introduction"
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Endpoint not found",
        "available_endpoints": {
            "docs": "/docs",
            "health": "/health",
            "generate_quiz": "/quiz/generate",
            "evaluate_quiz": "/quiz/evaluate",
            "generate_feedback": "/feedback/generate",
            "complete_workflow": "/quiz/complete-workflow",
            "submit_and_get_results": "/quiz/submit-and-get-results",
            "compat_generate_quiz": "/generate-quiz",
            "compat_evaluate_quiz": "/evaluate-quiz",
            "compat_give_feedback": "/give-feedback"
        }
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    if not os.environ.get("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY environment variable not set")
        print("The API will work but may use fallback question generation")
    print("Starting Quiz Agent API...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    uvicorn.run(
        "api:app",  # Fixed from "main:app"
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )