import os
import json
from typing import Dict, Any, List
from groq import Groq
from crewai.tools import BaseTool
from feedback.FeedbackPrompts import FEEDBACK_PROMPT_TEMPLATE, INSIGHTS_PROMPT_TEMPLATE
from dotenv import load_dotenv

load_dotenv()

class CreateFeedbackTool(BaseTool):
    """Create comprehensive, personalized feedback based on quiz results using Groq API"""
    
    name: str = "Create Feedback Tool"
    description: str = "Creates detailed, personalized feedback based on quiz evaluation results using Groq API."
    
    def _run(self, results: str) -> str:
        """
        Run the feedback creation tool
        
        Args:
            results: JSON string of quiz evaluation results
        """
        try:
            # Parse results string to dictionary
            results_dict = json.loads(results) if isinstance(results, str) else results
            
            # Initialize Groq client
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            
            # Generate feedback using Groq API
            feedback = create_feedback(client, results_dict)
            
            return json.dumps({
                "feedback_summary": feedback,
                "success": True
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"Failed to create feedback: {str(e)}",
                "success": False
            })

class GenerateLearningInsightsTool(BaseTool):
    """Generate actionable learning insights from quiz results using Groq API"""
    
    name: str = "Generate Learning Insights Tool"
    description: str = "Generates actionable learning insights and recommendations based on quiz performance patterns using Groq API."
    
    def _run(self, results: str) -> str:
        """
        Run the learning insights generation tool
        
        Args:
            results: JSON string of quiz evaluation results
        """
        try:
            # Parse results string to dictionary
            results_dict = json.loads(results) if isinstance(results, str) else results
            
            # Initialize Groq client
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            
            # Generate insights using Groq API
            insights = generate_learning_insights(client, results_dict)
            
            return json.dumps({
                "learning_insights": insights,
                "success": True
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"Failed to generate insights: {str(e)}",
                "success": False
            })

def create_feedback(client: Groq, results: Dict[str, Any]) -> str:
    """
    Create comprehensive, personalized feedback based on quiz results using Groq API
    
    Args:
        client: Initialized Groq client
        results: Dictionary of quiz evaluation results
    """
    # Format the results as JSON string for the prompt
    results_json = json.dumps(results, indent=2)
    
    # Prepare the prompt using the template
    prompt = FEEDBACK_PROMPT_TEMPLATE.format(results_json=results_json)
    
    # Make API call to Groq
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful and encouraging educational assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

def generate_learning_insights(client: Groq, results: Dict[str, Any]) -> List[str]:
    """
    Generate actionable learning insights from quiz results using Groq API
    
    Args:
        client: Initialized Groq client
        results: Dictionary of quiz evaluation results
    """
    # Format the results as JSON string for the prompt
    results_json = json.dumps(results, indent=2)
    
    # Prepare the prompt using the template
    prompt = INSIGHTS_PROMPT_TEMPLATE.format(results_json=results_json)
    
    # Make API call to Groq
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are an expert educational assistant specializing in generating actionable learning insights."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=500
    )
    
    # Split the response into a list of insights
    insights = response.choices[0].message.content.strip().split("\n")
    return [insight.strip("• ").strip() for insight in insights if insight.strip()]

# Create tool instances that can be imported
create_feedback_tool = CreateFeedbackTool()
generate_learning_insights_tool = GenerateLearningInsightsTool()

# For backward compatibility, also export the functions
__all__ = ['CreateFeedbackTool', 'GenerateLearningInsightsTool', 'create_feedback_tool', 
           'generate_learning_insights_tool', 'create_feedback', 'generate_learning_insights']