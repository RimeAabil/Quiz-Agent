"""
Quiz Generation Prompts
Contains all prompts used for quiz question generation with LLM
Enhanced with content-matched fallback templates
"""

def get_quiz_generation_prompt(module_content: str, profile: dict, num_questions: int, 
                              easy_count: int, medium_count: int, hard_count: int) -> str:
    """
    Generate the main prompt for quiz question creation
    
    Args:
        module_content: Educational content to base questions on
        profile: Student profile information
        num_questions: Total number of questions to generate
        easy_count: Number of easy questions
        medium_count: Number of medium questions  
        hard_count: Number of hard questions
        
    Returns:
        Formatted prompt string for the LLM
    """
    
    prompt = f"""You are an expert educational assessment creator. Generate exactly {num_questions} high-quality multiple choice questions based on the provided educational content.

EDUCATIONAL CONTENT:
{module_content[:2000]}

STUDENT PROFILE:
- Learning Level: {profile.get('initial_level', 'beginner')}
- Learning Style: {profile.get('learning_style', 'visual')}
- Name: {profile.get('name', 'Student')}

QUESTION DISTRIBUTION REQUIREMENTS:
- Questions 1-{easy_count}: EASY level (basic recall, definitions, simple identification)
- Questions {easy_count+1}-{easy_count+medium_count}: MEDIUM level (application, analysis, comparison)
- Questions {easy_count+medium_count+1}-{num_questions}: HARD level (evaluation, synthesis, critical thinking)

FORMATTING REQUIREMENTS:
- Each question must have exactly 4 options labeled A, B, C, D
- Only ONE correct answer per question
- Include a brief but clear explanation for the correct answer
- Extract relevant topics from the content for categorization
- Ensure questions progressively increase in cognitive complexity

QUALITY STANDARDS:
- Questions should be clear, unambiguous, and grammatically correct
- Distractors (wrong answers) should be plausible but clearly incorrect
- Cover different aspects of the content, not just repeated concepts
- Align with the student's learning level and style
- Test understanding, not just memorization

OUTPUT FORMAT:
Return ONLY a valid JSON array with this exact structure:

[
  {{
    "question": "Clear, specific question text here",
    "options": [
      "A) First option",
      "B) Second option", 
      "C) Third option",
      "D) Fourth option"
    ],
    "answer": "B",
    "difficulty": "easy",
    "topic": "Specific topic from content",
    "explanation": "Brief explanation of why this answer is correct and others are wrong."
  }}
]

CRITICAL INSTRUCTIONS:
- Generate exactly {num_questions} questions
- Follow the difficulty distribution exactly ({easy_count} easy, {medium_count} medium, {hard_count} hard)
- Return only valid JSON, no additional text or formatting
- Ensure each question tests different concepts from the material
- Make explanations educational and helpful for learning

Begin generating the {num_questions} questions now:"""

    return prompt


def get_simplified_retry_prompt(module_content: str, num_questions: int) -> str:
    """
    Generate a simplified prompt for retry attempts when the main prompt fails
    
    Args:
        module_content: Educational content (truncated for retry)
        num_questions: Total number of questions needed
        
    Returns:
        Simplified prompt string
    """
    
    prompt = f"""Create exactly {num_questions} quiz questions about this content:

CONTENT: {module_content[:800]}

Requirements:
- {num_questions} multiple choice questions (A, B, C, D format)
- Mix of easy, medium, and hard difficulty
- Include topic and explanation for each
- Return only JSON array format

Example format:
[
  {{
    "question": "What is the main concept discussed?",
    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
    "answer": "B",
    "difficulty": "easy",
    "topic": "Main Concepts", 
    "explanation": "The content clearly states this concept."
  }}
]

Generate {num_questions} questions now:"""

    return prompt


def get_fallback_question_templates():
    """
    Get question templates for fallback generation when API fails
    Now matches the three content types available in the interface:
    - Python Basics
    - Data Science  
    - Web Development
    
    Returns:
        Dictionary with question templates organized by difficulty and content type
    """
    
    templates = {
        'python_basics': {
            'easy': [
                {
                    "template": "What keyword is used to define a function in Python?",
                    "options": [
                        "A) def",
                        "B) function", 
                        "C) define",
                        "D) func"
                    ],
                    "answer": "A",
                    "explanation": "The 'def' keyword is used to define functions in Python, followed by the function name and parameters."
                },
                {
                    "template": "Which data type is used for whole numbers in Python?",
                    "options": [
                        "A) float",
                        "B) integer",
                        "C) string",
                        "D) boolean"
                    ],
                    "answer": "B",
                    "explanation": "Integers (int) are used for whole numbers in Python, while floats are for decimal numbers."
                },
                {
                    "template": "What symbol is used to create a list in Python?",
                    "options": [
                        "A) Square brackets []",
                        "B) Curly braces {}",
                        "C) Parentheses ()",
                        "D) Angle brackets <>"
                    ],
                    "answer": "A",
                    "explanation": "Square brackets [] are used to create lists in Python. Curly braces {} are for dictionaries and sets."
                }
            ],
            'medium': [
                {
                    "template": "How do you handle errors in Python?",
                    "options": [
                        "A) Using try-except blocks",
                        "B) Using if-else statements",
                        "C) Using while loops",
                        "D) Using for loops"
                    ],
                    "answer": "A",
                    "explanation": "Try-except blocks are used for error handling in Python, allowing you to catch and handle exceptions gracefully."
                },
                {
                    "template": "What is the difference between a list and a dictionary in Python?",
                    "options": [
                        "A) Lists store key-value pairs, dictionaries store ordered items",
                        "B) Lists store ordered items, dictionaries store key-value pairs",
                        "C) Lists are immutable, dictionaries are mutable",
                        "D) There is no difference"
                    ],
                    "answer": "B",
                    "explanation": "Lists store ordered collections of items accessed by index, while dictionaries store key-value pairs accessed by keys."
                },
                {
                    "template": "What is the purpose of the '__init__' method in Python classes?",
                    "options": [
                        "A) To delete an object",
                        "B) To initialize object attributes when created",
                        "C) To define static methods",
                        "D) To handle errors"
                    ],
                    "answer": "B",
                    "explanation": "The '__init__' method is a constructor that initializes object attributes when an instance of a class is created."
                }
            ],
            'hard': [
                {
                    "template": "Analyze the use of decorators in Python. What is their primary purpose?",
                    "options": [
                        "A) To modify or extend the behavior of functions or classes without permanently modifying them",
                        "B) To create new data types",
                        "C) To handle file operations",
                        "D) To manage memory allocation"
                    ],
                    "answer": "A",
                    "explanation": "Decorators are a powerful feature that allows you to modify or extend the behavior of functions or classes without permanently modifying their code."
                },
                {
                    "template": "Evaluate the benefits of using list comprehensions over traditional for loops in Python.",
                    "options": [
                        "A) List comprehensions are more readable and often faster for creating lists",
                        "B) List comprehensions use less memory but are slower",
                        "C) For loops are always better than list comprehensions",
                        "D) There is no difference in performance or readability"
                    ],
                    "answer": "A",
                    "explanation": "List comprehensions provide a concise, readable way to create lists and are often faster than traditional for loops for list creation tasks."
                }
            ]
        },
        
        'data_science': {
            'easy': [
                {
                    "template": "What does CSV stand for in data science?",
                    "options": [
                        "A) Comma Separated Values",
                        "B) Computer Science Variables",
                        "C) Central Statistical Values",
                        "D) Calculated Summary Values"
                    ],
                    "answer": "A",
                    "explanation": "CSV stands for Comma Separated Values, a common file format for storing tabular data."
                },
                {
                    "template": "Which Python library is most commonly used for data manipulation?",
                    "options": [
                        "A) matplotlib",
                        "B) pandas",
                        "C) numpy",
                        "D) scipy"
                    ],
                    "answer": "B",
                    "explanation": "Pandas is the most popular Python library for data manipulation and analysis, providing powerful data structures like DataFrames."
                },
                {
                    "template": "What does EDA stand for in data science?",
                    "options": [
                        "A) Enhanced Data Acquisition",
                        "B) Exploratory Data Analysis",
                        "C) Extended Data Application",
                        "D) Experimental Data Approach"
                    ],
                    "answer": "B",
                    "explanation": "EDA stands for Exploratory Data Analysis, the process of analyzing and visualizing data to understand its characteristics."
                }
            ],
            'medium': [
                {
                    "template": "What is the main purpose of data cleaning in the data science pipeline?",
                    "options": [
                        "A) To remove inconsistencies and prepare data for analysis",
                        "B) To increase the size of the dataset",
                        "C) To create visualizations",
                        "D) To perform statistical tests"
                    ],
                    "answer": "A",
                    "explanation": "Data cleaning involves removing inconsistencies, handling missing values, and preparing data for accurate analysis."
                },
                {
                    "template": "Which visualization is best for showing correlation between two continuous variables?",
                    "options": [
                        "A) Bar chart",
                        "B) Pie chart",
                        "C) Scatter plot",
                        "D) Line graph"
                    ],
                    "answer": "C",
                    "explanation": "Scatter plots are ideal for visualizing the relationship and correlation between two continuous variables."
                },
                {
                    "template": "What is the difference between supervised and unsupervised machine learning?",
                    "options": [
                        "A) Supervised uses labeled data, unsupervised finds patterns in unlabeled data",
                        "B) Supervised is faster, unsupervised is more accurate",
                        "C) Supervised uses more data, unsupervised uses less",
                        "D) There is no difference"
                    ],
                    "answer": "A",
                    "explanation": "Supervised learning uses labeled training data to make predictions, while unsupervised learning finds hidden patterns in unlabeled data."
                }
            ],
            'hard': [
                {
                    "template": "Evaluate the trade-offs between bias and variance in machine learning models.",
                    "options": [
                        "A) High bias leads to underfitting, high variance leads to overfitting; optimal models balance both",
                        "B) High bias is always better than high variance",
                        "C) Variance doesn't affect model performance",
                        "D) Bias and variance are unrelated to model performance"
                    ],
                    "answer": "A",
                    "explanation": "The bias-variance tradeoff is fundamental: high bias causes underfitting, high variance causes overfitting, and optimal models balance both for best generalization."
                },
                {
                    "template": "Synthesize the role of feature engineering in improving model performance across different domains.",
                    "options": [
                        "A) Feature engineering creates informative inputs that help models learn better patterns and improve predictive accuracy",
                        "B) Feature engineering only works for specific types of data",
                        "C) Raw data is always better than engineered features",
                        "D) Feature engineering decreases model performance"
                    ],
                    "answer": "A",
                    "explanation": "Feature engineering transforms raw data into informative features that help machine learning models learn better patterns and achieve higher accuracy across various domains."
                }
            ]
        },
        
        'web_development': {
            'easy': [
                {
                    "template": "What does HTML stand for?",
                    "options": [
                        "A) HyperText Markup Language",
                        "B) High Tech Modern Language",
                        "C) Home Tool Markup Language",
                        "D) Hyperlink Text Management Language"
                    ],
                    "answer": "A",
                    "explanation": "HTML stands for HyperText Markup Language, the standard language for creating web pages and web applications."
                },
                {
                    "template": "Which language is primarily used for styling web pages?",
                    "options": [
                        "A) HTML",
                        "B) JavaScript",
                        "C) CSS",
                        "D) Python"
                    ],
                    "answer": "C",
                    "explanation": "CSS (Cascading Style Sheets) is used to control the presentation and styling of web pages, including colors, layouts, and fonts."
                },
                {
                    "template": "What makes a website interactive?",
                    "options": [
                        "A) HTML",
                        "B) CSS",
                        "C) JavaScript",
                        "D) SQL"
                    ],
                    "answer": "C",
                    "explanation": "JavaScript is the programming language that adds interactivity to websites, handling user interactions and dynamic content updates."
                }
            ],
            'medium': [
                {
                    "template": "What is the purpose of a database in web development?",
                    "options": [
                        "A) To store and manage application data persistently",
                        "B) To style web pages",
                        "C) To add interactivity",
                        "D) To structure HTML content"
                    ],
                    "answer": "A",
                    "explanation": "Databases store and manage application data persistently, allowing web applications to save, retrieve, and manipulate user information and content."
                },
                {
                    "template": "How do front-end frameworks like React improve web development?",
                    "options": [
                        "A) They provide reusable components and better state management for complex applications",
                        "B) They replace the need for HTML and CSS",
                        "C) They only work for mobile applications",
                        "D) They make websites slower but more secure"
                    ],
                    "answer": "A",
                    "explanation": "Front-end frameworks like React provide component-based architecture, efficient state management, and reusable code, making complex web applications easier to build and maintain."
                },
                {
                    "template": "What is the difference between front-end and back-end development?",
                    "options": [
                        "A) Front-end handles user interface and experience, back-end manages server logic and data",
                        "B) Front-end is harder than back-end development",
                        "C) Front-end uses databases, back-end creates user interfaces",
                        "D) There is no significant difference"
                    ],
                    "answer": "A",
                    "explanation": "Front-end development focuses on user interface and user experience (client-side), while back-end development handles server-side logic, databases, and application architecture."
                }
            ],
            'hard': [
                {
                    "template": "Analyze the benefits and trade-offs of Single Page Applications (SPAs) versus Multi-Page Applications (MPAs).",
                    "options": [
                        "A) SPAs provide smoother user experience but have slower initial load times; MPAs have faster initial loads but less seamless navigation",
                        "B) SPAs are always better than MPAs in every situation",
                        "C) MPAs provide better performance in all scenarios",
                        "D) There are no significant differences between SPAs and MPAs"
                    ],
                    "answer": "A",
                    "explanation": "SPAs offer smoother user experiences with faster subsequent page loads but have slower initial loads and SEO challenges, while MPAs have faster initial loads and better SEO but less seamless user interactions."
                },
                {
                    "template": "Evaluate the role of web security measures like HTTPS, CSRF protection, and input validation in modern web development.",
                    "options": [
                        "A) These measures are essential for protecting user data, preventing attacks, and maintaining trust in web applications",
                        "B) Security measures are optional and only needed for large applications",
                        "C) Security measures slow down development without significant benefits",
                        "D) Only HTTPS is necessary; other measures are redundant"
                    ],
                    "answer": "A",
                    "explanation": "Web security measures like HTTPS, CSRF protection, and input validation are crucial for protecting sensitive data, preventing common attacks, and maintaining user trust in web applications."
                }
            ]
        },
        
        'general': {
            'easy': [
                {
                    "template": "What is the main concept discussed in this educational content?",
                    "options": [
                        "A) A fundamental principle that students should understand",
                        "B) An advanced technique for experts only",
                        "C) A historical reference with limited relevance",
                        "D) A controversial topic with no consensus"
                    ],
                    "answer": "A",
                    "explanation": "The content presents fundamental principles that form the foundation for student understanding in this subject area."
                },
                {
                    "template": "According to the material, which statement best describes the key learning objective?",
                    "options": [
                        "A) To build foundational knowledge for further learning",
                        "B) To memorize specific details without context",
                        "C) To focus only on theoretical aspects",
                        "D) To avoid practical applications"
                    ],
                    "answer": "A",
                    "explanation": "The material emphasizes building foundational knowledge that serves as a basis for more advanced learning and practical application."
                }
            ],
            'medium': [
                {
                    "template": "How do the concepts in this material relate to practical applications?",
                    "options": [
                        "A) They provide essential frameworks for real-world problem solving",
                        "B) They are purely theoretical with no practical use",
                        "C) They only apply in very specific, limited scenarios",
                        "D) They contradict real-world practices"
                    ],
                    "answer": "A",
                    "explanation": "The concepts presented provide essential frameworks and principles that can be applied to solve real-world problems and challenges."
                },
                {
                    "template": "What is the relationship between different concepts presented in this content?",
                    "options": [
                        "A) They build upon each other to create comprehensive understanding",
                        "B) They are completely independent and unrelated",
                        "C) They contradict each other frequently",
                        "D) Only one concept is important, others are distractions"
                    ],
                    "answer": "A",
                    "explanation": "The concepts are interconnected and build upon each other, creating a comprehensive understanding of the subject matter."
                }
            ],
            'hard': [
                {
                    "template": "Evaluate the long-term implications of mastering these concepts for future learning and development.",
                    "options": [
                        "A) Mastery provides a strong foundation for advanced study and professional application",
                        "B) These concepts will become obsolete quickly",
                        "C) Mastery is only useful for academic purposes",
                        "D) The concepts have no long-term value"
                    ],
                    "answer": "A",
                    "explanation": "Mastering these fundamental concepts provides a strong foundation that enables advanced study, professional application, and continued learning in the field."
                },
                {
                    "template": "Synthesize how these concepts contribute to the broader field of study.",
                    "options": [
                        "A) They represent core principles that shape current understanding and future developments",
                        "B) They are outdated approaches that hinder progress",
                        "C) They only have historical significance",
                        "D) They create unnecessary complexity in the field"
                    ],
                    "answer": "A",
                    "explanation": "These concepts represent core principles that not only shape current understanding but also provide the foundation for future developments and innovations in the field."
                }
            ]
        }
    }
    
    return templates


def detect_content_type(content: str) -> str:
    """
    Detect the type of educational content to select appropriate templates
    
    Args:
        content: Educational content to analyze
        
    Returns:
        Content type string ('python_basics', 'data_science', 'web_development', or 'general')
    """
    content_lower = content.lower()
    
    # Python keywords and concepts
    python_keywords = ['python', 'variable', 'function', 'def', 'class', 'object', 'list', 'dictionary', 
                       'loop', 'for', 'while', 'if', 'else', 'try', 'except', 'import', 'module']
    
    # Data science keywords
    data_science_keywords = ['data science', 'data analysis', 'pandas', 'numpy', 'matplotlib', 'machine learning',
                            'statistics', 'visualization', 'dataset', 'csv', 'dataframe', 'algorithm', 'model']
    
    # Web development keywords  
    web_dev_keywords = ['html', 'css', 'javascript', 'web development', 'website', 'browser', 'frontend',
                       'backend', 'react', 'node.js', 'database', 'server', 'http', 'api', 'dom']
    
    # Count matches for each category
    python_count = sum(1 for keyword in python_keywords if keyword in content_lower)
    data_science_count = sum(1 for keyword in data_science_keywords if keyword in content_lower) 
    web_dev_count = sum(1 for keyword in web_dev_keywords if keyword in content_lower)
    
    # Determine content type based on highest match count
    max_count = max(python_count, data_science_count, web_dev_count)
    
    if max_count == 0:
        return 'general'
    elif python_count == max_count:
        return 'python_basics'
    elif data_science_count == max_count:
        return 'data_science' 
    else:
        return 'web_development'


def get_content_analysis_prompt(content: str) -> str:
    """
    Generate prompt for content analysis to extract key concepts
    
    Args:
        content: Educational content to analyze
        
    Returns:
        Prompt for content analysis
    """
    
    prompt = f"""Analyze this educational content and identify the key concepts, terms, and topics that would be appropriate for quiz questions:

CONTENT:
{content[:1500]}

Please identify:
1. Key concepts that students should understand
2. Important terms and definitions
3. Main topics covered
4. Practical applications mentioned
5. Relationships between different ideas

Focus on extracting concepts that can be tested at different cognitive levels (recall, understanding, application, analysis, evaluation).

Return the analysis in a structured format that highlights the most important elements for assessment."""

    return prompt