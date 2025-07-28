# Assessment Generation System

A comprehensive educational assessment system that automatically generates learning objectives, creates questions based on Bloom's Taxonomy, evaluates student answers, and provides personalized feedback using AI and vector databases.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Database Schema](#database-schema)
- [Core Components](#core-components)
- [Detailed Function Documentation](#detailed-function-documentation)
- [Usage Guide](#usage-guide)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)

## Overview

This system processes educational PDF documents, generates learning objectives, creates assessments across different Bloom's Taxonomy levels, and provides automated evaluation with detailed feedback. It uses Google Gemini AI, Zilliz vector database, and PostgreSQL for comprehensive educational assessment automation.

## Features

- **Automated Learning Objective Generation**: Creates 4-6 measurable learning objectives from course content
- **Bloom's Taxonomy Assessment**: Generates questions across all 6 levels (Remember, Understand, Apply, Analyze, Evaluate, Create)
- **Vector-Based Content Retrieval**: Uses semantic search to find relevant content for question generation
- **Automated Evaluation**: AI-powered answer evaluation with detailed rubrics
- **Personalized Feedback**: Provides criterion-specific feedback and study recommendations
- **Database Persistence**: Stores all assessments, answers, and evaluations in PostgreSQL
- **Batch Processing**: Handles multiple PDF documents across different courses and modules

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Files     │───▶│  Vector Storage  │───▶│   AI Models     │
│   (Course       │    │   (Zilliz)       │    │   (Gemini)      │
│   Content)      │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PostgreSQL    │◀───│  Assessment      │◀───│  Question       │
│   Database      │    │  Generator       │    │  Generation     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Prerequisites

- Python 3.8+
- PostgreSQL database
- Google API key for Gemini models
- Zilliz cloud account and credentials
- LangSmith account (optional, for tracing)

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd assessment-system
```

2. **Install required packages**:
```bash
pip install -r requirements.txt
```

Required packages:
```
pydantic
pymilvus
langchain-core
langchain-community
langchain-text-splitters
langchain-milvus
langchain-google-genai
psycopg2-binary
PyPDF2
pathlib
```

## Configuration

### Environment Variables

Set up the following configuration in your script or environment:

```python
# Google API Configuration
api_key = "YOUR_GOOGLE_API_KEY"
os.environ["GOOGLE_API_KEY"] = api_key

# LangSmith Configuration (Optional)
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "YOUR_LANGSMITH_API_KEY"

# Zilliz Configuration
ZILLIZ_URI = "YOUR_ZILLIZ_URI"
ZILLIZ_TOKEN = "YOUR_ZILLIZ_TOKEN"
```

### Database Configuration

Update the database connection details in the `get_db_connection()` function:

```python
def get_db_connection():
    return psycopg2.connect(
        dbname="questionsdb",
        user="your_username",
        password="your_password",
        host="localhost",
        port="5432"
    )
```

## Database Schema

The system uses a normalized PostgreSQL schema with the following tables:

### Tables Structure

```sql
-- Courses table
CREATE TABLE courses (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL
);

-- Modules table (linked to courses)
CREATE TABLE modules (
    id SERIAL PRIMARY KEY,
    course_id INTEGER REFERENCES courses(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    UNIQUE(course_id, name)
);

-- Learning objectives table (linked to modules)
CREATE TABLE learning_objectives (
    id SERIAL PRIMARY KEY,
    module_id INTEGER REFERENCES modules(id) ON DELETE CASCADE,
    objective_text TEXT NOT NULL
);

-- Assessments/Questions table (linked to objectives)
CREATE TABLE assessments (
    id SERIAL PRIMARY KEY,
    objective_id INTEGER REFERENCES learning_objectives(id) ON DELETE CASCADE,
    bloom_level VARCHAR(50) NOT NULL,
    question_text TEXT NOT NULL
);

-- Rubric criteria table (linked to assessments)
CREATE TABLE rubric_criteria (
    id SERIAL PRIMARY KEY,
    assessment_id INTEGER REFERENCES assessments(id) ON DELETE CASCADE,
    description TEXT NOT NULL,
    weight FLOAT NOT NULL
);

-- User answers table (linked to assessments)
CREATE TABLE user_answers (
    id SERIAL PRIMARY KEY,
    assessment_id INTEGER REFERENCES assessments(id) ON DELETE CASCADE,
    answer_text TEXT NOT NULL,
    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Evaluation results table (linked to answers)
CREATE TABLE evaluation_results (
    id SERIAL PRIMARY KEY,
    answer_id INTEGER UNIQUE REFERENCES user_answers(id) ON DELETE CASCADE,
    score FLOAT NOT NULL,
    justification TEXT NOT NULL
);

-- Criterion feedback table (linked to evaluations)
CREATE TABLE criterion_feedback (
    id SERIAL PRIMARY KEY,
    evaluation_id INTEGER REFERENCES evaluation_results(id) ON DELETE CASCADE,
    criterion_description TEXT NOT NULL,
    feedback TEXT NOT NULL
);
```

## Core Components

### 1. Pydantic Models

The system uses Pydantic models for structured data validation and AI response parsing:

#### Questions Model
```python
class Questions(BaseModel):
    '''Questions based on each Bloom's Taxonomy level for each learning objectives.'''
    objective: str = Field(description="Learning objective for which questions are generated")
    bloom_level: str = Field(description="Bloom's Taxonomy level for which questions are generated")
    questions: List[str] = Field(description="List of questions for the given Learning Objective and Bloom Level")
```

#### Learning Objectives Model
```python
class LearningObjectives(BaseModel):
    """A list of learning objectives for a course module."""
    objectives: List[str] = Field(description="A list of 4 to 6 clear and measurable learning objectives.")
```

#### Evaluation Models
```python
class EvaluationResult(BaseModel):
    """The result of evaluating a single answer against a rubric."""
    question_text: str = Field(description="The question that was answered.")
    score: float = Field(description="The final calculated score for the answer, typically out of 100.")
    justification: str = Field(description="A detailed, overall explanation of how the score was determined based on the rubric criteria.")
    criterion_feedback: List[CriterionFeedback] = Field(description="A list of specific feedback for each criterion in the rubric.")
```

### 2. AI Models Configuration

```python
# Initialize Google Gemini models
llm = GoogleGenerativeAI(model="gemini-2.0-flash", timeout=600.0, temperature=0.2)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")
VECTOR_DIM = 768
```

## Detailed Function Documentation

### Document Indexing Functions

#### `index_source_documents_structured()`
**Purpose**: Processes PDF documents and stores them in vector database for semantic search.

**Process**:
1. **Document Loading**: Uses LangChain's DirectoryLoader to recursively load PDF files
2. **Metadata Extraction**: Extracts course, module, and topic information from file paths
3. **Text Chunking**: Splits documents into 1000-character chunks with 200-character overlap
4. **Vector Storage**: Creates embeddings and stores in Zilliz collections

**Code Example**:
```python
def index_source_documents_structured():
    # Setup the loader for PDF files
    loader = DirectoryLoader(
        path=root_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    
    # Process documents and add metadata
    for doc in documents:
        source_path = doc.metadata.get('source')
        if source_path:
            p = Path(source_path)
            doc.metadata['course_name'] = p.parent.parent.name
            doc.metadata['module_name'] = p.parent.name
            doc.metadata['file_name'] = p.name
```

**Collection Structure**: Each course-module combination gets its own collection with partitions for different topics.

#### `get_vector_store(course: str, module: str)`
**Purpose**: Creates a connection to a specific Milvus collection for retrieval.

**Parameters**:
- `course`: Course name
- `module`: Module name

**Returns**: Configured Milvus vector store instance

### Learning Objective Generation

#### `get_learning_objectives_sync(course: str, module: str)`
**Purpose**: Generates 4-6 measurable learning objectives from course content.

**Process**:
1. **Content Retrieval**: Searches vector store for relevant content
2. **AI Generation**: Uses Gemini to create learning objectives
3. **Validation**: Ensures objectives are measurable and clear

**Code Example**:
```python
def get_learning_objectives_sync(course: str, module: str) -> list[str]:
    vector_store = get_vector_store(course, module)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 25}
    )
    
    prompt_template = """
    You are an expert instructional designer tasked with creating learning objectives.
    Based on the following content from a course module, generate between 4 and 6 clear,
    concise, and measurable learning objectives.
    
    A learning objective should describe what a student will be able to DO after completing the module.
    
    Context: {context}
    {format_instructions}
    """
```

### Question Generation

#### `generate_questions(course: str, module: str, learning_objective: str, bloom_level: str, objective_id: int, cur)`
**Purpose**: Generates assessment questions for specific learning objectives and Bloom's levels.

**Key Features**:
- **Duplication Prevention**: Checks existing questions to avoid repetition
- **Context-Aware**: Uses relevant course content for question generation
- **Bloom's Alignment**: Ensures questions match the specified cognitive level

**Process**:
1. **Existing Question Check**: Queries database for previously generated questions
2. **Context Retrieval**: Gets relevant content from vector store
3. **Question Generation**: Creates 2 new questions using AI
4. **Database Storage**: Saves questions to assessments table

**Code Example**:
```python
def generate_questions(course: str, module: str, learning_objective: str, bloom_level: str, objective_id: int, cur):
    # Fetch existing questions to avoid duplication
    cur.execute(
        "SELECT question_text FROM assessments WHERE objective_id = %s AND bloom_level = %s;",
        (objective_id, bloom_level)
    )
    existing_questions = [row[0] for row in cur.fetchall()]
    
    # Generate new questions with context awareness
    prompt_template = """
    Generate 2 new and distinct assessment questions based on the provided context.
    The new questions must be tailored to the specific learning objective and Bloom's Taxonomy level.
    
    Crucially, you MUST NOT generate questions that are the same as or too similar to the "EXISTING QUESTIONS" listed below.
    
    Learning Objective: {learning_objective}
    Bloom's Taxonomy Level: {bloom_level}
    
    EXISTING QUESTIONS (DO NOT REPEAT THESE):
    {existing_questions}
    
    RELEVANT CONTEXT:
    {context}
    """
```

### Rubric Generation

#### `generate_evaluation_rubrics(course: str, module: str, learning_objective: str, bloom_level: str, question: str)`
**Purpose**: Creates evaluation rubrics with weighted criteria for each question.

**Features**:
- **Multiple Criteria**: Generates 2-5 evaluation criteria per question
- **Weighted Scoring**: Assigns weights that sum to 1.0
- **Content-Aligned**: Criteria based on course content and learning objectives

**Output Structure**:
```python
class Rubric(BaseModel):
    question_text: str
    criteria: List[Criterion]  # Each criterion has description and weight
```

### Answer Evaluation

#### `evaluate_answer_sync(question: str, user_answer: str, rubric: Rubric)`
**Purpose**: Evaluates student answers against generated rubrics using AI.

**Evaluation Process**:
1. **Rubric Application**: Applies each criterion to the student's answer
2. **Score Calculation**: Generates weighted score out of 100
3. **Detailed Feedback**: Provides specific feedback for each criterion
4. **Justification**: Explains the overall scoring decision

**Code Example**:
```python
def evaluate_answer_sync(question: str, user_answer: str, rubric: Rubric) -> EvaluationResult:
    prompt_template = """
    You are an expert teaching assistant. Your task is to evaluate a student's answer 
    based on a provided question and its evaluation rubric.
    
    You must provide a final score out of 100, a detailed justification for the score, 
    and specific, constructive feedback for EACH criterion in the rubric.
    
    ASSESSMENT QUESTION: {question}
    EVALUATION RUBRIC: {rubric}
    STUDENT'S ANSWER: {user_answer}
    """
```

### Study Content Generation

#### `get_suggested_content_sync(course: str, module: str, learning_objective: str)`
**Purpose**: Generates personalized study content for students who need improvement.

**Features**:
- **Targeted Content**: Focuses on specific learning objectives where student struggled
- **Relevant Material**: Pulls from course content related to the objective
- **Concise Summary**: Provides key concepts in digestible format

### Database Management

#### `setup_database()`
**Purpose**: Creates all necessary database tables with proper relationships and constraints.

**Tables Created**:
- `courses`: Stores course information
- `modules`: Links modules to courses
- `learning_objectives`: Stores generated learning objectives
- `assessments`: Stores questions with Bloom's level classification
- `rubric_criteria`: Stores evaluation criteria for each question
- `user_answers`: Stores student responses
- `evaluation_results`: Stores AI evaluation results
- `criterion_feedback`: Stores detailed feedback per criterion

#### `get_db_connection()`
**Purpose**: Establishes PostgreSQL database connection with error handling.

### Utility Functions

#### `sanitize_name(name: str)`
**Purpose**: Converts strings to valid Milvus collection/partition names.

**Process**:
- Removes special characters
- Ensures valid starting character
- Limits length to 255 characters

**Code Example**:
```python
def sanitize_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    if not re.match(r'^[a-zA-Z_]', name):
        name = '_' + name
    return name[:255]
```

#### `format_docs(docs: List)`
**Purpose**: Formats retrieved documents into a single context string for AI processing.

## Usage Guide

### 1. Initial Setup

1. **Run the system**:
```bash
python main.py
```

2. **Choose indexing**: When prompted, decide whether to index new documents:
```
Do you want to run the indexing process? (y/n): y
```

3. **Provide course details**:
```
Enter the course name (e.g., 'Reacts'): React Development
Enter the module name (e.g., '01 - Getting started with React'): Introduction to React
```

### 2. Learning Objective Selection

The system will either:
- **Use existing objectives** from the database, or
- **Generate new objectives** if none exist

Example output:
```
Please select a learning objective to generate questions for:
  1. Understand the core concepts of React components
  2. Apply JSX syntax in React applications
  3. Create functional components with props
  4. Implement state management using hooks
```

### 3. Bloom's Taxonomy Level Selection

Choose the cognitive level for question generation:
```
Please select a Bloom's Taxonomy level:
  1. Remember
  2. Understand
  3. Apply
  4. Analyze
  5. Evaluate
  6. Create
```

### 4. Question Generation and Display

The system generates 2 questions with detailed rubrics:
```
--- Generated Assessment ---

--- Question 1 ---
Explain how React components communicate through props and provide an example of parent-child component interaction.

  --- Rubric ---
  - Understanding of props concept (Weight: 0.4)
  - Code example quality (Weight: 0.3)
  - Explanation clarity (Weight: 0.3)
```

### 5. Answer Submission

Provide the path to a text file containing your answers:
```
Please provide the path to your text file containing the answers for the questions above:
> /path/to/answers.txt
```

### 6. Evaluation Results

The system provides detailed feedback:
```
--- Evaluation for Question 1 ---
Your Answer: Props are used to pass data from parent to child components...

  --- Evaluation Result ---
  Score: 85/100
  Justification: The answer demonstrates good understanding of props concept...
  Feedback per Criterion:
    - Understanding of props concept: Excellent explanation of the props mechanism
    - Code example quality: Good example but could include more detail
    - Explanation clarity: Clear and well-structured explanation
```

## File Structure

```
assessment-system/
├── main.py                 # Main application file
├── Parse_Answers_File.py   # Answer file parsing utility
├── requirements.txt        # Python dependencies
├── README.md              # This documentation
└── Course2/               # Course content directory
    ├── Course1/
    │   ├── Module1/
    │   │   ├── document1.pdf
    │   │   └── document2.pdf
    │   └── Module2/
    └── Course2/
```

### Answer File Format

Create a text file with answers separated by question markers:
```
Question 1:
Your answer to the first question goes here...

Question 2:
Your answer to the second question goes here...
```

## Troubleshooting

### Common Issues

1. **Database Connection Error**:
   - Verify PostgreSQL is running
   - Check connection credentials in `get_db_connection()`
   - Ensure database exists

2. **Vector Store Connection Error**:
   - Verify Zilliz credentials
   - Check network connectivity
   - Ensure collection exists after indexing

3. **API Rate Limits**:
   - Google API has rate limits for Gemini models
   - Implement delays if encountering rate limit errors
   - Monitor API usage in Google Cloud Console

4. **Memory Issues with Large Documents**:
   - Reduce chunk size in text splitter
   - Process documents in smaller batches
   - Increase system memory allocation

### Error Handling

The system includes comprehensive error handling:
- Database transaction rollbacks on errors
- API timeout handling (600 seconds)
- File path validation
- Missing document handling

### Performance Optimization

1. **Batch Processing**: Process multiple documents simultaneously
2. **Caching**: Store frequently accessed content in memory
3. **Indexing**: Create database indexes on frequently queried fields
4. **Connection Pooling**: Use connection pooling for database access

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the detailed function documentation above

---

*This README provides comprehensive documentation for the Assessment Generation System. For additional technical details, refer to the inline code comments and function docstrings.*