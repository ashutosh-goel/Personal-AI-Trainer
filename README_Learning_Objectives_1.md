# Learning Objectives AI Trainer - Comprehensive Documentation

## Purpose and Overview

The `Learning_Objectives_1.py` file implements a sophisticated **Personalized AI Trainer** for educational content that automatically extracts learning objectives and generates comprehensive assessments from course materials. This system functions as an intelligent educational assistant that can:

- **Extract Learning Objectives**: Automatically identify and formulate broad learning objectives from course content
- **Generate Bloom's Taxonomy Questions**: Create questions across all six levels of Bloom's taxonomy for each learning objective
- **Create Evaluation Rubrics**: Generate detailed, weighted rubrics for assessment questions
- **Evaluate Student Responses**: Automatically assess student answers against generated rubrics with detailed feedback

The system is designed to work with structured educational content and provides a complete workflow from content analysis to student evaluation, making it an invaluable tool for educators and educational institutions.

## Technical Architecture

### RAG (Retrieval-Augmented Generation) Pipeline

The system implements a sophisticated RAG pipeline that combines document retrieval with large language model generation:

```python
# Core RAG Components
reader = SimpleDirectoryReader(
    input_dir=r"C:\Courses1",
    recursive=True,
    file_extractor={".pdf": PyMuPDFReader()},
    file_metadata=metadata
)

# Vector storage and indexing
index = VectorStoreIndex.from_documents(documents, show_progress=True)
query_engine = index.as_query_engine(llm=structured_llm)
```

#### Structured Metadata System

The system uses a hierarchical metadata structure to organize educational content:

```python
def metadata(filepath: str):
    path = Path(filepath)
    course = path.parts[-3]  # Course name (e.g., "NodeJS")
    module = path.parts[-2]  # Module name (e.g., "04 - Servers")
    topic = path.name       # Topic/file name (e.g., "express_basics.pdf")
    return {"course": course, "module": module, "topic": topic}
```

**Expected Directory Structure:**
```
C:\Courses1/
├── NodeJS/
│   ├── 01 - Introduction/
│   │   ├── nodejs_basics.pdf
│   │   └── installation_guide.pdf
│   ├── 02 - Modules/
│   └── 04 - Servers/
│       ├── express_framework.pdf
│       └── server_setup.pdf
├── Python/
│   ├── 01 - Basics/
│   └── 02 - Advanced/
```

#### LLM Integration

The system integrates with Google's Gemini models for enhanced performance:

```python
# Primary LLM Configuration
Settings.llm = GoogleGenAI(
    model="gemini-2.5-flash",
    request_timeout=600.0
)

# Embedding Model
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-exp-03-07",
    api_key=api_key
)
```

### Data Models and Workflow State

#### Core Data Structures

**Questions Model** - Structures questions by Bloom's taxonomy levels:
```python
class Questions(BaseModel):
    objective: str = Field(description="Learning objective")
    remember: List[str] = Field(description="Remember level questions")
    understand: List[str] = Field(description="Understand level questions")
    apply: List[str] = Field(description="Apply level questions")
    analyze: List[str] = Field(description="Analyze level questions")
    evaluate: List[str] = Field(description="Evaluate level questions")
    create: List[str] = Field(description="Create level questions")
```

**Evaluation System**:
```python
class EvaluationResult(BaseModel):
    question_text: str
    score: float = Field(description="Score out of 100")
    justification: str = Field(description="Detailed scoring explanation")
    criterion_feedback: List[Criterion_Feedback]
```

## Core Functionality

### 1. Learning Objective Extraction (`get_objectives` step)

The system automatically extracts broad learning objectives from course content:

```python
@step
async def get_objectives(self, ctx: Context, ev: GetObjectives) -> SetAssessment:
    query = f"Extract 3-6 broad learning objectives for {ev.module} module in {ev.course} course"
    response = query_engine1.query(query)
    await ctx.store.set("objectives", response.objectives)
```

**Example Output:**
```
Course: NodeJS
Module: 04 - Servers

Objectives:
  1. Understand the fundamentals of server-side programming with Node.js
  2. Apply Express.js framework to build web applications
  3. Implement RESTful API endpoints and middleware
  4. Analyze server performance and optimization techniques
```

### 2. Bloom's Taxonomy Question Generation (`assessment` step)

For each learning objective, the system generates 3 questions across all six Bloom's taxonomy levels:

```python
@step
async def assessment(self, ctx: Context, ev: SetAssessment) -> RubrikEvent:
    query = f"Generate 3 questions based on each Bloom's Taxonomy level for each learning objective"
    response = query_engine2.query(query).response
```

**Example Output Structure:**
```
Objective: Understand Express.js framework fundamentals

  Remember Questions:
    1. What is Express.js and what is its primary purpose?
    2. List the main components of an Express.js application
    3. Define middleware in the context of Express.js

  Apply Questions:
    1. Create a basic Express.js server that handles GET requests
    2. Implement middleware for request logging
    3. Build a simple REST API endpoint for user data
```

### 3. Rubric Generation (`evaluation_rubriks` step)

The system creates detailed, weighted evaluation rubrics for assessment questions:

```python
@step
async def evaluation_rubriks(self, ctx: Context, ev: RubrikEvent) -> EvalEvent:
    prompt = (
        f"Create a detailed, weighted evaluation rubric for the following question.\n"
        f"**Learning Objective:** {objective_text}\n"
        f"**Bloom's Taxonomy Level:** {bloom_level}\n"
        f"**Question:** {question_text}\n"
        f"The rubric must have 2-4 weighted criteria with weights summing to 1.0"
    )
```

### 4. Student Assessment (`input_answers` step)

The final step evaluates student responses against generated rubrics:

```python
@step
async def input_answers(self, ctx: Context, ev: EvalEvent) -> StopEvent:
    answers_file_path = input("Please provide the path to your answers file:")
    student_answers = Parse_Answers_File.parse_answers_file(answers_file_path)
    
    # Evaluate each answer against corresponding rubric
    for rubric, answer in zip(rubrics_data, student_answers):
        evaluation_response = query_engine4.query(evaluation_prompt).response
```

## Usage Instructions

### Basic Workflow Execution

```python
import asyncio
from Learning_Objectives_1 import Objective_Workflow

async def run_assessment():
    workflow = Objective_Workflow()
    await workflow.run(query={"course": "NodeJS", "module": "04 - Servers"})

# Execute the workflow
asyncio.run(run_assessment())
```

### Step-by-Step Usage

#### 1. Setup Course Materials
Organize your course materials in the expected directory structure:
```
C:\Courses1\{CourseName}\{ModuleName}\{materials.pdf}
```

#### 2. Configure the System
Update the course directory path and API keys:
```python
# Update input directory
reader = SimpleDirectoryReader(
    input_dir=r"C:\Your_Course_Directory",
    recursive=True,
    file_extractor={".pdf": PyMuPDFReader()},
    file_metadata=metadata
)

# Set your Google API key
api_key = "your_google_api_key_here"
```

#### 3. Run the Workflow
```python
# Execute for specific course and module
workflow = Objective_Workflow()
await workflow.run(query={
    "course": "Python Programming", 
    "module": "02 - Data Structures"
})
```

#### 4. Prepare Student Answers
Create a text file with student answers (one answer per line) and provide the path when prompted.

### Expected Input/Output Formats

**Input Query Format:**
```python
query = {
    "course": "Course Name",
    "module": "Module Name"
}
```

**Student Answers File Format:**
```
Answer to question 1 goes here...

Answer to question 2 goes here...

Answer to question 3 goes here...
```

**Output Evaluation Format:**
```json
{
  "question_text": "What is Express.js?",
  "score": 85.0,
  "justification": "The answer demonstrates good understanding...",
  "criterion_feedback": [
    {
      "criterion": "Accuracy",
      "feedback": "Correctly identified Express.js as a web framework"
    }
  ]
}
```

## File Structure and Dependencies

### Required Imports
```python
# Core dependencies
from pydantic import BaseModel, Field
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.workflow import Workflow, step, Context

# Custom modules
import Parse_Answers_File  # For parsing student answer files
```

### Installation Requirements
```bash
pip install llama-index
pip install pydantic
pip install pymupdf
pip install google-generativeai
```

### File Organization
```
project/
├── Learning_Objectives_1.py    # Main workflow implementation
├── Parse_Answers_File.py        # Student answer parsing utility
├── storage_L1/                  # Vector index storage
└── C:\Courses1/                 # Course materials directory
```

## Personalization Features

### Adaptive Course Context
The AI trainer adapts to different educational contexts through:

1. **Course-Specific Objectives**: Extracts objectives tailored to specific course content
2. **Module-Focused Assessment**: Generates questions relevant to particular modules
3. **Bloom's Taxonomy Alignment**: Ensures questions match appropriate cognitive levels

### Memory and State Management
The system maintains context throughout the workflow:

```python
# Store course context
await ctx.store.set("course", course_name)
await ctx.store.set("module", module_name)
await ctx.store.set("objectives", extracted_objectives)

# Retrieve context in later steps
objectives = await ctx.store.get("objectives", [])
```

### Customizable Assessment Criteria
The rubric generation can be customized for different evaluation needs:
- Weighted criteria based on learning importance
- Flexible scoring levels (poor to excellent)
- Subject-specific evaluation standards

## Troubleshooting Guide

### Common Issues and Solutions

**1. "No documents found" Error**
```python
# Check directory path and file permissions
reader = SimpleDirectoryReader(input_dir=r"C:\Courses1", recursive=True)
documents = reader.load_data()
print(f"Loaded {len(documents)} documents")
```

**2. API Key Issues**
```python
# Verify API key is set correctly
import os
print(f"API Key set: {'GOOGLE_API_KEY' in os.environ}")
```

**3. Storage Directory Issues**
```python
# Ensure storage directory exists and is writable
import os
Persist_Dir = "./storage_L1"
if not os.path.exists(Persist_Dir):
    os.makedirs(Persist_Dir)
```

**4. Student Answer File Parsing**
- Ensure answers file exists and is readable
- Check that Parse_Answers_File.py is in the same directory
- Verify file format matches expected structure

### Performance Optimization
- Use persistent storage to avoid re-indexing documents
- Implement caching for frequently accessed course materials
- Consider batch processing for multiple assessments

## Advanced Features

### Custom Question Types
Extend the Questions model to include custom question types:
```python
class ExtendedQuestions(Questions):
    practical: List[str] = Field(description="Hands-on practical questions")
    case_study: List[str] = Field(description="Case study questions")
```

### Multi-Language Support
The system can be extended to support multiple languages by configuring appropriate embedding models and LLMs for different languages.

### Integration Capabilities
The workflow can be integrated with:
- Learning Management Systems (LMS)
- Student Information Systems (SIS)
- Automated grading platforms
- Educational analytics tools

This comprehensive AI trainer provides a complete solution for automated educational content analysis, assessment generation, and student evaluation, making it an invaluable tool for modern educational environments.
