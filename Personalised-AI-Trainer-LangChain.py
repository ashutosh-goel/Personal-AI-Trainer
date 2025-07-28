import os
import re
from pathlib import Path
from typing import List
from collections import defaultdict
import platform

from pydantic import BaseModel, Field
from pymilvus import MilvusClient, DataType

# LangChain Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus.vectorstores import Milvus
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

import psycopg2
from psycopg2 import sql

# Local import remains the same
import Parse_Answers_File

# --- Pydantic Objects -----

class Questions(BaseModel):
    '''Questions based on each Bloom's Taxonomy level for each learning objectives.'''
    objective: str = Field(description="Learning objective for which questions are generated")
    bloom_level: str = Field(description="Bloom's Taxonomy level for which questions are generated")
    questions: List[str] = Field(description="List of questions for the given Learning Objective and Bloom Level")

class LearningObjectives(BaseModel):
    """A list of learning objectives for a course module."""
    objectives: List[str] = Field(description="A list of 4 to 6 clear and measurable learning objectives.")

class Criterion(BaseModel):
    """A specific criterion for evaluating an answer (e.g., Accuracy, Completeness)."""
    description: str = Field(description="The criterion that is being evaluated.")
    weight: float = Field(description="The weight of this criterion in the total score (e.g., 0.5 for 50%).")
    #levels: List[ScoringLevel] = Field(description="A list of scoring levels from poor to excellent.")

class Rubric(BaseModel):
    """A complete evaluation rubric for a single question."""
    question_text: str = Field(description="The full text of the question being evaluated.")
    criteria: List[Criterion] = Field(description="A list of criteria to evaluate the answer against.")

class CriterionFeedback(BaseModel):
    """Feedback for a specific criterion."""
    criterion: str = Field(description="The name of the criterion being evaluated.")
    feedback: str = Field(description="Specific, constructive feedback on the user's answer for this criterion.")

class EvaluationResult(BaseModel):
    """The result of evaluating a single answer against a rubric."""
    question_text: str = Field(description="The question that was answered.")
    score: float = Field(description="The final calculated score for the answer, typically out of 100.")
    justification: str = Field(description="A detailed, overall explanation of how the score was determined based on the rubric criteria.")
    criterion_feedback: List[CriterionFeedback] = Field(description="A list of specific feedback for each criterion in the rubric.")

class SuggestedContent(BaseModel):
    """Targeted content to help a user improve on a learning objective."""
    learning_objective: str = Field(description="The learning objective the user needs to work on.")
    suggested_summary: str = Field(description="A concise summary of the key concepts from the source material related to the learning objective.")


# --- 1. Global Setup & Configuration ---
api_key = "AIzaSyAlABTPGnaSlYkrWsmUwfmPjyq9tIjMsOs" #"AIzaSyAv15Rnl3h1SjUT2Pa7d1Ci5LJwJ_0qctE" #"AIzaSyAlABTPGnaSlYkrWsmUwfmPjyq9tIjMsOs"
os.environ["GOOGLE_API_KEY"] = api_key

# Models and Embeddings
llm = GoogleGenerativeAI(model="gemini-2.0-flash", timeout=600.0, temperature=0.2)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")
VECTOR_DIM = 768

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"

# Milvus/Zilliz Configuration
ZILLIZ_URI = "https://in03-540f880b2b2a98e.serverless.gcp-us-west1.cloud.zilliz.com"
ZILLIZ_TOKEN = "9326f5b4421b923f2649bda2c60f9b2f8fe20339831a0eed5c8d8a5e60d410fa511c1c8ad944b6ea6a4a645f622d367e0b5d42b2"

try:
    milvus_client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
    print("Successfully connected to Zilliz with pymilvus client.")
except Exception as e:
    print(f"Failed to connect to Zilliz with pymilvus client: {e}")
    exit()

# --- Database Configuration & Setup ---

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    
    return psycopg2.connect(
        dbname= "questionsdb",
        user="cubastion",
        password="999forever",
        host="localhost", # or your db host
        port="5432"      # or your db port
    )

def setup_database():
    """Creates the necessary tables in the database if they don't already exist."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Table for Courses
    cur.execute("""
        CREATE TABLE IF NOT EXISTS courses (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) UNIQUE NOT NULL
        );
    """)
    
    # Table for Modules, linked to Courses
    cur.execute("""
        CREATE TABLE IF NOT EXISTS modules (
            id SERIAL PRIMARY KEY,
            course_id INTEGER REFERENCES courses(id) ON DELETE CASCADE,
            name VARCHAR(255) NOT NULL,
            UNIQUE(course_id, name)
        );
    """)

    # Table for Learning Objectives, linked to Modules
    cur.execute("""
        CREATE TABLE IF NOT EXISTS learning_objectives (
            id SERIAL PRIMARY KEY,
            module_id INTEGER REFERENCES modules(id) ON DELETE CASCADE,
            objective_text TEXT NOT NULL
        );
    """)

    # Table for Assessments (Questions), linked to Learning Objectives
    cur.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            id SERIAL PRIMARY KEY,
            objective_id INTEGER REFERENCES learning_objectives(id) ON DELETE CASCADE,
            bloom_level VARCHAR(50) NOT NULL,
            question_text TEXT NOT NULL
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS rubric_criteria (
            id SERIAL PRIMARY KEY,
            assessment_id INTEGER REFERENCES assessments(id) ON DELETE CASCADE,
            description TEXT NOT NULL,
            weight FLOAT NOT NULL
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_answers (
            id SERIAL PRIMARY KEY,
            assessment_id INTEGER REFERENCES assessments(id) ON DELETE CASCADE,
            answer_text TEXT NOT NULL,
            submitted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_results (
            id SERIAL PRIMARY KEY,
            answer_id INTEGER UNIQUE REFERENCES user_answers(id) ON DELETE CASCADE,
            score FLOAT NOT NULL,
            justification TEXT NOT NULL
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS criterion_feedback (
            id SERIAL PRIMARY KEY,
            evaluation_id INTEGER REFERENCES evaluation_results(id) ON DELETE CASCADE,
            criterion_description TEXT NOT NULL,
            feedback TEXT NOT NULL
        );
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Database tables checked and set up successfully.")


def sanitize_name(name: str) -> str:
    """Sanitizes a string to be a valid Milvus collection or partition name."""
    name = name.strip()
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    if not re.match(r'^[a-zA-Z_]', name):
        name = '_' + name
    return name[:255]

def format_docs(docs: List) -> str:
    """Helper function to format retrieved documents into a single string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# --- 3. Data Indexing Function (LangChain Version) ---
def index_source_documents_structured():
    """
    Reads source documents, chunks them, and stores them in Zilliz
    using LangChain loaders and splitters.
    """
    print("\n--- Starting Structured Indexing of Source Documents (LangChain) ---")

    # Group documents by their target collection and partition
    docs_by_partition = {}
    root_dir = Path(r"C:\Course2")

    glob_pattern = "**/*.pdf"

    # Setup the loader
    loader = DirectoryLoader(
        path=root_dir,
        glob=glob_pattern,
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )

    documents = loader.lazy_load()

    processed_docs = []

    # 2. Iterate through each document to add custom metadata
    for doc in documents:
        # Get the full source path from the existing metadata
        source_path = doc.metadata.get('source')

        if source_path:
            # Use pathlib to easily handle the path
            p = Path(source_path)

            # 3. Extract the parts based on the structure
            # Assumes .../course_name/module_name/file.pdf
            module_name = p.parent.name
            course_name = p.parent.parent.name
            file_name = p.name

            # 4. Update the metadata dictionary
            doc.metadata['course_name'] = course_name
            doc.metadata['module_name'] = module_name
            doc.metadata['file_name'] = file_name

        processed_docs.append(doc)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True # Helpful for identifying chunk position
    )

    # The split_documents method splits the text and preserves the metadata
    chunks = text_splitter.split_documents(processed_docs)

    # --- 3. Group Chunks by (Collection, Partition) ---
    # This is the most efficient way to process the data
    grouped_data = defaultdict(list)
    for chunk in chunks:
        course = chunk.metadata.get("course_name")
        module = chunk.metadata.get("module_name")
        topic = chunk.metadata.get("file_name")

        if not all([course, module, topic]):
            print(f"Skipping chunk with missing metadata: {chunk.metadata.get('source')}")
            continue

        collection_name = sanitize_name(f"{course}_{module}_source_docs")
        partition_name = sanitize_name(topic)
        
        grouped_data[(collection_name, partition_name)].append(chunk)

    # --- 4. Process Each Group and Insert into Milvus ---
    # Keep track of created collections to avoid redundant checks
    created_collections = set()

    for (collection_name, partition_name), chunk_list in grouped_data.items():
        
        # A. Create the Collection if it's the first time we've seen it
        if collection_name not in created_collections:
            if not milvus_client.has_collection(collection_name):
                print(f"Creating new collection: '{collection_name}'")
                # Your schema definition here...
                schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
                schema.add_field("id", DataType.INT64, is_primary=True)
                schema.add_field("text", DataType.VARCHAR, max_length=4000) # Store the text chunk
                schema.add_field("course", DataType.VARCHAR, max_length=256)
                schema.add_field("module", DataType.VARCHAR, max_length=256)
                schema.add_field("topic", DataType.VARCHAR, max_length=256)
                #schema.add_field("metadata", DataType.JSON)
                schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
                milvus_client.create_collection(collection_name=collection_name, schema=schema)

                print(f"Creating index for collection '{collection_name}'...")
                index_params = milvus_client.prepare_index_params()
                index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="L2")
                milvus_client.create_index(collection_name=collection_name, index_params=index_params)
            
            created_collections.add(collection_name)

        # B. Create the Partition within the collection
        if not milvus_client.has_partition(collection_name, partition_name):
            print(f"Creating partition '{partition_name}' in collection '{collection_name}'")
            milvus_client.create_partition(collection_name, partition_name)

        # C. Embedding
        # 1. Extract all the text content into a list
        texts_to_embed = [chunk.page_content for chunk in chunk_list]

        # 2. Call embed_documents ONCE with the list of all texts
        all_embeddings = embeddings.embed_documents(texts_to_embed)

        # D. Prepare and Insert Data into the correct partition
        print(f"Preparing to insert {len(chunk_list)} chunks into partition '{partition_name}'...")
        
        data_to_insert = []
        for i,chunk in enumerate(chunk_list):
            data_to_insert.append({
                "text": chunk.page_content,
                "course": chunk.metadata.get("course_name"),
                "module": chunk.metadata.get("module_name"),
                "topic": chunk.metadata.get("file_name"),
                "embedding": all_embeddings[i] # Your embedding logic
            })
            
        milvus_client.insert(
            collection_name=collection_name,
            data=data_to_insert,
            partition_name=partition_name # <-- This is the key change
        )

    print("\n Vector storage Processing complete")

def get_vector_store(course: str, module: str) -> Milvus:
    """Helper function to connect to a specific Milvus collection with improved configuration."""
    collection_name = sanitize_name(f"{course}_{module}_source_docs")
    if not milvus_client.has_collection(collection_name):
        raise ValueError(f"Collection '{collection_name}' not found. Please index the documents first.")
    
    # Load the collection to ensure it's available
    milvus_client.load_collection(collection_name)
    
    return Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_args={"uri": ZILLIZ_URI, "token": ZILLIZ_TOKEN, "secure": True},
        auto_id=True,
        text_field="text",
        vector_field="embedding",
        # Add these parameters to fix the ID issue
        primary_field="id",
        metadata_field=None
    )

# --- SOLUTION 2: Use sync version instead of async ---
def get_learning_objectives_sync(course: str, module: str) -> list[str]:
    """Synchronous version of get_learning_objectives"""
    print(f"\n--- Sync Generating Learning Objectives for {course} - {module} ---")
    
    try:
        vector_store = get_vector_store(course, module)
        
        # Use similarity search instead of MMR to avoid the KeyError
        retriever = vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={'k': 25}
        )
        
        parser = PydanticOutputParser(pydantic_object=LearningObjectives)

        prompt_template = """
        You are an expert instructional designer tasked with creating learning objectives.
        Based on the following content from a course module, generate between 4 and 6 clear, 
        concise, and measurable learning objectives.

        A learning objective should describe what a student will be able to DO after completing the module.

        Context: {context}

        {format_instructions}
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)

        chain = (
            {"context": retriever | format_docs}
            | prompt.partial(format_instructions=parser.get_format_instructions())
            | llm
            | parser
        )
        
        # The input query here is a placeholder to trigger the retriever.
        relevant_query = f"A summary of the content for the course '{course}' in the module '{module}'"
        response = chain.invoke(relevant_query)
        
        print("--- Objectives Generated ---")
        return response.objectives
        
    except Exception as e:
        print(f"Error in get_learning_objectives_sync: {e}")
        exit()

def generate_questions(course: str, module: str, learning_objective: str, bloom_level: str, objective_id: int, cur: psycopg2.extensions.cursor) -> Questions:
    """Synchronously generates questions for a given objective and Bloom's level."""
    print(f"\n--- Generating '{bloom_level}' questions for: '{learning_objective}' ---")

    # Fetch existing questions for this specific objective and level
    cur.execute(
        "SELECT question_text FROM assessments WHERE objective_id = %s AND bloom_level = %s;",
        (objective_id, bloom_level)
    )
    existing_questions_rows = cur.fetchall()
    existing_questions = [row[0] for row in existing_questions_rows]
    
    if existing_questions:
        print(f"Found {len(existing_questions)} existing questions. Will generate new ones.")

    vector_store = get_vector_store(course, module)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 10})
    parser = PydanticOutputParser(pydantic_object=Questions)

    print("Retrieving relevant documents...")
    docs = retriever.invoke(learning_objective)
    context = format_docs(docs)

    prompt_template = """You are an expert in creating educational assessments. Your task is to generate 2 new and distinct assessment questions based on the provided context.
    
    The new questions must be tailored to the specific learning objective and Bloom's Taxonomy level.

    Crucially, you MUST NOT generate questions that are the same as or too similar to the "EXISTING QUESTIONS" listed below. If the context is exhausted, paraphrase the existing questions to create new-sounding ones.

    {format_instructions}

    Learning Objective: {learning_objective}
    Bloom's Taxonomy Level: {bloom_level}

    ---
    EXISTING QUESTIONS (DO NOT REPEAT THESE):
    {existing_questions}
    ---
    RELEVANT CONTEXT:
    {context}
    ---
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        prompt.partial(format_instructions= parser.get_format_instructions())
        | llm
        | parser
    )

    print("Invoking LLM to generate questions...")
    response = chain.invoke({
        "context": context,
        "learning_objective": learning_objective,
        "bloom_level": bloom_level,
        "existing_questions": "\n".join(f"- {q}" for q in existing_questions) if existing_questions else "None"
    })
    return response

def generate_evaluation_rubrics(course: str, module: str, learning_objective: str, bloom_level: str, question: str):
    """Synchronously generates evaluation rubrics for a given objective and Bloom's level."""
    print(f"\n--- Generating evaluation rubric for: '{learning_objective}' ---")

    # Generate evaluation rubrics for each new generated question
    vector_store = get_vector_store(course, module)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 10})
    parser = PydanticOutputParser(pydantic_object=Rubric)

    print("Retrieving relevant documents...")
    docs = retriever.invoke(learning_objective)
    context = format_docs(docs)

    prompt_template = """You are an expert in creating educational assessments. Your task is to generate an evaluation rubric for the provided assessment question.
    
    The rubric should include 2 to 5 weighted criterias that will be used to evaluate student responses to the question. Each criterion should be measurable and aligned with the learning objective.
    The sum of weights must equal 1.0.

    {format_instructions}

    Assessment Question: {question}
    Learning Objective: {learning_objective}
    Bloom's Taxonomy Level: {bloom_level}


    RELEVANT CONTEXT:
    {context}
    ---
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        prompt.partial(format_instructions= parser.get_format_instructions())
        | llm
        | parser
    )

    print("Invoking LLM to generate rubric...")
    response = chain.invoke({
        "question": question,
        "learning_objective": learning_objective,
        "bloom_level": bloom_level,
        "context": context
    })

    return response

def evaluate_answer_sync(question: str, user_answer: str, rubric: Rubric) -> EvaluationResult:
    """Evaluates a user's answer against a rubric using an LLM."""
    print(f"\n--- Evaluating Answer for: '{question[:50]}...' ---")

    parser = PydanticOutputParser(pydantic_object=EvaluationResult)

    rubric_text = "\n".join(f"- {c.description} (Weight: {c.weight})" for c in rubric.criteria)

    prompt_template = """You are an expert teaching assistant. Your task is to evaluate a student's answer based on a provided question and its evaluation rubric.

    You must provide a final score out of 100, a detailed justification for the score, and specific, constructive feedback for EACH criterion in the rubric.

    {format_instructions}

    ---
    ASSESSMENT QUESTION:
    {question}
    ---
    EVALUATION RUBRIC:
    {rubric}
    ---
    STUDENT'S ANSWER:
    {user_answer}
    ---
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        prompt.partial(format_instructions= parser.get_format_instructions())
        | llm
        | parser
    )

    print("Invoking LLM to evaluate the answer...")
    response = chain.invoke({
        "question": question,
        "rubric": rubric_text,
        "user_answer": user_answer
    })
    return response

def get_suggested_content_sync(course: str, module: str, learning_objective: str) -> SuggestedContent:
    """Generates a targeted summary of content for a user who failed an assessment."""
    print(f"\n--- Generating Suggested Study Content for: '{learning_objective}' ---")
    
    vector_store = get_vector_store(course, module)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 5})

    print("Retrieving relevant documents for study guide...")
    docs = retriever.invoke(learning_objective)
    context = format_docs(docs)

    parser = PydanticOutputParser(pydantic_object=SuggestedContent)

    prompt_template = """You are an expert tutor. A student has failed an assessment on a specific learning objective. 
    
    Your task is to synthesize the provided "RELEVANT CONTENT" into a clear and concise summary. This summary should directly help the student understand the key concepts they missed, enabling them to pass the assessment next time.

    {format_instructions}

    Learning Objective the Student Failed: {learning_objective}

    ---
    RELEVANT CONTENT FROM THE COURSE:
    {context}
    ---
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        prompt.partial(format_instructions= parser.get_format_instructions())
        | llm
        | parser
    )

    print("Invoking LLM to generate study guide...")
    response = chain.invoke({
        "context": context,
        "learning_objective": learning_objective
    })
    return response


def main_sync():
    """Synchronous main function"""
    print("--- Assessment Generation System Initializing ---")

    # Setup the database tables on startup
    setup_database()

    run_indexing = input("Do you want to run the indexing process? (y/n): ").lower()
    if run_indexing == 'y':
        index_source_documents_structured()
    
    # Step 2: Get user input for course and module
    course = input("\nEnter the course name (e.g., 'Reacts'): ").strip()
    module = input("Enter the module name (e.g., '01 - Getting started with React'): ").strip()

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Get or create Course and Module IDs
        cur.execute("INSERT INTO courses (name) VALUES (%s) ON CONFLICT (name) DO UPDATE SET name=EXCLUDED.name RETURNING id;", (course,))
        course_id = cur.fetchone()[0]
        cur.execute("INSERT INTO modules (course_id, name) VALUES (%s, %s) ON CONFLICT (course_id, name) DO UPDATE SET name=EXCLUDED.name RETURNING id;", (course_id, module))
        module_id = cur.fetchone()[0]
        conn.commit()

        # Check for existing learning objectives
        cur.execute("SELECT id, objective_text FROM learning_objectives WHERE module_id = %s;", (module_id,))
        objectives_from_db = cur.fetchall()

        if objectives_from_db:
            print("\nFound existing learning objectives in the database.")
            objective_db_ids = {text: obj_id for obj_id, text in objectives_from_db}
            objectives_text = list(objective_db_ids.keys())
        else:
            print("\nNo existing learning objectives found. Generating new ones...")
            objectives_text = get_learning_objectives_sync(course, module)
            if not objectives_text:
                print("Could not generate learning objectives. Exiting.")
                return

            # Save newly generated objectives to DB
            objective_db_ids = {}
            for obj_text in objectives_text:
                cur.execute("INSERT INTO learning_objectives (module_id, objective_text) VALUES (%s, %s) ON CONFLICT (module_id, objective_text) DO NOTHING RETURNING id;", (module_id, obj_text))
                result = cur.fetchone()
                if result:
                    objective_db_ids[obj_text] = result[0]
            conn.commit()
            print("Successfully saved new learning objectives to the database.")

        
        print("\nPlease select a learning objective to generate questions for:")
        for i, obj in enumerate(objectives_text):
            print(f"  {i+1}. {obj}")

        selected_objective_text = ""
        while True:
            try:
                selected_idx = int(input(f"\nEnter objective number (1-{len(objectives_text)}): ")) - 1
                if 0 <= selected_idx < len(objectives_text):
                    selected_objective_text = objectives_text[selected_idx]
                    break
                else:
                    print("Invalid number.")
            except ValueError:
                print("Please enter a valid number.")

        bloom_levels = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
        print("\nPlease select a Bloom's Taxonomy level:")
        for i, level in enumerate(bloom_levels):
            print(f"  {i+1}. {level}")
        
        selected_level = ""
        while True:
            try:
                selected_level_idx = int(input(f"\nEnter level number (1-{len(bloom_levels)}): ")) - 1
                if 0 <= selected_level_idx < len(bloom_levels):
                    selected_level = bloom_levels[selected_level_idx]
                    break
                else:
                    print("Invalid number.")
            except ValueError:
                print("Please enter a valid number.")
        
        selected_objective_id = objective_db_ids[selected_objective_text]
        assessment = generate_questions(course, module, selected_objective_text, selected_level, selected_objective_id, cur)

        # Store generated questions and rubrics temporarily
        questions_to_evaluate = []
        for q_text in assessment.questions:
            cur.execute("INSERT INTO assessments (objective_id, bloom_level, question_text) VALUES (%s, %s, %s) RETURNING id;", (selected_objective_id, assessment.bloom_level, q_text))
            assessment_id = cur.fetchone()[0]
            rubric = generate_evaluation_rubrics(course, module, selected_objective_text, selected_level, q_text)
            for criterion in rubric.criteria:
                cur.execute("INSERT INTO rubric_criteria (assessment_id, description, weight) VALUES (%s, %s, %s);", (assessment_id, criterion.description, criterion.weight))
            conn.commit()
            questions_to_evaluate.append({"id": assessment_id, "text": q_text, "rubric": rubric})

        # Display all questions and rubrics first
        print("\n\n--- Generated Assessment ---")
        for i, q_data in enumerate(questions_to_evaluate):
            print(f"\n--- Question {i+1} ---")
            print(q_data["text"])
            print("\n  --- Rubric ---")
            for criterion in q_data["rubric"].criteria:
                 print(f"  - {criterion.description} (Weight: {criterion.weight})")
        
        # Get user answers from file
        print("\n--------------------------------------")
        answer_file_path = input("Please provide the path to your text file containing the answers for the questions above:\n> ")
        user_answers = Parse_Answers_File.parse_answers_file(answer_file_path)

        if len(user_answers) != len(questions_to_evaluate):
            print(f"\nWarning: Found {len(user_answers)} answers, but {len(questions_to_evaluate)} questions were generated. Evaluating the answers that were provided.")

        # Evaluate each answer and store scores
        scores = []
        for i, q_data in enumerate(questions_to_evaluate):
            if i < len(user_answers):
                user_answer = user_answers[i]
                assessment_id = q_data["id"]
                q_text = q_data["text"]
                rubric = q_data["rubric"]

                cur.execute("INSERT INTO user_answers (assessment_id, answer_text) VALUES (%s, %s) RETURNING id;", (assessment_id, user_answer))
                answer_id = cur.fetchone()[0]
                conn.commit()

                evaluation = evaluate_answer_sync(q_text, user_answer, rubric)
                scores.append(evaluation.score)

                cur.execute("INSERT INTO evaluation_results (answer_id, score, justification) VALUES (%s, %s, %s) RETURNING id;", (answer_id, evaluation.score, evaluation.justification))
                evaluation_id = cur.fetchone()[0]
                for fb in evaluation.criterion_feedback:
                    cur.execute("INSERT INTO criterion_feedback (evaluation_id, criterion_description, feedback) VALUES (%s, %s, %s);", (evaluation_id, fb.criterion, fb.feedback))
                conn.commit()

                print(f"\n--- Evaluation for Question {i+1} ---")
                print(f"Your Answer: {user_answer}")
                print("\n  --- Evaluation Result ---")
                print(f"  Score: {evaluation.score}/100")
                print(f"  Justification: {evaluation.justification}")
                print("  Feedback per Criterion:")
                for fb in evaluation.criterion_feedback:
                    print(f"    - {fb.criterion}: {fb.feedback}")
            else:
                print(f"\n--- No answer provided for Question {i+1}. ---")

        if scores:
            average_score = sum(scores) / len(scores)
            print("\n\n--- FINAL RESULT ---")
            print(f"Your average score for the '{selected_level}' level is: {average_score:.2f}%")
            if average_score >= 50:
                print("Result: PASS")
            else:
                print("Result: FAIL")
                
            suggested_content = get_suggested_content_sync(course, module, selected_objective_text)
            print("\n--- Suggested Study Content ---")
            print("Based on your performance, here are some key concepts from the course material to review:")
            print(suggested_content.suggested_summary)

        print("\n--------------------------------------")
        print(f"Successfully evaluated and saved {len(user_answers)} answers.")

    except psycopg2.Error as e:
        print(f"\nDatabase Error: {e}")
        print("Please check your database connection details and ensure PostgreSQL is running.")
    except ValueError as e:
        print(f"\nError: {e}")
        print("Please ensure the course/module names are correct and the documents have been indexed.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        if conn:
            cur.close()
            conn.close()

if __name__ == "__main__":
        
    main_sync()