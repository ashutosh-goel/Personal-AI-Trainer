import os
import re
from pathlib import Path
from typing import List
from collections import defaultdict
import platform

from pydantic import BaseModel, Field
from pymilvus import MilvusClient, DataType

import streamlit as st

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

import asyncio

# This is the fix
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

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

# --- Database Configuration & Setup (with Streamlit caching) ---

@st.cache_resource
def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    return psycopg2.connect(
        dbname="questionsdb",
        user="cubastion",
        password="999forever",
        host="localhost",
        port="5432"
    )

def sanitize_name(name: str) -> str:
    """Sanitizes a string to be a valid Milvus collection or partition name."""
    name = name.strip()
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    if not re.match(r'^[a-zA-Z_]', name):
        name = '_' + name
    return name[:255]

import io

def parse_answers_from_uploaded_file(uploaded_file):
    """Parses answers from a Streamlit UploadedFile object."""
    if uploaded_file is None:
        return []
    # Read the file content as a string
    stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    content = stringio.read()
    # Split answers based on a delimiter, e.g., '---' on a new line
    answers = re.split(r'\n---\n', content.strip())
    return [ans.strip() for ans in answers if ans.strip()]

def format_docs(docs: List) -> str:
    """Helper function to format retrieved documents into a single string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def setup_database():
    """Creates the necessary tables in the database if they don't already exist."""
    # This function is called once at the start of the app.
    # It's fine without caching as it runs only once.
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
            objective_text TEXT NOT NULL,
            UNIQUE(module_id, objective_text)
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

# --- Core Logic Functions (with minor tweaks for Streamlit) ---

@st.cache_resource
def get_milvus_client():
    """Establishes and caches a connection to Milvus."""
    try:
        client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
        # st.toast("Successfully connected to Zilliz.")
        return client
    except Exception as e:
        st.error(f"Failed to connect to Zilliz: {e}")
        st.stop()
        
milvus_client = get_milvus_client()


@st.cache_resource
def get_vector_store(course: str, module: str) -> Milvus:
    """Helper function to connect to a specific Milvus collection."""
    collection_name = sanitize_name(f"{course}_{module}_source_docs")
    if not milvus_client.has_collection(collection_name):
        raise ValueError(f"Collection '{collection_name}' not found. Please index the documents first.")
    
    milvus_client.load_collection(collection_name)
    
    return Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_args={"uri": ZILLIZ_URI, "token": ZILLIZ_TOKEN, "secure": True},
        auto_id=True,
        text_field="text",
        vector_field="embedding",
        primary_field="id",
        metadata_field=None
    )

def index_source_documents_structured(root_dir_path: str):
    """Reads, chunks, and stores source documents in Zilliz from a given path."""
    # Using st.status for a cleaner loading state
    with st.status("Starting structured indexing process...", expanded=True) as status:
        
        status.update(label="Scanning directory for PDF files...")
        root_dir = Path(root_dir_path)
        loader = DirectoryLoader(
            path=root_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=False, # Show progress in Streamlit instead
            use_multithreading=True,
        )
        documents = loader.lazy_load()

        processed_docs = []
        status.update(label="Extracting metadata from file paths...")
        for doc in documents:
            source_path = doc.metadata.get('source')
            if source_path:
                p = Path(source_path)
                try:
                    module_name = p.parent.name
                    course_name = p.parent.parent.name
                    file_name = p.name
                    doc.metadata['course_name'] = course_name
                    doc.metadata['module_name'] = module_name
                    doc.metadata['file_name'] = file_name
                    processed_docs.append(doc)
                except IndexError:
                    st.warning(f"Skipping file with unexpected path structure: {source_path}")

        status.update(label="Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        chunks = text_splitter.split_documents(processed_docs)

        status.update(label="Grouping chunks by course and module...")
        grouped_data = defaultdict(list)
        for chunk in chunks:
            course = chunk.metadata.get("course_name")
            module = chunk.metadata.get("module_name")
            topic = chunk.metadata.get("file_name")
            if not all([course, module, topic]):
                continue
            collection_name = sanitize_name(f"{course}_{module}_source_docs")
            partition_name = sanitize_name(topic)
            grouped_data[(collection_name, partition_name)].append(chunk)

        created_collections = set()
        status.update(label="Embedding and inserting data into Milvus...")
        progress_bar = st.progress(0.0)
        total_groups = len(grouped_data)

        for i, ((collection_name, partition_name), chunk_list) in enumerate(grouped_data.items()):
            
            if collection_name not in created_collections:
                if not milvus_client.has_collection(collection_name):
                    st.write(f"Creating new collection: '{collection_name}'")
                    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
                    schema.add_field("id", DataType.INT64, is_primary=True)
                    schema.add_field("text", DataType.VARCHAR, max_length=4000)
                    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
                    milvus_client.create_collection(collection_name=collection_name, schema=schema)

                    index_params = milvus_client.prepare_index_params()
                    index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="L2")
                    milvus_client.create_index(collection_name=collection_name, index_params=index_params)
                created_collections.add(collection_name)

            if not milvus_client.has_partition(collection_name, partition_name):
                milvus_client.create_partition(collection_name, partition_name)

            texts_to_embed = [chunk.page_content for chunk in chunk_list]
            all_embeddings = embeddings.embed_documents(texts_to_embed)
            
            data_to_insert = [
                {"text": chunk.page_content, "embedding": all_embeddings[j]}
                for j, chunk in enumerate(chunk_list)
            ]
                
            milvus_client.insert(
                collection_name=collection_name,
                data=data_to_insert,
                partition_name=partition_name
            )
            progress_bar.progress((i + 1) / total_groups, text=f"Processed group {i+1}/{total_groups}: {collection_name}/{partition_name}")

        status.update(label="Indexing complete!", state="complete", expanded=False)
    st.success("Vector storage processing complete.")

# --- The core generation and evaluation functions are mostly the same ---
# They will be called by the Streamlit UI logic.

def get_learning_objectives(course: str, module: str) -> list[str]:
    vector_store = get_vector_store(course, module)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 25})
    parser = PydanticOutputParser(pydantic_object=LearningObjectives)
    prompt = ChatPromptTemplate.from_template(
        """You are an expert instructional designer. Based on the following content from a course module, generate 4-6 clear, concise, and measurable learning objectives. A learning objective should describe what a student will be able to DO after completing the module.
        Context: {context}
        {format_instructions}"""
    )
    chain = ({"context": retriever | format_docs} | prompt.partial(format_instructions=parser.get_format_instructions()) | llm | parser)
    relevant_query = f"A summary of the content for the course '{course}' in the module '{module}'"
    response = chain.invoke(relevant_query)
    return response.objectives

def generate_questions(course: str, module: str, learning_objective: str, bloom_level: str, objective_id: int, cur: psycopg2.extensions.cursor) -> Questions:
    cur.execute("SELECT question_text FROM assessments WHERE objective_id = %s AND bloom_level = %s;", (objective_id, bloom_level))
    existing_questions = [row[0] for row in cur.fetchall()]
    
    vector_store = get_vector_store(course, module)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 10})
    parser = PydanticOutputParser(pydantic_object=Questions)

    context = format_docs(retriever.invoke(learning_objective))
    
    prompt_template = """You are an expert in creating educational assessments. Generate 2 new and distinct assessment questions based on the provided context, tailored to the specific learning objective and Bloom's Taxonomy level.
    You MUST NOT generate questions that are the same as or too similar to the "EXISTING QUESTIONS" listed below.
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
    chain = (prompt.partial(format_instructions= parser.get_format_instructions()) | llm | parser)
    
    response = chain.invoke({
        "context": context,
        "learning_objective": learning_objective,
        "bloom_level": bloom_level,
        "existing_questions": "\n".join(f"- {q}" for q in existing_questions) if existing_questions else "None"
    })
    return response

def generate_evaluation_rubrics(course: str, module: str, learning_objective: str, bloom_level: str, question: str):
    vector_store = get_vector_store(course, module)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 10})
    parser = PydanticOutputParser(pydantic_object=Rubric)
    context = format_docs(retriever.invoke(learning_objective))

    prompt_template = """You are an expert in creating educational assessments. Generate an evaluation rubric for the provided assessment question.
    The rubric should include 2 to 5 weighted criteria that will be used to evaluate student responses. The sum of weights must equal 1.0.
    {format_instructions}
    Assessment Question: {question}
    Learning Objective: {learning_objective}
    Bloom's Taxonomy Level: {bloom_level}
    RELEVANT CONTEXT: {context}
    ---
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = (prompt.partial(format_instructions= parser.get_format_instructions()) | llm | parser)

    response = chain.invoke({
        "question": question,
        "learning_objective": learning_objective,
        "bloom_level": bloom_level,
        "context": context
    })
    return response

def evaluate_answer(question: str, user_answer: str, rubric: Rubric) -> EvaluationResult:
    parser = PydanticOutputParser(pydantic_object=EvaluationResult)
    rubric_text = "\n".join(f"- {c.description} (Weight: {c.weight})" for c in rubric.criteria)
    prompt = ChatPromptTemplate.from_template(
        """You are an expert teaching assistant. Evaluate a student's answer based on a provided question and its rubric.
        Provide a final score out of 100, a detailed justification, and specific, constructive feedback for EACH criterion.
        {format_instructions}
        ---
        ASSESSMENT QUESTION: {question}
        ---
        EVALUATION RUBRIC: {rubric}
        ---
        STUDENT'S ANSWER: {user_answer}
        ---
        """
    )
    chain = (prompt.partial(format_instructions= parser.get_format_instructions()) | llm | parser)
    response = chain.invoke({"question": question, "rubric": rubric_text, "user_answer": user_answer})
    return response

def get_suggested_content(course: str, module: str, learning_objective: str) -> SuggestedContent:
    vector_store = get_vector_store(course, module)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 5})
    context = format_docs(retriever.invoke(learning_objective))
    parser = PydanticOutputParser(pydantic_object=SuggestedContent)

    prompt_template = """You are an expert tutor. A student has performed poorly on an assessment for a specific learning objective. 
    Synthesize the provided "RELEVANT CONTENT" into a clear and concise summary to help the student understand the key concepts they missed.
    {format_instructions}
    Learning Objective the Student Needs Help With: {learning_objective}
    ---
    RELEVANT CONTENT FROM THE COURSE:
    {context}
    ---
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = (prompt.partial(format_instructions= parser.get_format_instructions()) | llm | parser)
    response = chain.invoke({"context": context, "learning_objective": learning_objective})
    return response

# --- Streamlit Application Main Function ---

def main():
    st.set_page_config(page_title="Dynamic Assessment Generator", layout="wide")
    st.title("📚 Dynamic Assessment Generation System")

    # Initialize database
    try:
        setup_database()
    except psycopg2.Error as e:
        st.error(f"Database Connection Error: {e}")
        st.info("Please ensure your PostgreSQL server is running and connection details are correct.")
        st.stop()
        
    # --- Initialize Session State ---
    if 'course' not in st.session_state:
        st.session_state.course = ""
        st.session_state.module = ""
        st.session_state.objectives = []
        st.session_state.assessment = None
        st.session_state.evaluation_results = []

    # --- Sidebar for Setup and Indexing ---
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.info("Start by entering your course and module details. The data should already be indexed for this combination.")

        st.session_state.course = st.text_input("Course Name", value=st.session_state.course, placeholder="e.g., Reacts")
        st.session_state.module = st.text_input("Module Name", value=st.session_state.module, placeholder="e.g., 01 - Getting started with React")

        with st.expander("Advanced: Index Documents"):
            st.warning("Only run this if you've added new PDF files.")
            # In a real app, you might use st.file_uploader to upload a zip of the course structure
            # For this example, we'll stick to a local path.
            source_docs_path = st.text_input("Local Path to Course Folder", placeholder=r"C:\Course2")
            if st.button("Run Indexing Process"):
                if source_docs_path and os.path.isdir(source_docs_path):
                    index_source_documents_structured(source_docs_path)
                else:
                    st.error("Please provide a valid local directory path.")

    # --- Main Application Workflow ---
    if not st.session_state.course or not st.session_state.module:
        st.info("Please enter a course and module name in the sidebar to begin.")
        st.stop()

    st.header(f"Course: `{st.session_state.course}` | Module: `{st.session_state.module}`")
    
    conn = get_db_connection()
    cur = conn.cursor()

    # --- Step 1: Get or Generate Learning Objectives ---
    try:
        # Get or create Course and Module IDs
        cur.execute("INSERT INTO courses (name) VALUES (%s) ON CONFLICT (name) DO UPDATE SET name=EXCLUDED.name RETURNING id;", (st.session_state.course,))
        course_id = cur.fetchone()[0]
        cur.execute("INSERT INTO modules (course_id, name) VALUES (%s, %s) ON CONFLICT (course_id, name) DO UPDATE SET name=EXCLUDED.name RETURNING id;", (course_id, st.session_state.module))
        module_id = cur.fetchone()[0]
        conn.commit()

        # Check for existing learning objectives
        cur.execute("SELECT id, objective_text FROM learning_objectives WHERE module_id = %s;", (module_id,))
        objectives_from_db = cur.fetchall()

        if objectives_from_db:
            st.session_state.objective_db_ids = {text: obj_id for obj_id, text in objectives_from_db}
            st.session_state.objectives = list(st.session_state.objective_db_ids.keys())
        
        if not st.session_state.objectives:
            if st.button("Generate Learning Objectives"):
                with st.spinner("Generating learning objectives from source documents... This may take a moment."):
                    objectives_text = get_learning_objectives(st.session_state.course, st.session_state.module)
                    st.session_state.objectives = objectives_text
                    # Save newly generated objectives to DB
                    st.session_state.objective_db_ids = {}
                    for obj_text in objectives_text:
                        cur.execute("INSERT INTO learning_objectives (module_id, objective_text) VALUES (%s, %s) ON CONFLICT (module_id, objective_text) DO NOTHING RETURNING id;", (module_id, obj_text))
                        result = cur.fetchone()
                        if result:
                            st.session_state.objective_db_ids[obj_text] = result[0]
                    conn.commit()
                st.success("Learning objectives generated and saved!")
                st.rerun() # Rerun to display the new objectives

    except ValueError as e:
        st.error(f"Error: {e}")
        st.info("Please ensure the course/module names are correct and that the documents for them have been indexed.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.stop()

    if not st.session_state.objectives:
        st.info("Click the button above to generate learning objectives for this module.")
        st.stop()

    # --- Step 2: Select Objective and Generate Assessment ---
    st.subheader("1. Select a Learning Objective & Assessment Level")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_objective = st.radio(
            "Learning Objectives:",
            st.session_state.objectives,
            key="selected_objective"
        )
    with col2:
        bloom_levels = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
        selected_level = st.selectbox(
            "Bloom's Taxonomy Level:",
            bloom_levels,
            key="selected_level"
        )
    
    if st.button("Generate Assessment", type="primary"):
        st.session_state.assessment = None # Clear previous assessment
        st.session_state.evaluation_results = [] # Clear previous results
        selected_objective_id = st.session_state.objective_db_ids[selected_objective]
        
        with st.spinner(f"Generating '{selected_level}' questions for the selected objective..."):
            assessment = generate_questions(st.session_state.course, st.session_state.module, selected_objective, selected_level, selected_objective_id, cur)
        
        questions_to_evaluate = []
        with st.spinner("Generating evaluation rubrics for each question..."):
            for q_text in assessment.questions:
                cur.execute("INSERT INTO assessments (objective_id, bloom_level, question_text) VALUES (%s, %s, %s) RETURNING id;", (selected_objective_id, assessment.bloom_level, q_text))
                assessment_id = cur.fetchone()[0]
                rubric = generate_evaluation_rubrics(st.session_state.course, st.session_state.module, selected_objective, selected_level, q_text)
                for criterion in rubric.criteria:
                    cur.execute("INSERT INTO rubric_criteria (assessment_id, description, weight) VALUES (%s, %s, %s);", (assessment_id, criterion.description, criterion.weight))
                conn.commit()
                questions_to_evaluate.append({"id": assessment_id, "text": q_text, "rubric": rubric})
        
        st.session_state.assessment = questions_to_evaluate
        st.success("Assessment and rubrics generated successfully!")
        st.rerun()

    # --- Step 3: Display Assessment and Get User Answers ---
    if st.session_state.assessment:
        st.subheader("2. Generated Assessment")
        for i, q_data in enumerate(st.session_state.assessment):
            with st.expander(f"**Question {i+1}:** {q_data['text']}", expanded=True):
                st.markdown("**Rubric:**")
                for criterion in q_data["rubric"].criteria:
                    st.write(f"- {criterion.description} (Weight: {criterion.weight*100:.0f}%)")
        
        st.subheader("3. Provide Your Answers")
        st.markdown("Create a single text file with your answers. Separate each answer with a line containing only three dashes (`---`).")

        uploaded_file = st.file_uploader("Upload your answers file (.txt)", type=["txt"])

        if uploaded_file is not None:
            user_answers = parse_answers_from_uploaded_file(uploaded_file)
            st.write(f"Found {len(user_answers)} answers in the file.")
            
            if st.button("Evaluate My Answers", type="primary"):
                if len(user_answers) != len(st.session_state.assessment):
                     st.warning(f"Found {len(user_answers)} answers, but {len(st.session_state.assessment)} questions were generated. Only the provided answers will be evaluated.")
                
                eval_results = []
                with st.spinner("Evaluating answers... This might take some time."):
                    for i, q_data in enumerate(st.session_state.assessment):
                        if i < len(user_answers):
                            user_answer = user_answers[i]
                            assessment_id = q_data["id"]
                            
                            # Save answer to DB
                            cur.execute("INSERT INTO user_answers (assessment_id, answer_text) VALUES (%s, %s) RETURNING id;", (assessment_id, user_answer))
                            answer_id = cur.fetchone()[0]
                            conn.commit()

                            # Evaluate
                            evaluation = evaluate_answer(q_data["text"], user_answer, q_data["rubric"])
                            eval_results.append(evaluation)

                            # Save evaluation to DB
                            cur.execute("INSERT INTO evaluation_results (answer_id, score, justification) VALUES (%s, %s, %s) RETURNING id;", (answer_id, evaluation.score, evaluation.justification))
                            evaluation_id = cur.fetchone()[0]
                            for fb in evaluation.criterion_feedback:
                                cur.execute("INSERT INTO criterion_feedback (evaluation_id, criterion_description, feedback) VALUES (%s, %s, %s);", (evaluation_id, fb.criterion, fb.feedback))
                            conn.commit()
                st.session_state.evaluation_results = eval_results
                st.success("Evaluation complete!")
                st.rerun()
    
    # --- Step 4: Display Evaluation Results ---
    if st.session_state.evaluation_results:
        st.subheader("4. Evaluation Results")
        scores = []
        for i, evaluation in enumerate(st.session_state.evaluation_results):
            scores.append(evaluation.score)
            st.markdown(f"---")
            st.markdown(f"#### Results for Question {i+1}")
            st.metric(label="Score", value=f"{evaluation.score}/100")
            with st.expander("View Detailed Feedback", expanded=False):
                st.markdown(f"**Question:** {evaluation.question_text}")
                st.markdown(f"**Justification:** {evaluation.justification}")
                st.markdown("**Feedback per Criterion:**")
                for fb in evaluation.criterion_feedback:
                    st.write(f"- **{fb.criterion}:** {fb.feedback}")
        
        if scores:
            average_score = sum(scores) / len(scores)
            st.markdown("---")
            st.subheader("Final Result")
            st.metric(label=f"Average Score for '{st.session_state.selected_level}' Level", value=f"{average_score:.2f}%")
            if average_score >= 50:
                st.success("🎉 Result: PASS")
            else:
                st.error("Result: FAIL")
                with st.spinner("Generating suggested study content..."):
                    suggested_content = get_suggested_content(st.session_state.course, st.session_state.module, st.session_state.selected_objective)
                    st.info("Based on your performance, here are some key concepts from the course material to review:")
                    st.markdown(suggested_content.suggested_summary)
    
    cur.close()


if __name__ == "__main__":
    main()