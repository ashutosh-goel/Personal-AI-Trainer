from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Union
from pathlib import Path
import os
import re
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
# from llama_index.llms.ollama import Ollama
# from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import MilvusClient, DataType
import json
#from workflows.context.state_store import DictState
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Event,
    InputRequiredEvent,
    HumanResponseEvent,
    Workflow,
    step,
    Context
)
import asyncio
import time

import os

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
import Parse_Answers_File

# import google.generativeai as genai
from google.genai.errors import ServerError
from google.api_core import exceptions as api_exceptions

api_key = "AIzaSyAlABTPGnaSlYkrWsmUwfmPjyq9tIjMsOs"  # Replace with your actual Google API key
os.environ["GOOGLE_API_KEY"] = api_key
llm = GoogleGenAI(model="gemini-2.5-pro")
Settings.embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-001", api_key=api_key)
Settings.llm = GoogleGenAI(
    model="gemini-2.5-pro",  # Use "gemma:7b" for the 7B model
    request_timeout=600.0,  # Set a longer timeout for larger models
)

'''Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",  # Use "gemma:7b" for the 7B model
)

Settings.llm = Ollama(
    model="gemma:2b",  # Use "gemma:7b" for the 7B model
    request_timeout=600.0,  # Set a longer timeout for larger models
)'''

ZILLIZ_URI = "https://in03-540f880b2b2a98e.serverless.gcp-us-west1.cloud.zilliz.com"
ZILLIZ_TOKEN = "9326f5b4421b923f2649bda2c60f9b2f8fe20339831a0eed5c8d8a5e60d410fa511c1c8ad944b6ea6a4a645f622d367e0b5d42b2"
VECTOR_DIM = 3072

try:
    milvus_client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
    print("Successfully connected to Zilliz with pymilvus client.")
except Exception as e:
    print(f"Failed to connect to Zilliz with pymilvus client: {e}")
    exit()

def sanitize_name(name: str) -> str:
    """Sanitizes a string to be a valid Milvus collection or partition name."""
    # Remove leading/trailing whitespace
    name = name.strip()
    # Replace invalid characters with underscores
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Ensure it starts with a letter or underscore
    if not re.match(r'^[a-zA-Z_]', name):
        name = '_' + name
    # Truncate to a reasonable length (Milvus has limits)
    return name[:255]


#------ Metadata Function ------
def metadata(filepath: str):
    path = Path(filepath)
    course = path.parts[-3]  # Assuming structure: /Courses/CourseX/CourseX/ModuleY/Topic.pdf
    module = path.parts[-2]
    topic = path.stem
    return {"course": course, "module": module, "topic": topic}

def index_source_documents_structured():
    """
    Reads all source documents, chunks them, and stores them in Zilliz
    with a collection per module and a partition per topic (file).
    """
    print("\n--- Starting Structured Indexing of Source Documents ---")
    reader = SimpleDirectoryReader(
        input_dir=r"C:\Users\ashut\Downloads\Courses",
        recursive=True,
        file_extractor={".pdf": PDFReader()},
        file_metadata=metadata
    )
    documents = reader.load_data(show_progress=True)
    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)

    for doc in documents:
        course_name = doc.metadata.get("course")
        module_name = doc.metadata.get("module")
        topic_name = doc.metadata.get("topic")

        if not module_name or not topic_name:
            print(f"Skipping document with missing metadata: {doc.metadata.get('file_path')}")
            continue

        collection_name = sanitize_name(f"{course_name}_{module_name}_source_docs")
        partition_name = sanitize_name(topic_name)

        expected_fields = {"id", "text", "course", "module", "topic", "embedding"}

        # **FIXED**: Add a check for schema mismatch in existing collections.
        if milvus_client.has_collection(collection_name):
            description = milvus_client.describe_collection(collection_name)
            existing_fields = {field['name'] for field in description['fields']}
            if not expected_fields.issubset(existing_fields):
                print(f"\n[FATAL ERROR] Collection '{collection_name}' has an outdated schema.")
                print(f"--> Expected to find fields: {expected_fields}")
                print(f"--> But only found fields: {existing_fields}")
                print("\nSOLUTION: Please delete this collection from your Zilliz Cloud dashboard and run the script again.")
                exit() # Stop the script to prevent further errors.

        # Create collection for the module if it doesn't exist
        if not milvus_client.has_collection(collection_name):
            print(f"Creating new source collection for module: '{collection_name}'")
            schema = MilvusClient.create_schema(auto_id=True)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("text", DataType.VARCHAR, max_length=4000) # Store the text chunk
            schema.add_field("metadata", DataType.JSON)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
            milvus_client.create_collection(collection_name=collection_name, schema=schema)

            print(f"Creating index for collection '{collection_name}'...")
            index_params = milvus_client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="AUTOINDEX",
                metric_type="L2"
            )
            milvus_client.create_index(collection_name=collection_name, index_params=index_params)


        # Create partition for the topic if it doesn't exist
        if not milvus_client.has_partition(collection_name, partition_name):
            print(f"Creating new partition for topic: '{partition_name}' in '{collection_name}'")
            milvus_client.create_partition(collection_name, partition_name)
        
        # Check if partition already has content to avoid re-indexing
        partition_stats = milvus_client.get_partition_stats(collection_name, partition_name)
        if int(partition_stats.get('row_count', 0)) > 0:
            print(f"Partition '{partition_name}' already contains data. Skipping indexing.")
            continue

        # Process and insert the document chunks
        nodes = node_parser.get_nodes_from_documents([doc])
        texts = [node.get_content() for node in nodes]

        # Implement retry logic with exponential backoff for embedding generation
        batch_size = 32
        all_embeddings = []
        max_retries = 3
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            for attempt in range(max_retries):
                try:
                    print(f"Generating embeddings for batch {i//batch_size + 1}, attempt {attempt + 1}...")
                    batch_embeddings = Settings.embed_model.get_text_embedding_batch(batch_texts, show_progress=False)
                    all_embeddings.extend(batch_embeddings)
                    time.sleep(1) # Add a small delay to be polite to the API
                    break # If successful, exit the retry loop
                # **FIXED**: Catch the correct ServerError exception using the module alias
                except (ServerError, api_exceptions.InternalServerError) as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** (attempt + 1)
                        print(f"Server error encountered: {e}. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"Failed to generate embeddings after {max_retries} attempts. Aborting.")
                        raise # Re-raise the exception to stop the script

        data_to_insert = []
        for text, emb in zip(texts, all_embeddings):
            metadata_payload = {
                "_node_content": json.dumps({"text": text}), # Serialize the node content
                "course": course_name,
                "module": module_name,
                "topic": topic_name,
            }
            data_to_insert.append({
                "text": text, 
                "embedding": emb,
                "metadata": metadata_payload
            })

        if data_to_insert:
            print(f"Inserting {len(data_to_insert)} chunks for topic '{topic_name}'...")
            milvus_client.insert(collection_name, data_to_insert, partition_name=partition_name)
    
    print("--- Finished Structured Indexing ---")



        # batch_size = 32
        # all_embeddings = []
        # max_retries = 3
        # for i in range(0, len(texts), batch_size):
        #     batch_texts = texts[i:i + batch_size]

        # embeddings = Settings.embed_model.get_text_embedding_batch(texts, show_progress=True)

        # data_to_insert = [{"text": text, "embedding": emb} for text, emb in zip(texts, embeddings)]


# ------ Indexing and Querying ------
# reader = SimpleDirectoryReader(
#     input_dir=r"C:\Courses1",
#     recursive=True,
#     file_extractor={".pdf": PyMuPDFReader()},
#     file_metadata=metadata
# )


# # Check if storage already exists
# Persist_Dir = "./storage_L1"
# if not os.path.exists(Persist_Dir):
#     os.makedirs(Persist_Dir)
#     # Load the documents, create the index
#     documents = reader.load_data()
#     index = VectorStoreIndex.from_documents(documents, show_progress=True)
#     # storing it
#     index.storage_context.persist(persist_dir=Persist_Dir)
# else:
#     # Load the index from storage
#     storage_context = StorageContext.from_defaults(persist_dir=Persist_Dir)
#     index = load_index_from_storage(storage_context)

# ------ Output Structure  ------

class Questions(BaseModel):
    '''Questions based on each Bloom's Taxonomy level for each learning objectives.'''
    objective: str = Field(description="Learning objective for which questions are generated")
    remember: List[str] = Field(description="List of questions for Remember level")
    understand: List[str] = Field(description="List of questions for Understand level")
    apply: List[str] = Field(description="List of questions for Apply level")
    analyze: List[str] = Field(description="List of questions for Analyze level")
    evaluate: List[str] = Field(description="List of questions for Evaluate level")
    create: List[str] = Field(description="List of questions for Create level")

class ModuleAssessment(BaseModel):
    assessment: list[Questions] = Field(description="List of all the questions of all the learning objectives of the module")

class Output(BaseModel):
    course: str
    module: str
    objectives: List[str] = []
    """questions: List[Questions] = Field(description="List of list of questions for each learning objective")"""

class Criterion(BaseModel):
    """A specific criterion for evaluating an answer (e.g., Accuracy, Completeness)."""
    name: str = Field(description="The name of the criterion being evaluated.")
    weight: float = Field(description="The weight of this criterion in the total score (e.g., 0.5 for 50%).")
    #levels: List[ScoringLevel] = Field(description="A list of scoring levels from poor to excellent.")

class Rubric(BaseModel):
    """A complete evaluation rubric for a single question."""
    objective: str = Field(description="The learning objective this question belongs to.")
    question_text: str = Field(description="The full text of the question being evaluated.")
    bloom_level: str = Field(description="The Bloom's Taxonomy level of the question.")
    criteria: List[Criterion] = Field(description="A list of criteria to evaluate the answer against.")

class Criterion_Feedback(BaseModel):
    """A specific criterion for evaluating an answer (e.g., Accuracy, Completeness)."""
    criterion: str = Field(description="The name of the criterion being evaluated.")
    feedback: str = Field(description="feedback on the user's answer")

class EvaluationResult(BaseModel):
    """The result of evaluating a single answer against a rubric."""
    question_text: str = Field(description="The question that was answered.")
    score: float = Field(description="The final calculated score for the answer, typically out of 100.")
    justification: str = Field(description="A detailed explanation of how the score was determined based on the rubric criteria.")
    criterion_feedback: List[Criterion_Feedback] = Field(description="Specific feedback for each criterion in the rubric.")

# llm1 = Settings.llm.as_structured_llm(Output)
# llm2 = Settings.llm.as_structured_llm(Questions)
# llm3 = Settings.llm.as_structured_llm(Rubric)
# llm4 = Settings.llm.as_structured_llm(EvaluationResult)
# query_engine1 = index.as_query_engine(llm=llm1)
# query_engine2 = index.as_query_engine(llm=llm2)
# query_engine3 = index.as_query_engine(llm = llm3)
# query_engine4 = index.as_query_engine(llm=llm4)
# suggestion_query_engine = index.as_query_engine(llm=Settings.llm)

# ------ Events ------

class GetObjectives(Event):
    course : str
    module : str

class SetAssessment(Event):
    """Event to set the assessment based on learning objectives."""
    query : str

class RubrikEvent(Event):
    query : str

class EvalEvent(Event):
    query : str

class SuggestionEvent(Event):
    """Event to trigger remedial content suggestions."""
    pass

# ------ Workflow Definition ------

class Objective_Workflow(Workflow):
    @step
    async def set_course(self, ctx: Context, ev: StartEvent) -> GetObjectives :
        """Extract course and module from the query and set them in the context. """
        query = ev.get("query")
        Output.course = query['course']
        Output.module = query['module']
        await ctx.store.set("course", Output.course)
        await ctx.store.set("module", Output.module)


        return GetObjectives(course = Output.course, module=Output.module)
    
    @step
    async def get_objectives(self, ctx: Context, ev: GetObjectives) -> SetAssessment :
        """Query the index for the objectives based on course and module."""

        # Dynamically connect to the correct source collection for the module
        module_collection_name = sanitize_name(f"{ev.course}_{ev.module}_source_docs")

        await ctx.store.set("module_collection_name", module_collection_name)
        
        if not milvus_client.has_collection(module_collection_name):
            raise ValueError(f"Source collection '{module_collection_name}' not found. Please run the indexing script.")

        print(f"\nConnecting to source collection for module: '{module_collection_name}'")
        vector_store = MilvusVectorStore(
            uri=ZILLIZ_URI, token=ZILLIZ_TOKEN, collection_name='React_01___Getting_started_with_React_source_docs', dim=VECTOR_DIM, text_field="text"
        )

        client = MilvusClient(
            uri=ZILLIZ_URI,
            token=ZILLIZ_TOKEN
        )

        # Use query() to fetch the first 3 entities from the partition
        res = client.query(
            collection_name="React_01___Getting_started_with_React_source_docs",
            partition_names=["_01_01_Introduction_to_React___The_Power_of_Modern_UI_Development"],
            # An empty filter fetches all, limit restricts the count
            filter="",
            limit=3,
            output_fields=["embedding", "text", "course", "module", "topic"]
        )

        print(res)

        module_index = VectorStoreIndex.from_vector_store(vector_store)
        
        
        # Store the module-specific index in the context for the next step
        await ctx.store.set("module_index", module_index)

        query_engine1 = module_index.as_query_engine(llm=Settings.llm.as_structured_llm(Output))
        

        course_name = ev.course.replace(" ", "_")
        module_name = ev.module.replace(" ", "_")
        objectives_file = f"{course_name}_{module_name}_objectives.json"

        if os.path.exists(objectives_file):
            print(f"Found existing objectives file. Loading from '{objectives_file}'...")
            with open(objectives_file, 'r') as f:
                objectives = json.load(f)
        else:
            print("No objectives file found. Generating new objectives...")
            query = f"You're a curriculum designer. Generate 5-10 broad learning objectives for {ev.module} module in {ev.course} course. Focus on the word broad because later i will need to create questions based on each Bloom's Taxonomy Level for each Learning objective that you give. The learning objectives should be foused on covering the whole stored content of the module"
            response = query_engine1.query(query).response
            objectives = response.objectives

            with open(objectives_file, 'w') as f:
                json.dump(objectives, f, indent=4)
            print(f"Objectives saved to '{objectives_file}'.")
        
        await ctx.store.set("all_objectives", objectives)
        
        print(f"Course: {ev.course}")
        print(f"Module: {ev.module}")
        print("\nObjectives:")
        for i, obj in enumerate(objectives, 1):
            print(f"  {i}. {obj}\n")
       
        return SetAssessment(query = "Set Assessment") 
    
    @step
    async def assessment(self, ctx: Context, ev: SetAssessment) -> RubrikEvent:
        """Generate questions based on each Bloom's Taxonomy level for each learning objectives."""
        course = await ctx.store.get("course")
        module = await ctx.store.get("module")
        all_objectives = await ctx.store.get("all_objectives", [])
        if not all_objectives:
            print("No learning objectives found.")

        # Retrieve the module-specific index from the context
        #module_index = await ctx.store.get("module_index")

        module_collection_name = await ctx.store.get("module_collection_name")
        vector_store = MilvusVectorStore(
            uri=ZILLIZ_URI, token=ZILLIZ_TOKEN, collection_name=module_collection_name, dim=VECTOR_DIM, text_field="text"
        )
        module_index = VectorStoreIndex.from_vector_store(vector_store)

        if not module_index:
            raise ValueError("Module-specific index not found in workflow context.")
        
        query_engine2 = module_index.as_query_engine(llm=Settings.llm.as_structured_llm(Questions))

        choice = int(input("Please select the Learning Objective: "))-1
        level_choice = int(input("Please select the level between 1-6"))-1
        bloom_levels = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
        selected_level = bloom_levels[level_choice]
        selected_objective = all_objectives[choice]
        print(f"\nYou have selected: '{selected_objective}'")
        print(f"\nLevel: '{selected_level}'")
        await ctx.store.set("selected_objective", selected_objective)
        await ctx.store.set("selected_level", selected_level)

        course_name = course.replace(" ", "_")
        module_name = module.replace(" ", "_")
        questions_file = f"{course_name}_{module_name}_{selected_objective}_{selected_level}_questions.json"

        if os.path.exists(questions_file):
            print(f"Found existing questions file. Loading from '{questions_file}'...")
            with open(questions_file, 'r') as f:
                questions = json.load(f)
            query = f"Generate 3 questions based on '{selected_level}' Bloom's Taxonomy level for the learning objective: {selected_objective} for the module: {module} of the course: {course}. The generated questions should not be exactly similar to these questions {questions}, either paraphrase them or generate new questions."
        else:
            print("No questions file found. Generating new questions...")
            query = f"Generate 3 questions based on '{selected_level}' Bloom's Taxonomy level for the learning objective: {selected_objective} for the module: {module} of the course: {course}. "
        
        response = query_engine2.query(query).response

        with open(questions_file, 'a+') as f:
            json.dump(response, f, indent=4)
        print(f"Questions saved to '{questions_file}'.")


        # 1. Store the entire structured response in the context
        await ctx.store.set("generated_assessment", response)

        # 2. Nicely print the contents
        print("\n--- Generated Questions ---")

        print(f"Objective: {response.objective}\n")
        bloom_levels = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
        for level in bloom_levels:
            questions_for_level = getattr(response, level)
            if questions_for_level:
                print(f"  {level.capitalize()} Questions:")
                for i, q in enumerate(questions_for_level, 1):
                    print(f"    {i}. {q}")
                print()
        print("="*60 + "\n")
            
        return RubrikEvent(query = "Make Evaluation Rubrik")
    
    @step
    async def evaluation_rubriks(self, ctx: Context, ev: RubrikEvent) -> EvalEvent:
        """Generate an evaluation rubrik for the generated questions"""

        module_collection_name = await ctx.store.get("module_collection_name")
        vector_store = MilvusVectorStore(
            uri=ZILLIZ_URI, token=ZILLIZ_TOKEN, collection_name=module_collection_name, dim=VECTOR_DIM
        )
        module_index = VectorStoreIndex.from_vector_store(vector_store)
        query_engine3 = module_index.as_query_engine(Settings.llm.as_structured_llm(Rubric))

        # Fetch the module assessment from context store
        question_obj = await ctx.store.get("generated_assessment")
        
        print(f"Targeting objective: '{question_obj.objective}'")

        all_rubrics_for_objective = []
        objective_text = question_obj.objective
        
        for bloom_level in ['Apply']:
            questions_for_level = getattr(question_obj, bloom_level, [])
            
            for question_text in questions_for_level:
                if not question_text.strip(): continue

                print(f"--- Generating rubric for: '{question_text[:50]}...'")
                prompt = (
                    f"You are an expert instructional designer. Create a detailed, weighted evaluation rubric for the following question.\n\n"
                    f"**Learning Objective:** {objective_text}\n"
                    f"**Bloom's Taxonomy Level:** {bloom_level.capitalize()}\n"
                    f"**Question:** \"{question_text}\"\n\n"
                    f"The rubric must have 2-4 weighted criteria. The sum of weights must equal 1.0. "
                    f"For each criterion, provide at least 3 scoring levels."
                    f"Maximum marks for each question must be 10"
                )

                rubric_response = query_engine3.query(prompt).response
                print(rubric_response.model_dump_json(indent=2))
                all_rubrics_for_objective.append(rubric_response)
                print(f"--- Rubric generated successfully.")

        await ctx.store.set("generated_rubrics", all_rubrics_for_objective)
        print(f"\nSuccessfully generated and stored {len(all_rubrics_for_objective)} rubrics for the selected objective.")

        return EvalEvent(query="Start the evaluation")
    
    @step
    async def input_answers(self, ctx:Context, ev: EvalEvent)-> StopEvent|SuggestionEvent:
        """Evaluate a student's answers against the generated rubrics."""

        module_collection_name = await ctx.store.get("module_collection_name")
        vector_store = MilvusVectorStore(
            uri=ZILLIZ_URI, token=ZILLIZ_TOKEN, collection_name=module_collection_name, dim=VECTOR_DIM
        )
        module_index = VectorStoreIndex.from_vector_store(vector_store)
        query_engine4 = module_index.as_query_engine(Settings.llm.as_structured_llm(EvaluationResult))

        print("Taking input from user...")
        answers_file_path = input("Please provide the path to your answers file:")
        print(f"Received file path from user: {answers_file_path}")

         # 1. Retrieve the rubrics from the context store
        rubrics_data = await ctx.store.get("generated_rubrics", [])
        if not rubrics_data:
            print("ERROR: No rubrics found in context to evaluate against.")
            return StopEvent(error="Rubrics not found.")

        # 2. Define path to answers file and parse it
        # You can change this path to point to the user's file.
        
        student_answers = Parse_Answers_File.parse_answers_file(answers_file_path)

        final_evaluations = []
        total_score = 0
        
        # 3. Loop through each rubric object
        for i, rubric_data in enumerate(rubrics_data):
            try:
                rubric = Rubric.model_validate(rubric_data)
            except Exception as e:
                print(f"Skipping an item that could not be parsed into a Rubric object: {e}")
                continue
            
            question = rubric.question_text
            
            # Get the corresponding answer from the parsed file list
            # If the index is out of bounds or the answer is empty, treat as skipped.
            answer = student_answers[i] if i < len(student_answers) else ""
            
            print(f"--- Evaluating answer for: '{question[:50]}...'")

            # 4. Handle skipped or unanswered questions
            if not answer:
                print("--- Question SKIPPED by user (or no answer provided). Score: 0")
                skipped_evaluation = EvaluationResult(
                    question_text=question,
                    score=0.0,
                    justification="Question was skipped by the user or no answer was provided in the file.",
                    criterion_feedback=[]
                )
                final_evaluations.append(skipped_evaluation)
                continue # Move to the next rubric

            # 5. If an answer exists, proceed with LLM evaluation
            rubric_json_string = rubric.model_dump_json(indent=2)
            prompt = (
                f"You are an expert teaching assistant. Evaluate the student's answer based on the provided rubric. "
                f"Provide a final score and a detailed justification.\n\n"
                f"**QUESTION:**\n{question}\n\n"
                f"**EVALUATION RUBRIC (in JSON format):**\n{rubric_json_string}\n\n"
                f"**STUDENT'S ANSWER:**\n{answer}\n\n"
                f"Your evaluation must follow the rubric's criteria and weighting precisely. "
                f"Provide specific feedback for each criterion."
            )

            evaluation_response = query_engine4.query(prompt).response
            final_evaluations.append(evaluation_response)
            total_score += evaluation_response.score
            
            print(f"--- Evaluation complete.")
            print(evaluation_response.model_dump_json(indent=2))

        # 6. Store the final results
        await ctx.store.set("final_evaluations", final_evaluations)
        print(f"\nSuccessfully evaluated {len(final_evaluations)} answers.")

        # 7. Performance Analysis
        average_score = total_score / (10*len(final_evaluations)) if final_evaluations else 0
        print(f"\nCollective Average Score: {average_score:.2f}%")

        if average_score < 60:
            print("Performance is below 60%. Triggering remediation...")
            return SuggestionEvent()
        else:
            print("Great work! You've mastered this section.")
            return StopEvent() 
    
    @step
    async def provide_suggestion(self, ctx: Context, ev: SuggestionEvent) -> StopEvent:
        """Provide targeted reading suggestions based on poor performance."""

        module_collection_name = await ctx.store.get("module_collection_name")
        vector_store = MilvusVectorStore(
            uri=ZILLIZ_URI, token=ZILLIZ_TOKEN, collection_name=module_collection_name, dim=VECTOR_DIM
        )
        module_index = VectorStoreIndex.from_vector_store(vector_store)
        query_engine5 = module_index.as_query_engine(Settings.llm)

        print("\n" + "="*50)
        print("STEP: PROVIDING REMEDIATION")
        print("="*50 + "\n")
        
        objective = await ctx.store.get("selected_objective")
        bloom_level = "remember"  

        print("Based on your performance, here are some suggested readings from the course material to help you improve:\n")
        
        # Create a query to find relevant content in the indexed documents
        remediation_query = f"Find content related to the learning objective '{objective}' focusing on the Bloom's Taxonomy level: '{bloom_level}' level of learning."
        
        # Use a standard query engine to get text results
        response = query_engine5.query(remediation_query)

        print(response.response)

        # Print the source nodes (the relevant text chunks)
        # for i, source_node in enumerate(response.source_nodes):
        #     print(f"--- Reading Suggestion {i+1} (from: {source_node.metadata.get('topic', 'N/A')}) ---\n")
        #     print(source_node.get_content())
        #     print("\n" + "-"*50 + "\n")
            
        return StopEvent()

# ------ Workflow Execution ------
async def main():

    index_source_documents_structured()
    workflow = Objective_Workflow()
    await workflow.run(query = {"course": 'React', "module": '01___Getting_started_with_React'})

if __name__ == "__main__":
    asyncio.run(main())
