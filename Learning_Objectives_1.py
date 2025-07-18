from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Union
from pathlib import Path
import os
from llama_index.readers.file import PyMuPDFReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
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

import os

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
import Parse_Answers_File

api_key = "AIzaSyAlABTPGnaSlYkrWsmUwfmPjyq9tIjMsOs"  # Replace with your actual Google API key
os.environ["GOOGLE_API_KEY"] = api_key
llm = GoogleGenAI(model="gemini-2.5-flash")
Settings.embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-exp-03-07", api_key=api_key)
Settings.llm = GoogleGenAI(
    model="gemini-2.5-flash",  # Use "gemma:7b" for the 7B model
    request_timeout=600.0,  # Set a longer timeout for larger models
)

'''Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",  # Use "gemma:7b" for the 7B model
)

Settings.llm = Ollama(
    model="gemma:2b",  # Use "gemma:7b" for the 7B model
    request_timeout=600.0,  # Set a longer timeout for larger models
)'''

#------ Metadata Function ------
def metadata(filepath: str):
    path = Path(filepath)
    course = path.parts[-3]  # Assuming structure: /Courses/CourseX/CourseX/ModuleY/Topic.pdf
    module = path.parts[-2]
    topic = path.name
    return {"course": course, "module": module, "topic": topic}

# ------ Indexing and Querying ------
reader = SimpleDirectoryReader(
    input_dir=r"C:\Courses1",
    recursive=True,
    file_extractor={".pdf": PyMuPDFReader()},
    file_metadata=metadata
)


# Check if storage already exists
Persist_Dir = "./storage_L1"
if not os.path.exists(Persist_Dir):
    os.makedirs(Persist_Dir)
    # Load the documents, create the index
    documents = reader.load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    # storing it
    index.storage_context.persist(persist_dir=Persist_Dir)
else:
    # Load the index from storage
    storage_context = StorageContext.from_defaults(persist_dir=Persist_Dir)
    index = load_index_from_storage(storage_context)

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

llm1 = Settings.llm.as_structured_llm(Output)
llm2 = Settings.llm.as_structured_llm(Questions)
llm3 = Settings.llm.as_structured_llm(Rubric)
llm4 = Settings.llm.as_structured_llm(EvaluationResult)
query_engine1 = index.as_query_engine(llm=llm1)
query_engine2 = index.as_query_engine(llm=llm2)
query_engine3 = index.as_query_engine(llm = llm3)
query_engine4 = index.as_query_engine(llm=llm4)
suggestion_query_engine = index.as_query_engine(llm=Settings.llm)

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

        course_name = ev.course.replace(" ", "_")
        module_name = ev.module.replace(" ", "_")
        objectives_file = f"{course_name}_{module_name}_objectives.json"

        if os.path.exists(objectives_file):
            print(f"Found existing objectives file. Loading from '{objectives_file}'...")
            with open(objectives_file, 'r') as f:
                objectives = json.load(f)
        else:
            print("No objectives file found. Generating new objectives...")
            query = f"You're a curriculum designer. Generate 3-6 broad learning objectives for {ev.module} module in {ev.course} course. Focus on the word broad because later i will need to create questions based on each Bloom's Taxonomy Level for each Learning objective that you give. The learning objectives should be foused on covering the whole stored content of the module"
            response = query_engine1.query(query).response
            objectives = response.objectives
            with open(objectives_file, 'w') as f:
                json.dump(objectives, f, indent=4)
            print(f"Objectives saved to '{objectives_file}'.")
            
        await ctx.store.set("all_objectives", objectives)
        
        print(f"Course: {course_name}")
        print(f"Module: {module_name}")
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

        choice = int(input("Please select the Learning Objective: "))-1
        selected_objective = all_objectives[choice]
        print(f"\nYou have selected: '{selected_objective}'")
        await ctx.store.set("selected_objective", selected_objective)

        query = f"Generate 3 questions based on each Bloom's Taxonomy level for the learning objective: {selected_objective} for the module: {module} of the course: {course}. "
        response = query_engine2.query(query).response

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

        # Fetch the module assessment from context store
        question_obj = await ctx.store.get("generated_assessment")
        
        print(f"Targeting objective: '{question_obj.objective}'")

        all_rubrics_for_objective = []
        objective_text = question_obj.objective
        
        for bloom_level in ['Create']:
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
        print("\n" + "="*50)
        print("STEP: PROVIDING REMEDIATION")
        print("="*50 + "\n")
        
        objective = await ctx.store.get("selected_objective")
        bloom_level = "remember"  

        print("Based on your performance, here are some suggested readings from the course material to help you improve:\n")
        
        # Create a query to find relevant content in the indexed documents
        remediation_query = f"Find content related to the learning objective '{objective}' focusing on the Bloom's Taxonomy level: '{bloom_level}' level of learning."
        
        # Use a standard query engine to get text results
        response = suggestion_query_engine.query(remediation_query)

        print(response.response)

        # Print the source nodes (the relevant text chunks)
        # for i, source_node in enumerate(response.source_nodes):
        #     print(f"--- Reading Suggestion {i+1} (from: {source_node.metadata.get('topic', 'N/A')}) ---\n")
        #     print(source_node.get_content())
        #     print("\n" + "-"*50 + "\n")
            
        return StopEvent()

# ------ Workflow Execution ------
async def main():

    workflow = Objective_Workflow()
    await workflow.run(query = {"course": 'NodeJS', "module": '04 - Servers'})

if __name__ == "__main__":
    asyncio.run(main())