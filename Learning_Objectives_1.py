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
    model_config = {"arbitrary_types_allowed": True}

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


llm1 = Settings.llm.as_structured_llm(Output)
llm2 = Settings.llm.as_structured_llm(ModuleAssessment)
llm3 = Settings.llm.as_structured_llm(Rubric)
query_engine1 = index.as_query_engine(llm=llm1)
query_engine2 = index.as_query_engine(llm=llm2)
query_engine3 = index.as_query_engine(llm = llm3)

# Format questions into a readable string
def format_questions(questions_obj):
    if not isinstance(questions_obj, Questions):
        print("ERROR: Expected Questions object, got:", type(questions_obj))
        return f"Invalid questions object: {str(questions_obj)}"
    
    if not questions_obj.objective:
        print("WARNING: Questions object has empty objective.")
        return "Objective: None\n  No questions available."
    
    result = [f"Objective: {questions_obj.objective}"]
    has_questions = False
    for level in ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']:
        questions_list = [q for q in getattr(questions_obj, level, []) if q.strip()]  # Filter out whitespace-only questions
        print(f"DEBUG: Questions for {level}: {questions_list}")
        if questions_list:
            has_questions = True
            result.append(f"  {level.capitalize()} Questions:")
            for i, q in enumerate(questions_list, 1):
                result.append(f"    {i}. {q}")
        else:
            result.append(f"  {level.capitalize()} Questions: None")
    if not has_questions:
        print("WARNING: No valid questions found in any Bloom's level.")
        result.append("  No valid questions available for evaluation.")
    return "\n".join(result).strip()


# ------ Events ------

class GetObjectives(Event):
    course : str
    module : str

class SetAssessment(Event):
    """Event to set the assessment based on learning objectives."""
    query : str
class EvalEvent(Event):
    query : str

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
        query = f"You're a curriculum designer. Extract 3-6 broad learning objectives for {ev.module} module in {ev.course} course. Focus on the word broad because later i will need to create questions based on each Bloom's Taxonomy Level for each Learning objective that you give."
        response = query_engine1.query(query)
        Response = response.response  # Settings.llm.complete(prompt).text.strip()
        #print(Response)
        await ctx.store.set("objectives", Response.objectives)
        
        print(f"Course: {Response.course}")
        print(f"Module: {Response.module}")
        print("\nObjectives:")
        for i, objective in enumerate(Response.objectives, 1):
            print(f"  {i}. {objective}")
       
        return SetAssessment(query = "Set Assessment") #EvalEvent(query = "Start Evaluation") 
    
    @step
    async def assessment(self, ctx: Context, ev: SetAssessment) -> EvalEvent:
        """Generate questions based on each Bloom's Taxonomy level for each learning objectives."""
        course = await ctx.store.get("course")
        module = await ctx.store.get("module")
        objectives = await ctx.store.get("objectives", [])
        if not objectives:
            print("No learning objectives found.")
        
        query = f"Generate 3 questions based on each Bloom's Taxonomy level for each learning objective in {objectives} for the module {module} of the course {course}. "
        response = query_engine2.query(query).response

        # 1. Store the entire structured response in the context
        print("Storing the generated assessment in the context store...")
        await ctx.store.set("generated_assessment", response)

        # 2. Nicely print the contents
        print("\n--- Generated Questions ---")
        # Ensure the attribute name matches your Pydantic model (e.g., response.assessment)
        list_of_questions = response.assessment 
        
        for question_obj in list_of_questions:
            print(f"Objective: {question_obj.objective}\n")
            bloom_levels = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
            for level in bloom_levels:
                questions_for_level = getattr(question_obj, level)
                if questions_for_level:
                    print(f"  {level.capitalize()} Questions:")
                    for i, q in enumerate(questions_for_level, 1):
                        print(f"    {i}. {q}")
                    print()
            print("="*60 + "\n")
            
        return EvalEvent(query = "Make Evaluation Rubrik")
    
    @step
    async def evaluate_answers(self, ctx: Context, ev: EvalEvent) -> StopEvent:
        """Generate an evaluation rubrik for the generated questions"""

        # Fetch the module assessment from context store
        module_assessment = await ctx.store.get("generated_assessment")

        target_index = 2

        question_obj = module_assessment.assessment[target_index]
        
        print(f"Targeting objective: '{question_obj.objective}'")

        all_rubrics_for_objective = []
        objective_text = question_obj.objective
        
        # The rest of the logic is the same, but it now operates only on the
        # single 'question_obj' instead of looping through all of them.
        for bloom_level in ['analyze', 'create']:
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
                )

                rubric_response = query_engine3.query(prompt).response
                print(rubric_response.model_dump_json(indent=2))
                all_rubrics_for_objective.append(rubric_response)
                print(f"--- Rubric generated successfully.")

        await ctx.store.set("generated_rubrics_for_one_objective", all_rubrics_for_objective)
        print(f"\nSuccessfully generated and stored {len(all_rubrics_for_objective)} rubrics for the selected objective.")


        '''all_rubrics = []
        
        # 2. Loop through every objective and every question
        for question_obj in module_assessment.assessment:
            objective_text = question_obj.objective
            
            for bloom_level in ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']:
                questions_for_level = getattr(question_obj, bloom_level, [])
                
                for question_text in questions_for_level:
                    if not question_text.strip(): continue # Skip empty questions

                    print(f"--- Generating rubric for: '{question_text[:50]}...' (Level: {bloom_level.capitalize()})")

                    # 3. Create a highly specific prompt for ONE question
                    prompt = (
                        f"You are an expert instructional designer. Your task is to create a detailed, "
                        f"weighted evaluation rubric for the following question.\n\n"
                        f"**Learning Objective:** {objective_text}\n"
                        f"**Bloom's Taxonomy Level:** {bloom_level.capitalize()}\n"
                        f"**Question:** \"{question_text}\"\n\n"
                        f"The rubric must have 2-4 weighted criteria (e.g., Accuracy, Completeness, Clarity, Depth of Analysis). "
                        f"The sum of weights must equal 1.0. "
                        f"For each criterion, provide at least 3 scoring levels (e.g., 0 for 'Does not meet expectations', "
                        f"1 for 'Partially meets', 2 for 'Fully meets')."
                    )

                    # 4. Call the rubric-specific query engine
                    rubric_response = query_engine3.query(prompt).response
                    all_rubrics.append(rubric_response)
                    print(f"--- Rubric generated successfully.")

        # 5. Store all the generated rubrics in the context for later use
        await ctx.store.set("generated_rubrics", all_rubrics)
        print(f"\nSuccessfully generated and stored {len(all_rubrics)} rubrics.")

        # Optional: Print the generated rubrics to verify
        for rubric in all_rubrics:
            print(rubric.model_dump_json(indent=2))'''


        '''# Step 1: Create a rubric prompt
        rubric_prompt = (
            f"You are an expert rubric designer for a personalized AI training system. Your task is to generate a detailed evaluation rubric for each question, taking into account:"

            f"1. The question's Bloom's taxonomy level (e.g., Remember, Understand, Apply, Analyze, Evaluate, Create). 2. The question type (short answer, long answer, or coding task). 3. The need for automated grading. 4. The system's mastery logic, based on the following scoring bands:"
            
            f"Recommended Scoring Bands (Per Bloom Level)"
            f"Mastered (≥ 80%, and no score < 60%) - skip this Bloom level"
            f"Proficient (65-79%) - suggest optional content or confidence boost"
            f"Needs Improvement (< 60%) - recommend targeted content based on rubric gap"

            "Your output should include:"

            "A. A **rubric table** with the following columns:"
            "- Criterion: What is being evaluated (e.g., Accuracy, Completeness, Code Output)."
            "- **Score Range**: The points range (e.g., 0-4)."
            "- **Scoring Guide**: What earns full, partial, or no credit."

            "B. **Rubric rules tailored to Bloom's Level**:"
            "- For *Remember/Understand*: Focus on correctness of facts, definitions, and completeness."
            "- For *Apply/Analyze*: Include application of concepts, logical structure, and process."
            "- For *Evaluate/Create*: Emphasize depth, justification, originality, or design reasoning."
            "- For *Coding Questions*: Include criteria such as Correct Output, Code Quality, and Problem Solving."

            "C. **Allow partial credit** where applicable."

            "D. **Support multiple solution paths** (especially for higher-order questions)."

            "E. At the end, provide:"
            "Minimum Passing Rule: Total score ≥ 70% AND no criterion < 50%."
            f"These are the questions for which rubrik needs to be generated: {questions_str}\n\n"
        )

        rubric_response = query_engine2.query(rubric_prompt).response
        print("\nGenerated Rubric:\n", rubric_response)'''

        return StopEvent()
        



# ------ Workflow Execution ------
async def main():

    workflow = Objective_Workflow()
    await workflow.run(query = {"course": 'NodeJS', "module": '04 - Servers'})

if __name__ == "__main__":
    asyncio.run(main())