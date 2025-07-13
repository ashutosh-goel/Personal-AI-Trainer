from llama_index.core import Settings
from llama_index.core.workflow import Workflow, step, Context
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Union
from pathlib import Path
from llama_index.core import VectorStoreIndex
import os
from llama_index.readers.file import PyMuPDFReader
from llama_index.llms.ollama import Ollama
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
import json
from workflows.context.state_store import DictState

from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Event,
    HumanResponseEvent
)
import asyncio

# Building the RAG Pipeline

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",  # Use "gemma:7b" for the 7B model
)

Settings.llm = Ollama(
    model="gemma:2b",  # Use "gemma:7b" for the 7B model
    request_timeout=600.0,  # Set a longer timeout for larger models
)
from pathlib import Path

def add_folder_metadata(filepath: str):
    path = Path(filepath)
    course = path.parts[-3]  # Assuming structure: /Courses/CourseX/CourseX/ModuleY/Topic.pdf
    module = path.parts[-2]
    topic = path.name
    return {"course": course, "module": module, "topic": topic}

reader = SimpleDirectoryReader(
    input_dir=r"C:\Courses1",
    recursive=True,
    file_extractor={".pdf": PyMuPDFReader()},
    file_metadata=add_folder_metadata
)

from llama_index.core import (StorageContext, load_index_from_storage)

# Check if storage already exists
Persist_Dir = "./storage_sample"
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

query_engine = index.as_query_engine(similarity_top_k=3)


# ---------- Models ----------

class ObjectiveWorkflowState(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    course: str
    module: str
    combined_text: str
    objectives: List[str] = []
    bloom_mapped: List[Dict] = []
    student_answers: Dict[str, str] = {}
    evaluation: Dict[str, Dict] = {}

# ---------- Events ----------

class Event1(Event):
    query: str = ""

class Event2(Event):
    query: str = ""

class SetupEvent(Event):
    query: str = ""

class StateStartEvent(StartEvent):
    """StartEvent that can carry initial state data."""
    def __init__(self, state: Dict = None, **kwargs):
        super().__init__(**kwargs)
        self.state = state or {}

# ---------- Workflow Definition ----------

class LearningAssessmentWorkflow(Workflow):

    # ---------- Step 1: Extract Learning Objectives ----------

    @step
    async def extract_learning_objectives(self, ctx: Context[ObjectiveWorkflowState], ev: StartEvent) -> Event1:
        # Get state from context or use the provided state in StartEvent
        state_data = await ctx.store.get_state(default=None)

        if state_data is None:
            # Check if state is provided in the StartEvent
            if hasattr(ev, 'state') and ev.state:
                state = ObjectiveWorkflowState(**ev.state)
                print(f"Using state from StartEvent: Course: {state.course}, Module: {state.module}")
            else:
                # If no state is provided, initialize it with default values
                state = ObjectiveWorkflowState(
                    course="",
                    module="",
                    combined_text="",
                    objectives=[],
                    bloom_mapped=[],
                    student_answers={},
                    evaluation={}
                )
                print("Using default empty state")
        else:
            # Convert stored state back to ObjectiveWorkflowState
            state = ObjectiveWorkflowState(**state_data)
            print(f"Using stored state: Course: {state.course}, Module: {state.module}")

        # Store the state in context
        await ctx.store.set_state(state.model_dump())

        # Only proceed if we have content to work with
        if not state.combined_text.strip():
            print("Warning: No content available for objective extraction")
            return Event1()

        prompt = f"""
        You're a curriculum designer. Given the module text below for the course "{state.course}" and module "{state.module}", generate 4-6 clear, measurable learning objectives for each module.

        Module Text:
        {state.combined_text[:2000]}...
        """

        print(f"Extracting objectives for: {state.course} - {state.module}")
        response = Settings.llm.complete(prompt).text.strip()
        objectives = [line.strip("1234567890.- ").strip() for line in response.split("\n") if line.strip()]
        state.objectives = objectives[:6]

        # Update the stored state
        updated_state_data = await ctx.store.get_state()
        updated_state = ObjectiveWorkflowState(**updated_state_data)
        updated_state.objectives = state.objectives
        await ctx.store.set_state(updated_state.model_dump())

        # Print the extracted objectives for debugging
        print("Extracted Learning Objectives:")
        for i, obj in enumerate(state.objectives, 1):
            print(f"{i}. {obj}")

        return Event1()

    # ---------- Step 2: Tag Bloom's Level and Generate Questions ----------

    @step
    async def map_blooms_and_generate_questions(self, ctx: Context[ObjectiveWorkflowState], ev: Event1) -> Event2:
        state_data = await ctx.store.get_state()
        state = ObjectiveWorkflowState(**state_data)

        objectives = state.objectives
        result = []
        taxonomy_levels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

        for obj in objectives:
            questions = {}
            for level in taxonomy_levels:
                prompt_questions = f"""
                Write 3 {level}-Bloom's Taxonomy level assessment questions for this learning objective:
                "{obj}"
                """
                q_resp = Settings.llm.complete(prompt_questions).text.strip()
                q_lines = [line.strip("1234567890.- ") for line in q_resp.split("\n") if line.strip()]
                questions[level] = q_lines[:3]

            result.append({
                "objective": obj,
                "questions_by_level": questions
            })

        state.bloom_mapped = result
        await ctx.store.set_state(state.model_dump())
        return Event2()

    # ---------- Step 3: Print Questions and wait for Student Answers ----------

    @step
    async def print_questions(self, ctx: Context[ObjectiveWorkflowState], ev: Event2) -> StopEvent:
        state_data = await ctx.store.get_state()
        state = ObjectiveWorkflowState(**state_data)

        for item in state.bloom_mapped:
            obj = item["objective"]
            for level, questions in item["questions_by_level"].items():
                for q in questions:
                    print(f"Objective: {obj} | Bloom Level: {level} | Question: {q}")

        return StopEvent()

# ---------- Utility ----------

'''def save_output(ctx: Context, folder="./results"):
    Path(folder).mkdir(parents=True, exist_ok=True)
    with open(Path(folder) / f"{ctx.state.course}_{ctx.state.module}_output.json", "w", encoding="utf-8") as f:
        json.dump(ctx.state.dict(), f, indent=2, ensure_ascii=False)'''

'''def load_student_answers(file_path: str) -> Dict[str, str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)'''

'''# ---------- Example Run ----------

if __name__ == "__main__":
    from llama_index.llms.ollama import Ollama

    Settings.llm = Ollama(model="gemma:2b")

    combined_text = """
    Recursion is a method of solving problems where a function calls itself...
    """
    #student_answers = load_student_answers("./student_answers.json")

    state = ObjectiveWorkflowState(
        course="JavaScript",
        module="Working with Arrays",
        combined_text=combined_text,
        #student_answers=student_answers
    )

    wf = LearningAssessmentWorkflow()
    ctx = wf.run(state)

    #save_output(ctx)'''



# ---------- Enhanced Query Functions with Metadata ----------

def query_course_module_content(course: str, module: str) -> str:
    """Query RAG pipeline for specific course and module content using metadata-aware queries."""
    # Create a targeted query that leverages the metadata structure
    query_text = f"""
    Provide a comprehensive summary of all content from the module '{module}'
    in the course '{course}'. Include all key concepts, topics, and learning materials
    covered in this specific module.
    """

    # Query the RAG pipeline
    response = query_engine.query(query_text)
    return response.response.strip()

def get_available_courses_modules():
    """Get available courses and modules from the indexed documents."""
    from collections import defaultdict

    # Load documents to extract metadata
    documents = reader.load_data()
    module_docs = defaultdict(list)

    for doc in documents:
        course = doc.metadata.get("course", "Unknown")
        module = doc.metadata.get("module", "Unknown")
        key = (course, module)
        module_docs[key].append(doc.text[:500])  # Sample text for verification

    return dict(module_docs)

# ---------- Utility Functions for State Management ----------

def create_workflow_state(course: str, module: str) -> ObjectiveWorkflowState:
    """Create a properly initialized ObjectiveWorkflowState with RAG-queried content."""
    combined_text = query_course_module_content(course, module)

    return ObjectiveWorkflowState(
        course=course,
        module=module,
        combined_text=combined_text,
        objectives=[],
        bloom_mapped=[],
        student_answers={},
        evaluation={}
    )

def save_workflow_state(state: ObjectiveWorkflowState, output_dir: str = "./workflow_outputs"):
    """Save the workflow state to a JSON file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"{state.course}_{state.module}_workflow_state.json"
    filepath = Path(output_dir) / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state.model_dump(), f, indent=2, ensure_ascii=False)

    print(f"Workflow state saved to: {filepath}")
    return filepath

def load_workflow_state(filepath: str) -> ObjectiveWorkflowState:
    """Load a workflow state from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return ObjectiveWorkflowState(**data)

# ---------- Enhanced Workflow Execution ----------

async def extract_objectives_manually(state: ObjectiveWorkflowState) -> ObjectiveWorkflowState:
    """Manually extract objectives as a fallback when workflow fails."""
    if not state.combined_text.strip():
        print("No content available for objective extraction")
        return state

    prompt = f"""
    You're a curriculum designer. Given the module text below for the course "{state.course}" and module "{state.module}", generate 4-6 clear, measurable learning objectives.

    Module Text:"""
    #{state.combined_text[:2000]}...


    print("Manually extracting learning objectives...")
    response = query_engine.query(prompt).response.strip() #Settings.llm.complete(prompt).text.strip()
    objectives = [line.strip("1234567890.- ").strip() for line in response.split("\n") if line.strip()]
    state.objectives = objectives[:6]

    print(f"Extracted {len(state.objectives)} learning objectives:")
    for i, obj in enumerate(state.objectives, 1):
        print(f"  {i}. {obj}")

    return state

async def run_learning_assessment_workflow(course: str, module: str) -> ObjectiveWorkflowState:
    """Run the complete learning assessment workflow for a specific course and module."""

    # Create initial state with RAG-queried content
    initial_state = create_workflow_state(course, module)

    print(f"Starting workflow for Course: {course}, Module: {module}")
    print(f"Content length: {len(initial_state.combined_text)} characters")

    # For now, use manual extraction as it's more reliable
    # The workflow system seems to have compatibility issues
    try:
        final_state = await extract_objectives_manually(initial_state)

        # Save the final state
        save_workflow_state(final_state)

        return final_state

    except Exception as e:
        print(f"Error during objective extraction: {e}")
        # Return the initial state as fallback
        save_workflow_state(initial_state)
        return initial_state

async def run_workflow_with_context(course: str, module: str) -> ObjectiveWorkflowState:
    """Alternative workflow execution method using direct context management."""

    # Create initial state with RAG-queried content
    initial_state = create_workflow_state(course, module)

    print(f"Starting workflow with context for Course: {course}, Module: {module}")

    try:
        # Initialize workflow
        workflow = LearningAssessmentWorkflow(timeout=300, verbose=False)

        # Create context and set initial state
        ctx = Context(workflow)
        await ctx.store.set_state(initial_state)

        # Create a StartEvent with state
        start_event = StateStartEvent(state=initial_state)
        # Run the workflow
        await workflow.run(start_event)

        # Get final state
        final_state_data = await ctx.store.get_state(default=initial_state)
        final_state = ObjectiveWorkflowState(**final_state_data)

        # Save the final state
        save_workflow_state(final_state)

        return final_state

    except Exception as e:
        print(f"Workflow execution error: {e}")
        print("Falling back to manual objective extraction...")
        return await extract_objectives_manually(initial_state)

# ---------- Example Usage ----------

async def main():
    """Main function demonstrating the enhanced workflow usage."""

    # Example 1: List available courses and modules
    print("Available Courses and Modules:")
    available = get_available_courses_modules()
    for (course, module), _ in list(available.items())[:5]:  # Show first 5
        print(f"  Course: {course}, Module: {module}")

    # Example 2: Run workflow for specific course and module
    target_course = "Angular"
    target_module = "03 - Typescript Basics"

    try:
        # Use the more reliable manual extraction approach
        print(f"\nRunning learning assessment for: {target_course} - {target_module}")
        final_state = await run_workflow_with_context(target_course, target_module)
        #await run_learning_assessment_workflow(target_course, target_module)

        print(f"\nWorkflow completed successfully!")
        print(f"Generated {len(final_state.objectives)} learning objectives")
        print(f"Generated {len(final_state.bloom_mapped)} Bloom's taxonomy mappings")

        # Display some results
        if final_state.objectives:
            print("\nExtracted Learning Objectives:")
            for i, obj in enumerate(final_state.objectives, 1):
                print(f"  {i}. {obj}")
        else:
            print("No objectives were extracted. Check the content and try again.")

    except Exception as e:
        print(f"Error running workflow: {e}")
        import traceback
        traceback.print_exc()

        # Fallback: Create and save a basic state for testing
        print("\nCreating basic state for testing...")
        try:
            basic_state = create_workflow_state(target_course, target_module)
            save_workflow_state(basic_state)
            print("Basic state created and saved.")
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")

if __name__ == "__main__":
    asyncio.run(main())



