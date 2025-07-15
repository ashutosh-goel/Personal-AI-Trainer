from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Union
from pathlib import Path
import os
from llama_index.readers.file import PyMuPDFReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage, VectorStoreIndex
import json
from workflows.context.state_store import DictState
import pprint

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

api_key = "AIzaSyAv15Rnl3h1SjUT2Pa7d1Ci5LJwJ_0qctE"  # Replace with your actual Google API key
os.environ["GOOGLE_API_KEY"] = api_key
llm = GoogleGenAI(model="gemini-2.0-flash")
Settings.embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-exp-03-07", api_key=api_key)
Settings.llm = GoogleGenAI(
    model="gemini-2.0-flash",  # Use "gemma:7b" for the 7B model
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

class Output(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    course: str
    module: str
    objectives: List[str] = []
    questions: List[Questions] = Field(description="List of list of questions for each learning objective")

llm = Settings.llm.as_structured_llm(Output)
query_engine = index.as_query_engine(llm=llm)

# ------ Events ------

class GetObjectives(Event):
    course : str
    module : str

class SetAssessment(Event):
    """Event to set the assessment based on learning objectives."""
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
    async def get_objectives(self, ctx: Context, ev: GetObjectives) -> StopEvent:#SetAssessment :
        """Query the index for the objectives based on course and module."""
        query = f"You're a curriculum designer. Extract 3-6 broad learning objectives for {ev.module} module in {ev.course} course."
        response = query_engine.query(query)
        Response = response.response  # Settings.llm.complete(prompt).text.strip()
        #print(Response)
        await ctx.store.set("objectives", Response.objectives)
        print("The context store has the following objectives:")
        #print(await ctx.store.get("objectives", []))

        #print all the content
        '''print(f"Course: {Response.course}")
        print(f"Module: {Response.module}")
        print("\nObjectives:")
        for i, objective in enumerate(Response.objectives, 1):
            print(f"  {i}. {objective}")
    
        print("\nQuestions:")
        for i, question in enumerate(Response.questions, 1):
            print(f"\n  Objective {i}: {question.objective}")
            for category in ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']:
                if category in question and question.category:  # Check if category exists and is non-empty
                    print(f"    {category.capitalize()}:")
                    for j, item in enumerate(question.category, 1):
                        print(f"      {j}. {item}")'''
        
        print(f"Course: {Response.course}")
        print(f"Module: {Response.module}")
        print("\nObjectives:")
        for i, objective in enumerate(Response.objectives, 1):
            print(f"  {i}. {objective}")
        
        print("\nQuestions:")
        for i, question in enumerate(Response.questions, 1):
            print(f"\n  Objective {i}: {question.objective}")
            print(f"\n  Remember: {question.remember}")
            print(f"\n  Understand: {question.understand}")
            print(f"\n  Apply: {question.apply}")
            print(f"\n  Analyze: {question.analyze}")
            print(f"\n  Evaluate: {question.evaluate}")
            print(f"\n  Create: {question.create}")
       
        return StopEvent() #SetAssessment() 
    
    '''@step
    async def assessment(self, ctx: Context, ev: SetAssessment) -> StopEvent:
        """Generate questions based on each Bloom's Taxonomy level for each learning objectives."""
        objectives = await ctx.store.get("objectives", [])
        if not objectives:
            print("No learning objectives found.")
        
        query = "Generate questions based on each Bloom's Taxonomy level for each learning objective. "
        response = query_engine.query(query).response
        print(response)
        return StopEvent()'''
    
    '''@step
    async def input_answer_file(self, ctx: Context, ev: HumanResponseEvent) -> StopEvent:
        """Handle user input for the answer file."""
        answer_file = ev.get("answer_file")
        if not answer_file:
            print("No answer file provided.")
            return StopEvent()
        
        # Process the answer file as needed
        print(f"Received answer file: {answer_file}")
        
        return StopEvent()'''
        
        



# ------ Workflow Execution ------
async def main():

    workflow = Objective_Workflow()
    await workflow.run(query = {"course": 'NodeJS', "module": '04 - Servers'})

if __name__ == "__main__":
    asyncio.run(main())