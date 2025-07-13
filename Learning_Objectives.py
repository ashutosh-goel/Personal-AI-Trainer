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

from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Event,
    HumanResponseEvent,
    Workflow,
    step,
    Context
)
import asyncio

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",  # Use "gemma:7b" for the 7B model
)

Settings.llm = Ollama(
    model="gemma:2b",  # Use "gemma:7b" for the 7B model
    request_timeout=600.0,  # Set a longer timeout for larger models
)

# ------ Events ------

class IngestEvent(Event):
    documents: list[Document]

# ------ Workflow Definition ------

class Workflow(Workflow):
    @step
    async def ingest(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        """Ingest step (for ingesting docs and initializing index)."""
        documents: list[Document] | None = ev.get("documents")

        if documents is None:
            return None

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

        return StopEvent(result=index)   

#------ Metadata Function ------

def metadata(filepath: str):
    path = Path(filepath)
    course = path.parts[-3]  # Assuming structure: /Courses/CourseX/CourseX/ModuleY/Topic.pdf
    module = path.parts[-2]
    topic = path.name
    return {"course": course, "module": module, "topic": topic}

# ------ Workflow Execution ------
async def main():

    reader = SimpleDirectoryReader(
        input_dir=r"C:\Courses1",
        recursive=True,
        file_extractor={".pdf": PyMuPDFReader()},
        file_metadata=metadata
    )

    workflow = Workflow()
    await workflow.start()