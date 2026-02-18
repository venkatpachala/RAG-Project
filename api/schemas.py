from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid

class DocumentUploadReasponse(BaseModel):
    """
    Returned after a successful file upload and ingestion.
    The doc_id is what clients use to reference this document later.
    """
    doc_id:str=Field(description="Unique ID assigned to this document")
    filename: str =Field(description="Original filename")
    status: str =Field(description="'success' or 'failed'")
    chunk_count: int =Field