from pypdf import PdfReader
from typing import List

def load_pdf(file_path: str) -> List[dict]:
    """Loads a pdf and ectract text page by page
    return:
        List of dictionaries with page content and metadata"""
    
    reader=PdfReader(file_path)
    documents=[]

    for page_number,page in enumerate(reader.pages):
        text=page.extract_text()
        if text:
            documents.append(
                {
                    "page_content": text.strip(),
                    "metadata": {
                        "page": page_number +1,
                        "source": file_path
                    },
                }
            )

    return documents