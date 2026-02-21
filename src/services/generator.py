"""
LLM Generator Module
====================
Uses Google Gemini to generate answers with inline citations.
"""

import google.generativeai as genai
from src.core.logger import get_logger
from typing import List, Dict

logger = get_logger(__name__)


class GeminiGenerator:
    """Generates answers using Google Gemini with context and citations."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        logger.info(f"🤖 Gemini generator initialized: {model_name}")

    def generate(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Generate a response using Gemini with retrieved context.

        Args:
            query: The user's question
            context_chunks: List of dicts with 'text', 'source_file', 'page_number'

        Returns:
            Generated answer string with inline citations
        """
        if not context_chunks:
            return "No relevant documents found in the knowledge base. Please upload some PDFs first."

        # Format context with source labels
        context_blocks = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get("source_file", "Unknown")
            page = chunk.get("page_number", "N/A")
            text = chunk.get("text", "")
            context_blocks.append(
                f"[Document {i} | Source: {source}, Page: {page}]\n{text}"
            )

        context_text = "\n\n".join(context_blocks)

        prompt = f"""You are a highly knowledgeable AI research assistant. Your task is to answer a question using ONLY the provided reference documents.

REFERENCE DOCUMENTS:
{context_text}

USER QUESTION: {query}

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

## [Main Topic Title]

Provide a brief overview paragraph that directly addresses the question (2-3 sentences).

### [Subtopic 1]
- Key point with detail and explanation [1]
- Another key point drawn from the documents [2]
- Continue with relevant information

### [Subtopic 2]
- More structured insights from the documents [1]
- Additional details as needed [3]

(Add more subtopics as needed to thoroughly cover the answer)

### Key Takeaways
- Summarize the 2-3 most important points

---

**References:**
1. *Document Name* — Page X
2. *Document Name* — Page Y
3. *Document Name* — Page Z

RULES:
1. Use ONLY information from the provided documents. Never fabricate facts.
2. Use inline citation numbers like [1], [2], [3] throughout your answer to reference specific documents.
3. The **References** section at the end MUST list every cited document with its exact filename and page number.
4. Structure your answer with a clear ## heading, ### subheadings, and bullet points.
5. Be thorough and detailed — aim for comprehensive coverage of the topic.
6. If the documents lack enough information, state that clearly and answer with what is available."""

        logger.info(f"📝 Generating response for: '{query[:80]}...'")

        try:
            response = self.model.generate_content(prompt)
            logger.info("✅ Response generated successfully")
            return response.text
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return f"Error generating response: {str(e)}"
