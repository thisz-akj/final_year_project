import os
from typing import List

# =========================
# ENV
# =========================
os.environ["GOOGLE_API_KEY"] = "AIzaSyDZ4qKg3JN69NY2Sb9ZkhwCNMVn-9ibzg0"

# =========================
# LANGCHAIN IMPORTS
# =========================
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =========================
# SENTENCE TRANSFORMER
# =========================
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings:
    """
    Custom embedding wrapper (NO langchain-huggingface)
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str):
        return self.model.encode(text).tolist()


# =========================
# RAG RETRIEVER
# =========================
class RagRetriever:

    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.embeddings = SentenceTransformerEmbeddings()

    def load_documents(self):
        loader = DirectoryLoader(
            "Dataset",
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        return loader.load()

    def split_documents(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        return splitter.split_documents(documents)

    def build_retriever(self):
        documents = self.load_documents()
        chunks = self.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

        return vectorstore.as_retriever(
            search_kwargs={"k": self.top_k}
        )


# =========================
# LLM HANDLER
# =========================
class LLM:

    def __init__(self):
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            transport="rest"   # 🔥 REQUIRED FOR STREAMLIT
        )

        self.ner_prompt = """
        Extract up to 3 important proper noun entities.
        Return strictly as:
        entity1;entity2;entity3

        Text: {text}
        """

        self.answer_prompt = """
        You are an epigraphist.
        Answer ONLY from the context.
        If not present, say you do not know.

        Question: {question}
        Context: {context}
        Answer:
        """

    def extract_entities(self, query: str):
        prompt = ChatPromptTemplate.from_template(self.ner_prompt)
        chain = prompt | self.model | StrOutputParser()
        response = chain.invoke({"text": query})
        entities = [e.strip() for e in response.split(";") if e.strip()]
        return [query] + entities

    def generate_answer(self, query: str, context: str):
        prompt = ChatPromptTemplate.from_template(self.answer_prompt)
        chain = prompt | self.model | StrOutputParser()
        return chain.invoke({
            "question": query,
            "context": context
        })


# =========================
# CONTEXT COLLECTOR
# =========================
class ContextCollector:

    def __init__(self, documents):
        self.documents = documents

    def collect(self, retriever, queries):
        selected_sources = set()

        for q in queries:
            docs = retriever.invoke(q)
            if docs:
                selected_sources.add(docs[0].metadata["source"])

        context = ""
        for doc in self.documents:
            if doc.metadata["source"] in selected_sources:
                context += doc.page_content + "\n"

        return context, sorted(selected_sources)


# =========================
# PIPELINE
# =========================
class RAGPipeline:

    def run_single_query(self, query: str):
        print("\n🔍 Query:", query)

        llm = LLM()
        retriever_obj = RagRetriever(top_k=3)
        retriever = retriever_obj.build_retriever()

        documents = retriever_obj.load_documents()
        entities = llm.extract_entities(query)

        collector = ContextCollector(documents)
        context, sources = collector.collect(retriever, entities)

        answer = llm.generate_answer(query, context)

        print("\n✅ ANSWER:\n", answer)
        print("📂 SOURCES:", sources)

        return answer, sources


# =========================
# HELPER (OPTIONAL)
# =========================
def answer_query(query: str):
    pipeline = RAGPipeline()
    return pipeline.run_single_query(query)


# =========================
# MAIN (CLI MODE)
# =========================
if __name__ == "__main__":
    pipeline = RAGPipeline()
    query = input("\nEnter your question: ")
    answer, sources = pipeline.run_single_query(query)

    print("\n--- FINAL ANSWER ---\n", answer)
    print("\n--- SOURCES ---")
    for s in sources:
        print("•", s)
