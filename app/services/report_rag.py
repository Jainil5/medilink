import sys
from pathlib import Path
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================= PATH OPTIMIZATION =================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

model = ChatOllama(
    model="gpt-oss:20b-cloud",
    temperature=0,
)

loader = PyPDFLoader(str(BASE_DIR / "datasets/report/jainil-report.pdf"))

docs = loader.load()
print(f"Total characters: {len(docs[0].page_content)}")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  
    chunk_overlap=200,  
    add_start_index=True,  
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")

vector_store = Chroma(
    collection_name="medical_report",
    embedding_function=embeddings,
    persist_directory=str(BASE_DIR / "chroma_langchain_db"), 
)

document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


from langchain.agents import create_agent

tools = [retrieve_context]

prompt = (
    "You have access to a tool that retrieves context from a blog post. ",
    "Use the tool to help answer user queries."
)
agent = create_agent(model, tools, system_prompt=prompt)

query = (
    "What is langchain?",
    "what are the data sources of the project?"
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()