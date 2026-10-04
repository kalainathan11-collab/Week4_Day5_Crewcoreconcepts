import os
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# -------------------------------------------------------------------
# 1. Setup Sample Data & Build FAISS Vector Database
# -------------------------------------------------------------------
# Sample domain knowledge / facts to store in the vector DB
tech_facts = [
    "Project Alpha is a cloud-native platform using microservices architecture built on Kubernetes.",
    "Project Alpha uses FAISS for high-speed semantic search over documentation.",
    "The primary database for Project Alpha is PostgreSQL, with Redis used for caching.",
    "Security policy requires OAuth2 for authentication and AES-256 for data encryption at rest.",
    "Deployment pipelines are automated using GitHub Actions with a required 80% test coverage."
]

# Initialize OpenAI Embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create the FAISS index from text documents
vector_db = FAISS.from_texts(texts=tech_facts, embedding=embeddings)

# (Optional) Save index locally:
# vector_db.save_local("faiss_index")
# vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# -------------------------------------------------------------------
# 2. Define Custom CrewAI Tool using FAISS
# -------------------------------------------------------------------
@tool("FAISS Vector Search Tool")
def search_vector_db(query: str) -> str:
    """
    Searches the internal FAISS vector database for relevant documentation and facts 
    based on semantic similarity to the query.
    """
    # Perform similarity search retrieving top 2 matching documents
    docs = vector_db.similarity_search(query, k=2)
    
    if not docs:
        return "No relevant information found in the vector database."
    
    # Format and return retrieved document content
    results = "\n".join([f"- {doc.page_content}" for doc in docs])
    return f"Retrieved Context:\n{results}"

# -------------------------------------------------------------------
# 3. Define CrewAI Agent, Task, and Crew
# -------------------------------------------------------------------
# Knowledge Retrieval Agent equipped with the FAISS Tool
retriever_agent = Agent(
    role="Technical Knowledge Specialist",
    goal="Retrieve accurate technical facts about Project Alpha using the FAISS vector database.",
    backstory=(
        "You are an expert internal support specialist. You look up technical "
        "specifications and infrastructure details from the vector database to answer queries."
    ),
    tools=[search_vector_db],
    verbose=True
)

# Task to research and answer specific questions
retrieval_task = Task(
    description=(
        "Search the FAISS vector database for details regarding Project Alpha's "
        "security policies, database architecture, and search capabilities. "
        "Summarize the findings clearly."
    ),
    expected_output=(
        "A concise summary of Project Alpha's database, security, and search setup."
    ),
    agent=retriever_agent
)

# Assemble and Kick off Crew
crew = Crew(
    agents=[retriever_agent],
    tasks=[retrieval_task],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    # Ensure OPENAI_API_KEY environment variable is set
    result = crew.kickoff()
    print("\n--- FINAL OUTPUT ---\n")
    print(result)