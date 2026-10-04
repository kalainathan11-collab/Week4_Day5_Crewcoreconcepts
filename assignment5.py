import os
import logging
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# -------------------------------------------------------------------
# 1. SETUP LOGGING & TRACE RECORDING
# -------------------------------------------------------------------
LOG_FILENAME = "agent_trace.log"
logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    filemode="a"
)

def log_event(message: str):
    """Utility function to log runtime events into terminal and local logs."""
    print(f"[TRACE]: {message}")
    logging.info(message)


# -------------------------------------------------------------------
# 2. INTERNAL VECTOR DB & RAG SETUP
# -------------------------------------------------------------------
def initialize_vector_db():
    """Reads internal policy document, creates chunks, embeds them, and persists into ChromaDB."""
    sample_file = "company_policy.txt"
    if not os.path.exists(sample_file):
        raise FileNotFoundError(
            f"Please ensure '{sample_file}' exists in the current directory before running!"
        )

    log_event("Initializing local Vector Database pipeline...")

    # Load and split document
    loader = TextLoader(sample_file)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)
    log_event(f"Created {len(docs)} document chunks.")

    # Embed and save to vector db
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db_directory = "./chroma_db"
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=db_directory
    )
    log_event("Vector store populated and persisted to disk.")
    return vectorstore


# Initialize Vector DB on startup
vectorstore_instance = initialize_vector_db()

# Custom RAG Tool
@tool("Internal RAG Search Tool")
def internal_rag_search(query: str) -> str:
    """Searches the internal document vector store for relevant company info using semantic embeddings."""
    log_event(f"Executing Internal Vector DB Query: '{query}'")
    results = vectorstore_instance.similarity_search(query, k=2)
    if not results:
        return "No relevant internal company policy records found."
    return "\n\n".join([doc.page_content for doc in results])


# -------------------------------------------------------------------
# 3. WEB SEARCH & FILE LOGGING TOOLS
# -------------------------------------------------------------------
web_search_engine = DuckDuckGoSearchRun()

@tool("Live Web Search Tool")
def web_search_tool(query: str) -> str:
    """Searches the internet for external data, news, and up-to-date industry information."""
    log_event(f"Executing External Web Search Query: '{query}'")
    return web_search_engine.run(query)


@tool("File Report Writer Tool")
def file_writer_tool(data: str) -> str:
    """Writes the final synthesized response report into a local text file."""
    output_filename = "final_rag_report.txt"
    log_event(f"Writing query findings to local file '{output_filename}'...")
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(data)
        
    log_event("Report successfully generated and persisted.")
    return f"Report successfully written to {output_filename}."


# -------------------------------------------------------------------
# 4. CREWAI AGENTS DEFINITION
# -------------------------------------------------------------------
internal_data_agent = Agent(
    role="Internal Knowledge Specialist",
    goal="Retrieve relevant facts from internal company documents using Vector DB search.",
    backstory="An expert librarian specialized in navigating internal knowledge bases.",
    tools=[internal_rag_search],
    verbose=True,
    allow_delegation=False
)

web_search_agent = Agent(
    role="Web Intelligence Specialist",
    goal="Search the web for relevant external information or industry standards.",
    backstory="A web researcher who finds relevant public standards and online references.",
    tools=[web_search_tool],
    verbose=True,
    allow_delegation=False
)

logger_writer_agent = Agent(
    role="Technical Writer and System Logger",
    goal="Synthesize internal findings and web research into a final response file.",
    backstory="A diligent publisher ensuring findings are compiled and saved safely.",
    tools=[file_writer_tool],
    verbose=True,
    allow_delegation=False
)


# -------------------------------------------------------------------
# 5. MAIN TERMINAL INTERACTION LOOP
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=======================================================")
    print("      CrewAI Multi-Agent RAG System (Terminal UI)      ")
    print("=======================================================\n")

    # Ask the user for their question dynamically
    user_query = input("👉 Enter your question for the Crew agents: ").strip()

    if not user_query:
        print("❌ Question cannot be empty. Exiting execution.")
        exit()

    log_event(f"Received User Question: '{user_query}'")

    # Define tasks dynamically using the user's question
    task_internal = Task(
        description=f"Search internal vector store data relevant to this user question: '{user_query}'",
        expected_output="Summary of relevant findings from internal documentation.",
        agent=internal_data_agent
    )

    task_web = Task(
        description=f"Search the web for additional external facts or benchmarks regarding: '{user_query}'",
        expected_output="Summary of external web findings.",
        agent=web_search_agent
    )

    task_logging_writing = Task(
        description=(
            f"Answer the user's question: '{user_query}'. "
            "Combine context from both the internal vector search and the web search. "
            "Save the complete detailed response into 'final_rag_report.txt' using the File Report Writer Tool."
        ),
        expected_output="Confirmation message that the response was saved to final_rag_report.txt.",
        agent=logger_writer_agent,
        context=[task_internal, task_web]
    )

    # Initialize Crew
    rag_crew = Crew(
        agents=[internal_data_agent, web_search_agent, logger_writer_agent],
        tasks=[task_internal, task_web, task_logging_writing],
        process=Process.sequential,
        verbose=True
    )

    # Execute agents
    print("\n🚀 Starting Crew execution...\n")
    result = rag_crew.kickoff()

    print("\n================ FINAL SYSTEM OUTPUT ================")
    print(result)
    print("\n✅ Check 'final_rag_report.txt' and 'agent_trace.log' for saved outputs.")