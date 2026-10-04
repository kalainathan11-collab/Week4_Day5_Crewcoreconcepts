import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_community.utilities import SerpAPIWrapper

# Load environment variables from .env file
load_dotenv()

# Verify that the keys are loaded correctly
SERP_KEY = os.getenv("SERPER_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not SERP_KEY:
    raise ValueError("SERPER_API_KEY is missing from your .env file!")

# Define SerpApi search tool using CrewAI's @tool decorator
@tool("SerpApi Web Search")
def serp_search_tool(query: str) -> str:
    """Useful for searching the web for current real-time information, research facts, and news."""
    search = SerpAPIWrapper(serpapi_api_key=SERP_KEY)
    return search.run(query)

# Define Researcher Agent
researcher = Agent(
    role="Senior Technology Researcher",
    goal="Uncover cutting-edge developments and factual insights on {topic}",
    backstory=(
        "You are a seasoned technology analyst known for meticulous research. "
        "You gather up-to-date facts, trends, and architectural breakthroughs."
    ),
    tools=[serp_search_tool],
    verbose=True
)

# Define Tech Writer Agent
writer = Agent(
    role="Tech Content Strategist & Writer",
    goal="Transform technical research findings into clear, structured, executive reports",
    backstory=(
        "You are an expert technology communicator who distills complex research "
        "into structured markdown reports with key takeaways and actionable insights."
    ),
    verbose=True
)

# Define Tasks
research_task = Task(
    description="Conduct thorough search on {topic} and compile key findings.",
    expected_output="Detailed factual summary covering key technical breakthroughs, trends, and real-world applications.",
    agent=researcher
)

write_task = Task(
    description="Review the researcher's summary and write a polished executive report on {topic}.",
    expected_output="A well-formatted Markdown report featuring an Executive Summary, Key Breakthroughs, and Industry Outlook.",
    agent=writer
)

# Define and execute Crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential
)

if __name__ == "__main__":
    result = crew.kickoff(inputs={"topic": "Autonomous Multi-Agent AI Systems"})
    print("\n==========================================")
    print("           FINAL WRITTEN REPORT           ")
    print("==========================================\n")
    print(result.raw)   