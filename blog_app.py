
import os
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

# --------------------------------------------------
# Load environment variables from .env
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# Check API key from .env
# --------------------------------------------------
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not configured in the .env file.")
    st.stop()

# --------------------------------------------------
# Streamlit Page Setup
# --------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent AI Blog Generator",
    page_icon="✍️",
    layout="wide"
)

st.title("✍️ Multi-Agent AI Blog Generator")
st.markdown("Powered by **CrewAI** & **Streamlit**")

# --------------------------------------------------
# Sidebar - Model Selection Only
# --------------------------------------------------
st.sidebar.header("Configuration")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["gpt-4o-mini", "gpt-4o"],
    index=0
)

# --------------------------------------------------
# Input Controls
# --------------------------------------------------
topic = st.text_input(
    "Enter the topic for your blog post:",
    placeholder="e.g., The Future of Autonomous AI Agents in 2026"
)

target_audience = st.selectbox(
    "Target Audience:",
    [
        "General Public",
        "Software Developers",
        "Business Executives",
        "Tech Enthusiasts"
    ]
)

# --------------------------------------------------
# Define CrewAI Multi-Agent Workflow
# --------------------------------------------------
def run_blog_crew(blog_topic, audience, model_name):

    # 1. Researcher Agent
    researcher = Agent(
        role="Senior Content Researcher",
        goal=(
            f"Discover accurate, relevant, and compelling research "
            f"points about '{blog_topic}' tailored for {audience}."
        ),
        backstory=(
            "You are an elite researcher with a knack for identifying "
            "key insights, architectural details, and trends."
        ),
        llm=model_name,
        verbose=True
    )

    # 2. Tech Writer Agent
    writer = Agent(
        role="Lead Blog Writer",
        goal=(
            f"Write an engaging, well-structured, narrative-driven "
            f"blog post about '{blog_topic}'."
        ),
        backstory=(
            "You are an expert tech columnist skilled at making "
            "complex technical concepts simple and engaging."
        ),
        llm=model_name,
        verbose=True
    )

    # 3. SEO Editor Agent
    editor = Agent(
        role="SEO & Content Editor",
        goal=(
            "Optimize the draft for search engines, ensure clear "
            "headings, and refine tone and readability."
        ),
        backstory=(
            "You are an experienced SEO editor who ensures high "
            "keyword density, catchy title tags, and readable "
            "Markdown formatting."
        ),
        llm=model_name,
        verbose=True
    )

    # --------------------------------------------------
    # Tasks
    # --------------------------------------------------

    research_task = Task(
        description=(
            f"Conduct thorough research on '{blog_topic}'. "
            f"Outline 5 key takeaways, facts, or technical insights "
            f"for a target audience of {audience}."
        ),
        expected_output=(
            "A bulleted summary of key findings, core trends, and facts."
        ),
        agent=researcher
    )

    writing_task = Task(
        description=(
            f"Use the research summary to compose a comprehensive "
            f"blog post on '{blog_topic}'. Include an introduction, "
            f"body sections with subheadings, and a conclusion."
        ),
        expected_output=(
            "A complete full-length blog post draft in Markdown."
        ),
        agent=writer
    )

    seo_task = Task(
        description=(
            "Review and optimize the blog draft. Add SEO recommendations "
            "(Meta Title, Meta Description, Key Keywords) at the top, "
            "and format the post neatly in clean Markdown."
        ),
        expected_output=(
            "A polished, publication-ready Markdown document with SEO metadata."
        ),
        agent=editor
    )

    # --------------------------------------------------
    # Assemble Crew
    # --------------------------------------------------

    blog_crew = Crew(
        agents=[
            researcher,
            writer,
            editor
        ],
        tasks=[
            research_task,
            writing_task,
            seo_task
        ],
        process=Process.sequential,
        verbose=True
    )

    # --------------------------------------------------
    # Execute Workflow
    # --------------------------------------------------

    return blog_crew.kickoff()


# --------------------------------------------------
# Main Action Button
# --------------------------------------------------

if st.button("Generate Blog Post", type="primary"):

    if not topic.strip():

        st.warning(
            "Please enter a blog topic before generating."
        )

    else:

        with st.spinner(
            "🤖 The AI Agents are researching, writing, and editing your post..."
        ):

            try:

                # Run multi-agent pipeline
                result = run_blog_crew(
                    topic,
                    target_audience,
                    model_choice
                )

                st.success(
                    "Blog Post Generated Successfully!"
                )

                # Display output
                st.subheader("Generated Blog Post")

                st.markdown(result)

                # Download Markdown
                st.download_button(
                    label="📥 Download Markdown",
                    data=str(result),
                    file_name=(
                        f"{topic.lower().replace(' ', '_')}_blog.md"
                    ),
                    mime="text/markdown"
                )

            except Exception as e:

                st.error(
                    f"An error occurred during execution: {e}"
                )
