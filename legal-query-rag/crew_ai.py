from crewai import Agent, Task, Crew
from crewai import LLM
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


llm_crew = LLM(
    model="groq/llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=512,
    top_p=0.9,
    frequency_penalty=0.1,
    presence_penalty=0.1,
    stop=["END"],
    seed=42
)

Prompt_Engineer = Agent(
    role="Expert Prompt Engineer",
    goal="Craft and optimize prompts to improve AI's performance.",
    backstory="""As an expert in prompt engineering, you specialize in refining and optimizing queries 
                to enhance AI-generated responses. Your deep understanding of language processing and 
                user intent allows you to create clear, effective, and contextually appropriate prompts.""",
    allow_delegation=False,
    llm=llm_crew,
    max_retry_limit=1,
    verbose=True
)

LLM_Agent = Agent(
    role="General AI Assistant",
    goal="Generate responses to user queries based on the model own knowledge.",
    backstory="""An AI assistant designed to provide information and answer questions based on its training data.""",
    allow_delegation=False,
    llm=llm_crew,
    max_retry_limit=1,
    verbose=True
)

Prompt_Engineering_Task = Task(
    description=(
        "1. Take the initial query {topic} that did not pass evaluation.\n"
        "2. Generate a refined versions of that query that maintain relevance but improve clarity.\n"
        "3. Keep the new query short, simple, concise and clear."
        "4. Each iteration generate only one query."
        "5. Always write the refined query inside the <answer></answer> tag."
    ),
    expected_output="The refined query: 'Refined query statement here'",
    agent=Prompt_Engineer,
)

LLM_Agent_Task = Task(
    description=(
        "1. Take the initial query {topic} that did not pass evaluation.\n"
        "2. Generate answer from models pre-trained knowledge.\n"
        "3. Keep the response short, simple, concise and clear within 3 sentencs."
        "4. Always write the response inside the <answer>  </answer> tags."
    ),
    expected_output="<answer> ... </answer>",
    agent=LLM_Agent,
)

crew_prompt = Crew(
    agents=[Prompt_Engineer],
    tasks=[Prompt_Engineering_Task],
    verbose=True
)

crew_llm = Crew(
    agents=[LLM_Agent],
    tasks=[LLM_Agent_Task],
    verbose=True
)
