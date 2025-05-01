# Import necessary libraries
import os
import re
import gradio as gr
import faiss
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import load_index_from_storage
from llama_index.core.schema import IndexNode
from llama_index.core.retrievers import BaseRetriever, RecursiveRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from prompt import custom_prompt
from trulens.core import TruSession
from trulens.apps.llamaindex import TruLlama
from trulens.core import Feedback
from trulens.providers.openai import OpenAI
from trulens.dashboard.display import get_feedback_result
from trulens.apps.llamaindex.guardrails import WithFeedbackFilterNodes
import numpy as np
from crewai import Agent, Task, Crew
from crewai import LLM
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

# Global variables
llm = None
embed_model = None
query_engine = None
faiss_index_documents = None
vector_store = None
storage_context = None

# Data directory
data_path = "./data"
os.makedirs(data_path, exist_ok=True)

# Initialize FAISS index
def initialize_faiss_index():
    global faiss_index_documents, vector_store, storage_context
    d = 1024
    faiss_index_documents = faiss.IndexFlatL2(d)
    vector_store = FaissVectorStore(faiss_index=faiss_index_documents)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

initialize_faiss_index()


# # Initialize Models from local directory
# llm = HuggingFaceLLM(
#     model_name="wahid028/llama-3-merged-linear",
#     tokenizer_name="wahid028/llama-3-merged-linear",
#     max_new_tokens=512,
#     generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95}
# )
# embed_model = HuggingFaceEmbedding(model_name="wahid028/GIST-Law-Embed")


# Initialize Models from remote directory and local directory
llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
embed_model = HuggingFaceEmbedding(model_name="avsolatorio/GIST-large-Embedding-v0")

Settings.llm = llm
Settings.embed_model = embed_model


# Function to process and index PDFs
def process_pdf(files):
    global query_engine, faiss_index_documents, vector_store, storage_context

    # Clear the data directory
    for file in os.listdir(data_path):
        os.remove(os.path.join(data_path, file))

    indexed_files = []
    try:
        # File handling
        for file_path in files:
            new_path = os.path.join(data_path, os.path.basename(file_path))
            os.rename(file_path, new_path)
            indexed_files.append(new_path)

        # Load documents
        document_reader = SimpleDirectoryReader(data_path)
        documents = document_reader.load_data() # newly added num_workers=2

        print("Loading document done")

        # Node splitting
        text_splitter = SentenceSplitter(chunk_size=512)
        nodes = text_splitter.get_nodes_from_documents(documents)

        print("Splitting document done")

        # Check if documents are indexed properly
        if not nodes:
            return "❌ Error: No valid text found in the uploaded documents. Please check the file format and content."

        # Prepare sub-nodes
        sub_chunk_sizes = [256, 512]
        sub_node_parsers = [SentenceSplitter(chunk_size=c) for c in sub_chunk_sizes]

        all_nodes = []
        for base_node in nodes:
            for n in sub_node_parsers:
                sub_nodes = n.get_nodes_from_documents([base_node])
                sub_inodes = [
                    IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
                ]
                all_nodes.extend(sub_inodes)
            original_node = IndexNode.from_text_node(base_node, base_node.node_id)
            all_nodes.append(original_node)

        all_nodes_dict = {n.node_id: n for n in all_nodes}

        print("Sub-nodes creation done")

        # Reinitialize FAISS index
        initialize_faiss_index()

        print("Indexing documents...")
        # Store nodes directly
        storage_context.docstore.add_documents(all_nodes)
        index = VectorStoreIndex(all_nodes, storage_context=storage_context)
        index.storage_context.persist("storage_documents")
        vector_store = FaissVectorStore.from_persist_dir("./storage_documents")
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir="./storage_documents")
        index = load_index_from_storage(storage_context=storage_context)

        print("Document indexing done")

        # Ensure the retriever doesn't exceed the number of available nodes
        top_k = min(5, len(nodes))

        # Simplified retriever setup
        vector_retriever_documents = index.as_retriever(similarity_top_k=top_k)
        vector_retriever = RecursiveRetriever("vector", retriever_dict={"vector": vector_retriever_documents}, node_dict=all_nodes_dict, verbose=True)
        bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)
        hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)

        print("Hybrid retriever creation done")

        # Initialize SentenceTransformerRerank
        reranker = SentenceTransformerRerank(top_n=min(3, len(nodes)), model="BAAI/bge-reranker-large")

        # Configure response synthesizer
        response_synthesizer = get_response_synthesizer(response_mode="compact")

        print("Initializing query engine, this may take a while...")
        # Initialize query engine
        query_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
            node_postprocessors=[reranker],
            response_synthesizer=response_synthesizer,
            # streaming = True, # Newly added, Enable streaming mode for large datasets
        )
        print("Query engine initialization done")

        # Update query engine prompts with custom RAG prompt
        query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": custom_prompt}
        )

        return f"✅ Indexed {len(indexed_files)} document(s) successfully."

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return f"❌ Error: {str(e)}"



# Initialize LLM and CrewAI agents
llm = LLM(
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
    llm=llm,
    max_retry_limit=1,
    verbose=True
)

LLM_Agent = Agent(
    role="General AI Assistant",
    goal="Generate responses to user queries based on the model own knowledge.",
    backstory="""An AI assistant designed to provide information and answer questions based on its training data.""",
    allow_delegation=False,
    llm=llm,
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


# Function to refine query

def refine_query(query):
    """
    Refines the given query using CrewAI's prompt engineering agent.

    Parameters:
    query (str): The original user query that needs refinement.

    Returns:
    str: The refined query extracted from <answer> tags, or the original query if not found.
    """
    result = crew_prompt.kickoff(inputs={"topic": query})  # Run the task

    # Debugging: Print CrewAI output structure
    print("🔍 CrewAI Output:", result.tasks_output)

    if result.tasks_output and isinstance(result.tasks_output, list):
        task_output = result.tasks_output[0]  # Extract first task output

        if hasattr(task_output, "raw") and isinstance(task_output.raw, str):
            raw_text = task_output.raw.strip()

            # Use regex to extract content inside <answer> tags
            match = re.search(r'<answer>(.*?)</answer>', raw_text, re.DOTALL)

            if match:
                refined_query = match.group(1).strip()  # Extract and clean the refined query
                return refined_query  # Return the extracted refined query
            else:
                print("⚠️ No content found inside <answer> tags. Returning original query.")
                return query  
        else:
            print("⚠️ No 'raw' attribute found in TaskOutput. Returning original query.")
            return query  
    else:
        print("⚠️ Empty tasks_output. Returning original query.")
        return query


# Function to process user query with evaluation

def query_model(user_query):
    if query_engine is None:
        return "⚠️ Please upload documents first."
    
    max_attempts = 3
    attempt = 0
    
    while attempt < max_attempts:
        response = query_engine.query(user_query)
        
        # Initialize TruLens session
        session = TruSession()
        session.reset_database()
        provider = OpenAI(api_key=OPENAI_API_KEY)
        context = TruLlama.select_context(query_engine)
        
        f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness").on(context.collect()).on_output()
        f_answer_relevance = Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance").on_input_output()
        f_context_relevance = Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance").on_input().on(context).aggregate(np.mean)
        
        tru_query_engine_recorder = TruLlama(
            query_engine,
            app_name="LlamaIndex_App",
            app_version="base",
            feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
        )
        
        with tru_query_engine_recorder as recording:
            response = query_engine.query(user_query)
            print(response)

        last_record = recording.records[-1]
        get_feedback_result(last_record, "Context Relevance")
        
        leaderboard_df = session.get_leaderboard()
        print(leaderboard_df)

        # Check if leaderboard_df is empty
        if leaderboard_df.empty:
            return "⚠️ No evaluation data available. Unable to determine response quality."

        # Safely retrieve values with .get() to avoid KeyErrors
        answer_relevance = leaderboard_df.iloc[0].get("Answer Relevance", None)
        context_relevance = leaderboard_df.iloc[0].get("Context Relevance", None)
        groundedness = leaderboard_df.iloc[0].get("Groundedness", None)

        # If context_relevance is 0.0, use LLM_Agent to generate a response
        if (context_relevance is not None and context_relevance <= 0.09) or (groundedness is not None and groundedness <= 0.1):
            print("Context relevance is 0.0. Using LLM_Agent to generate a response.")
            result = crew_llm.kickoff(inputs={"topic": user_query})
            
            if result.tasks_output and isinstance(result.tasks_output, list):
                task_output = result.tasks_output[0]  # Extract first task output

                if hasattr(task_output, "raw") and isinstance(task_output.raw, str):
                    raw_text = task_output.raw.strip()

                    # Use regex to extract content inside <LLM_answer> tags
                    match = re.search(r'<answer>(.*?)</answer>', raw_text, re.DOTALL)

                    if match:
                        llm_response = match.group(1).strip()  # Extract and clean the LLM response
                        # Add a warning message to inform the user
                        warning_message = (
                            "⚠️ Warning: The response is generated by LLM. "
                            "The provided document does not have enough information about the query. "
                            "This response may not be fully accurate.\n\n"
                            f"## Final Response: {llm_response} "
                        )
                        return warning_message  # Return the extracted LLM response
                        
                    else:
                        print("⚠️ No content found inside <answer> tags. Returning original query.")
                        return user_query  
                else:
                    print("⚠️ No 'raw' attribute found in TaskOutput. Returning original query.")
                    return user_query  
            else:
                print("⚠️ Empty tasks_output. Returning original query.")
                return user_query

        # Count available values
        available_values = [v for v in [answer_relevance, context_relevance, groundedness] if v is not None]
        num_available = len(available_values)

        # Make decision based on available values
        if num_available == 3:
            if context_relevance >= 0.5 and answer_relevance >= 0.6 and groundedness >= 0.6:
                return str(f"🎯 ## Final Response: {response}")
                # return str(response)  
            elif num_available == 3:
                if context_relevance <= 0.3 and answer_relevance >= 0.8 and groundedness >= 0.8:
                    return str(f"🎯 ## Final Response: {response}")
        elif num_available == 2:
            if (context_relevance and context_relevance >= 0.5) and (answer_relevance and answer_relevance >= 0.6):
                return str(f"🎯 ## Final Response: {response}")
                # return str(response)
            elif (context_relevance and context_relevance >= 0.5) and (groundedness and groundedness >= 0.6):
                return str(f"🎯 ## Final Response: {response}")
                # return str(response)
            elif (answer_relevance and answer_relevance >= 0.6) and (groundedness and groundedness >= 0.6):
                return str(f"🎯 ## Final Response: {response}")
                # return str(response)
        elif num_available == 1:
            if context_relevance and context_relevance >= 0.5:
                return str(f"🎯 ## Final Response: {response}")
                # return str(response)
            elif answer_relevance and answer_relevance >= 0.6:
                return str(f"🎯 ## Final Response: {response}")
                # return str(response)
            elif groundedness and groundedness >= 0.6:
                return str(f"🎯 ## Final Response: {response}")
                # return str(response)

        # If no decision could be made, refine query and retry
        print(f"Attempt {attempt+1}: Query refinement needed.")
        user_query = refine_query(user_query)
        attempt += 1
    
    # return "⚠️ Sorry, we couldn't find your answer."
    # return str(response)
    return str(f"⚠️ Warning: The response doesn't pass the evaluation critaria properly. This response may not be fully accurate.\n\n ## Final Response: {response}")


# Hybrid Retriever class
class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()
    
    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)
        
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🏛️ Legal Query RAG - Your Trusted AI Assistant")

    gr.Markdown("""
    ### ℹ️ System Overview: 
    Legal Query RAG (LQ-RAG) is an advanced Retrieval-Augmented Generation (RAG) 
    system designed specifically for legal domain Q&A. It enhances traditional RAG approaches by integrating 
    an agent-based iterative refinement mechanism. During inference, it first generates an initial response 
    to a user query and then utilizes an evaluation agent to assess its quality based on contextual relevance 
    and factual grounding. If the response does not meet predefined criteria, the evaluation agent provides 
    feedback to the prompt engineering agent, which modifies the query to improve the next response. 
    This iterative feedback loop continues until the evaluation scores approach optimal values.

    📄 **Read our IEEE Paper:** [Click here to access the paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10887211)

    - 📂 **Upload your documents** in PDF format and do indexing.
    - 🔎 **Ask your query** related to the uploaded documents.
    - 🤖 **Our AI assistant** will analyze and provide relevant insights.
    """)

    with gr.Row():
        pdf_input = gr.Files(file_types=[".pdf"], label="Upload PDF(s)")
        upload_button = gr.Button("Upload & Index")

    upload_output = gr.Textbox(label="Upload Status", interactive=False)
    upload_button.click(process_pdf, inputs=[pdf_input], outputs=[upload_output])

    user_query = gr.Textbox(label="Enter Your Legal Query")
    submit_button = gr.Button("Submit")
    response_output = gr.Textbox(label="Response", interactive=False)
    submit_button.click(query_model, inputs=[user_query], outputs=[response_output])
    user_query.submit(query_model, inputs=[user_query], outputs=[response_output])

demo.launch(share=True)