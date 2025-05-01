from langchain_core.prompts import PromptTemplate
from llama_index.core.prompts import LangchainPromptTemplate
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# Few-shot examples and prompt templates
examples = [
    {
        "question": "What is the statute of limitations for breach of contract in California?",
        "answer": "The statute of limitations for breach of contract in California is generally four years under Code of Civil Procedure section 337. Thanks for asking!"
    },
    {
        "question": "Can a minor enter into a legally binding contract?",
        "answer": "Minors typically cannot enter into binding contracts except for necessities. The contract is voidable at the minor's discretion. Thanks for asking!"
    }
]

example_template = """
Example Question: {question}
Example Answer: {answer}"""

example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template=example_template
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="""You are a highly experienced Legal Expert with over 10 years of practice in the field. Your task is to provide accurate, concise, and professional answers based on the provided context. Follow these guidelines strictly:       
            1. Limit your response to 3-4 sentences. Never repeat the final response.
            2. Never generate any new query by yourself.
            3. Maintain a formal and professional tone throughout your response.
            4. Always end your response with "Thanks for asking!" to maintain a polite and professional demeanor.
        """,
    suffix="""\n\nContext: {context}
Question: {question}

Final response:""",
    input_variables=["context", "question"],
    example_separator="\n\n"
)

custom_prompt = LangchainPromptTemplate(
    template=few_shot_prompt,
    template_var_mappings={"query_str": "question", "context_str": "context"},
)