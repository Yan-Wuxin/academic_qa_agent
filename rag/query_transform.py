from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.prompt_loader import load_rewrite_prompt, load_hyde_prompt

def build_query_rewrite_chain(llm):
    rewrite_prompt = PromptTemplate.from_template(
        load_rewrite_prompt()
    )
    return rewrite_prompt | llm | StrOutputParser()

def build_hyde_chain(llm):
    rewrite_prompt = PromptTemplate.from_template(
        load_hyde_prompt()
    )
    return rewrite_prompt | llm | StrOutputParser()
