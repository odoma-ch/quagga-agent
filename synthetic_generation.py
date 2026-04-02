import os
import httpx
import random
import numpy as np

from rdflib import Graph
from dotenv import load_dotenv
from langchain.tools import tool
from langgraph.runtime import get_runtime
from rdflib.plugins.stores import sparqlstore
from langchain_core.runnables import RunnableConfig

from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model

from models import ContextSchema, SPARQLValidationInput

load_dotenv()


def get_synthetic_generation_prompt(shex_schema: str) -> str:
    """
    Prompt for generating a new pair of question and sparql query.

    params: shex_schema - the SHEX schema to generate the question and sparql query for

    returns: the prompt for generating a new pair of question and sparql query
    """

    instruction = """
    Given the following examples of natural language question-sparql query pairs and the SHEX schema; 
    Generate a new pair of question and sparql query.
    Add all the prefixes necessary for the query to work.
    Use the validate_sparql_tool to validate the query before returning the result.
    The generated natural language question and sparql query pair should be concise and add some creativity to it.
    """
    examples = """
    Question: generate a sparql query to get the title and publication date of the article with the DOI: 10.1162/qss_a_00023
    SPARQL Query: 
    SELECT ?id ?title ?pub_date { 
        ?identifier literal:hasLiteralValue '10.1162/qss_a_00023'. 
        ?br purlSD:hasIdentifier ?identifier; 
            dcterms:title ?title; 
            prismstandardNB2:publicationDate ?publicationDate. 
            BIND(STR(?publicationDate) AS ?pub_date) 
            BIND((CONCAT('doi:', '10.1162/qss_a_00023')) AS ?id) 
    }


    Question: how many journal articles in OCM were published in 2024?
    SPARQL Query: 
    PREFIX prismstandardNB2: <http://prismstandard.org/namespaces/basic/2.0/> 
    SELECT (COUNT(?publication) AS ?count_2024) WHERE { 
        ?publication a purlSF:JournalArticle ; prismstandardNB2:publicationDate 
        ?date . FILTER(STRSTARTS(STR(?date), '2024')) 
    }
    """
    SYSTEM_PROMPT = f"""You are an exceptionally intelligent SPARQL query writing assistant that consistently delivers accurate and reliable responses to user instructions.

    @@ Instruction:
{instruction}

    @@ Response:
    The response should be in the following format:
    Generated Question:
    Generated SPARQL Query:

    @@Examples:
{examples}

    @@ SHEX Schema:
{shex_schema}
    """
    return SYSTEM_PROMPT


def instantiate_llm(temperature: float = 0.7, top_p: float = 1, http_client: httpx.Client = None, use_claude: bool = False):
    """"
    Instantiate the LLM for generating a new pair of question and sparql query.

    params: temperature: the temperature for the LLM
    params: top_p: the top_p for the LLM
    params: http_client: the http client for the LLM
    params: use_claude: whether to use the Claude LLM

    returns: the instantiated LLM
    """"
    print(f"Instantiating LLM with temperature: {temperature}, top_p: {top_p}, use_claude: {use_claude}")
    http_client = httpx.Client(verify=False)
    if use_claude:
        # https://docs.anthropic.com/en/docs/about-claude/models/overview
        anthropic_llm = init_chat_model(
            "anthropic:claude-3-7-sonnet-latest",
            temperature=temperature,
            top_p=top_p,
            max_tokens=32896,
        )
        return anthropic_llm

    llm = ChatOpenAI(
        base_url=os.getenv("PSNC_HPC_BASE_URL"),
        model="DeepSeek-V3.1",
        api_key=os.getenv("PSNC_HPC_API_KEY"),
        temperature=temperature,
        top_p=top_p,
        seed=42,
        max_tokens=None,
        http_client=http_client,
    )
    return llm


def generate_synthetic_data(shex_schema: str, num_samples: int = 10, temperature: float = 0.7, top_p: float = 1, use_claude: bool = False) -> list[str]:
    """
    Generate synthetic data for the given SHEX schema.

    params: shex_schema: the SHEX schema to generate the synthetic data for
    params: num_samples: the number of samples to generate
    params: temperature: the temperature for the LLM
    params: top_p: the top_p for the LLM
    params: use_claude: whether to use the Claude LLM

    returns: the synthetic question and sparql query pair
    """
    llm = instantiate_llm(temperature=temperature, top_p=top_p, use_claude=use_claude)
    prompt = get_synthetic_generation_prompt(shex_schema)
    config_with_model = {
        "configurable": {
            "sparql_endpoint": "https://sparql.opencitations.net/meta",
        }
    }
    results = []
    for _ in range(num_samples):
        response = llm.invoke(prompt, config=config_with_model)
        results.append(response.content)
    return results


if __name__ == "__main__":
    shex_file_path = "../test/artifacts/shex/shex_2025-08-11T17-01-11_oc_meta.ttl"
    with open(shex_file_path, "r") as f:
        shex_schema = f.read()

    # have some variation in the top_p and temperature values
    temperature_value = round(random.choice(np.linspace(0.1, 1.0, 10)), 1)
    top_p_value = round(random.choice(np.linspace(0.7, 1, 5)), 2)
    results = generate_synthetic_data(
        shex_schema, num_samples=1, temperature=temperature_value, top_p=top_p_value, use_claude=True)
    print(results[0][0]["text"])
    print(results[0][1]["input"]["query"])
