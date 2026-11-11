import os
import httpx
import requests
import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

tqdm.pandas()


class OutputSchema(BaseModel):
    """Schema for extracted information from the domain content."""

    humanities_or_not: bool


def get_main_domain(url: str) -> str:
    """
    Get the main domain from a URL.

    Args:
        url (str): The endpoint URL to get the main domain from.

    Returns:
        str: The main domain which is part of the URL.
    """
    parsed = urlparse(url)
    domain_parts = parsed.netloc.split(".")
    if len(domain_parts) >= 2:
        # Get last two parts (domain.com)
        main_domain = ".".join(domain_parts[-2:])
    else:
        main_domain = parsed.netloc
    return main_domain


def get_domain_content(domain: str) -> str:
    """
    Get the content of the domain.

    Args:
        domain (str): The domain to get the content of.

    Returns:
        str: The content of the domain.
    """
    try:
        domain_url = f"https://{domain}"
        domain_content = requests.get(domain_url).text
        return domain_content
    except requests.exceptions.RequestException:
        return "Unable to get the content of the domain"


def get_humanities_or_not(domain_content: str) -> bool:
    """
    Check if the domain content is about humanities.

    Args:
        domain_content (str): The content of the domain.

    Returns:
        bool: True if the domain content is about humanities, False otherwise.
    """
    http_client = httpx.Client(verify=False)
    llm = ChatOpenAI(
        base_url=os.getenv("PSNC_HPC_BASE_URL"),
        model="DeepSeek-V3.1-test",
        api_key=os.getenv("PSNC_HPC_API_KEY"),
        temperature=0.1,
        top_p=1,
        http_client=http_client,
        max_tokens=None,
    )
    structured_llm = llm.with_structured_output(OutputSchema, include_raw=True)
    result = structured_llm.invoke(f"""
Given some content and decide if it is about social sciences and humanities topics based on definition below: {domain_content}
If it is not about social sciences and humanities topics, return False for the humanities_or_not.
If it is about social sciences and humanities topics, return True for the humanities_or_not.

Definition of Social Sciences and Humanities:
Social Sciences and Humanities (SSH) encompass scholarly disciplines that systematically study human society, culture, behavior, thought, and creative expression. This includes social sciences such as sociology, anthropology, political science, economics, psychology, geography, education, communication studies, demographics, law, and social policy, which examine social structures, human behavior, and societal systems. It also includes humanities disciplines such as history, philosophy, languages and linguistics, literature, arts (including visual arts, performing arts, music, and theatre), religious studies, archaeology, cultural studies, and classics, which explore human culture, values, meaning-making, and creative works. SSH also encompasses interdisciplinary fields that bridge these areas, including gender studies, environmental humanities, digital humanities, urban studies, and development studies. Research classified as SSH focuses on understanding human experiences, social phenomena, cultural production, historical developments, and the ideas and values that shape societies, rather than natural phenomena or technological systems studied in natural sciences, engineering, or computer science.
    """)
    parsed_result = dict(result["parsed"])
    return parsed_result["humanities_or_not"]


def check_sparql_endpoint(endpoint_uri: str) -> bool:
    """
    Check if the SPARQL endpoint is accessible.

    Args:
        endpoint_uri (str): The URI of the SPARQL endpoint.

    Returns:
        bool: True if the endpoint is accessible, False otherwise.
        str: The endpoint URI if accessible, None otherwise.
    """
    try:
        # put a timeout of 30 seconds to avoid long wait for a simple query
        test_query = "SELECT * WHERE { ?s ?p ?o } LIMIT 1"
        response = requests.get(
            endpoint_uri,
            params={"query": test_query, "format": "json"},
            headers={"Accept": "application/sparql-results+json"},
            timeout=30,
        )
        if response.status_code != 200:
            return False
        return True
    except requests.exceptions.RequestException:
        return False


def filter_github_data(filename: str):
    """
    Filter the GitHub data to only include accessible SPARQL endpoints.

    Args:
        filename (str): The path to the GitHub data file.

    Returns:
        pandas.DataFrame: The filtered GitHub data.
    """
    sparql_github_data = pd.read_json(filename)
    # remove rows where the endpoint_url is None
    sparql_github_data = sparql_github_data[~sparql_github_data["endpoint_url"].isna()]
    # remove rows where the endpoint_url is the example.com/sparql
    sparql_github_data = sparql_github_data[
        sparql_github_data["endpoint_url"] != "http://example.com/sparql"
    ]
    # remove rows where the endpoint_url is not accessible
    sparql_github_data = sparql_github_data[
        sparql_github_data["endpoint_url"].progress_apply(check_sparql_endpoint)
    ]
    sparql_github_data["domain"] = sparql_github_data["endpoint_url"].progress_apply(
        get_main_domain
    )

    unique_domains = sparql_github_data["domain"].unique().tolist()
    domains_humanities_or_not = {}

    for d in unique_domains:
        domain_content = get_domain_content(d)
        if domain_content != "Unable to get the content of the domain":
            all_summaries = sparql_github_data[sparql_github_data["domain"] == d][
                "summary"
            ].tolist()
            humanities_or_not = get_humanities_or_not(all_summaries)
            domains_humanities_or_not[d] = humanities_or_not
        else:
            humanities_or_not = get_humanities_or_not(domain_content)
            domains_humanities_or_not[d] = humanities_or_not

    sparql_github_data["humanities_or_not"] = sparql_github_data[
        "domain"
    ].progress_apply(lambda d: domains_humanities_or_not[d])

    return sparql_github_data


if __name__ == "__main__":
    filter_github_data("sparql_github_data.json")
