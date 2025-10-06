import os
import time
import json
import httpx
import requests
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel
from rich.console import Console
from langchain_openai import ChatOpenAI

load_dotenv()

console = Console()


class OutputSchema(BaseModel):
    """Schema for extracted information from the code."""
    summary: str
    endpoint_url: Optional[str] = None


def search_github_code(query, token, sort="indexed", order="desc", per_page=30):
    """
    Search GitHub code using the Search API with pagination to get all results
    
    Args:
        query (str): Search query
        token (str, optional): GitHub personal access token
        sort (str): Sort by 'indexed', 'created', or 'updated'
        order (str): Order 'asc' or 'desc'
        per_page (int): Results per page (max 30)
    
    Returns:
        dict: Combined JSON response with all items from all pages
    """
    url = "https://api.github.com/search/code"
    
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "GitHub-Search-Client"
    }
    
    if token:
        headers["Authorization"] = f"token {token}"
    
    # First request to get total count
    params = {
        "q": query,
        "sort": sort,
        "order": order,
        "per_page": per_page,
        "page": 1
    }
    
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    
    first_response = response.json()
    total_count = first_response.get("total_count", 0)
    total_items = first_response.get("items", [])
    
    console.print(f"Total results found: {total_count}", style="bold blue")
    console.print(f"Retrieved {len(total_items)} items from page 1", style="bold green")
    
    # Calculate total pages needed
    total_pages = (total_count + per_page - 1) // per_page
    
    # Fetch remaining pages if there are more than 1 page
    for page in range(2, total_pages + 1):
        params["page"] = page
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, params=params)
                
                # Handle rate limit
                if response.status_code == 403:
                    if 'rate limit' in response.text.lower():
                        if attempt < max_retries - 1:  # Don't wait on last attempt
                            console.print(f"Rate limit hit on page {page}, attempt {attempt + 1}. Waiting 30 seconds...", style="bold yellow")
                            time.sleep(60)  # Wait 60 seconds
                            continue
                        else:
                            console.print(f"Rate limit hit on page {page} after all retries, skipping remaining pages", style="bold red")
                            return {
                                "total_count": total_count,
                                "incomplete_results": True,
                                "items": total_items
                            }
                    else:
                        response.raise_for_status()
                
                response.raise_for_status()
                
                time.sleep(2)  # add delay to avoid rate limiting
                page_response = response.json()
                page_items = page_response.get("items", [])
                total_items.extend(page_items)
                
                console.print(f"Retrieved {len(page_items)} items from page {page}", style="bold green")
                break  # Success, move to next page
                
            except requests.exceptions.HTTPError as e:
                if response.status_code == 403:
                    if "rate limit" in response.text.lower() and attempt < max_retries - 1:
                        console.print(f"Rate limit hit on page {page}, attempt {attempt + 1}. Waiting 30 seconds...", style="bold yellow")
                        time.sleep(60)
                        continue
                    else:
                        console.print(f"Hit rate limit or pagination limit at page {page} after all retries, stopping", style="bold red")
                        return {
                            "total_count": total_count,
                            "incomplete_results": True,
                            "items": total_items
                        }
                else:
                    raise e
        else:
            # If we exhausted all retries without success
            console.print(f"Failed to retrieve page {page} after {max_retries} attempts", style="bold red")
            break
    
    # Return combined response
    total_responses = {
        "total_count": total_count,
        "incomplete_results": first_response.get("incomplete_results", False),
        "items": total_items
    }
    
    console.print(f"Total items collected: {len(total_items)}", style="bold yellow")
    
    return total_responses


def get_file_content(repo_full_name, file_path, token):
    """
    Get raw file content from GitHub
    
    Args:
        repo_full_name (str): Repository name like "owner/repo"
        file_path (str): Path to file in repo
        token (str, optional): GitHub personal access token
    
    Returns:
        str: Raw file content
    """
    url = f"https://api.github.com/repos/{repo_full_name}/contents/{file_path}"
    
    headers = {
        "Accept": "application/vnd.github.v3.raw",
        "User-Agent": "GitHub-Search-Client"
    }
    
    if token:
        headers["Authorization"] = f"token {token}"
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    return (url, response.text)


def check_endpoint_exists_and_extract(content: str):
    """
    Check if the code contains an endpoint URL and extract the summary from the code.

    Args:
        content (str): The code to check and extract the endpoint URL and the summary from.

    Returns:
        dict: The endpoint URL and the summary.
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
    result = structured_llm.invoke(
        f"""
Extract the endpoint URL and the summary from the code: {content}
If there is no endpoint URL, return None for the endpoint URL.
If there is no summary, return None for the summary.
    """
    )
    console.print(f"Raw result returned: {result['raw']}", style="bold green")
    parsed_result = dict(result["parsed"])
    parsed_result["sparql_content"] = content
    return parsed_result


if __name__ == "__main__":
    # "cidoc crm language:SPARQL"
    query = "endpoint summary language:SPARQL"
    github_token = os.getenv("GITHUB_DATA_TOKEN")
    if not github_token:
        raise ValueError("GITHUB_TOKEN is not set")
    
    # Get all results using pagination
    total_responses = search_github_code(query, token=github_token, per_page=30)
    
    console.print(f"Processing {len(total_responses['items'])} items...", style="bold magenta")

    total_results = []
    for i, r in enumerate(total_responses["items"], 1):
        console.print(f"Processing item {i}/{len(total_responses['items'])}", style="bold cyan")
        url, file_content = get_file_content(r["repository"]["full_name"], r["path"], token=github_token)
        parsed_result = check_endpoint_exists_and_extract(file_content)
        parsed_result["url"] = url
        total_results.append(parsed_result)

    with open("sparql_github_data.json", "w+") as f:
        json.dump(total_results, f, indent=4)
    
    console.print(f"Saved {len(total_results)} results to sparql_github_data.json", style="bold green")
