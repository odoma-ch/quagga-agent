
from mcp.server.fastmcp import FastMCP
from SPARQLWrapper import SPARQLWrapper, JSON

mcp = FastMCP("run_sparql_tool")

@mcp.tool()
def run_sparql(query: str, endpoint: str) -> dict:
    """Run a SPARQL query against a potential SPARQL endpoint for exploration of a knowledge graph"""
    # your logic here
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results

if __name__ == "__main__":
    mcp.run()
