EVAL_QUESTIONS = [
    "What is Quagga?",
    "How many publications are in the OpenCitations dataset?",
    "List all authors who published more than 5 papers.",
    "Are there other ways to categorize publications in GESIS, perhaps by field or research area?",
    "What is the most cited article in OpenCitations?",
    "Give me a list of organizations in GESIS together with the number of datasets they published.",
    "I want to query OpenCitations. How many publications are indexed in there? break them down by publication typology",
    "who is the most prolific Humanities author in terms of number of publications?",
]

# Questions that may be answered directly by the model
# without going through a tool call/result path.
TRACE_ASSERT_EXCEPTIONS = {
    "What is Quagga?",
}

# Interesting conversation extracts:
# https://quagga-agent-webui-graphia-app1-staging.apps.bst2.paas.psnc.pl/s/957b2ffc-6872-4008-a733-49953d116cb2
# https://quagga-agent-webui-graphia-app1-staging.apps.bst2.paas.psnc.pl/s/4df38006-07c6-4017-8b71-1338cd9edf8f


TEST_ANSWER_TOOLS = {
    "What is Quagga?": [],
    "How many publications are in the OpenCitations dataset?": [
        "**Selecting knowledge graph(s)**",
        "**Retrieving knowledge graph metadata**",
        "**Generating, validating and executing SPARQL query**",
    ],
    "List all authors who published more than 5 papers.": [
        "**Checking available knowledge graphs**",
        "**Selecting knowledge graph(s)**",
    ],
    "Are there other ways to categorize publications in GESIS, perhaps by field or research area?": [
        "**Selecting knowledge graph(s)**",
        "**Retrieving knowledge graph metadata**",
        "**Generating, validating and executing SPARQL query**",
    ],
    "What is the most cited article in OpenCitations?": [
        "**Selecting knowledge graph(s)**",
        "**Retrieving knowledge graph metadata**",
        "**Generating, validating and executing SPARQL query**",
    ],
    "Give me a list of organizations in GESIS together with the number of datasets they published.": [
        "**Selecting knowledge graph(s)**",
        "**Retrieving knowledge graph metadata**",
        "**Generating, validating and executing SPARQL query**",
    ],
    "I want to query OpenCitations. How many publications are indexed in there? break them down by publication typology": [
        "**Selecting knowledge graph(s)**",
        "**Retrieving knowledge graph metadata**",
        "**Generating, validating and executing SPARQL query**",
    ],
    "who is the most prolific Humanities author in terms of number of publications?": [
        "**Selecting knowledge graph(s)**",
        "**Retrieving knowledge graph metadata**",
        "**Generating, validating and executing SPARQL query**",
    ],
}
