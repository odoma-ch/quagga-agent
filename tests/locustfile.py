import json
import random
import uuid
import time
import logging
from locust import HttpUser, task, between

KNOWLEDGE_GRAPHS = {
    "GESIS KG": {
        "endpoint": "https://data.gesis.org/gesiskg/sparql",
        "questions": [
            "How many datasets are available in GESIS KG?",
            "List all organizations in GESIS together with the number of datasets they published.",
            "What research areas are covered in the GESIS Knowledge Graph?",
            "How many publications are linked to datasets in GESIS KG?",
            "Give me a list of the most recent datasets added to GESIS.",
            "What are the main topics studied in GESIS datasets?",
            "Which institutions contribute the most data to GESIS KG?",
            "How many survey series are in the GESIS Knowledge Graph?",
            "What methodologies are used in GESIS datasets?",
            "List all GESIS datasets related to social inequality.",
        ],
    },
    "EHRI - KG": {
        "endpoint": "https://lod.ehri-project-test.eu/sparql",
        "questions": [
            "How many Holocaust-related archival institutions are in the EHRI KG?",
            "List all countries that have EHRI archival institutions.",
            "What types of archival material are described in EHRI?",
            "How many archival descriptions exist in the EHRI Knowledge Graph?",
            "Which repositories in EHRI have the most collections?",
            "Find all EHRI institutions in Poland.",
            "What languages are represented in EHRI archival materials?",
            "How many EHRI collections relate to Jewish history?",
            "List the top contributing institutions in the EHRI KG.",
            "What is the geographic coverage of EHRI archival data?",
        ],
    },
    "DNB KB": {
        "endpoint": "https://sparql.dnb.de/api/dnbgnd",
        "questions": [
            "How many bibliographic records are in the DNB Knowledge Graph?",
            "List all German publishers with more than 100 titles in DNB KB.",
            "What types of publications are catalogued in DNB?",
            "How many authors are registered in the DNB Knowledge Graph?",
            "Find all publications from 2024 in the DNB KB.",
            "What subject categories exist in the DNB Knowledge Graph?",
            "Which German libraries contribute data to DNB KB?",
            "How many serial publications are in the DNB Knowledge Graph?",
            "List the most published authors in the DNB dataset.",
            "What languages are represented in DNB publications?",
        ],
    },
    "Wikidata": {
        "endpoint": "https://qlever.dev/api/wikidata",
        "questions": [
            "How many human settlements are in Wikidata with a population over 1 million?",
            "List all UNESCO World Heritage Sites in Italy from Wikidata.",
            "What countries have the most universities listed in Wikidata?",
            "Find all SpaceX launches documented in Wikidata.",
            "Which languages have the most native speakers according to Wikidata?",
            "List all Nobel Prize winners in Physics from Wikidata.",
            "What are the tallest buildings in the world according to Wikidata?",
        ],
    },
}


logger = logging.getLogger("quagga.locust")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class QuaggaAgentUser(HttpUser):
    """Simulates a user interacting with the Quagga SPARQL Agent API."""
    
    # Wait between 1 and 5 seconds between tasks
    wait_time = between(1, 5)
    
    def on_start(self):
        """Called when a simulated user starts."""
        # Each user gets a unique chat_id to simulate separate conversations
        self.chat_id = f"load-test-user-{uuid.uuid4()}"
        
    def _get_random_question(self):
        """Get a random question from a random knowledge graph."""
        kg_name = random.choice(list(KNOWLEDGE_GRAPHS.keys()))
        kg_data = KNOWLEDGE_GRAPHS[kg_name]
        question = random.choice(kg_data["questions"])
        return kg_name, question
    
    @task(weight=10)
    def ask_question_streaming(self):
        """Ask a question with streaming enabled."""
        kg_name, question = self._get_random_question()
        run_id = uuid.uuid4().hex[:8]
        start_time = time.perf_counter()
        logger.info("[run=%s] START kg=%s prompt=%r", run_id, kg_name, question)
        
        payload = {
            "chat_id": self.chat_id,
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ],
            "stream": True
        }
        
        # Track metrics
        total_chunks = 0
        total_content_length = 0
        
        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=(10, 600),
            catch_response=True,
            name=f"/v1/chat/completions [streaming] - {kg_name}"
        ) as response:
            try:
                if response.status_code != 200:
                    duration_ms = int((time.perf_counter() - start_time) * 1000)
                    failure_msg = (
                        f"run={run_id} kg={kg_name} status={response.status_code} prompt={question}"
                    )
                    logger.error("[run=%s] FAIL status=%s kg=%s duration_ms=%s prompt=%r", run_id, response.status_code, kg_name, duration_ms, question)
                    response.failure(failure_msg)
                    return

                done_received = False
                for line in response.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if decoded.startswith('data: '):
                            data_str = decoded[6:]

                            if data_str == '[DONE]':
                                done_received = True
                                break

                            try:
                                chunk = json.loads(data_str)
                                delta = chunk.get('choices', [{}])[0].get('delta', {})
                                content = delta.get('content', '')

                                if content:
                                    total_chunks += 1
                                    total_content_length += len(content)

                            except json.JSONDecodeError:
                                pass

                duration_ms = int((time.perf_counter() - start_time) * 1000)
                # Log stats
                if total_chunks == 0:
                    failure_msg = (
                        f"run={run_id} kg={kg_name} no_content done={done_received} "
                        f"duration_ms={duration_ms} prompt={question}"
                    )
                    logger.error("[run=%s] FAIL kg=%s no_content done=%s duration_ms=%s prompt=%r", run_id, kg_name, done_received, duration_ms, question)
                    response.failure(failure_msg)
                else:
                    logger.info("[run=%s] OK kg=%s chunks=%s chars=%s done=%s duration_ms=%s prompt=%r", run_id, kg_name, total_chunks, total_content_length, done_received, duration_ms, question)
                    response.success()

            except Exception as e:
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                logger.exception("[run=%s] EXCEPTION kg=%s duration_ms=%s prompt=%r", run_id, kg_name, duration_ms, question)
                response.failure(
                    f"run={run_id} kg={kg_name} exception={str(e)} duration_ms={duration_ms} prompt={question}"
                )
    
    # @task(weight=1)
    # def health_check(self):
    #     """Check the health endpoint."""
    #     with self.client.get("/health", catch_response=True, name="/health") as response:
    #         if response.status_code != 200:
    #             response.failure(f"Health check failed with status {response.status_code}")
    #         else:
    #             response.success()
    
    # @task(weight=1)
    # def list_models(self):
    #     """List available models."""
    #     with self.client.get("/v1/models", catch_response=True, name="/v1/models") as response:
    #         if response.status_code != 200:
    #             response.failure(f"Models endpoint failed with status {response.status_code}")
    #         else:
    #             response.success()
