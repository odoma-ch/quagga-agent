import time
import requests


ENDPOINT = "https://sparql.digital.skkg.ch/sparql"
OUTPUT_FILE = "skkg_dump.nt"
LIMIT = 100000
HEADERS = {"Accept": "application/n-triples"}


def download_triples():
    """Download triples from endpoint by doing offset queries."""
    offset = 0
    total_triples = 0

    with open(OUTPUT_FILE, "w") as f:
        while True:
            query = f"CONSTRUCT {{ ?s ?p ?o }} WHERE {{ ?s ?p ?o }} LIMIT {LIMIT} OFFSET {offset}"
            print(f"Downloading triples at offset {offset}...")

            try:
                response = requests.post(ENDPOINT, data={"query": query}, headers=HEADERS, timeout=300)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"Error at offset {offset}: {e}")
                print("Retrying in 10 seconds...")
                time.sleep(10)
                continue

            triples = response.text.strip()
            if not triples:
                print("Empty response. Done.")
                break

            count = triples.count("\n") + 1
            f.write(triples + "\n")
            f.flush()

            total_triples += count
            print(f"Got {count} triples | Total so far: {total_triples}")

            if count < LIMIT:
                print("Reached end of dataset.")
                break

            offset += LIMIT

    print(f"\nDone! Total triples saved: {total_triples}")
    print(f"Output file: {OUTPUT_FILE}")
