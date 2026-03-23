- `CHOOSE_BY_WEIGHTS=True HYBRID_SEMANTIC_AND_DEGREE=False SELECTIVELY_CHOOSE_BY_DEGREE=False SELECTIVELY_CHOOSE_BY_SIMILARITY=False PYTHONWARNINGS="ignore" && python generate_sample/get_sample.py   --num_nodes 150   --batch_size 15   --query_type "less_random"   --depth 2   --prune_percentage 0.50   --max_neighbors 50000   --output_filename "samples/artresearch_hybrid_150_d2_PP=0.50_max_n=50000.nt^C  --endpoint "https://artresearch.net/sparql"`
- `brew install raptor`
- `rapper -i ntriples -o turtle input.nt > output.ttl`
- `java -jar target/void-generator-*-uber.jar --from-test-file='/Users/hdsingh/quagga-agent/mvp/dumps/artresearch_v1.ttl' --void-file /tmp/artresearch-15M.ttl --iri-of-void 'https://artresearch.net/.well-known/void#' --repository='https://artresearch.net/sparql' --max-concurrency=200`
- `java -jar target/void-generator-*-uber.jar --from-test-file='/Users/hdsingh/quagga-agent/mvp/dumps/dnb_v1.ttl' --void-file /tmp/dnb-14M.ttl --iri-of-void 'https://sparql.dnb.de/.well-known/void#' --repository='https://sparql.dnb.de/api/gnd' --max-concurrency=200`


- `java -jar target/void-generator-*-uber.jar --from-test-file='/Users/hdsingh/quagga-agent/mvp/dumps/dnb_v1.ttl' --void-file /tmp/dnbgnd-14M.ttl --iri-of-void 'https://sparql.dnb.de/.well-known/void#' --repository='https://sparql.dnb.de/api/dnbgnd' --max-concurrency=200`


https://sparql.dnb.de/api/dnbgnd


CHOOSE_BY_WEIGHTS=True HYBRID_SEMANTIC_AND_DEGREE=False SELECTIVELY_CHOOSE_BY_DEGREE=False SELECTIVELY_CHOOSE_BY_SIMILARITY=False PYTHONWARNINGS="ignore" && python generate_sample/get_sample.py   --num_nodes 300   --batch_size 15   --query_type "less_random"   --depth 2   --prune_percentage 0.25   --max_neighbors 50000   --output_filename "samples/dnbgnd_hybrid_300_d2_PP=0.25_max_n=50000.nt^C  --endpoint "https://sparql.dnb.de/api/dnbgnd"