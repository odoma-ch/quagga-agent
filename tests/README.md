# Load Testing with Locust

## Installation

```bash
pip install locust
```

## Running the Load Test

### Basic Usage (50 users)

```bash
locust -f tests/locustfile.py --host=http://localhost:8004 --users 50 --spawn-rate 5
```

Then open http://localhost:8089 to start the test.

### Headless Mode

Run without the web UI:

```bash
locust -f tests/locustfile.py \
    --host=http://localhost:8004 \
    --users 50 \
    --spawn-rate 5 \
    --run-time 5m \
    --headless
```

### With Results Export

```bash
mkdir -p tests/results

locust -f tests/locustfile.py \
    --host=http://localhost:8004 \
    --users 50 \
    --spawn-rate 5 \
    --run-time 5m \
    --headless \
    --csv=tests/results/load_test \
    --html=tests/results/report.html
```

## Test Scenarios

- **Streaming Questions (83%)**: Questions from 4 knowledge graphs
- **Health Check (8%)**: API health validation
- **List Models (8%)**: Model listing validation

### Knowledge Graphs (37 questions total)

- GESIS KG: 10 questions
- EHRI KG: 10 questions  
- DNB KB: 10 questions
- Wikidata: 7 questions

## Example Output

```
Name                                              # reqs      # fails  |     Avg     Min     Max    Med  |   req/s
---------------------------------------------------------------------------------------------------------------------------------------------------------------
/health                                                 45     0(0.00%)  |      12       8      45     11  |    0.90
/v1/chat/completions [streaming] - GESIS KG            120     2(1.67%)  |   15234    8901   28456  14523  |    2.40
/v1/models                                              41     0(0.00%)  |      18      10      52     15  |    0.82
---------------------------------------------------------------------------------------------------------------------------------------------------------------
Aggregated                                             206     2(0.97%)  |    9234      8   28456   9234  |    4.12
```
