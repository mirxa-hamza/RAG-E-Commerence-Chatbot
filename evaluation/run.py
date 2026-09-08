"""Live contract evaluation. Use a dedicated test account; calls spend LLM tokens.

Run after ingestion and backend startup:
    python evaluation/run.py --username evaluation-user

Optional case fields expected_product_ids and expected_review_ids enable hit-rate/MRR.
These need human-labelled ground truth from your own catalog; absence reports null.
"""
import argparse
import getpass
import json
import statistics
import time
from pathlib import Path
import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--username", required=True)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.json"))
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    rows, reciprocal_ranks = [], []
    with httpx.Client(base_url=args.url, timeout=150) as client:
        auth = client.post("/api/login", json={"username":args.username,"password":getpass.getpass("Evaluation account password: ")})
        auth.raise_for_status()
        client.headers["Authorization"] = "Bearer " + auth.json()["access_token"]
        for case in cases:
            start = time.perf_counter()
            response = client.post("/chat", json={"question":case["question"]})
            elapsed = time.perf_counter()-start
            failures = []
            if response.status_code != 200:
                rows.append({"question":case["question"],"passed":False,"http_status":response.status_code,"seconds":round(elapsed,2)})
                continue
            data = response.json()
            products = data["products"]
            if len(products) < case.get("min_products",0): failures.append("too_few_products")
            if "max_price" in case and any(p["price"] is None or p["price"] > case["max_price"] for p in products): failures.append("price_constraint")
            if "min_rating" in case and any(p["average_rating"] is None or p["average_rating"] < case["min_rating"] for p in products): failures.append("rating_constraint")
            if "preference" in case or "required_preference_field" in case:
                memory = client.get("/api/preferences"); memory.raise_for_status(); memory = memory.json()
                if any(memory.get(k) != v for k,v in case.get("preference",{}).items()): failures.append("preference_not_saved")
                if case.get("required_preference_field") and not memory.get(case["required_preference_field"]): failures.append("implied_preference_not_saved")
            for expected_key, returned_ids in [("expected_product_ids",[p["parent_asin"] for p in products]),
                                               ("expected_review_ids",[c["review_id"] for c in data["citations"]])]:
                if case.get(expected_key):
                    rank = next((i for i,p in enumerate(returned_ids,1) if p in case[expected_key]),None)
                    reciprocal_ranks.append(1/rank if rank else 0)
                    if not rank: failures.append(expected_key+"_miss")
            rows.append({"question":case["question"],"passed":not failures,"failures":failures,"seconds":round(elapsed,2),"products":len(products),"citations":len(data["citations"])})
    print(json.dumps({"cases":rows,"contract_pass_rate":sum(r["passed"] for r in rows)/len(rows) if rows else None,
                      "mean_latency_seconds":statistics.mean(r["seconds"] for r in rows) if rows else None,
                      "labelled_hit_rate":sum(r>0 for r in reciprocal_ranks)/len(reciprocal_ranks) if reciprocal_ranks else None,
                      "labelled_MRR":statistics.mean(reciprocal_ranks) if reciprocal_ranks else None,
                      "note":"Contract checks do not measure prose truthfulness. Review answers against source excerpts separately."},indent=2))
    return 0 if rows and all(r["passed"] for r in rows) else 1


if __name__ == "__main__": raise SystemExit(main())
