import os
import json
from collections import defaultdict

def build_passage_score_mapping_per_file(test_query_dir, qrels_dir, output_dir):
    """
    Build a mapping between user queries and their passages grouped by qrels score.
    Each output file will include all queries with passages organized by score levels (0–3).
    """

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(test_query_dir):
        if not filename.endswith(".json"):
            continue

        # Load test_query file
        with open(os.path.join(test_query_dir, filename), 'r', encoding='utf-8') as f:
            test_queries = json.load(f)

        # Load corresponding qrels file
        qrels_file = filename.replace(".json", "_qrels.json")
        qrels_path = os.path.join(qrels_dir, qrels_file)
        if not os.path.exists(qrels_path):
            print(f"[Skip] Corresponding qrels file not found: {qrels_file}")
            continue

        with open(qrels_path, 'r', encoding='utf-8') as f:
            qrels_entries = json.load(f)

        # Build user_query -> list of (score, passage)
        qrels_dict = defaultdict(list)
        for entry in qrels_entries:
            qrels_dict[entry["user_query"]].append((entry["score"], entry["passage"]))

        # Process each query in the test set
        results = []
        for query_entry in test_queries:
            user_query = query_entry["user_query"]
            general_type = query_entry.get("general_type", "")

            if user_query not in qrels_dict:
                continue

            score_buckets = defaultdict(list)
            for score, passage in qrels_dict[user_query]:
                score_buckets[str(score)].append(passage)

            if not score_buckets["3"]:
                print(f"[Skip] No documents with score=3 found in: {filename}")
                continue

            # Ensure all score levels (0–3) exist
            full_bucket = {str(s): score_buckets.get(str(s), []) for s in [0, 1, 2, 3]}

            results.append({
                "user_query": user_query,
                "general_type": general_type,
                "passages_by_score": full_bucket
            })

        # Write output JSON file for each domain
        output_path = os.path.join(output_dir, filename.replace(".json", "_by_score.json"))
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[Done] Written: {output_path}")

# Example usage
if __name__ == "__main__":
    test_query_dir = "/home/shared/RAG_DATA/benchmark/test_query"
    qrels_dir = "/home/shared/RAG_DATA/benchmark/qrels"
    output_dir = "/home/shared/RAG_DATA/benchmark/data_prepare"

    build_passage_score_mapping_per_file(test_query_dir, qrels_dir, output_dir)
