import os
import json
from collections import defaultdict

def build_passage_score_mapping_per_file(test_query_dir, qrels_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(test_query_dir):
        if not filename.endswith(".json"):
            continue

        # 加载 test_query 和 qrels 文件
        with open(os.path.join(test_query_dir, filename), 'r', encoding='utf-8') as f:
            test_queries = json.load(f)

        qrels_file = filename.replace(".json", "_qrels.json")
        qrels_path = os.path.join(qrels_dir, qrels_file)
        if not os.path.exists(qrels_path):
            print(f"[跳过] 未找到对应的 qrels 文件: {qrels_file}")
            continue

        with open(qrels_path, 'r', encoding='utf-8') as f:
            qrels_entries = json.load(f)

        # 构建 user_query -> list of (score, passage)
        qrels_dict = defaultdict(list)
        for entry in qrels_entries:
            qrels_dict[entry["user_query"]].append((entry["score"], entry["passage"]))

        # 按照 test_query 遍历
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
                print(f"[跳过] 没有score=3的文档: {filename}")
                continue

            # 确保所有分数段都存在
            full_bucket = {str(s): score_buckets.get(str(s), []) for s in [0, 1, 2, 3]}

            results.append({
                "user_query": user_query,
                "general_type": general_type,
                "passages_by_score": full_bucket
            })

        # 输出为每个文件一个 JSON
        output_path = os.path.join(output_dir, filename.replace(".json", "_by_score.json"))
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[完成] 写入：{output_path}")

# 用法
test_query_dir = "/home/shared/RAG_DATA/benchmark/test_query"
qrels_dir = "/home/shared/RAG_DATA/benchmark/qrels"
output_dir = "/home/shared/RAG_DATA/benchmark/data_prepare"

build_passage_score_mapping_per_file(test_query_dir, qrels_dir, output_dir)
