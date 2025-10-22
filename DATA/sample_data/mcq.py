import json
import os
from tqdm import tqdm

# Step 1: 建立 user_query → type_category 映射
data_dir = "/home/shared/RAG_DATA/DATA/data_prepare"
query2label = {}

file_list = [fn for fn in os.listdir(data_dir) if fn.endswith(".json")]
for filename in tqdm(file_list, desc="遍历data_prepare文件"):
    path = os.path.join(data_dir, filename)
    parts = filename.replace(".json", "").split("_")
    file_type = parts[0]
    category = "_".join(parts[1:-2])
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
        for item in data:
            if isinstance(item, dict):
                query = item.get("user_query")
                if query:
                    query2label[query] = f"{file_type}_{category}"

# Step 2: 读取 test_set_golden_simple.json，生成新json
input_path = "/home/shared/RAG_DATA/DATA/test_set_golden_simple.json"
output_path = "/home/shared/RAG_DATA/DATA/query2type_category.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

result = []
for item in tqdm(data, desc="处理test_set_golden_simple.json"):
    query = item.get("original_query")
    label = query2label.get(query)
    result.append({"original_query": query, "type_category": label if label else "[标签未找到]"})

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"已生成 query2type_category.json，条数: {len(result)}")
