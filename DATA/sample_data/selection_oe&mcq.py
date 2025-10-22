import json
import random
from collections import defaultdict
import os
from collections import OrderedDict

# 读取标签表
with open("query2type_category.json", "r", encoding="utf-8") as f:
    query2type = json.load(f)

# 读取MCQ和OE原始数据
with open("test_set_golden_simple.json", "r", encoding="utf-8") as f:
    mcq_data = json.load(f)
with open("OE_test_set_golden_simple.json", "r", encoding="utf-8") as f:
    oe_data = json.load(f)

# 建立original_query到数据的映射
mcq_lookup = {item["original_query"]: item for item in mcq_data}
oe_lookup = {item["original_query"]: item for item in oe_data}

# 按type_category分组
bucket = defaultdict(list)
for entry in query2type:
    query = entry["original_query"]
    type_category = entry["type_category"]
    bucket[type_category].append(query)

# 采样
sampled_mcq = []
sampled_oe = []
for type_category, queries in bucket.items():
    if len(queries) < 50:
        print(f"⚠️ {type_category} 不足50条，仅有{len(queries)}条")
    sampled_queries = random.sample(queries, min(50, len(queries)))
    sampled_mcq.extend([mcq_lookup[q] for q in sampled_queries if q in mcq_lookup])

    # OE采样
    oe_queries = [q for q in sampled_queries if q in oe_lookup]
    if len(oe_queries) < 3:
        print(f"⚠️ {type_category} OE不足3条，仅有{len(oe_queries)}条")
    sampled_oe.extend([oe_lookup[q] for q in random.sample(oe_queries, min(3, len(oe_queries)))])

# === 新增：添加human_answer和human_choice字段 ===
for item in sampled_oe:
    try:
        item["open_ended"]["gpt40"]["content"]["human_answer"] = ""
    except Exception as e:
        print(f"OE数据添加human_answer失败: {e}")

for item in sampled_mcq:
    try:
        content = item["multiple_choice"]["gpt40"]["content"]
        new_content = OrderedDict()
        for k, v in content.items():
            if k == "correct_option":
                new_content["human_choice"] = ""
            new_content[k] = v
        item["multiple_choice"]["gpt40"]["content"] = new_content
    except Exception as e:
        print(f"MCQ数据添加human_choice失败: {e}")

# 保存全集
with open("sampled_mcq.json", "w", encoding="utf-8") as f:
    json.dump(sampled_mcq, f, ensure_ascii=False, indent=2)
with open("sampled_oe.json", "w", encoding="utf-8") as f:
    json.dump(sampled_oe, f, ensure_ascii=False, indent=2)

# 平均分成4份并保存到新文件夹
for name, data, outdir in [
    ("mcq", sampled_mcq, "DATA/sampled_mcq_split"),
    ("oe", sampled_oe, "DATA/sampled_oe_split")
]:
    n = len(data)
    size = n // 4
    os.makedirs(outdir, exist_ok=True)  # 确保目录存在
    for i in range(4):
        start = i * size
        end = (i + 1) * size if i < 3 else n  # 最后一份包含所有剩余
        part = data[start:end]
        outpath = os.path.join(outdir, f"{name}_part{i+1}.json")
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(part, f, ensure_ascii=False, indent=2)
    print(f"{name} 已分成4份，分别保存在 {outdir}/ 下")