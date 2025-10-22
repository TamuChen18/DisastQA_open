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

# 读取之前MCQ采样中使用的queries，用于排除
with open("sampled_mcq.json", "r", encoding="utf-8") as f:
    sampled_mcq_data = json.load(f)
used_mcq_queries = {item["original_query"] for item in sampled_mcq_data}

print(f"已使用的MCQ queries数量: {len(used_mcq_queries)}")

# 按type_category分组，但排除已用于MCQ的queries
bucket = defaultdict(list)
for entry in query2type:
    query = entry["original_query"]
    type_category = entry["type_category"]
    # 只添加不在MCQ采样中的queries
    if query not in used_mcq_queries:
        bucket[type_category].append(query)

# 统计每个类别的可用OE数量
print("\n各类型可用OE数量统计:")
for type_category, queries in bucket.items():
    oe_queries = [q for q in queries if q in oe_lookup]
    print(f"{type_category}: {len(oe_queries)} 个OE可用")

# 计算每个类别应该采样的数量
total_categories = len(bucket)
target_per_category = 1000 // total_categories
print(f"\n目标总数: 1000个OE")
print(f"类别总数: {total_categories}")
print(f"每个类别目标采样: {target_per_category}个")

# 采样
sampled_oe = []
for type_category, queries in bucket.items():
    # 筛选出有OE数据的queries
    oe_queries = [q for q in queries if q in oe_lookup]
    
    if len(oe_queries) < target_per_category:
        print(f"⚠️ {type_category} OE不足{target_per_category}条，仅有{len(oe_queries)}条")
        # 如果不足，全部采样
        sampled_queries = oe_queries
    else:
        # 随机采样目标数量
        sampled_queries = random.sample(oe_queries, target_per_category)
    
    sampled_oe.extend([oe_lookup[q] for q in sampled_queries])
    print(f"✓ {type_category}: 采样了 {len(sampled_queries)} 个OE")

print(f"\n实际采样总数: {len(sampled_oe)} 个OE")

# 验证与MCQ样本不重叠
sampled_oe_queries = {item["original_query"] for item in sampled_oe}
overlap = sampled_oe_queries.intersection(used_mcq_queries)
print(f"与MCQ样本重叠的queries数量: {len(overlap)}")
if len(overlap) > 0:
    print("⚠️ 警告：存在重叠！")
else:
    print("✓ 验证通过：与MCQ样本无重叠")

# === 数据清洗：删除answer、key_points、reasoning，添加human_answer字段 ===
for item in sampled_oe:
    try:
        content = item["open_ended"]["gpt40"]["content"]
        # 删除answer内容
        if "answer" in content:
            content["answer"] = ""
        
        # 删除key_points
        if "key_points" in content:
            del content["key_points"]
        
        # 删除reasoning
        if "reasoning" in content:
            del content["reasoning"]
        
        # 添加human_answer字段
        content["human_answer"] = ""
        
    except Exception as e:
        print(f"OE数据清洗失败: {e}")

# 保存全集
with open("sampled_oe_independent.json", "w", encoding="utf-8") as f:
    json.dump(sampled_oe, f, ensure_ascii=False, indent=2)

# 平均分成4份并保存到新文件夹
outdir = "DATA/sampled_oe_independent_split"
n = len(sampled_oe)
size = n // 4
os.makedirs(outdir, exist_ok=True)  # 确保目录存在
for i in range(4):
    start = i * size
    end = (i + 1) * size if i < 3 else n  # 最后一份包含所有剩余
    part = sampled_oe[start:end]
    outpath = os.path.join(outdir, f"oe_independent_part{i+1}.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(part, f, ensure_ascii=False, indent=2)
print(f"OE独立样本已分成4份，分别保存在 {outdir}/ 下")

# 保存统计信息
stats = {
    "total_sampled": len(sampled_oe),
    "categories_count": total_categories,
    "target_per_category": target_per_category,
    "overlap_with_mcq": len(overlap),
    "category_breakdown": {}
}

for type_category, queries in bucket.items():
    oe_queries = [q for q in queries if q in oe_lookup]
    sampled_count = len([item for item in sampled_oe if item["original_query"] in oe_queries])
    stats["category_breakdown"][type_category] = {
        "available": len(oe_queries),
        "sampled": sampled_count
    }

with open("sampled_oe_independent_stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print(f"\n统计信息已保存到 sampled_oe_independent_stats.json") 