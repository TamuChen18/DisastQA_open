import json
import os
from collections import defaultdict, Counter

def load_data(file_path):
    """加载JSON数据文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_type_category(original_query, query2type_category):
    """根据original_query获取type_category"""
    for item in query2type_category:
        if item["original_query"] == original_query:
            return item["type_category"]
    return "unknown"

def analyze_insufficient_samples():
    """分析137个'信息不足'样本的分布"""
    
    # 加载数据
    print("加载数据文件...")
    ground_truth_data = load_data('ground_truth_check_answers.json')
    query2type_category = load_data('query2type_category.json')
    
    # 统计"信息不足"样本的类别分布
    insufficient_samples = []
    category_distribution = defaultdict(list)
    
    print("分析'信息不足'样本...")
    for i, item in enumerate(ground_truth_data):
        llm_answer = item["open_ended"]["gpt40"]["content"].get("llm_answer", "")
        if "The passage does not provide sufficient information" in llm_answer:
            original_query = item["original_query"]
            type_category = get_type_category(original_query, query2type_category)
            
            insufficient_samples.append({
                "index": i,
                "original_query": original_query,
                "type_category": type_category,
                "llm_answer": llm_answer
            })
            category_distribution[type_category].append(i)
    
    print(f"找到 {len(insufficient_samples)} 个'信息不足'样本")
    
    # 统计每个类别的分布
    category_counts = {cat: len(samples) for cat, samples in category_distribution.items()}
    
    print("\n=== 类别分布统计 ===")
    for category, count in sorted(category_counts.items()):
        print(f"{category}: {count} 个样本")
    
    # 保存分析结果
    analysis_result = {
        "total_insufficient": len(insufficient_samples),
        "category_distribution": category_counts,
        "insufficient_samples": insufficient_samples
    }
    
    with open('insufficient_samples_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n分析结果已保存到: insufficient_samples_analysis.json")
    
    return analysis_result

def check_available_replacement_samples():
    """检查每个类别可用的替换样本数量"""
    
    print("\n检查可用替换样本...")
    
    # 加载数据
    query2type_category = load_data('query2type_category.json')
    mcq_data = load_data('sampled_mcq.json')
    current_oe_data = load_data('sampled_oe_independent.json')
    
    # 获取已使用的original_query
    mcq_queries = set(item["original_query"] for item in mcq_data)
    current_oe_queries = set(item["original_query"] for item in current_oe_data)
    
    print(f"MCQ样本数: {len(mcq_queries)}")
    print(f"当前OE样本数: {len(current_oe_queries)}")
    
    # 统计每个类别的总样本数和已使用样本数
    total_samples_by_category = defaultdict(int)
    mcq_used_by_category = defaultdict(int)
    current_oe_used_by_category = defaultdict(int)
    
    # 统计总样本
    for item in query2type_category:
        original_query = item["original_query"]
        type_category = item["type_category"]
        total_samples_by_category[type_category] += 1
        
        if original_query in mcq_queries:
            mcq_used_by_category[type_category] += 1
        
        if original_query in current_oe_queries:
            current_oe_used_by_category[type_category] += 1
    
    # 计算可用样本
    available_samples = {}
    for category in total_samples_by_category:
        total = total_samples_by_category[category]
        mcq_used = mcq_used_by_category[category]
        current_oe_used = current_oe_used_by_category[category]
        available = total - mcq_used - current_oe_used
        
        available_samples[category] = {
            "total": total,
            "mcq_used": mcq_used,
            "current_oe_used": current_oe_used,
            "available": available
        }
    
    # 保存可用样本统计
    with open('available_replacement_samples.json', 'w', encoding='utf-8') as f:
        json.dump(available_samples, f, ensure_ascii=False, indent=2)
    
    print(f"可用样本统计已保存到: available_replacement_samples.json")
    
    return available_samples

def check_replacement_feasibility():
    """检查替换可行性"""
    
    print("\n检查替换可行性...")
    
    # 加载分析结果
    insufficient_analysis = load_data('insufficient_samples_analysis.json')
    available_samples = load_data('available_replacement_samples.json')
    
    category_counts = insufficient_analysis["category_distribution"]
    
    print("\n=== 替换可行性检查 ===")
    print("类别\t\t需要替换\t可用样本\t是否可行")
    print("-" * 50)
    
    feasible_categories = []
    infeasible_categories = []
    
    for category, need_count in sorted(category_counts.items()):
        if category in available_samples:
            available_count = available_samples[category]["available"]
            feasible = available_count >= need_count
            status = "✅" if feasible else "❌"
            
            print(f"{category:<20} {need_count:<10} {available_count:<10} {status}")
            
            if feasible:
                feasible_categories.append(category)
            else:
                infeasible_categories.append(category)
        else:
            print(f"{category:<20} {need_count:<10} {'N/A':<10} ❌")
            infeasible_categories.append(category)
    
    print(f"\n✅ 可行的类别: {len(feasible_categories)}")
    print(f"❌ 不可行的类别: {len(infeasible_categories)}")
    
    if infeasible_categories:
        print(f"\n不可行的类别: {infeasible_categories}")
    
    return feasible_categories, infeasible_categories

if __name__ == "__main__":
    # 分析"信息不足"样本
    analysis_result = analyze_insufficient_samples()
    
    # 检查可用替换样本
    available_samples = check_available_replacement_samples()
    
    # 检查替换可行性
    feasible_categories, infeasible_categories = check_replacement_feasibility()
    
    print(f"\n=== 总结 ===")
    print(f"总共有 {analysis_result['total_insufficient']} 个'信息不足'样本需要替换")
    print(f"其中 {len(feasible_categories)} 个类别可以找到足够的替换样本")
    print(f"{len(infeasible_categories)} 个类别需要特殊处理") 