import json
import os
from collections import OrderedDict

# 路径配置
sampled_mcq_path = os.path.join(os.path.dirname(__file__), '../sampled_mcq.json')
base_path = os.path.join(os.path.dirname(__file__), '../MCQ_settings/test_set_base_simple.json')
mix_path = os.path.join(os.path.dirname(__file__), '../MCQ_settings/test_set_mix_simple.json')
output_dir = os.path.join(os.path.dirname(__file__), '../final_mcq')
os.makedirs(output_dir, exist_ok=True)

# 读取sampled_mcq，获取需要的original_query集合
def load_sampled_queries(path):
    with open(path, 'r', encoding='utf-8') as f:
        sampled_mcq = json.load(f)
    # original_query 字段
    queries = set()
    for item in sampled_mcq:
        if 'original_query' in item:
            queries.add(item['original_query'])
        else:
            print('警告：条目缺少original_query字段')
    return queries

def filter_by_queries(input_path, queries):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    filtered = [item for item in data if item.get('original_query') in queries]
    print(f"{input_path} 匹配到 {len(filtered)} 条")
    return filtered

def add_human_choice_field(data):
    for item in data:
        try:
            content = item["multiple_choice"]["gpt40"]["content"]
            if "human_choice" not in content:
                new_content = OrderedDict()
                for k, v in content.items():
                    if k == "correct_option":
                        new_content["human_choice"] = ""
                    new_content[k] = v
                item["multiple_choice"]["gpt40"]["content"] = new_content
        except Exception as e:
            print(f"添加human_choice失败: {e}")
    return data

def main():
    queries = load_sampled_queries(sampled_mcq_path)
    base_filtered = filter_by_queries(base_path, queries)
    mix_filtered = filter_by_queries(mix_path, queries)
    base_filtered = add_human_choice_field(base_filtered)
    mix_filtered = add_human_choice_field(mix_filtered)
    # 保存
    with open(os.path.join(output_dir, 'base_2000.json'), 'w', encoding='utf-8') as f:
        json.dump(base_filtered, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, 'mix_2000.json'), 'w', encoding='utf-8') as f:
        json.dump(mix_filtered, f, ensure_ascii=False, indent=2)
    print('提取完成，已保存到final_mcq目录下')

if __name__ == '__main__':
    main() 