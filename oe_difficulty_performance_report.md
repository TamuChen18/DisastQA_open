# OE模型按Difficulty性能分析报告

## 总体统计
- 总模型数量: 18
- 测试类型: mix, base, golden
- 难度级别: extremely_complex, medium, easy, hard
- 总测试样本: 10800

## Difficulty级别说明
- **easy**: Keypoint个数范围 1-2
- **medium**: Keypoint个数范围 3-5
- **hard**: Keypoint个数范围 6-7
- **extremely_complex**: Keypoint个数范围 8-14

## 各Difficulty级别平均Coverage
- **easy**: 79.80%
- **medium**: 82.84%
- **hard**: 83.56%
- **extremely_complex**: 83.40%

## 详细性能表
| 模型 | Easy | Medium | Hard | Extremely Complex | 平均 |
|------|------|--------|------|-------------------|------|
| gpt-4o | 89.64% | 89.78% | 89.11% | 90.48% | 89.75% |
| gemini-1.5-pro | 84.55% | 88.11% | 89.78% | 84.23% | 86.67% |
| Llama-3.2-1B-Instruct | 84.24% | 83.06% | 84.64% | 92.28% | 86.06% |
| Mistral-7B-Instruct-v0.2 | 82.45% | 85.02% | 83.54% | 91.39% | 85.60% |
| qwen-2.5-3b-instruct | 83.64% | 85.90% | 84.28% | 87.56% | 85.34% |
| qwen-3-8b | 82.85% | 87.63% | 87.23% | 83.23% | 85.24% |
| llama-3.2-3b-instruct | 78.96% | 84.52% | 87.32% | 85.73% | 84.13% |
| llama-3-8b | 84.23% | 85.47% | 86.30% | 78.53% | 83.63% |
| qwen-3-4b | 81.60% | 83.65% | 86.65% | 80.14% | 83.01% |
| Falcon3-1B-Instruct | 79.16% | 82.49% | 84.69% | 83.44% | 82.45% |
| AceMath-1.5B-Instruct | 78.80% | 82.55% | 83.64% | 84.58% | 82.39% |
| qwen-3-0.6b | 78.35% | 82.03% | 83.33% | 85.06% | 82.19% |
| Yi-6B-Chat | 76.49% | 82.28% | 82.00% | 86.39% | 81.79% |
| phi-2 | 76.44% | 81.31% | 82.79% | 85.83% | 81.59% |
| Hunyuan-4B-Instruct | 79.43% | 80.75% | 82.79% | 81.65% | 81.15% |
| deepseek-v3-7b | 80.19% | 84.89% | 81.07% | 76.73% | 80.72% |
| gemma-7b | 76.69% | 77.45% | 77.94% | 76.81% | 77.22% |
| Hunyuan-7B-Instruct | 58.72% | 64.24% | 66.97% | 67.08% | 64.26% |

## 模型性能排名 (按总体平均Coverage)
1. **gpt-4o**: 89.75%
2. **gemini-1.5-pro**: 86.67%
3. **Llama-3.2-1B-Instruct**: 86.06%
4. **Mistral-7B-Instruct-v0.2**: 85.60%
5. **qwen-2.5-3b-instruct**: 85.34%
6. **qwen-3-8b**: 85.24%
7. **llama-3.2-3b-instruct**: 84.13%
8. **llama-3-8b**: 83.63%
9. **qwen-3-4b**: 83.01%
10. **Falcon3-1B-Instruct**: 82.45%
11. **AceMath-1.5B-Instruct**: 82.39%
12. **qwen-3-0.6b**: 82.19%
13. **Yi-6B-Chat**: 81.79%
14. **phi-2**: 81.59%
15. **Hunyuan-4B-Instruct**: 81.15%
16. **deepseek-v3-7b**: 80.72%
17. **gemma-7b**: 77.22%
18. **Hunyuan-7B-Instruct**: 64.26%

## 各Difficulty级别详细分析

### EASY级别
| 排名 | 模型 | Coverage |
|------|------|----------|
| 1 | gpt-4o | 89.64% |
| 2 | gemini-1.5-pro | 84.55% |
| 3 | Llama-3.2-1B-Instruct | 84.24% |
| 4 | llama-3-8b | 84.23% |
| 5 | qwen-2.5-3b-instruct | 83.64% |
| 6 | qwen-3-8b | 82.85% |
| 7 | Mistral-7B-Instruct-v0.2 | 82.45% |
| 8 | qwen-3-4b | 81.60% |
| 9 | deepseek-v3-7b | 80.19% |
| 10 | Hunyuan-4B-Instruct | 79.43% |
| 11 | Falcon3-1B-Instruct | 79.16% |
| 12 | llama-3.2-3b-instruct | 78.96% |
| 13 | AceMath-1.5B-Instruct | 78.80% |
| 14 | qwen-3-0.6b | 78.35% |
| 15 | gemma-7b | 76.69% |
| 16 | Yi-6B-Chat | 76.49% |
| 17 | phi-2 | 76.44% |
| 18 | Hunyuan-7B-Instruct | 58.72% |

### MEDIUM级别
| 排名 | 模型 | Coverage |
|------|------|----------|
| 1 | gpt-4o | 89.78% |
| 2 | gemini-1.5-pro | 88.11% |
| 3 | qwen-3-8b | 87.63% |
| 4 | qwen-2.5-3b-instruct | 85.90% |
| 5 | llama-3-8b | 85.47% |
| 6 | Mistral-7B-Instruct-v0.2 | 85.02% |
| 7 | deepseek-v3-7b | 84.89% |
| 8 | llama-3.2-3b-instruct | 84.52% |
| 9 | qwen-3-4b | 83.65% |
| 10 | Llama-3.2-1B-Instruct | 83.06% |
| 11 | AceMath-1.5B-Instruct | 82.55% |
| 12 | Falcon3-1B-Instruct | 82.49% |
| 13 | Yi-6B-Chat | 82.28% |
| 14 | qwen-3-0.6b | 82.03% |
| 15 | phi-2 | 81.31% |
| 16 | Hunyuan-4B-Instruct | 80.75% |
| 17 | gemma-7b | 77.45% |
| 18 | Hunyuan-7B-Instruct | 64.24% |

### HARD级别
| 排名 | 模型 | Coverage |
|------|------|----------|
| 1 | gemini-1.5-pro | 89.78% |
| 2 | gpt-4o | 89.11% |
| 3 | llama-3.2-3b-instruct | 87.32% |
| 4 | qwen-3-8b | 87.23% |
| 5 | qwen-3-4b | 86.65% |
| 6 | llama-3-8b | 86.30% |
| 7 | Falcon3-1B-Instruct | 84.69% |
| 8 | Llama-3.2-1B-Instruct | 84.64% |
| 9 | qwen-2.5-3b-instruct | 84.28% |
| 10 | AceMath-1.5B-Instruct | 83.64% |
| 11 | Mistral-7B-Instruct-v0.2 | 83.54% |
| 12 | qwen-3-0.6b | 83.33% |
| 13 | phi-2 | 82.79% |
| 14 | Hunyuan-4B-Instruct | 82.79% |
| 15 | Yi-6B-Chat | 82.00% |
| 16 | deepseek-v3-7b | 81.07% |
| 17 | gemma-7b | 77.94% |
| 18 | Hunyuan-7B-Instruct | 66.97% |

### EXTREMELY_COMPLEX级别
| 排名 | 模型 | Coverage |
|------|------|----------|
| 1 | Llama-3.2-1B-Instruct | 92.28% |
| 2 | Mistral-7B-Instruct-v0.2 | 91.39% |
| 3 | gpt-4o | 90.48% |
| 4 | qwen-2.5-3b-instruct | 87.56% |
| 5 | Yi-6B-Chat | 86.39% |
| 6 | phi-2 | 85.83% |
| 7 | llama-3.2-3b-instruct | 85.73% |
| 8 | qwen-3-0.6b | 85.06% |
| 9 | AceMath-1.5B-Instruct | 84.58% |
| 10 | gemini-1.5-pro | 84.23% |
| 11 | Falcon3-1B-Instruct | 83.44% |
| 12 | qwen-3-8b | 83.23% |
| 13 | Hunyuan-4B-Instruct | 81.65% |
| 14 | qwen-3-4b | 80.14% |
| 15 | llama-3-8b | 78.53% |
| 16 | gemma-7b | 76.81% |
| 17 | deepseek-v3-7b | 76.73% |
| 18 | Hunyuan-7B-Instruct | 67.08% |