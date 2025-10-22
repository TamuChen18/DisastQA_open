# OE模型性能分析报告

## 总体统计
- 总模型数量: 18
- 测试类型: base, golden, mix
- 总测试样本: 54000

## Key Point Coverage分析
| 模型 | Base Coverage | Golden Coverage | Mix Coverage | 平均 |
|------|---------------|-----------------|--------------|------|
| gpt-4o | 83.69% | 95.37% | 89.81% | 89.62% |
| gemini-1.5-pro | 84.52% | 95.06% | 83.98% | 87.85% |
| qwen-3-8b | 79.14% | 93.95% | 87.13% | 86.74% |
| llama-3-8b | 77.47% | 93.44% | 85.05% | 85.32% |
| qwen-2.5-3b-instruct | 76.91% | 93.43% | 85.40% | 85.25% |
| AceMath-1.5B-Instruct | 67.86% | 92.31% | 86.61% | 82.26% |
| Falcon3-1B-Instruct | 71.07% | 92.26% | 84.12% | 82.48% |
| deepseek-v3-7b | 73.49% | 92.01% | 84.09% | 83.20% |
| llama-3.2-3b-instruct | 76.62% | 92.01% | 84.31% | 84.31% |
| qwen-3-4b | 76.52% | 91.74% | 83.50% | 83.92% |
| Mistral-7B-Instruct-v0.2 | 76.27% | 91.63% | 85.43% | 84.44% |
| Hunyuan-4B-Instruct | 67.15% | 91.41% | 84.46% | 81.01% |
| Llama-3.2-1B-Instruct | 74.46% | 90.38% | 86.45% | 83.76% |
| qwen-3-0.6b | 69.72% | 90.07% | 85.66% | 81.82% |
| gemma-7b | 72.87% | 89.73% | 69.68% | 77.43% |
| phi-2 | 74.00% | 86.93% | 82.04% | 80.99% |
| Yi-6B-Chat | 73.05% | 86.54% | 84.71% | 81.43% |
| Hunyuan-7B-Instruct | 39.64% | 79.90% | 72.64% | 64.06% |

## 模型性能排名 (按综合分数)
1. **gemma-7b**: ROUGE-L=0.333, BLEU-4=0.118, BERTScore-F1=0.442, 综合=0.298
2. **llama-3-8b**: ROUGE-L=0.318, BLEU-4=0.104, BERTScore-F1=0.441, 综合=0.288
3. **deepseek-v3-7b**: ROUGE-L=0.320, BLEU-4=0.105, BERTScore-F1=0.437, 综合=0.287
4. **llama-3.2-3b-instruct**: ROUGE-L=0.315, BLEU-4=0.099, BERTScore-F1=0.435, 综合=0.283
5. **Llama-3.2-1B-Instruct**: ROUGE-L=0.286, BLEU-4=0.082, BERTScore-F1=0.417, 综合=0.261
6. **AceMath-1.5B-Instruct**: ROUGE-L=0.281, BLEU-4=0.076, BERTScore-F1=0.427, 综合=0.261
7. **Falcon3-1B-Instruct**: ROUGE-L=0.293, BLEU-4=0.071, BERTScore-F1=0.411, 综合=0.258
8. **Mistral-7B-Instruct-v0.2**: ROUGE-L=0.285, BLEU-4=0.070, BERTScore-F1=0.405, 综合=0.253
9. **Yi-6B-Chat**: ROUGE-L=0.278, BLEU-4=0.076, BERTScore-F1=0.397, 综合=0.250
10. **phi-2**: ROUGE-L=0.276, BLEU-4=0.071, BERTScore-F1=0.400, 综合=0.249
11. **gpt-4o**: ROUGE-L=0.271, BLEU-4=0.060, BERTScore-F1=0.395, 综合=0.242
12. **qwen-3-8b**: ROUGE-L=0.265, BLEU-4=0.053, BERTScore-F1=0.390, 综合=0.236
13. **qwen-2.5-3b-instruct**: ROUGE-L=0.268, BLEU-4=0.054, BERTScore-F1=0.385, 综合=0.236
14. **qwen-3-4b**: ROUGE-L=0.264, BLEU-4=0.051, BERTScore-F1=0.383, 综合=0.232
15. **qwen-3-0.6b**: ROUGE-L=0.243, BLEU-4=0.050, BERTScore-F1=0.393, 综合=0.228
16. **gemini-1.5-pro**: ROUGE-L=0.225, BLEU-4=0.024, BERTScore-F1=0.341, 综合=0.197
17. **Hunyuan-4B-Instruct**: ROUGE-L=0.209, BLEU-4=0.033, BERTScore-F1=0.339, 综合=0.194
18. **Hunyuan-7B-Instruct**: ROUGE-L=0.183, BLEU-4=0.023, BERTScore-F1=0.277, 综合=0.161

## 详细性能表
| 排名 | 模型 | ROUGE-L | BLEU-4 | BERTScore-F1 | 综合分数 |
|------|------|---------|--------|--------------|----------|
| 1 | gemma-7b | 0.333 | 0.118 | 0.442 | 0.298 |
| 2 | llama-3-8b | 0.318 | 0.104 | 0.441 | 0.288 |
| 3 | deepseek-v3-7b | 0.320 | 0.105 | 0.437 | 0.287 |
| 4 | llama-3.2-3b-instruct | 0.315 | 0.099 | 0.435 | 0.283 |
| 5 | Llama-3.2-1B-Instruct | 0.286 | 0.082 | 0.417 | 0.261 |
| 6 | AceMath-1.5B-Instruct | 0.281 | 0.076 | 0.427 | 0.261 |
| 7 | Falcon3-1B-Instruct | 0.293 | 0.071 | 0.411 | 0.258 |
| 8 | Mistral-7B-Instruct-v0.2 | 0.285 | 0.070 | 0.405 | 0.253 |
| 9 | Yi-6B-Chat | 0.278 | 0.076 | 0.397 | 0.250 |
| 10 | phi-2 | 0.276 | 0.071 | 0.400 | 0.249 |
| 11 | gpt-4o | 0.271 | 0.060 | 0.395 | 0.242 |
| 12 | qwen-3-8b | 0.265 | 0.053 | 0.390 | 0.236 |
| 13 | qwen-2.5-3b-instruct | 0.268 | 0.054 | 0.385 | 0.236 |
| 14 | qwen-3-4b | 0.264 | 0.051 | 0.383 | 0.232 |
| 15 | qwen-3-0.6b | 0.243 | 0.050 | 0.393 | 0.228 |
| 16 | gemini-1.5-pro | 0.225 | 0.024 | 0.341 | 0.197 |
| 17 | Hunyuan-4B-Instruct | 0.209 | 0.033 | 0.339 | 0.194 |
| 18 | Hunyuan-7B-Instruct | 0.183 | 0.023 | 0.277 | 0.161 |

## 测试类型分析

### BASE测试
- 平均ROUGE-L: 0.209
- 平均BLEU-4: 0.035
- 平均BERTScore-F1: 0.308

### GOLDEN测试
- 平均ROUGE-L: 0.322
- 平均BLEU-4: 0.093
- 平均BERTScore-F1: 0.461

### MIX测试
- 平均ROUGE-L: 0.288
- 平均BLEU-4: 0.076
- 平均BERTScore-F1: 0.417