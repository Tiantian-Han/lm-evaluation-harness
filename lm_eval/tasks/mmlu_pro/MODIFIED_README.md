# 修改后的 MMLU Pro 评估说明

本文档说明了根据要求对 MMLU Pro 评估逻辑所做的修改。

## 修改依据

### 原始修改（适用于早期 DeepSeek-R1 版本）
本次修改基于 **DeepSeek-R1 官方使用建议**：
- 参考链接：[DeepSeek-R1 Usage Recommendations](https://github.com/deepseek-ai/DeepSeek-R1?tab=readme-ov-file#usage-recommendations)
- 修改目的：优化 MMLU Pro 评估以更好地支持 DeepSeek-R1 系列模型及其他大语言模型
- 实现了官方建议的温度设置、提示格式、数学问题处理等关键改进

### DeepSeek-R1-0528 版本更新
**重要更新**：DeepSeek-R1-0528 版本的使用建议发生了重要变化：
- 参考链接：[DeepSeek-R1-0528 Usage Instructions](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528#4-how-to-run-locally)
- **新特性**：
  1. **系统提示现已支持** - 可以使用系统提示来指导模型行为
  2. **无需强制思维模式** - 不再需要在输出开头添加 `<think>\n` 来强制模型进入思维模式

## 主要修改

### 1. 温度设置（0.5-0.7 范围，推荐 0.6）
- 在所有模板中设置 `temperature: 0.6`
- 启用了采样 `do_sample: true`
- 这有助于防止无限重复或不连贯的输出
- **依据**：DeepSeek-R1 官方建议在 0.5-0.7 范围内设置温度，推荐 0.6

### 2. 系统提示支持（R1-0528 新特性）
- **R1-0528 版本**：创建了支持系统提示的新模板
  - `_r1_0528_template_yaml`: 通用任务模板，包含系统提示
  - `_r1_0528_math_template_yaml`: 数学任务模板，包含数学专用系统提示
- **早期版本**：避免系统提示，所有指令包含在用户提示中
- **依据**：R1-0528 官方说明现在支持系统提示

### 3. 数学问题的特殊处理
- 为数学问题添加了特殊指令："Please reason step by step, and put your final answer within \\boxed{}."
- **R1-0528 版本**：通过系统提示提供数学专用指导
- **早期版本**：通过用户提示提供指导
- **依据**：官方建议对数学问题添加逐步推理指令和 `\boxed{}` 格式要求

### 4. 多次测试支持
- 创建了多次运行评估脚本
- 支持运行多次评估并计算平均结果
- 提供标准差和置信区间信息
- **依据**：官方建议进行多次测试并平均结果以获得更可靠的性能评估

### 5. 思维模式处理
- **R1-0528 版本**：移除了强制思维模式的功能，因为不再需要
- **早期版本**：自动检测 DeepSeek-R1 模型并强制以 `<think>\n` 开始响应
- **依据**：R1-0528 官方说明不再需要强制思维模式

## 使用方法

### DeepSeek-R1-0528 版本

#### 标准单次评估
```bash
python -m lm_eval --model hf --model_args pretrained=deepseek-ai/DeepSeek-R1-0528 --tasks mmlu_pro_r1_0528 --num_fewshot 5
```

#### 多次评估并平均结果
```bash
python lm_eval/tasks/mmlu_pro/multi_run_evaluator_r1_0528.py --model "hf --model_args pretrained=deepseek-ai/DeepSeek-R1-0528" --runs 3
```

#### 数学任务专用评估
```bash
python -m lm_eval --model hf --model_args pretrained=deepseek-ai/DeepSeek-R1-0528 --tasks mmlu_pro_math_r1_0528 --num_fewshot 5
```

### 早期 DeepSeek-R1 版本

#### 标准单次评估
```bash
python -m lm_eval --model hf --model_args pretrained=your_model_name --tasks mmlu_pro --num_fewshot 5
```

#### 多次评估并平均结果
```bash
python lm_eval/tasks/mmlu_pro/multi_run_evaluator.py --model "hf --model_args pretrained=your_model_name" --runs 3
```

#### 针对早期 DeepSeek-R1 模型的评估
```bash
# 设置环境变量
export MODEL_NAME="deepseek-r1"

# 使用多次评估脚本
python lm_eval/tasks/mmlu_pro/multi_run_evaluator.py --model "hf --model_args pretrained=deepseek-r1-model" --runs 3 --force_thinking
```

## 文件结构

### DeepSeek-R1-0528 版本文件
- `_r1_0528_template_yaml`: R1-0528 通用模板，支持系统提示
- `_r1_0528_math_template_yaml`: R1-0528 数学模板，包含数学专用系统提示
- `mmlu_pro_*_r1_0528.yaml`: 各学科的 R1-0528 优化配置
- `_mmlu_pro_r1_0528.yaml`: R1-0528 组配置文件
- `multi_run_evaluator_r1_0528.py`: R1-0528 多次运行评估脚本

### 早期版本文件
- `_default_template_yaml`: 默认模板，温度设置为 0.6
- `_math_template_yaml`: 数学任务专用模板，包含特殊的推理指令
- `utils.py`: 修改后的工具函数，支持数学指令和模型检测
- `multi_run_evaluator.py`: 多次运行评估脚本

## 配置参数

### 温度设置
- 推荐温度：0.6
- 范围：0.5-0.7
- 在模板文件中的 `generation_kwargs` 部分设置

### 系统提示（R1-0528 新特性）
- 通用任务：专业的多选题回答指导
- 数学任务：数学专家角色和逐步推理要求
- 早期版本：不使用系统提示

### 思维模式
- R1-0528：自然的思维模式，无需强制
- 早期版本：通过环境变量检测并强制 `<think>\n`

## 注意事项

1. **版本选择**：请根据您使用的 DeepSeek-R1 版本选择相应的配置
2. **系统提示**：R1-0528 支持系统提示，早期版本避免使用
3. **思维模式**：R1-0528 不需要强制思维模式，早期版本需要
4. **温度设置**：所有版本都建议使用 0.6 的温度
5. **数学任务**：所有版本都支持 `\boxed{}` 格式要求

## 示例输出

### R1-0528 版本多次运行评估示例：
```
=== FINAL AVERAGED RESULTS (DeepSeek-R1-0528) ===
mmlu_pro_math_r1_0528: 0.7456 ± 0.0134 (n=3)
mmlu_pro_physics_r1_0528: 0.7123 ± 0.0187 (n=3)
mmlu_pro_r1_0528: 0.7234 ± 0.0156 (n=3)
```

### 早期版本多次运行评估示例：
```
=== FINAL AVERAGED RESULTS ===
mmlu_pro_math: 0.7234 ± 0.0156 (n=3)
mmlu_pro_physics: 0.6891 ± 0.0203 (n=3)
mmlu_pro: 0.6945 ± 0.0123 (n=3)
```

## 参考资料

- [DeepSeek-R1 GitHub Repository](https://github.com/deepseek-ai/DeepSeek-R1)
- [DeepSeek-R1 Usage Recommendations](https://github.com/deepseek-ai/DeepSeek-R1?tab=readme-ov-file#usage-recommendations)
- [DeepSeek-R1-0528 Hugging Face Model Card](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)
- [DeepSeek-R1-0528 Usage Instructions](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528#4-how-to-run-locally)
- [lm-evaluation-harness Documentation](https://github.com/EleutherAI/lm-evaluation-harness) 