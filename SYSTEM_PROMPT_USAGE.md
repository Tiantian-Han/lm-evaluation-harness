# System Prompt 使用说明

## 概述

R1-0528分支现在支持system prompt功能，可以为DeepSeek-R1-0528模型提供系统级别的指导信息。

## 功能特点

- **原生支持**：在TaskConfig中直接支持system_prompt字段
- **优先级控制**：system_prompt具有最高优先级
- **模板集成**：预置的R1-0528模板已包含适当的system prompt
- **聊天模板兼容**：支持聊天模板格式的system prompt

## 使用方法

### 方法1：在YAML配置中使用

#### 普通问题模板
```yaml
# lm_eval/tasks/mmlu_pro/_r1_0528_template_yaml
system_prompt: "You are an expert in multiple-choice question answering. Please analyze each question carefully and provide your reasoning step by step."
```

#### 数学问题模板
```yaml
# lm_eval/tasks/mmlu_pro/_r1_0528_math_template_yaml
system_prompt: "You are an expert mathematician. Please solve mathematical problems step by step, showing your reasoning clearly. Put your final answer within \\boxed{}."
```

### 方法2：动态配置

```python
from lm_eval.api.task import TaskConfig

# 创建包含system prompt的配置
config = TaskConfig(
    task="mmlu_pro_custom",
    system_prompt="You are an AI assistant specialized in academic evaluation.",
    dataset_path="TIGER-Lab/MMLU-Pro",
    # ... 其他配置
)
```

### 方法3：使用预置模板

```bash
# 使用包含system prompt的R1-0528模板
CUDA_VISIBLE_DEVICES="0,1,2,3" lm_eval \
    --model vllm \
    --model_args pretrained=/mnt/yrfs/llm_weights/DeepSeek-R1-0528-Qwen3-8B,tensor_parallel_size=4,trust_remote_code=True \
    --tasks mmlu_pro_math_r1_0528 \
    --num_fewshot 5 \
    --batch_size auto
```

## 优先级规则

System prompt的优先级从高到低：

1. **config.system_prompt**（最高优先级）
2. **system_instruction 参数**
3. **description 字段**

```python
# 示例：优先级演示
config = {
    "system_prompt": "I am the system prompt",      # 最高优先级
    "description": "I am the description"           # 较低优先级
}

# 结果：使用 "I am the system prompt"
```

## 聊天模板支持

当使用聊天模板时，system prompt会被正确格式化：

```python
# 聊天格式
[
    {"role": "system", "content": "You are an expert in multiple-choice questions."},
    {"role": "user", "content": "Question: What is the capital of France?"},
    # ...
]
```

## 实际效果

### 不使用System Prompt
```
Question:
What is the capital of France?
Options:
A. London
B. Berlin
C. Paris
D. Madrid
Answer: Let's think step by step.
```

### 使用System Prompt
```
System: You are an expert in multiple-choice question answering. Please analyze each question carefully and provide your reasoning step by step.

Question:
What is the capital of France?
Options:
A. London
B. Berlin
C. Paris
D. Madrid
Answer: Let's think step by step.
```

## 预置模板

### R1-0528标准模板
- **文件**: `_r1_0528_template_yaml`
- **System Prompt**: "You are an expert in multiple-choice question answering. Please analyze each question carefully and provide your reasoning step by step."
- **适用**: 一般性多选题

### R1-0528数学模板
- **文件**: `_r1_0528_math_template_yaml`
- **System Prompt**: "You are an expert mathematician. Please solve mathematical problems step by step, showing your reasoning clearly. Put your final answer within \\boxed{}."
- **适用**: 数学相关问题

## 自定义System Prompt

您可以根据需要自定义system prompt：

```yaml
# 自定义示例
system_prompt: |
  You are an AI assistant with expertise in the following areas:
  - Multiple-choice question analysis
  - Step-by-step reasoning
  - Academic evaluation
  
  Please provide clear, logical reasoning for each answer.
```

## 验证方法

运行测试脚本验证system prompt功能：

```bash
python test_system_prompt.py
```

## 注意事项

1. **模型兼容性**: System prompt主要为DeepSeek-R1-0528等支持系统提示的模型设计
2. **优先级**: 确保理解优先级规则，避免配置冲突
3. **格式要求**: System prompt应该是清晰、具体的指导信息
4. **性能影响**: 较长的system prompt可能会影响推理速度

## 故障排除

### 问题1: System prompt不生效
- 检查TaskConfig是否正确设置了system_prompt字段
- 确认没有被其他配置覆盖
- 验证模型是否支持系统提示

### 问题2: 配置文件错误
- 检查YAML语法是否正确
- 确认system_prompt字段拼写无误
- 验证引号和缩进

### 问题3: 聊天模板问题
- 确认使用的是支持系统消息的聊天模板
- 检查模板格式是否正确

## 总结

System prompt功能为DeepSeek-R1-0528模型提供了更好的指导能力，通过合理配置可以显著提升模型在特定任务上的表现。建议根据具体任务需求选择合适的system prompt内容。 