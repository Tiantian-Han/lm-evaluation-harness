from functools import partial
import os


choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def format_cot_example(example, including_answer=True, is_math=False, force_thinking=False):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"

    for i, opt in enumerate(options):
        if i >= len(choices):
            break
        prompt += "{}. {}\n".format(choices[i], opt)

    if including_answer:
        cot_content = example["cot_content"].replace(
            "A: Let's think step by step.", "Answer: Let's think step by step."
        )
        prompt += cot_content + "\n\n"
    else:
        if is_math:
            prompt += "Answer: Please reason step by step, and put your final answer within \\boxed{}. Let's think step by step."
        else:
            prompt += "Answer: Let's think step by step."
        
        # Force thinking pattern for DeepSeek-R1 models
        if force_thinking:
            prompt += "\n<think>\n"

    return prompt


def check_if_deepseek_r1_model():
    """Check if the model being used is a DeepSeek-R1 series model"""
    # This can be enhanced to check model name/path from environment or config
    model_name = os.environ.get('MODEL_NAME', '').lower()
    return 'deepseek-r1' in model_name or 'deepseek_r1' in model_name


# Check if we should force thinking pattern
force_thinking = check_if_deepseek_r1_model()

doc_to_text = partial(format_cot_example, including_answer=False, force_thinking=force_thinking)
fewshot_to_text = partial(format_cot_example, including_answer=True, force_thinking=False)

# Math-specific versions
doc_to_text_math = partial(format_cot_example, including_answer=False, is_math=True, force_thinking=force_thinking)
fewshot_to_text_math = partial(format_cot_example, including_answer=True, is_math=True, force_thinking=False)


def process_docs(dataset, subject):
    return dataset.filter(lambda x: x["category"] == subject)


process_biology = partial(process_docs, subject="biology")
process_business = partial(process_docs, subject="business")
process_chemistry = partial(process_docs, subject="chemistry")
process_computer_science = partial(process_docs, subject="computer science")
process_economics = partial(process_docs, subject="economics")
process_engineering = partial(process_docs, subject="engineering")
process_health = partial(process_docs, subject="health")
process_history = partial(process_docs, subject="history")
process_law = partial(process_docs, subject="law")
process_math = partial(process_docs, subject="math")
process_other = partial(process_docs, subject="other")
process_philosophy = partial(process_docs, subject="philosophy")
process_physics = partial(process_docs, subject="physics")
process_psychology = partial(process_docs, subject="psychology")
