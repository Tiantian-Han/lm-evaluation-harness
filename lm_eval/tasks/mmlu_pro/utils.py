from functools import partial
import os
import re


choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def format_cot_example(example, including_answer=True, is_math=False, force_thinking=False):
    """
    Format Chain-of-Thought example for MMLU Pro.
    
    Args:
        example: The example data
        including_answer: Whether to include the answer
        is_math: Whether this is a math problem
        force_thinking: Whether to force thinking pattern with "<think>\n"
    """
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
        # Force thinking pattern for traditional DeepSeek-R1 models
        if force_thinking:
            prompt += "Answer: <think>\n"
        else:
            if is_math:
                prompt += "Answer: Please reason step by step, and put your final answer within \\boxed{}. Let's think step by step."
            else:
                prompt += "Answer: Let's think step by step."

    return prompt


def check_if_deepseek_r1_0528_model():
    """
    Check if the model being used is DeepSeek-R1-0528.
    This version supports system prompts and doesn't require forced thinking pattern.
    """
    model_name = os.environ.get('MODEL_NAME', '').lower()
    model_path = os.environ.get('MODEL_PATH', '').lower()
    pretrained_model = os.environ.get('PRETRAINED_MODEL', '').lower()
    
    # Check for R1-0528 specific patterns
    r1_0528_patterns = [
        'deepseek-r1-0528', 'deepseek_r1_0528', 'r1-0528', 'r1_0528'
    ]
    
    sources = [model_name, model_path, pretrained_model]
    
    for source in sources:
        for pattern in r1_0528_patterns:
            if pattern in source:
                return True
    
    return False


def check_if_legacy_deepseek_r1_model():
    """
    Check if the model being used is a legacy DeepSeek-R1 model (not 0528).
    Legacy versions require forced thinking pattern and avoid system prompts.
    """
    model_name = os.environ.get('MODEL_NAME', '').lower()
    model_path = os.environ.get('MODEL_PATH', '').lower()
    pretrained_model = os.environ.get('PRETRAINED_MODEL', '').lower()
    
    # Check for general R1 patterns but exclude 0528
    r1_patterns = ['deepseek-r1', 'deepseek_r1', 'deepseekr1']
    
    sources = [model_name, model_path, pretrained_model]
    
    for source in sources:
        for pattern in r1_patterns:
            if pattern in source and '0528' not in source:
                return True
    
    return False


def should_force_thinking():
    """
    Determine if we should force thinking pattern based on model type and environment settings.
    
    According to official documentation:
    - DeepSeek-R1-0528: Does NOT require forced thinking pattern
    - Legacy DeepSeek-R1: Requires forced thinking pattern with "<think>\n"
    """
    # Check if explicitly controlled via environment variable
    force_thinking_env = os.environ.get('FORCE_THINKING', '').lower()
    if force_thinking_env in ['true', '1', 'yes', 'on']:
        return True
    elif force_thinking_env in ['false', '0', 'no', 'off']:
        return False
    
    # Auto-detect based on model type
    # For R1-0528, do NOT force thinking (official requirement)
    if check_if_deepseek_r1_0528_model():
        return False  # R1-0528 does not require forced thinking
    
    # For legacy R1 models, force thinking (official requirement)
    if check_if_legacy_deepseek_r1_model():
        return True
    
    return False


# Check if we should force thinking pattern
force_thinking = should_force_thinking()

# Standard document conversion functions
doc_to_text = partial(format_cot_example, including_answer=False, force_thinking=force_thinking)
fewshot_to_text = partial(format_cot_example, including_answer=True, force_thinking=False)

# Math-specific versions with \boxed{} instruction
doc_to_text_math = partial(format_cot_example, including_answer=False, is_math=True, force_thinking=force_thinking)
fewshot_to_text_math = partial(format_cot_example, including_answer=True, is_math=True, force_thinking=False)


def add_choices_to_doc(doc):
    """Adds a 'choices' field to the document, which is a list of
    letters (A, B, C, ...) corresponding to the options."""
    num_options = len(doc["options"])
    doc["choices"] = choices[:num_options]
    return doc


def process_docs(dataset, subject):
    """
    Filters a dataset by subject and adds a 'choices' field to each document,
    which is required by the multi_choice_regex filter.
    """
    filtered_dataset = dataset.filter(lambda x: x["category"] == subject)
    return filtered_dataset.map(add_choices_to_doc)


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


def process_results_pass_at_k(doc, results):
    """
    Custom process_results function to implement pass@k logic for MMLU.
    It checks if any of the generated results match the correct answer.

    :param doc: The document object.
    :param results: A list of generated strings from the model.
    :return: A dictionary with a "pass@k" metric.
    """
    # The 'gold' is the correct choice, e.g., 'A', 'B', etc.
    gold = doc["answer"]
    
    # The 'results' are the model's generated answers, after being
    # processed by the 'filter_list' regex.
    # We check if the correct answer is present in any of the generations.
    is_correct = any([result == gold for result in results])
    
    return {
        "pass@1": is_correct
    }


def generate_choices(doc):
    """
    Generates a list of choice letters (A, B, C, ...) based on the number
    of options in the document.
    """
    num_options = len(doc["options"])
    return choices[:num_options]


def process_results_robust(doc, results):
    """
    Processes results and implements a robust fallback regex strategy.
    This function is used to extract the final answer from the model's
    output and calculate the exact match score.

    :param doc: The document object.
    :param results: A list containing the model's generated text.
    :return: A dictionary with the exact_match score.
    """
    # The first result is the one we want to process.
    completion = results[0]
    
    # The 'gold' is the correct choice, e.g., 'A', 'B', etc.
    gold = doc["answer"]
    
    # List of regex patterns to try in order of preference.
    regex_patterns = [
        r"answer is \((A-J)\)",
        r"ANSWER:\s?\(?(A-J)\)?",
        r"[aA]nswer:\s*\(?(A-J)\)?",
        r"(?:the\s+)?answer\s+is\s+\(?(A-J)\)?",
        # Final fallback: look for the last occurrence of (A), (B), etc.
        r".*\((A-J)\)",
    ]

    extracted_answer = None
    for pattern in regex_patterns:
        match = re.search(pattern, completion, re.IGNORECASE)
        if match:
            extracted_answer = match.group(1).upper()
            break
    
    # If no pattern matched, try a final, simple fallback.
    if extracted_answer is None:
        # This regex looks for the last single capital letter A-J on its own.
        match = re.findall(r"\b([A-J])\b", completion)
        if match:
            extracted_answer = match[-1]

    is_correct = (extracted_answer == gold)

    return {
        "exact_match": is_correct
    }


process_psychology = partial(process_docs, subject="psychology")
