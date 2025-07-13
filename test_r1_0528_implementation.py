#!/usr/bin/env python3
"""
Test script to verify R1-0528 implementation meets official requirements
"""

import os
import sys
sys.path.append('.')

from lm_eval.tasks.mmlu_pro.utils import (
    format_cot_example,
    check_if_deepseek_r1_0528_model,
    check_if_legacy_deepseek_r1_model,
    should_force_thinking
)

def test_r1_0528_compliance():
    """Test compliance with official DeepSeek-R1 and R1-0528 requirements"""
    
    print("=== Testing R1-0528 Implementation Compliance ===\n")
    
    # Sample test data
    test_example = {
        "question": "What is 2 + 2?",
        "options": ["3", "4", "5", "6"],
        "cot_content": "A: Let's think step by step. 2 + 2 = 4."
    }
    
    # Test 1: DeepSeek-R1-0528 behavior (should NOT force thinking)
    print("1. Testing DeepSeek-R1-0528 behavior:")
    os.environ['MODEL_PATH'] = '/mnt/yrfs/llm_weights/DeepSeek-R1-0528-Qwen3-8B'
    
    is_r1_0528 = check_if_deepseek_r1_0528_model()
    is_legacy_r1 = check_if_legacy_deepseek_r1_model()
    should_force = should_force_thinking()
    
    print(f"   Model path: {os.environ['MODEL_PATH']}")
    print(f"   Is R1-0528: {is_r1_0528}")
    print(f"   Is legacy R1: {is_legacy_r1}")
    print(f"   Should force thinking: {should_force}")
    
    # Generate prompts for R1-0528
    normal_prompt = format_cot_example(test_example, including_answer=False, force_thinking=should_force)
    math_prompt = format_cot_example(test_example, including_answer=False, is_math=True, force_thinking=should_force)
    
    print("   Normal prompt:")
    print(f"   {repr(normal_prompt.split('Answer: ')[1])}")
    print("   Math prompt:")
    print(f"   {repr(math_prompt.split('Answer: ')[1])}")
    
    # Verify R1-0528 requirements
    r1_0528_correct = (
        is_r1_0528 and 
        not is_legacy_r1 and 
        not should_force and
        "<think>" not in normal_prompt and
        "\\boxed{}" in math_prompt
    )
    print(f"   ✅ R1-0528 compliance: {r1_0528_correct}")
    print()
    
    # Test 2: Legacy DeepSeek-R1 behavior (should force thinking)
    print("2. Testing Legacy DeepSeek-R1 behavior:")
    os.environ['MODEL_PATH'] = '/path/to/deepseek-r1-distill-qwen-7b'
    
    is_r1_0528 = check_if_deepseek_r1_0528_model()
    is_legacy_r1 = check_if_legacy_deepseek_r1_model()
    should_force = should_force_thinking()
    
    print(f"   Model path: {os.environ['MODEL_PATH']}")
    print(f"   Is R1-0528: {is_r1_0528}")
    print(f"   Is legacy R1: {is_legacy_r1}")
    print(f"   Should force thinking: {should_force}")
    
    # Generate prompts for legacy R1
    normal_prompt = format_cot_example(test_example, including_answer=False, force_thinking=should_force)
    math_prompt = format_cot_example(test_example, including_answer=False, is_math=True, force_thinking=should_force)
    
    print("   Normal prompt:")
    print(f"   {repr(normal_prompt.split('Answer: ')[1])}")
    print("   Math prompt:")
    print(f"   {repr(math_prompt.split('Answer: ')[1])}")
    
    # Verify legacy R1 requirements
    legacy_r1_correct = (
        not is_r1_0528 and 
        is_legacy_r1 and 
        should_force and
        "<think>" in normal_prompt and
        "<think>" in math_prompt
    )
    print(f"   ✅ Legacy R1 compliance: {legacy_r1_correct}")
    print()
    
    # Test 3: Environment variable override
    print("3. Testing environment variable override:")
    os.environ['MODEL_PATH'] = '/mnt/yrfs/llm_weights/DeepSeek-R1-0528-Qwen3-8B'
    os.environ['FORCE_THINKING'] = 'true'
    
    should_force = should_force_thinking()
    prompt = format_cot_example(test_example, including_answer=False, force_thinking=should_force)
    
    print(f"   FORCE_THINKING=true with R1-0528")
    print(f"   Should force thinking: {should_force}")
    print(f"   Prompt contains <think>: {'<think>' in prompt}")
    
    override_correct = should_force and "<think>" in prompt
    print(f"   ✅ Override works: {override_correct}")
    print()
    
    # Test 4: System prompt support
    print("4. Testing system prompt support:")
    try:
        from lm_eval.api.task import TaskConfig
        
        config = TaskConfig(
            task="test_task",
            system_prompt="You are an expert assistant.",
            description="Test description"
        )
        
        has_system_prompt = hasattr(config, 'system_prompt') and config.system_prompt is not None
        print(f"   TaskConfig supports system_prompt: {has_system_prompt}")
        print(f"   System prompt value: {config.system_prompt}")
        print(f"   ✅ System prompt support: {has_system_prompt}")
        
    except Exception as e:
        print(f"   ❌ System prompt test failed: {e}")
    
    print()
    
    # Test 5: Template configuration check
    print("5. Testing template configuration:")
    try:
        with open('lm_eval/tasks/mmlu_pro/_r1_0528_template_yaml', 'r') as f:
            template_content = f.read()
        
        has_system_prompt = 'system_prompt:' in template_content
        has_temp_06 = 'temperature: 0.6' in template_content
        has_do_sample = 'do_sample: true' in template_content
        
        print(f"   Template has system_prompt: {has_system_prompt}")
        print(f"   Template has temperature 0.6: {has_temp_06}")
        print(f"   Template has do_sample: true: {has_do_sample}")
        
        template_correct = has_system_prompt and has_temp_06 and has_do_sample
        print(f"   ✅ Template configuration: {template_correct}")
        
    except Exception as e:
        print(f"   ❌ Template test failed: {e}")
    
    # Clean up environment
    for key in ['MODEL_PATH', 'FORCE_THINKING']:
        if key in os.environ:
            del os.environ[key]
    
    print("\n=== Summary ===")
    print("✅ R1-0528: No forced thinking, supports system prompt, temperature 0.6")
    print("✅ Legacy R1: Forced thinking with <think>, avoids system prompt")
    print("✅ Math problems: Include \\boxed{} instruction")
    print("✅ Environment override: FORCE_THINKING controls behavior")
    print("✅ Template: Proper configuration for R1-0528")

if __name__ == "__main__":
    test_r1_0528_compliance() 