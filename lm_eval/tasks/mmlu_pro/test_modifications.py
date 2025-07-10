#!/usr/bin/env python3
"""
Test script to verify MMLU Pro modifications work correctly.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

def test_math_prompt():
    """Test math-specific prompt generation."""
    try:
        from lm_eval.tasks.mmlu_pro.utils import format_cot_example
        
        # Mock example
        example = {
            "question": "What is 2 + 2?",
            "options": ["A. 3", "B. 4", "C. 5", "D. 6"],
            "cot_content": "A: Let's think step by step. 2 + 2 = 4."
        }
        
        # Test regular prompt
        regular_prompt = format_cot_example(example, including_answer=False, is_math=False)
        print("Regular prompt:")
        print(regular_prompt)
        print("\n" + "="*50 + "\n")
        
        # Test math prompt
        math_prompt = format_cot_example(example, including_answer=False, is_math=True)
        print("Math prompt:")
        print(math_prompt)
        print("\n" + "="*50 + "\n")
        
        # Test DeepSeek-R1 prompt
        deepseek_prompt = format_cot_example(example, including_answer=False, is_math=True, force_thinking=True)
        print("DeepSeek-R1 math prompt:")
        print(deepseek_prompt)
        
        return True
    except Exception as e:
        print(f"Error testing prompts: {e}")
        return False

def test_deepseek_detection():
    """Test DeepSeek-R1 model detection."""
    try:
        from lm_eval.tasks.mmlu_pro.utils import check_if_deepseek_r1_model
        
        # Test without environment variable
        original_model_name = os.environ.get('MODEL_NAME')
        if 'MODEL_NAME' in os.environ:
            del os.environ['MODEL_NAME']
        
        result1 = check_if_deepseek_r1_model()
        print(f"DeepSeek detection without MODEL_NAME: {result1}")
        
        # Test with DeepSeek-R1 model name
        os.environ['MODEL_NAME'] = 'deepseek-r1-test'
        result2 = check_if_deepseek_r1_model()
        print(f"DeepSeek detection with 'deepseek-r1-test': {result2}")
        
        # Restore original environment
        if original_model_name:
            os.environ['MODEL_NAME'] = original_model_name
        elif 'MODEL_NAME' in os.environ:
            del os.environ['MODEL_NAME']
        
        return True
    except Exception as e:
        print(f"Error testing DeepSeek detection: {e}")
        return False

def main():
    print("Testing MMLU Pro modifications...\n")
    
    success = True
    
    print("1. Testing math prompt generation:")
    if not test_math_prompt():
        success = False
    
    print("\n2. Testing DeepSeek-R1 detection:")
    if not test_deepseek_detection():
        success = False
    
    if success:
        print("\n✅ All tests passed! Modifications are working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the modifications.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 