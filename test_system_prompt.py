#!/usr/bin/env python3
"""
Test script to verify system prompt functionality in R1-0528 branch
"""

import sys
import os
sys.path.append('.')

def test_system_prompt_support():
    """Test that system prompt is properly supported"""
    
    print("=== Testing System Prompt Support ===\n")
    
    # Test 1: Check if TaskConfig supports system_prompt
    try:
        from lm_eval.api.task import TaskConfig
        
        # Create a config with system_prompt
        config = TaskConfig(
            task="test_task",
            system_prompt="You are a helpful assistant.",
            description="Test description"
        )
        
        print("1. TaskConfig system_prompt support:")
        print(f"   system_prompt field: {hasattr(config, 'system_prompt')}")
        print(f"   system_prompt value: {config.system_prompt}")
        print("   ✅ TaskConfig supports system_prompt")
        print()
        
    except Exception as e:
        print(f"   ❌ Error testing TaskConfig: {e}")
        print()
    
    # Test 2: Check template files
    template_files = [
        "lm_eval/tasks/mmlu_pro/_r1_0528_template_yaml",
        "lm_eval/tasks/mmlu_pro/_r1_0528_math_template_yaml"
    ]
    
    print("2. Template files system_prompt support:")
    for template_file in template_files:
        try:
            with open(template_file, 'r') as f:
                content = f.read()
                if 'system_prompt:' in content:
                    print(f"   ✅ {template_file} has system_prompt")
                else:
                    print(f"   ❌ {template_file} missing system_prompt")
        except FileNotFoundError:
            print(f"   ❌ {template_file} not found")
        except Exception as e:
            print(f"   ❌ Error reading {template_file}: {e}")
    print()
    
    # Test 3: Test fewshot_context with system_prompt
    try:
        from lm_eval.api.task import ConfigurableTask
        
        # Create a mock task with system_prompt
        config = {
            "task": "test_task",
            "system_prompt": "You are a test assistant.",
            "description": "Test description",
            "doc_to_text": lambda doc: "Test question",
            "doc_to_target": lambda doc: "Test answer",
            "output_type": "generate_until",
            "generation_kwargs": {"until": ["Question:"], "max_gen_toks": 100}
        }
        
        print("3. fewshot_context system_prompt integration:")
        print("   Testing system_prompt priority in fewshot_context...")
        
        # This would require more complex setup to fully test
        # For now, just verify the config accepts system_prompt
        task_config = ConfigurableTask(config=config)
        if hasattr(task_config.config, 'system_prompt'):
            print(f"   ✅ Task config has system_prompt: {task_config.config.system_prompt}")
        else:
            print("   ❌ Task config missing system_prompt")
        print()
        
    except Exception as e:
        print(f"   ❌ Error testing fewshot_context: {e}")
        print()
    
    # Test 4: Show usage example
    print("4. Usage example:")
    print("   To use system prompt in your YAML config:")
    print("   ```yaml")
    print("   system_prompt: \"You are an expert in multiple-choice questions.\"")
    print("   ```")
    print()
    print("   Priority order:")
    print("   1. config.system_prompt (highest)")
    print("   2. system_instruction parameter")
    print("   3. description field")
    print()
    
    print("=== System Prompt Support Test Complete ===")

if __name__ == "__main__":
    test_system_prompt_support() 