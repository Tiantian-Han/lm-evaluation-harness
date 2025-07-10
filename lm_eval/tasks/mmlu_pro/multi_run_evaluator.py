#!/usr/bin/env python3
"""
Multi-run evaluator for MMLU Pro tasks.
This script runs MMLU Pro evaluation multiple times and computes average results.
"""

import json
import statistics
import subprocess
import sys
import argparse
from pathlib import Path


def run_single_evaluation(model_args, task_name="mmlu_pro", num_fewshot=5, additional_args=None):
    """Run a single evaluation and return the results."""
    cmd = [
        "python", "-m", "lm_eval",
        "--model", model_args,
        "--tasks", task_name,
        "--num_fewshot", str(num_fewshot),
        "--batch_size", "auto",
        "--output_path", "temp_results"
    ]
    
    if additional_args:
        cmd.extend(additional_args)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse the results from the output file
        results_file = Path("temp_results") / "results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        else:
            print("Warning: Results file not found")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None


def extract_scores(results):
    """Extract accuracy scores from evaluation results."""
    scores = {}
    if results and "results" in results:
        for task, metrics in results["results"].items():
            if "exact_match" in metrics:
                scores[task] = metrics["exact_match"]
    return scores


def compute_average_results(all_results):
    """Compute average results across multiple runs."""
    if not all_results:
        return {}
    
    # Collect all task names
    all_tasks = set()
    for result in all_results:
        all_tasks.update(result.keys())
    
    averaged_results = {}
    for task in all_tasks:
        task_scores = [result.get(task, 0) for result in all_results if task in result]
        if task_scores:
            averaged_results[task] = {
                "mean": statistics.mean(task_scores),
                "std": statistics.stdev(task_scores) if len(task_scores) > 1 else 0,
                "runs": len(task_scores),
                "individual_scores": task_scores
            }
    
    return averaged_results


def main():
    parser = argparse.ArgumentParser(description="Run MMLU Pro evaluation multiple times")
    parser.add_argument("--model", required=True, help="Model arguments")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs (default: 3)")
    parser.add_argument("--task", default="mmlu_pro", help="Task name (default: mmlu_pro)")
    parser.add_argument("--num_fewshot", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--output", default="mmlu_pro_multi_run_results.json", help="Output file")
    parser.add_argument("--force_thinking", action="store_true", 
                       help="Force thinking pattern for DeepSeek-R1 models")
    
    args = parser.parse_args()
    
    # Set environment variable for DeepSeek-R1 detection if requested
    if args.force_thinking:
        import os
        os.environ["MODEL_NAME"] = "deepseek-r1"
    
    print(f"Running MMLU Pro evaluation {args.runs} times...")
    
    all_results = []
    for run_idx in range(args.runs):
        print(f"\nRun {run_idx + 1}/{args.runs}")
        result = run_single_evaluation(
            model_args=args.model,
            task_name=args.task,
            num_fewshot=args.num_fewshot
        )
        
        if result:
            scores = extract_scores(result)
            all_results.append(scores)
            print(f"Run {run_idx + 1} completed. Overall accuracy: {scores.get('mmlu_pro', 'N/A')}")
        else:
            print(f"Run {run_idx + 1} failed")
    
    if all_results:
        averaged_results = compute_average_results(all_results)
        
        # Save results
        output_data = {
            "configuration": {
                "model": args.model,
                "task": args.task,
                "num_fewshot": args.num_fewshot,
                "runs": args.runs,
                "force_thinking": args.force_thinking
            },
            "averaged_results": averaged_results,
            "individual_runs": all_results
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n=== FINAL AVERAGED RESULTS ===")
        for task, metrics in averaged_results.items():
            print(f"{task}: {metrics['mean']:.4f} ± {metrics['std']:.4f} (n={metrics['runs']})")
        
        print(f"\nResults saved to {args.output}")
    else:
        print("No successful runs completed")
        sys.exit(1)


if __name__ == "__main__":
    main() 