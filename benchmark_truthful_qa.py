import os
import json
import torch
from dotenv import load_dotenv

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_all_direction_ablation_hooks, get_all_subspace_ablation_hooks
from pipeline.utils.hook_utils import add_hooks

import lm_eval
from lm_eval.models.huggingface import HFLM

load_dotenv()

def benchmark_truthfulqa():
    model_path = "google/gemma-2b-it"
    
    # 1. Check if all required prior info is available
    base_dir = r"pipeline\runs"
    baseline_dir = os.path.join(base_dir, "gemma-2b-it_baseline")
    multi_dir = os.path.join(base_dir, "gemma-2b-it_multivector")
    
    baseline_pt = os.path.join(baseline_dir, "direction.pt")
    baseline_meta = os.path.join(baseline_dir, "direction_metadata.json")
    
    multi_pt = os.path.join(multi_dir, "direction.pt")
    multi_meta = os.path.join(multi_dir, "direction_metadata.json")
    
    missing_files = []
    for f in [baseline_pt, baseline_meta, multi_pt, multi_meta]:
        if not os.path.exists(f):
            missing_files.append(f)
            
    if missing_files:
        print("ERROR: Missing the following required artifact files:")
        for f in missing_files:
            print(f"- {f}")
        print("\nPlease ensure you have fully run both the 'baseline' and 'multivector' pipelines before benchmarking.")
        return

    print("All required direction artifacts found! Loading model...")
    model_base = construct_model_base(model_path)
    
    # Wrap model for LM Eval
    print("Wrapping model for lm_eval harness...")
    lm = HFLM(pretrained=model_base.model, tokenizer=model_base.tokenizer, batch_size=32)
    
    # 2. Load configurations
    with open(baseline_meta, "r") as f: b_meta = json.load(f)
    with open(multi_meta, "r") as f: m_meta = json.load(f)
    
    baseline_direction = torch.load(baseline_pt).to(model_base.model.device)
    multi_subspace = torch.load(multi_pt).to(model_base.model.device)
    
    baseline_fwd_pre_hooks, baseline_fwd_hooks = get_all_direction_ablation_hooks(model_base, baseline_direction)
    multi_fwd_pre_hooks, multi_fwd_hooks = get_all_subspace_ablation_hooks(model_base, multi_subspace)
    
    tasks = ["truthfulqa_mc2"]
    
    def evaluate_harness(label, fwd_pre_hooks, fwd_hooks):
        print(f"\n--- Running LM Evaluation Harness for: {label} ---")
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            results = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=0)
            
            mc2_acc = results['results']['truthfulqa_mc2']['acc,none']
            
            print(f"\nResults for {label}:")
            print(f"TruthfulQA MC2 (Multi True):  {mc2_acc * 100:.1f}%")
            return {"mc2": mc2_acc}

    base_results = evaluate_harness("Base Model", [], [])
    baseline_results = evaluate_harness("Baseline Ablation", baseline_fwd_pre_hooks, baseline_fwd_hooks)
    multi_results = evaluate_harness("Multivector Ablation", multi_fwd_pre_hooks, multi_fwd_hooks)
    
    print("\n======= FINAL TRUTHFULQA EVALUATION SUMMARY =======")
    print(f"Base Model:           MC2 {base_results['mc2']*100:.1f}%")
    print(f"Baseline Ablation:    MC2 {baseline_results['mc2']*100:.1f}%")
    print(f"Multivector Ablation: MC2 {multi_results['mc2']*100:.1f}%")

if __name__ == "__main__":
    benchmark_truthfulqa()
