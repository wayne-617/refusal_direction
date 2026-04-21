import json
import torch
import os
from tqdm import tqdm

from pipeline.model_utils.model_base import ModelBase
from pipeline.utils.hook_utils import get_all_subspace_ablation_hooks
from pipeline.submodules.select_direction import get_last_position_logits, kl_div_fn, get_refusal_scores

def select_multivector_direction(
    model_base: ModelBase,
    harmful_instructions,
    harmless_instructions,
    candidate_subspaces, # [n_pos, n_layer, k_max, d_model]
    artifact_dir,
    kl_threshold=0.25,
    batch_size=32
):
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    n_pos, n_layer, k_max, d_model = candidate_subspaces.shape

    baseline_refusal_scores_harmful = get_refusal_scores(model_base.model, harmful_instructions, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_hooks=[], batch_size=batch_size)

    baseline_harmless_logits = get_last_position_logits(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        instructions=harmless_instructions,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        fwd_pre_hooks=[],
        fwd_hooks=[],
        batch_size=batch_size
    )

    results = []

    # Iterate over pos, layer, and k
    for source_pos in range(-n_pos, 0):
        for source_layer in range(n_layer):
            for k in range(1, k_max + 1):
                subspace = candidate_subspaces[source_pos, source_layer, :k, :]
                
                fwd_pre_hooks, fwd_hooks = get_all_subspace_ablation_hooks(model_base, subspace)

                # Compute KL divergence
                intervention_logits = get_last_position_logits(
                    model=model_base.model,
                    tokenizer=model_base.tokenizer,
                    instructions=harmless_instructions,
                    tokenize_instructions_fn=model_base.tokenize_instructions_fn,
                    fwd_pre_hooks=fwd_pre_hooks,
                    fwd_hooks=fwd_hooks,
                    batch_size=batch_size
                )
                kl_div = kl_div_fn(baseline_harmless_logits, intervention_logits, mask=None).mean(dim=0).item()

                # Compute Refusal
                refusal_scores = get_refusal_scores(model_base.model, harmful_instructions, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, batch_size=batch_size)
                mean_refusal = refusal_scores.mean().item()

                results.append({
                    'position': source_pos,
                    'layer': source_layer,
                    'k': k,
                    'refusal_score': mean_refusal,
                    'kl_div_score': kl_div
                })

    with open(f"{artifact_dir}/multivector_evaluations.json", 'w') as f:
        json.dump(results, f, indent=4)

    # Filter by KL and enforce selection from deep layers where semantic features live
    filtered_results = [r for r in results if r['kl_div_score'] <= kl_threshold and r['layer'] >= 8]
    if len(filtered_results) == 0:
        print("Warning: All multi-vector subspaces were filtered out by KL threshold. Relaxing it.")
        filtered_results = results
        
    # Sort by refusal score (lower is better, meaning closer to bypassing refusal)
    filtered_results = sorted(filtered_results, key=lambda x: x['refusal_score'])

    best = filtered_results[0]
    best_pos = best['position']
    best_layer = best['layer']
    best_k = best['k']

    print(f"Selected multivector subspace: position={best_pos}, layer={best_layer}, k={best_k}")
    print(f"Refusal score: {best['refusal_score']:.4f} (baseline: {baseline_refusal_scores_harmful.mean().item():.4f})")
    print(f"KL Divergence: {best['kl_div_score']:.4f}")

    best_subspace = candidate_subspaces[best_pos, best_layer, :best_k, :]
    return best_pos, best_layer, best_k, best_subspace
