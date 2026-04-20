import torch
import os

from typing import List
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from pipeline.utils.hook_utils import add_hooks
from pipeline.model_utils.model_base import ModelBase
from pipeline.submodules.generate_directions import get_mean_activations, get_mean_diff

def get_all_activations_pre_hook(layer, cache: Float[Tensor, "n_samples pos layer d_model"], start_idx, positions: List[int]):
    def hook_fn(module, input):
        activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone().to(cache)
        batch_size = activation.shape[0]
        cache[start_idx:start_idx+batch_size, :, layer, :] = activation[:, positions, :]
    return hook_fn

def get_all_activations(model, tokenizer, instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module], batch_size=32, positions=[-1]):
    torch.cuda.empty_cache()

    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    all_activations = torch.zeros((n_samples, n_positions, n_layers, d_model), dtype=torch.float64, device=model.device)

    for i in tqdm(range(0, len(instructions), batch_size), desc="Extracting Activations"):
        inputs = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])
        
        fwd_pre_hooks = [(block_modules[layer], get_all_activations_pre_hook(layer=layer, cache=all_activations, start_idx=i, positions=positions)) for layer in range(n_layers)]

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )

    return all_activations

def get_multivector_directions(
    model, tokenizer, harmful_instructions, harmless_instructions, truthful_instructions, 
    tokenize_instructions_fn, block_modules: List[torch.nn.Module], batch_size=32, positions=[-1], k=1
):
    print("Generating TruthfulQA mean activations...")
    # Get truthful mean activation
    mean_acts_truthful = get_mean_activations(model, tokenizer, truthful_instructions, tokenize_instructions_fn, block_modules, batch_size=batch_size, positions=positions)
    truthful_dirs = mean_acts_truthful / (mean_acts_truthful.norm(dim=-1, keepdim=True) + 1e-8)

    print("Generating Harmful and Harmless mean activations...")
    mean_acts_harmful = get_mean_activations(model, tokenizer, harmful_instructions, tokenize_instructions_fn, block_modules, batch_size=batch_size, positions=positions)
    mean_acts_harmless = get_mean_activations(model, tokenizer, harmless_instructions, tokenize_instructions_fn, block_modules, batch_size=batch_size, positions=positions)
    
    # Simple baseline approach with k=1: mean diff orthogonalized against truthful
    if k == 1:
        mean_diff = mean_acts_harmful - mean_acts_harmless
        # Orthogonalize against truthful
        projection = (mean_diff * truthful_dirs).sum(dim=-1, keepdim=True) * truthful_dirs
        ortho_diff = mean_diff - projection
        
        # Reshape to have k dimension: [n_positions, n_layers, k=1, d_model]
        return ortho_diff.unsqueeze(2)
        
    print("Extracting all Harmful activations for PCA...")
    acts_harmful = get_all_activations(model, tokenizer, harmful_instructions, tokenize_instructions_fn, block_modules, batch_size=batch_size, positions=positions)
    
    n_samples, n_positions, n_layers, d_model = acts_harmful.shape
    
    # Output tensor
    subspaces = torch.zeros((n_positions, n_layers, k, d_model), dtype=torch.float64, device=model.device)
    
    for pos in range(n_positions):
        for layer in range(n_layers):
            # Center harmful activations against harmless domain mean
            X = acts_harmful[:, pos, layer, :] - mean_acts_harmless[pos, layer, :].unsqueeze(0)
            
            # SVD to get principal components
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
            top_k_vecs = Vh[:k, :] # shape [k, d_model]
            
            # Orthogonalize each component against truthful_dirs using Gram-Schmidt
            t_dir = truthful_dirs[pos, layer, :].unsqueeze(0) # [1, d_model]
            
            ortho_vecs = []
            for i in range(k):
                v = top_k_vecs[i:i+1, :]
                # Project out truthful direction
                v = v - (v @ t_dir.T) * t_dir
                
                # Project out previous orthogonalized components to maintain orthonormal basis
                for prev_v in ortho_vecs:
                    v = v - (v @ prev_v.T) * prev_v
                
                # Normalize
                v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
                ortho_vecs.append(v)
                
            subspaces[pos, layer, :, :] = torch.cat(ortho_vecs, dim=0)

    return subspaces

def generate_multivector_directions(
    model_base: ModelBase, harmful_instructions, harmless_instructions, truthful_instructions, artifact_dir, k=1
):
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    positions = list(range(-len(model_base.eoi_toks), 0))
    subspaces = get_multivector_directions(
        model_base.model, model_base.tokenizer, harmful_instructions, harmless_instructions, truthful_instructions,
        model_base.tokenize_instructions_fn, model_base.model_block_modules, positions=positions, k=k
    )

    # Subspaces shape: (n_positions, n_layers, k, d_model)
    torch.save(subspaces, f"{artifact_dir}/mean_diffs_multivector_k{k}.pt")

    return subspaces
