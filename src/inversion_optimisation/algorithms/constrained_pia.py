import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
import os
import time
from datasets import load_dataset

from inversion_optimisation.utils import load_dataset_tokens, DotDict, DATA_PATH
from inversion_optimisation.algorithms.optimisers import CustomAdam
from inversion_optimisation.algorithms.embed_search import embed_search_initialisation

# Algorithm defined in https://arxiv.org/abs/2503.09022
# 'Prompt Inversion Attack against Collaborative Inference of Large Language Models'

def pia_text_search(cfg, model, device="cuda"):
    # Get the targets used for all experiments based on dataset
    if cfg.target_strategy == "random":
        with open(DATA_PATH / f"true_tokens_{cfg.num_targets}_{cfg.input_len}.pkl", 'rb') as file:
            loaded_true_tokens = pickle.load(file).to("cpu")
    elif cfg.target_strategy == "privacy":
        # Privacy dataset only allows num_targets == 500 currently
        privacy_ds = load_dataset("AdrSkapars/pii-inversion-test-5k", split=f"length_{cfg.input_len}")
        loaded_true_tokens = torch.cat([torch.tensor(item["tokens"]).to(torch.int64).unsqueeze(0) for item in privacy_ds], dim=0).to("cpu")
    else:
        loaded_true_tokens = load_dataset_tokens(cfg.target_strategy, cfg.input_len, cfg.num_targets, include_bos=False, random_sentence=True, random_start=False, model=model)

    # Get the targets output
    true_tokens_loc = str(DATA_PATH / f"true_tokens_{cfg.num_targets}_{cfg.input_len}_{cfg.output_len}")
    if cfg.target_sample == "random":
        true_tokens_loc += ".pkl"
    elif cfg.target_sample == "greedy":
        true_tokens_loc += "_greedy.pkl"
    with open(true_tokens_loc, 'rb') as file:
        loaded_true_outputs = pickle.load(file).to("cpu")

    # Get the initialisation based on strategy
    initialisation_embeds = embed_search_initialisation(
        cfg.input_len, cfg.num_targets, cfg.init_strategy, cfg.loaded_true_tokens, cfg.max_batch_size, cfg.max_chunk_size, model, device).to("cpu")

    # Initialise state variables
    state_path = DATA_PATH / f'{cfg.save_folder}/checkpoint_{cfg.input_len}_{cfg.num_targets}_{cfg.max_epochs}.pt'
    if os.path.exists(state_path):
        print("LOADING STATE")
        state = torch.load(state_path, weights_only=False)
        if len(state.batch_results) > 0:
            done_epochs = state.batch_results[-1]["done_epochs"]
    else:
        print("INITIALISING STATE")
        state = DotDict({
            "results" : [],
            "batch_results" : [],
            "true_outputs" : torch.tensor([]).to(device).to(torch.int64),
            "pred_embed" : torch.tensor([]).to(device),
            "loaded_i" : 0,
            "epoch" : 0,
            "num_remain_items" : cfg.num_targets,
            "num_success_items" : 0,
            "elapsed_time" : 0,
            "checkpoint_elapsed_time" : 0,
        })

    # Work out embedding space bounds for clamping
    embed_min = model.embed.W_E.min(dim=0, keepdim=True).values
    embed_max = model.embed.W_E.max(dim=0, keepdim=True).values

    while state.num_remain_items != 0 or len(state.batch_results) != 0:
        start_time = time.time()

        # Checkpoint current progress if hour has passed
        if state.elapsed_time - state.checkpoint_elapsed_time > (3600 * 0.5):
            print("\nSAVING STATE")
            state.checkpoint_elapsed_time = state.elapsed_time
            torch.save(state, state_path)

        # Print progress
        state.epoch += 1
        if state.epoch % 100 == 0:
            print(f"({state.num_success_items}/{cfg.num_targets})({cfg.num_targets-state.num_remain_items}/{cfg.num_targets}){state.epoch}", end=", ")

        ############### STAGE 1: Initial embedding search ###############
        with torch.no_grad():
            # Add whole new batch of items
            if len(state.batch_results) == 0 and state.num_remain_items != 0:
                done_epochs = 0
                num_new_items = min(cfg.max_batch_size, state.num_remain_items)
                state.num_remain_items -= num_new_items

                # Initialise new target and add to end
                true_tokens = loaded_true_tokens[state.loaded_i:state.loaded_i+num_new_items].to(device)
                new_true_outputs = loaded_true_outputs[state.loaded_i:state.loaded_i+num_new_items].to(device)
                state.true_outputs = torch.cat((state.true_outputs, new_true_outputs))

                for i in range(num_new_items):
                    # Initialise new results tracking and add to end
                    state.batch_results.append({
                        "true_tokens": true_tokens[i].to("cpu"),
                        "true_outputs": state.true_outputs[i].to("cpu"),
                        "pred_tokens": None,
                        "found_solution": False,
                        "done_epochs": 0,
                    })

                    # Initialise new prediction and add to end
                    new_pred_embed = initialisation_embeds[state.loaded_i+i:state.loaded_i+i+1].to(device)
                    state.pred_embed = torch.cat((state.pred_embed, new_pred_embed))

                state.loaded_i += num_new_items

        # Do one epoch of optimisation on batch
        pred_embed = state.pred_embed
        pred_embed.requires_grad = True
        pred_embed_full = torch.cat((pred_embed, model.embed(state.true_outputs[:,:-1])), dim=1)
        if "gpt" not in cfg.model_name and "tiny" not in cfg.model_name:
            pred_embed_full = pred_embed_full
        else:
            pred_embed_full = pred_embed_full + model.pos_embed(pred_embed_full[:,:,0].detach())
        pred_logits = model(pred_embed_full, start_at_layer=0)
        
        pred_logprobs = F.softmax(pred_logits[:,cfg.input_len-1:,:], dim=-1).clamp(min=1e-12).log()
        pred_logits_target = torch.gather(pred_logprobs, 2, state.true_outputs.unsqueeze(-1)).squeeze(-1)
        pred_logits_diff = (pred_logits_target - pred_logprobs.max(dim=-1).values)
        loss = - pred_logits_diff.mean()
        
        with torch.no_grad():
            # Find nearest neighbor indices
            nearest_neighbor_indices = []
            for i in range(0, pred_embed.size(0), cfg.max_chunk_size):
                batch_chunk = pred_embed[i:i+cfg.max_chunk_size]
                distances_chunk = torch.cdist(batch_chunk.squeeze(0), model.embed.W_E, p=2)
                nearest_idx_chunk = torch.argmin(distances_chunk, dim=-1)
                if nearest_idx_chunk.dim() == 1:
                    nearest_idx_chunk = nearest_idx_chunk.unsqueeze(0)
                nearest_neighbor_indices.append(nearest_idx_chunk)
            nearest_neighbor_indices = torch.cat(nearest_neighbor_indices, dim=0)

        # Get the actual nearest neighbor embeddings
        nearest_embeddings = model.embed.W_E[nearest_neighbor_indices]
        nn_loss = torch.norm(pred_embed - nearest_embeddings, p=2, dim=-1).mean()
        loss += cfg.nn_weight * nn_loss
        
        # Take gradient of loss with respect to pred_embed
        loss.backward()
        with torch.no_grad():
            state.pred_embed = state.pred_embed - (cfg.learn_rate * pred_embed.grad)
            
            # Clamp to embedding space bounds
            state.pred_embed = torch.clamp(state.pred_embed, min=embed_min, max=embed_max)
            
            done_epochs += 1
            for i in range(len(state.batch_results)-1,-1,-1):
                state.batch_results[i]["done_epochs"] += 1
                
            if done_epochs == cfg.max_epochs:
                ############### STAGE 2: Adaptive discretisation ###############
                batch_size, seq_len, d_model = pred_embed.shape
                
                # Find top-k nearest neighbors for each position in each sequence
                top_k_indices = []
                for i in range(0, batch_size, cfg.max_chunk_size):
                    batch_chunk = pred_embed[i:i+cfg.max_chunk_size]
                    # Shape: [chunk_size, seq_len, vocab_size]
                    distances = torch.cdist(batch_chunk, model.embed.W_E.unsqueeze(0).expand(batch_chunk.size(0), -1, -1), p=2)
                    # Get top-k closest tokens for each position
                    # Shape: [chunk_size, seq_len, k]
                    top_k_chunk = torch.topk(distances, k=cfg.top_k, dim=-1, largest=False).indices
                    top_k_indices.append(top_k_chunk)
                
                top_k_indices = torch.cat(top_k_indices, dim=0)  # Shape: [batch_size, seq_len, k]
                
                # Initialize pred_tokens with argmax (k=0, the closest neighbor)
                pred_tokens = top_k_indices[:, :, 0].clone()  # Shape: [batch_size, seq_len]
                
                # Iterate through each position in the sequence
                for pos in range(seq_len):
                    # Try each of the k substitutions
                    for k_idx in range(cfg.top_k):
                        # Create candidate sequences with the k-th option at current position
                        candidate_tokens = pred_tokens.clone()
                        candidate_tokens[:, pos] = top_k_indices[:, pos, k_idx]
                        
                        # Calculate loss (no regularization)
                        candidate_tokens_full = torch.cat((candidate_tokens, state.true_outputs[:,:-1]), dim=1)
                        candidate_logits = model(candidate_tokens_full)               
                        candidate_logprobs = F.softmax(candidate_logits[:,cfg.input_len-1:,:], dim=-1).clamp(min=1e-12).log()
                        candidate_logits_target = torch.gather(candidate_logprobs, 2, state.true_outputs.unsqueeze(-1)).squeeze(-1)
                        candidate_logits_diff = (candidate_logits_target - candidate_logprobs.max(dim=-1).values)
                        candidate_loss = - candidate_logits_diff.mean() # Shape: [batch_size]
                        
                        # Update best choice for each sequence in batch
                        if k_idx == 0:
                            best_losses = candidate_loss
                        else:
                            improved_mask = candidate_loss < best_losses
                            best_losses = torch.where(improved_mask, candidate_loss, best_losses)
                            pred_tokens[improved_mask, pos] = candidate_tokens[improved_mask, pos]
                
                # Update results and clear batch
                pred_tokens_full = torch.cat((pred_tokens, state.true_outputs[:,:-1]), dim=1)
                disc_pred_logits = model(pred_tokens_full)[:,cfg.input_len-1:,:]
                for i in range(len(state.batch_results)-1,-1,-1):
                    # Remove item if have found a solution or reached final epoch
                    have_inverted = torch.equal(state.batch_results[i]["true_outputs"].detach(), disc_pred_logits[i].argmax(dim=-1).to("cpu").detach())
                    if have_inverted:
                        state.batch_results[i]["found_solution"] = True
                        state.num_success_items += 1
                        
                    state.batch_results[i]["pred_tokens"] = pred_tokens[i].to("cpu")
                    state.results.append(state.batch_results.pop(i))
                    
                state.pred_embed = torch.tensor([]).to(device)
                state.true_outputs = torch.tensor([]).to(device).to(torch.int64)

            state.elapsed_time += time.time() - start_time

    return state.results, round(state.elapsed_time, 3)


def pia_search(cfg, model, device="cuda"):
    # Get the targets used for all experiments based on dataset
    if cfg.target_strategy == "random":
        with open(DATA_PATH / f"true_tokens_{cfg.num_targets}_{cfg.input_len}.pkl", 'rb') as file:
            loaded_true_tokens = pickle.load(file).to("cpu")
    elif cfg.target_strategy == "privacy":
        # Privacy dataset only allows num_targets == 500 currently
        privacy_ds = load_dataset("AdrSkapars/pii-inversion-test-5k", split=f"length_{cfg.input_len}")
        loaded_true_tokens = torch.cat([torch.tensor(item["tokens"]).to(torch.int64).unsqueeze(0) for item in privacy_ds], dim=0).to("cpu")
    else:
        loaded_true_tokens = load_dataset_tokens(cfg.target_strategy, cfg.input_len, cfg.num_targets, include_bos=False, random_sentence=True, random_start=False, model=model)

    # Get the initialisation based on strategy
    initialisation_embeds = embed_search_initialisation(
        cfg.input_len, cfg.num_targets, cfg.init_strategy, cfg.loaded_true_tokens, cfg.max_batch_size, cfg.max_chunk_size, model, device).to("cpu")

    # Initialise state variables
    state_path = DATA_PATH / f'{cfg.save_folder}/checkpoint_{cfg.input_len}_{cfg.num_targets}_{cfg.max_epochs}.pt'
    if os.path.exists(state_path):
        print("LOADING STATE")
        state = torch.load(state_path, weights_only=False)
        if len(state.batch_results) > 0:
            done_epochs = state.batch_results[-1]["done_epochs"]
    else:
        print("INITIALISING STATE")
        state = DotDict({
            "results" : [],
            "batch_results" : [],
            "true_logits" : torch.Tensor([]).to(device),
            "pred_embed" : torch.tensor([]).to(device),
            "loaded_i" : 0,
            "epoch" : 0,
            "num_remain_items" : cfg.num_targets,
            "num_success_items" : 0,
            "elapsed_time" : 0,
            "checkpoint_elapsed_time" : 0,
        })
        
    # Work out embedding space bounds for clamping
    embed_min = model.embed.W_E.min(dim=0, keepdim=True).values
    embed_max = model.embed.W_E.max(dim=0, keepdim=True).values

    while state.num_remain_items != 0 or len(state.batch_results) != 0:
        start_time = time.time()

        # Checkpoint current progress if hour has passed
        if state.elapsed_time - state.checkpoint_elapsed_time > (3600 * 0.5):
            print("\nSAVING STATE")
            state.checkpoint_elapsed_time = state.elapsed_time
            torch.save(state, state_path)

        # Print progress
        state.epoch += 1
        if state.epoch % 100 == 0:
            print(f"({state.num_success_items}/{cfg.num_targets})({cfg.num_targets-state.num_remain_items}/{cfg.num_targets}){state.epoch}", end=", ")

        ############### STAGE 1: Initial embedding search ###############
        with torch.no_grad():
            # Add whole new batch of items
            if len(state.batch_results) == 0 and state.num_remain_items != 0:
                done_epochs = 0
                num_new_items = min(cfg.max_batch_size, state.num_remain_items)
                state.num_remain_items -= num_new_items

                # Initialise new target and add to end
                true_tokens = loaded_true_tokens[state.loaded_i:state.loaded_i+num_new_items].to(device)
                new_true_logits = model(true_tokens).detach()[:,-1,:]
                state.true_logits = torch.cat((state.true_logits, new_true_logits))

                for i in range(num_new_items):
                    # Initialise new results tracking and add to end
                    state.batch_results.append({
                        "true_tokens": true_tokens[i].to("cpu"),
                        "pred_tokens": None,
                        "found_solution": False,
                        "done_epochs": 0,
                    })

                    # Initialise new prediction and add to end
                    new_pred_embed = initialisation_embeds[state.loaded_i+i:state.loaded_i+i+1].to(device)
                    state.pred_embed = torch.cat((state.pred_embed, new_pred_embed))

                state.loaded_i += num_new_items

        # Do one epoch of optimisation on batch
        pred_embed = state.pred_embed
        pred_embed.requires_grad = True
        pred_embed_full = pred_embed + model.pos_embed(pred_embed[:,:,0].detach())
        pred_logits = model(pred_embed_full, start_at_layer=0)
        loss = torch.nn.MSELoss()(state.true_logits.detach(), pred_logits[:,-1,:])
        with torch.no_grad():
            # Find nearest neighbor indices
            nearest_neighbor_indices = []
            for i in range(0, pred_embed.size(0), cfg.max_chunk_size):
                batch_chunk = pred_embed[i:i+cfg.max_chunk_size]
                distances_chunk = torch.cdist(batch_chunk.squeeze(0), model.embed.W_E, p=2)
                nearest_idx_chunk = torch.argmin(distances_chunk, dim=-1)
                if nearest_idx_chunk.dim() == 1:
                    nearest_idx_chunk = nearest_idx_chunk.unsqueeze(0)
                nearest_neighbor_indices.append(nearest_idx_chunk)
            nearest_neighbor_indices = torch.cat(nearest_neighbor_indices, dim=0)

        # Get the actual nearest neighbor embeddings
        nearest_embeddings = model.embed.W_E[nearest_neighbor_indices]
        nn_loss = torch.norm(pred_embed - nearest_embeddings, p=2, dim=-1).mean()
        loss += cfg.nn_weight * nn_loss
        
        # Take gradient of loss with respect to pred_embed
        loss.backward()
        with torch.no_grad():
            state.pred_embed = state.pred_embed - (cfg.learn_rate * pred_embed.grad)
            
            # Clamp to embedding space bounds
            state.pred_embed = torch.clamp(state.pred_embed, min=embed_min, max=embed_max)
            
            done_epochs += 1
            for i in range(len(state.batch_results)-1,-1,-1):
                state.batch_results[i]["done_epochs"] += 1
            
            if done_epochs == cfg.max_epochs:
                ############### STAGE 2: Adaptive discretisation ###############
                batch_size, seq_len, d_model = pred_embed.shape
                
                # Find top-k nearest neighbors for each position in each sequence
                top_k_indices = []
                for i in range(0, batch_size, cfg.max_chunk_size):
                    batch_chunk = pred_embed[i:i+cfg.max_chunk_size]
                    # Shape: [chunk_size, seq_len, vocab_size]
                    distances = torch.cdist(batch_chunk, model.embed.W_E.unsqueeze(0).expand(batch_chunk.size(0), -1, -1), p=2)
                    # Get top-k closest tokens for each position
                    # Shape: [chunk_size, seq_len, k]
                    top_k_chunk = torch.topk(distances, k=cfg.top_k, dim=-1, largest=False).indices
                    top_k_indices.append(top_k_chunk)
                
                top_k_indices = torch.cat(top_k_indices, dim=0)  # Shape: [batch_size, seq_len, k]
                
                # Initialize pred_tokens with argmax (k=0, the closest neighbor)
                pred_tokens = top_k_indices[:, :, 0].clone()  # Shape: [batch_size, seq_len]
                
                # Iterate through each position in the sequence
                for pos in range(seq_len):
                    # Try each of the k substitutions
                    for k_idx in range(cfg.top_k):
                        # Create candidate sequences with the k-th option at current position
                        candidate_tokens = pred_tokens.clone()
                        candidate_tokens[:, pos] = top_k_indices[:, pos, k_idx]
                        
                        # Calculate loss (only logit matching, no regularization)
                        candidate_logits = model(candidate_tokens)
                        candidate_loss = torch.nn.MSELoss(reduction='none')(
                            state.true_logits.detach(), 
                            candidate_logits[:, -1, :]
                        ).mean(dim=-1)  # Shape: [batch_size]
                        
                        # Update best choice for each sequence in batch
                        if k_idx == 0:
                            best_losses = candidate_loss
                        else:
                            improved_mask = candidate_loss < best_losses
                            best_losses = torch.where(improved_mask, candidate_loss, best_losses)
                            pred_tokens[improved_mask, pos] = candidate_tokens[improved_mask, pos]

                # Update results and clear batch
                disc_pred_logits = model(pred_tokens)[:,-1,:]
                for i in range(len(state.batch_results)-1,-1,-1):
                    have_inverted = torch.allclose(state.true_logits[i], disc_pred_logits[i], atol=1e-4, rtol=1e-4)
                    if have_inverted:
                        state.batch_results[i]["found_solution"] = True
                        state.num_success_items += 1
                        
                    state.batch_results[i]["pred_tokens"] = pred_tokens[i].to("cpu")
                    state.results.append(state.batch_results.pop(i))
                    
                state.pred_embed = torch.tensor([]).to(device)
                state.true_logits = torch.tensor([]).to(device)
                    
        state.elapsed_time += time.time() - start_time

    return state.results, round(state.elapsed_time, 3)
