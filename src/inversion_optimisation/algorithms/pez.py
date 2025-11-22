import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
import os
import time
from datasets import load_dataset

from inversion_optimisation.utils import load_dataset_tokens, DotDict
from inversion_optimisation.algorithms.embed_search import embed_search_initialisation

def pez_text_search(cfg, model, device="cuda"):
    # Get the targets used for all experiments based on dataset
    if cfg.target_strategy == "random":
        with open(f"/content/true_tokens_{cfg.num_targets}_{cfg.input_len}.pkl", 'rb') as file:
            loaded_true_tokens = pickle.load(file).to("cpu")
    else:
        loaded_true_tokens = load_dataset_tokens(cfg.target_strategy, cfg.input_len, cfg.num_targets, include_bos=False, random_sentence=True, random_start=False, model=model)

    # Get the targets output
    true_tokens_loc = f"/content/true_tokens_{cfg.num_targets}_{cfg.input_len}_{cfg.output_len}"
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
    state_path = f'/content/{cfg.save_folder}/checkpoint_{cfg.input_len}_{cfg.num_targets}_{cfg.max_epochs}.pt'
    if os.path.exists(state_path):
        print("LOADING STATE")
        state = torch.load(state_path, weights_only=False)
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

    while state.num_remain_items != 0 or len(state.batch_results) != 0:
        start_time = time.time()

        # Checkpoint current progress if hour has passed
        if state.elapsed_time - state.checkpoint_elapsed_time > (3600 * 3):
            print("\nSAVING STATE")
            state.checkpoint_elapsed_time = state.elapsed_time
            torch.save(state, state_path)

        # Print progress
        state.epoch += 1
        if state.epoch % 100 == 0:
            print(f"({state.num_success_items}/{cfg.num_targets})({cfg.num_targets-state.num_remain_items}/{cfg.num_targets}){state.epoch}", end=", ")

        with torch.no_grad():
            # Add new items to batch if have space and have more items to do
            if (cfg.max_batch_size - len(state.batch_results)) > 0 and state.num_remain_items != 0:
                num_new_items = min((cfg.max_batch_size - len(state.batch_results)), state.num_remain_items)
                state.num_remain_items -= num_new_items

                # Initialise new target and add to end (batched)
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

                    # Initialise new prediction and add to end, one optimiser per sequence
                    new_pred_embed = initialisation_embeds[state.loaded_i+i:state.loaded_i+i+1].to(device)
                    state.pred_embed = torch.cat((state.pred_embed, new_pred_embed))

                state.loaded_i += num_new_items

        with torch.no_grad():
            # Add new items to batch if have space and have more items to do
            if (cfg.max_batch_size - len(state.batch_results)) > 0 and state.num_remain_items != 0:
                num_new_items = min((cfg.max_batch_size - len(state.batch_results)), state.num_remain_items)
                state.num_remain_items -= num_new_items

                # Initialise new target and add to end (batched)
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

                    # Initialise new prediction and add to end, one optimiser per sequence
                    new_pred_embed = initialisation_embeds[state.loaded_i+i:state.loaded_i+i+1].to(device)
                    state.pred_embed = torch.cat((state.pred_embed, new_pred_embed))

                state.loaded_i += num_new_items

        with torch.no_grad():
            # Use nearest neighbour to get back closest tokens from continuous embeddings
            pred_embed_continuous = state.pred_embed
            pred_tokens = []
            for i in range(0, pred_embed_continuous.size(0), cfg.max_chunk_size):
                batch_chunk = pred_embed_continuous[i:i+cfg.max_chunk_size]
                # Cosine similarity
                if cfg.con_distance == "cos sim":
                    tokens_sim_chunk = F.cosine_similarity(
                        batch_chunk.unsqueeze(2),
                        model.embed.W_E.unsqueeze(0).unsqueeze(0),
                        dim=-1
                    )
                    pred_tokens_chunk = torch.argmax(tokens_sim_chunk, dim=-1)

                # Euclidean distance
                elif cfg.con_distance == "euc dist":
                    tokens_sim_chunk = torch.cdist(batch_chunk.squeeze(0), model.embed.W_E, p=2)
                    pred_tokens_chunk = torch.argmin(tokens_sim_chunk, dim=-1)
                    if pred_tokens_chunk.dim() == 1:
                        pred_tokens_chunk = pred_tokens_chunk.unsqueeze(0)

                pred_tokens.append(pred_tokens_chunk)
            pred_tokens = torch.cat(pred_tokens, dim=0)
        
            # Get closest possible discrete embeddings
            pred_embed_discrete = model.embed(pred_tokens)
            pred_embed_discrete = pred_embed_discrete + model.pos_embed(pred_embed_discrete[:,:,0].detach())
            pred_embed_discrete.requires_grad = True
            
        # Use gradients from discrete embeddings to update continuous embeddings
        pred_embed_discrete_full = torch.cat((pred_embed_discrete, model.embed(state.true_outputs[:,:-1])), dim=1)
        pred_embed_discrete_full = pred_embed_discrete_full + model.pos_embed(pred_embed_discrete_full[:,:,0].detach())
        pred_logits = model(pred_embed_discrete_full, start_at_layer=0)
        pred_logprobs = F.softmax(pred_logits[:,cfg.input_len-1:,:], dim=-1).clamp(min=1e-12).log()
        pred_logits_target = torch.gather(pred_logprobs, 2, state.true_outputs.unsqueeze(-1)).squeeze(-1)
        pred_logits_diff = (pred_logits_target - pred_logprobs.max(dim=-1).values)
        loss = - pred_logits_diff.mean()
        loss.backward()
        with torch.no_grad():
            state.pred_embed = state.pred_embed - (cfg.learn_rate * pred_embed_discrete.grad)
            
            # Update history of tokens over epochs
            for i in range(len(state.batch_results)-1,-1,-1):
                # state.batch_results[i]["done_epochs"] += 1
                state.batch_results[i]["done_epochs"] += cfg.check_con_epochs

                # Remove item if have found a solution or reached final epoch
                have_inverted = torch.equal(state.batch_results[i]["true_outputs"].detach(), pred_logits[i].argmax(dim=-1).to("cpu").detach())
                if have_inverted:
                    state.batch_results[i]["found_solution"] = True
                    state.num_success_items += 1
                if have_inverted or (cfg.max_epochs is not None and state.batch_results[i]["done_epochs"] >= cfg.max_epochs):
                    state.batch_results[i]["pred_tokens"] = pred_tokens[i].to("cpu")
                    del state.pred_embed[i]
                    state.true_outputs = torch.cat((state.true_outputs[:i], state.true_outputs[i+1:]))
                    state.results.append(state.batch_results.pop(i))

            state.elapsed_time += time.time() - start_time

    return state.results, round(state.elapsed_time, 3)


def pez_search(cfg, model, device="cuda"):
    # Get the targets used for all experiments based on dataset
    if cfg.target_strategy == "random":
        with open(f"/content/true_tokens_{cfg.num_targets}_{cfg.input_len}.pkl", 'rb') as file:
            loaded_true_tokens = pickle.load(file).to("cpu")
    else:
        loaded_true_tokens = load_dataset_tokens(cfg.target_strategy, cfg.input_len, cfg.num_targets, include_bos=False, random_sentence=True, random_start=False, model=model)

    # Get the initialisation based on strategy
    initialisation_embeds = embed_search_initialisation(
        cfg.input_len, cfg.num_targets, cfg.init_strategy, cfg.loaded_true_tokens, cfg.max_batch_size, cfg.max_chunk_size, model, device).to("cpu")

    # Initialise state variables
    state_path = f'/content/{cfg.save_folder}/checkpoint_{cfg.input_len}_{cfg.num_targets}_{cfg.max_epochs}.pt'
    if os.path.exists(state_path):
        print("LOADING STATE")
        state = torch.load(state_path, weights_only=False)
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

    while state.num_remain_items != 0 or len(state.batch_results) != 0:
        start_time = time.time()

        # Checkpoint current progress if hour has passed
        if state.elapsed_time - state.checkpoint_elapsed_time > (3600 * 3):
            print("\nSAVING STATE")
            state.checkpoint_elapsed_time = state.elapsed_time
            torch.save(state, state_path)

        # Print progress
        state.epoch += 1
        if state.epoch % 100 == 0:
            print(f"({state.num_success_items}/{cfg.num_targets})({cfg.num_targets-state.num_remain_items}/{cfg.num_targets}){state.epoch}", end=", ")

        with torch.no_grad():
            # Add new items to batch if have space and have more items to do
            if (cfg.max_batch_size - len(state.batch_results)) > 0 and state.num_remain_items != 0:
                num_new_items = min((cfg.max_batch_size - len(state.batch_results)), state.num_remain_items)
                state.num_remain_items -= num_new_items

                # Initialise new target and add to end (batched)
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

                    # Initialise new prediction and add to end, one optimiser per sequence
                    new_pred_embed = initialisation_embeds[state.loaded_i+i:state.loaded_i+i+1].to(device)
                    state.pred_embed = torch.cat((state.pred_embed, new_pred_embed))

                state.loaded_i += num_new_items

        with torch.no_grad():
            # Use nearest neighbour to get back closest tokens from continuous embeddings
            pred_embed_continuous = state.pred_embed
            pred_tokens = []
            for i in range(0, pred_embed_continuous.size(0), cfg.max_chunk_size):
                batch_chunk = pred_embed_continuous[i:i+cfg.max_chunk_size]
                # Cosine similarity
                if cfg.con_distance == "cos sim":
                    tokens_sim_chunk = F.cosine_similarity(
                        batch_chunk.unsqueeze(2),
                        model.embed.W_E.unsqueeze(0).unsqueeze(0),
                        dim=-1
                    )
                    pred_tokens_chunk = torch.argmax(tokens_sim_chunk, dim=-1)

                # Euclidean distance
                elif cfg.con_distance == "euc dist":
                    tokens_sim_chunk = torch.cdist(batch_chunk.squeeze(0), model.embed.W_E, p=2)
                    pred_tokens_chunk = torch.argmin(tokens_sim_chunk, dim=-1)
                    if pred_tokens_chunk.dim() == 1:
                        pred_tokens_chunk = pred_tokens_chunk.unsqueeze(0)

                pred_tokens.append(pred_tokens_chunk)
            pred_tokens = torch.cat(pred_tokens, dim=0)
        
            # Get closest possible discrete embeddings
            pred_embed_discrete = model.embed(pred_tokens)
            pred_embed_discrete = pred_embed_discrete + model.pos_embed(pred_embed_discrete[:,:,0].detach())
            pred_embed_discrete.requires_grad = True
            
        # Use gradients from discrete embeddings to update continuous embeddings
        pred_logits = model(pred_embed_discrete, start_at_layer=0)
        # loss = torch.nn.HuberLoss()(state.true_logits.detach(), pred_logits[:,-1,:])
        loss = torch.nn.MSELoss()(state.true_logits.detach(), pred_logits[:,-1,:])
        loss.backward()
        with torch.no_grad():
            state.pred_embed = state.pred_embed - (cfg.learn_rate * pred_embed_discrete.grad)

            # Update history of tokens over epochs
            for i in range(len(state.batch_results)-1,-1,-1):
                # state.batch_results[i]["done_epochs"] += 1
                state.batch_results[i]["done_epochs"] += cfg.check_con_epochs

                # Remove item if have found a solution or reached final epoch
                have_inverted = torch.allclose(state.true_logits[i], pred_logits[i], atol=1e-4, rtol=1e-4)
                if have_inverted:
                    state.batch_results[i]["found_solution"] = True
                    state.num_success_items += 1
                if have_inverted or (cfg.max_epochs is not None and state.batch_results[i]["done_epochs"] >= cfg.max_epochs):
                    state.batch_results[i]["pred_tokens"] = pred_tokens[i].to("cpu")
                    del state.pred_embed[i]
                    state.true_logits = torch.cat((state.true_logits[:i], state.true_logits[i+1:]))
                    state.results.append(state.batch_results.pop(i))

            state.elapsed_time += time.time() - start_time

    return state.results, round(state.elapsed_time, 3)