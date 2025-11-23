import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
import os
import time
from datasets import load_dataset

from inversion_optimisation.utils import load_dataset_tokens, DotDict, DATA_PATH
from inversion_optimisation.algorithms.optimisers import CustomAdam


def embed_search_initialisation(input_len, num_targets, init_strategy, loaded_true_tokens, max_batch_size, max_chunk_size, model, device):
    with torch.no_grad():
        # Loaded
        if init_strategy == "loaded":
            with open(DATA_PATH / f"initial_tokens_{num_targets}_{input_len}.pkl", 'rb') as file:
                initialisation_tokens = pickle.load(file).to(device)
            return model.embed(initialisation_tokens)

        # Normal
        elif "normal" in init_strategy:
            normal_embed = torch.empty((num_targets, input_len, model.cfg.d_model))
            _ = nn.init.normal_(normal_embed, std=model.cfg.initializer_range)
            if init_strategy == "normal":
                return normal_embed

        # Zeros
        elif init_strategy == "zeros":
            return torch.zeros((num_targets, input_len, model.cfg.d_model))

        else:
            raise ValueError("Invalid initialisation strategy")
        

def embed_text_search(cfg, model, device="cuda"):
    # Get the targets used for all experiments based on dataset
    if cfg.target_strategy == "random":
        with open(DATA_PATH / f"true_tokens_{cfg.num_targets}_{cfg.input_len}.pkl", 'rb') as file:
            loaded_true_tokens = pickle.load(file).to("cpu")
    else:
        loaded_true_tokens = load_dataset_tokens(cfg.target_strategy, cfg.input_len, cfg.num_targets, include_bos=False, random_sentence=True, random_start=False, model=model)

    # Get the targets output
    true_tokens_loc = DATA_PATH / f"true_tokens_{cfg.num_targets}_{cfg.input_len}_{cfg.output_len}"
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
    else:
        print("INITIALISING STATE")
        state = DotDict({
            "results" : [],
            "batch_results" : [],
            "true_outputs" : torch.tensor([]).to(device).to(torch.int64),
            "optimizers" : [],
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
                    for j in range(cfg.input_len):
                        new_pred_embed_pos = new_pred_embed[:,j:j+1]
                        new_pred_embed_pos.requires_grad = True
                        if j == 0:
                            state.optimizers.append(torch.optim.Adam([new_pred_embed_pos], lr=cfg.learn_rate, betas=cfg.betas))
                            # if cfg.bias_correction:
                            #     state.optimizers.append(torch.optim.Adam([new_pred_embed_pos], lr=cfg.learn_rate, betas=cfg.betas))
                            # else:
                            #     state.optimizers.append(CustomAdam([new_pred_embed_pos], lr=cfg.learn_rate, betas=cfg.betas))
                        else:
                            state.optimizers[-1].param_groups[0]['params'].append(new_pred_embed_pos)

                state.loaded_i += num_new_items

        # Do one epoch of optimisation on batch
        for optimizer in state.optimizers:
            optimizer.zero_grad()
        pred_embed = torch.cat([torch.cat([param for param in optimizer.param_groups[0]['params']], dim=1)
                                    for optimizer in state.optimizers], dim=0).to(device)
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

        loss.backward()
        for optimizer in state.optimizers:
            optimizer.step()

        with torch.no_grad():
            # # Add decay to embeddings
            # for i in range(len(state.optimizers)):
            #     for j in range(len(state.optimizers[i].param_groups[0]['params'])):
            #         state.optimizers[i].param_groups[0]['params'][j].mul_(cfg.decay_rate)

            # # Intervene if sequence not found yet
            # for i in range(len(state.batch_results)):
            #     targets_epoch = (state.batch_results[i]["done_epochs"]+1)
            #     # Reset optimiser state
            #     if targets_epoch % cfg.reset_epoch == 0:
            #         for j in range(cfg.input_len):
            #             del state.optimizers[i].state[state.optimizers[i].param_groups[0]['params'][j]]

            #     # Reinitialise sequence
            #     if targets_epoch % cfg.reinit_epoch == 0:
            #         for j in range(cfg.input_len):
            #             state.optimizers[i].param_groups[0]['params'][j].normal_(std=0.1)

            # Only check for stopping on some epochs
            if state.epoch % cfg.check_con_epochs != 0:
                state.elapsed_time += time.time() - start_time
                continue

            # Use nearest neighbour to get back tokens from embeddings
            pred_tokens = []
            for i in range(0, pred_embed.size(0), cfg.max_chunk_size):
                batch_chunk = pred_embed[i:i+cfg.max_chunk_size]
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
            
            # Update history of tokens over epochs
            pred_tokens_full = torch.cat((pred_tokens, state.true_outputs[:,:-1]), dim=1)
            disc_pred_logits = model(pred_tokens_full)[:,cfg.input_len-1:,:]
            for i in range(len(state.batch_results)-1,-1,-1):
                # state.batch_results[i]["done_epochs"] += 1
                state.batch_results[i]["done_epochs"] += cfg.check_con_epochs

                # Remove item if have found a solution or reached final epoch
                have_inverted = torch.equal(state.batch_results[i]["true_outputs"].detach(), disc_pred_logits[i].argmax(dim=-1).to("cpu").detach())
                if have_inverted:
                    state.batch_results[i]["found_solution"] = True
                    state.num_success_items += 1
                if have_inverted or (cfg.max_epochs is not None and state.batch_results[i]["done_epochs"] >= cfg.max_epochs):
                    state.batch_results[i]["pred_tokens"] = pred_tokens[i].to("cpu")
                    del state.optimizers[i]
                    state.true_outputs = torch.cat((state.true_outputs[:i], state.true_outputs[i+1:]))
                    state.results.append(state.batch_results.pop(i))

            state.elapsed_time += time.time() - start_time

    return state.results, round(state.elapsed_time, 3)


def embed_search(cfg, model, device="cuda"):
    # Get the targets used for all experiments based on dataset
    if cfg.target_strategy == "random":
        with open(DATA_PATH / f"true_tokens_{cfg.num_targets}_{cfg.input_len}.pkl", 'rb') as file:
            loaded_true_tokens = pickle.load(file).to("cpu")
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
    else:
        print("INITIALISING STATE")
        state = DotDict({
            "results" : [],
            "batch_results" : [],
            "true_logits" : torch.Tensor([]).to(device),
            "optimizers" : [],
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
                    for j in range(cfg.input_len):
                        new_pred_embed_pos = new_pred_embed[:,j:j+1]
                        new_pred_embed_pos.requires_grad = True
                        if j == 0:
                            state.optimizers.append(torch.optim.Adam([new_pred_embed_pos], lr=cfg.learn_rate, betas=cfg.betas))
                            # if cfg.bias_correction:
                            #     state.optimizers.append(torch.optim.Adam([new_pred_embed_pos], lr=cfg.learn_rate, betas=cfg.betas))
                            # else:
                            #     state.optimizers.append(CustomAdam([new_pred_embed_pos], lr=cfg.learn_rate, betas=cfg.betas))
                        else:
                            state.optimizers[-1].param_groups[0]['params'].append(new_pred_embed_pos)

                state.loaded_i += num_new_items

        # Do one epoch of optimisation on batch
        for optimizer in state.optimizers:
            optimizer.zero_grad()
        pred_embed = torch.cat([torch.cat([param for param in optimizer.param_groups[0]['params']], dim=1)
                                    for optimizer in state.optimizers], dim=0).to(device)
        pred_embed_full = pred_embed + model.pos_embed(pred_embed[:,:,0].detach())
        pred_logits = model(pred_embed_full, start_at_layer=0)
        # loss = torch.nn.HuberLoss()(state.true_logits.detach(), pred_logits[:,-1,:])
        loss = torch.nn.MSELoss()(state.true_logits.detach(), pred_logits[:,-1,:])

        loss.backward()
        for optimizer in state.optimizers:
            optimizer.step()

        with torch.no_grad():
            # # Add decay to embeddings
            # for i in range(len(state.optimizers)):
            #     for j in range(len(state.optimizers[i].param_groups[0]['params'])):
            #         state.optimizers[i].param_groups[0]['params'][j].mul_(cfg.decay_rate)

            # # Intervene if sequence not found yet
            # for i in range(len(state.batch_results)):
            #     targets_epoch = (state.batch_results[i]["done_epochs"]+1)
            #     # Reset optimiser state
            #     if targets_epoch % cfg.reset_epoch == 0:
            #         for j in range(cfg.input_len):
            #             del state.optimizers[i].state[state.optimizers[i].param_groups[0]['params'][j]]

            #     # Reinitialise sequence
            #     if targets_epoch % cfg.reinit_epoch == 0:
            #         for j in range(cfg.input_len):
            #             state.optimizers[i].param_groups[0]['params'][j].normal_(std=0.1)

            # Only check for stopping on some epochs
            if state.epoch % cfg.check_con_epochs != 0:
                state.elapsed_time += time.time() - start_time
                continue

            # Use nearest neighbour to get back tokens from embeddings
            pred_tokens = []
            for i in range(0, pred_embed.size(0), cfg.max_chunk_size):
                batch_chunk = pred_embed[i:i+cfg.max_chunk_size]
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

            # Update history of tokens over epochs
            disc_pred_logits = model(pred_tokens)[:,-1,:]
            for i in range(len(state.batch_results)-1,-1,-1):
                # state.batch_results[i]["done_epochs"] += 1
                state.batch_results[i]["done_epochs"] += cfg.check_con_epochs

                # Remove item if have found a solution or reached final epoch
                have_inverted = torch.allclose(state.true_logits[i], disc_pred_logits[i], atol=1e-4, rtol=1e-4)
                if have_inverted:
                    state.batch_results[i]["found_solution"] = True
                    state.num_success_items += 1
                if have_inverted or (cfg.max_epochs is not None and state.batch_results[i]["done_epochs"] >= cfg.max_epochs):
                    state.batch_results[i]["pred_tokens"] = pred_tokens[i].to("cpu")
                    del state.optimizers[i]
                    state.true_logits = torch.cat((state.true_logits[:i], state.true_logits[i+1:]))
                    state.results.append(state.batch_results.pop(i))

            state.elapsed_time += time.time() - start_time

    return state.results, round(state.elapsed_time, 3)