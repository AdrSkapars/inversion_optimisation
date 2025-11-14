import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
import os
import time
from datasets import load_dataset

from inversion_optimisation.utils import load_dataset_tokens, DotDict
from inversion_optimisation.algorithms.optimisers import CustomAdam


def onehot_text_search(cfg, model, device="cuda"):
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
    if cfg.init_strategy == "loaded":
        with open(f"/content/initial_tokens_{cfg.num_targets}_{cfg.input_len}.pkl", 'rb') as file:
            initialisation_tokens = pickle.load(file).to(device)
        initialisation_embeds = F.one_hot(initialisation_tokens, num_classes=model.embed.W_E.size(0)).to(model.embed.W_E.dtype).to("cpu")
    elif cfg.init_strategy == "normal":
        normal_embed = torch.empty((cfg.num_targets, cfg.input_len, model.embed.W_E.size(0)))
        _ = nn.init.normal_(normal_embed, std=0.05)
        initialisation_embeds = normal_embed.to("cpu")
    elif cfg.init_strategy == "zeros":
        initialisation_embeds = torch.zeros((cfg.num_targets, cfg.input_len, model.embed.W_E.size(0))).to("cpu")

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

                    # Initialise new prediction and add to end, one optimiser per sequence
                    new_pred_embed = initialisation_embeds[state.loaded_i+i:state.loaded_i+i+1].to(device)
                    for j in range(cfg.input_len):
                        new_pred_embed_pos = new_pred_embed[:,j:j+1]
                        new_pred_embed_pos.requires_grad = True
                        if j == 0:
                            if cfg.bias_correction:
                                state.optimizers.append(torch.optim.Adam([new_pred_embed_pos], lr=cfg.learn_rate, betas=cfg.betas))
                            else:
                                state.optimizers.append(CustomAdam([new_pred_embed_pos], lr=cfg.learn_rate, betas=cfg.betas))
                        else:
                            state.optimizers[-1].param_groups[0]['params'].append(new_pred_embed_pos)

                state.loaded_i += num_new_items

        # Do one epoch of optimisation on batch
        for optimizer in state.optimizers:
            optimizer.zero_grad()
        pred_embed_pre = torch.cat([torch.cat([param for param in optimizer.param_groups[0]['params']], dim=1)
                                    for optimizer in state.optimizers], dim=0).to(device)
        pred_one_hot = torch.softmax(pred_embed_pre / cfg.temp, dim=-1)
        pred_embed = (pred_one_hot @ model.embed.W_E)
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

        if cfg.reg_weight is not None:
            # Compute regularisation penalty
            if state.epoch >= 0:
                # # Size of input penalty
                # reg_penalty = (pred_one_hot).pow(2).sum(dim=-1).sqrt() * -1
                # reg_penalty = reg_penalty.mean(dim=-1).mean(dim=-1)

                # Fluency penalty
                reg_penalty = pred_logits[:,:-1,:].softmax(dim=-1).log().gather(2, pred_one_hot[:,1:,:].argmax(dim=-1).unsqueeze(-1)).squeeze(-1) * -1
                reg_penalty = reg_penalty.mean(dim=-1).mean(dim=-1)

                loss = loss + (cfg.reg_weight * reg_penalty)

        loss.backward()
        for optimizer in state.optimizers:
            optimizer.step()

        with torch.no_grad():
            # Add decay to embeddings
            for i in range(len(state.optimizers)):
                for j in range(len(state.optimizers[i].param_groups[0]['params'])):
                    state.optimizers[i].param_groups[0]['params'][j].mul_(cfg.decay_rate)

            # Intervene if sequence not found yet
            for i in range(len(state.batch_results)):
                targets_epoch = (state.batch_results[i]["done_epochs"]+1)
                # Reset optimiser state
                if targets_epoch % cfg.reset_epoch == 0:
                    for j in range(cfg.input_len):
                        del state.optimizers[i].state[state.optimizers[i].param_groups[0]['params'][j]]

                # Reinitialise sequence
                if targets_epoch % cfg.reinit_epoch == 0:
                    for j in range(cfg.input_len):
                        state.optimizers[i].param_groups[0]['params'][j].normal_(std=0.1)

            # Update history of tokens over epochs
            pred_tokens = torch.argmax(pred_one_hot, dim=-1)
            pred_tokens_full = torch.cat((pred_tokens, state.true_outputs[:,:-1]), dim=1)
            disc_pred_logits = model(pred_tokens_full)[:,cfg.input_len-1:,:]

            for i in range(len(state.batch_results)-1,-1,-1):
                state.batch_results[i]["done_epochs"] += 1

                # Remove item if have found a solution or reached final epoch
                have_inverted = torch.equal(state.batch_results[i]["true_outputs"].detach(), disc_pred_logits[i].argmax(dim=-1).to("cpu").detach())
                if have_inverted:
                    state.batch_results[i]["found_solution"] = True
                    state.batch_results[i]["test"] = disc_pred_logits[i].argmax(dim=-1).to("cpu").detach()
                    state.num_success_items += 1
                if have_inverted or (cfg.max_epochs is not None and state.batch_results[i]["done_epochs"] >= cfg.max_epochs):
                    state.batch_results[i]["pred_tokens"] = pred_tokens[i].to("cpu")
                    del state.optimizers[i]
                    state.true_outputs = torch.cat((state.true_outputs[:i], state.true_outputs[i+1:]))
                    state.results.append(state.batch_results.pop(i))

            state.elapsed_time += time.time() - start_time

    return state.results, round(state.elapsed_time, 3)


def onehot_search(cfg, model, device="cuda"):
    # Get the targets used for all experiments based on dataset
    if cfg.target_strategy == "random":
        with open(f"/content/true_tokens_{cfg.num_targets}_{cfg.input_len}.pkl", 'rb') as file:
            loaded_true_tokens = pickle.load(file).to("cpu")
    elif cfg.target_strategy == "privacy":
        # Privacy dataset only allows num_targets == 500 currently
        privacy_ds = load_dataset("AdrSkapars/pii-inversion-test-5k", split=f"length_{cfg.input_len}")
        loaded_true_tokens = torch.cat([torch.tensor(item["tokens"]).to(torch.int64).unsqueeze(0) for item in privacy_ds], dim=0).to("cpu")
    else:
        loaded_true_tokens = load_dataset_tokens(cfg.target_strategy, cfg.input_len, cfg.num_targets, include_bos=False, random_sentence=True, random_start=False, model=model)

    # Get the initialisation based on strategy
    if cfg.init_strategy == "loaded":
        with open(f"/content/initial_tokens_{cfg.num_targets}_{cfg.input_len}.pkl", 'rb') as file:
            initialisation_tokens = pickle.load(file).to(device)
        initialisation_embeds = F.one_hot(initialisation_tokens, num_classes=model.embed.W_E.size(0)).to(model.embed.W_E.dtype).to("cpu")
    elif cfg.init_strategy == "normal":
        normal_embed = torch.empty((cfg.num_targets, cfg.input_len, model.embed.W_E.size(0)))
        _ = nn.init.normal_(normal_embed, std=0.05)
        initialisation_embeds = normal_embed.to("cpu")
    elif cfg.init_strategy == "zeros":
        initialisation_embeds = torch.zeros((cfg.num_targets, cfg.input_len, model.embed.W_E.size(0))).to("cpu")

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
        # if state.elapsed_time - state.checkpoint_elapsed_time > (3600 * 3):
        if state.elapsed_time - state.checkpoint_elapsed_time > (3600 * 6):
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
                            if cfg.bias_correction:
                                state.optimizers.append(torch.optim.Adam([new_pred_embed_pos], lr=cfg.learn_rate, betas=cfg.betas))
                            else:
                                state.optimizers.append(CustomAdam([new_pred_embed_pos], lr=cfg.learn_rate, betas=cfg.betas))
                        else:
                            state.optimizers[-1].param_groups[0]['params'].append(new_pred_embed_pos)

                state.loaded_i += num_new_items

        # Do one epoch of optimisation on batch
        for optimizer in state.optimizers:
            optimizer.zero_grad()
        pred_embed_pre = torch.cat([torch.cat([param for param in optimizer.param_groups[0]['params']], dim=1)
                                    for optimizer in state.optimizers], dim=0).to(device)
        pred_one_hot = torch.softmax(pred_embed_pre / cfg.temp, dim=-1)
        pred_embed = (pred_one_hot @ model.embed.W_E)
        if "gpt" not in cfg.model_name and "tiny" not in cfg.model_name:
            pred_embed_full = pred_embed
        else:
            pred_embed_full = pred_embed + model.pos_embed(pred_embed[:,:,0].detach())
        pred_logits = model(pred_embed_full, start_at_layer=0)
        loss = torch.nn.HuberLoss()(state.true_logits.detach(), pred_logits[:,-1,:])

        if cfg.reg_weight is not None:
            # Compute regularisation penalty
            if state.epoch >= 0:
                # # Size of input penalty
                # reg_penalty = (pred_one_hot).pow(2).sum(dim=-1).sqrt() * -1
                # reg_penalty = reg_penalty.mean(dim=-1).mean(dim=-1)

                # Fluency penalty
                reg_penalty = pred_logits[:,:-1,:].softmax(dim=-1).log().gather(2, pred_one_hot[:,1:,:].argmax(dim=-1).unsqueeze(-1)).squeeze(-1) * -1
                reg_penalty = reg_penalty.mean(dim=-1).mean(dim=-1)

                loss = loss + (cfg.reg_weight * reg_penalty)

        loss.backward()
        for optimizer in state.optimizers:
            optimizer.step()

        with torch.no_grad():
            # Add decay to embeddings
            for i in range(len(state.optimizers)):
                for j in range(len(state.optimizers[i].param_groups[0]['params'])):
                    state.optimizers[i].param_groups[0]['params'][j].mul_(cfg.decay_rate)

            # Intervene if sequence not found yet
            for i in range(len(state.batch_results)):
                targets_epoch = (state.batch_results[i]["done_epochs"]+1)
                # Reset optimiser state
                if targets_epoch % cfg.reset_epoch == 0:
                    for j in range(cfg.input_len):
                        del state.optimizers[i].state[state.optimizers[i].param_groups[0]['params'][j]]

                # Reinitialise sequence
                if targets_epoch % cfg.reinit_epoch == 0:
                    for j in range(cfg.input_len):
                        state.optimizers[i].param_groups[0]['params'][j].normal_(std=0.1)

            # Assume largest one-hot token is the true one
            pred_tokens = torch.argmax(pred_one_hot, dim=-1)

            # Update history of tokens over epochs
            disc_pred_logits = model(pred_tokens)[:,-1,:]
            for i in range(len(state.batch_results)-1,-1,-1):
                state.batch_results[i]["done_epochs"] += 1

                # Remove item if have found a solution or reached final epoch
                threshold = 1e-4 if "tiny" in cfg.model_name else 1e-3
                have_inverted = torch.allclose(state.true_logits[i], disc_pred_logits[i], atol=threshold, rtol=threshold)
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