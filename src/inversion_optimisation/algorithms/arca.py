import torch
import torch.nn.functional as F
import pickle
import os
import time
from datasets import load_dataset

from inversion_optimisation.utils import load_dataset_tokens, DotDict


def arca_text_search(cfg, model, device="cuda"):
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
            loaded_initial_tokens = pickle.load(file)
    elif cfg.init_strategy == "zeros":
        loaded_initial_tokens = torch.zeros((cfg.num_targets, cfg.input_len)).to("cpu")

    # Initialise state variables
    state_path = f'/content/{cfg.save_folder}/checkpoint_{cfg.input_len}_{cfg.num_targets}_{cfg.adjusted_max_epochs}.pt'
    if os.path.exists(state_path):
        print("LOADING STATE")
        state = torch.load(state_path, weights_only=False)
    else:
        print("INITIALISING STATE")
        state = DotDict({
            "results" : [],
            "batch_results" : [],
            "true_outputs" : torch.tensor([]).to(device).to(torch.int64),
            "pred_tokens" : torch.tensor([]).to(device).to(torch.int64),
            "current_positions" : torch.tensor([]).to(device).to(torch.int64),  # Track current position for each batch item
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

        state.epoch +=1
        if state.epoch % 50 == 0:
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

                # Initialise new results tracking and add to end
                for i in range(num_new_items):
                    state.batch_results.append({
                        "true_tokens": true_tokens[i].to("cpu"),
                        "true_outputs": state.true_outputs[i].to("cpu"),
                        "pred_tokens": None,
                        "found_solution": False,
                        "done_epochs": 0,
                    })

                # Initialise new prediction and add to end
                if cfg.init_strategy == "loaded":
                    append_pred_tokens = loaded_initial_tokens[state.loaded_i:state.loaded_i+num_new_items].to(device)
                elif cfg.init_strategy == "zeros":
                    append_pred_tokens = torch.tensor([[0]]).repeat(num_new_items, cfg.input_len).to(device)
                state.pred_tokens = torch.cat((state.pred_tokens, append_pred_tokens))

                # Initialise position tracking (start at position 0 for each new item)
                new_positions = torch.zeros(num_new_items, dtype=torch.int64).to(device)
                state.current_positions = torch.cat((state.current_positions, new_positions))

                state.loaded_i += num_new_items

        # ARCA: Get averaged gradients by sampling P random substitutions at current position
        P = cfg.num_grad_samples

        accumulated_grad = torch.zeros((len(state.batch_results), cfg.input_len, len(model.tokenizer.vocab)),
                                       dtype=model.embed.W_E.dtype, device=device)

        for sample_idx in range(P):
            # Create sequences with random token at current position
            sampled_pred_tokens = state.pred_tokens.clone()
            random_tokens = torch.randint(0, len(model.tokenizer.vocab),
                                        (len(state.batch_results), 1), device=device)
            sampled_pred_tokens.scatter_(1, state.current_positions.unsqueeze(1), random_tokens)

            # Get one hot encoding and multiply by embedding
            pred_one_hot = F.one_hot(sampled_pred_tokens, num_classes=len(model.tokenizer.vocab))
            pred_one_hot = pred_one_hot.to(model.embed.W_E.dtype).to(device).requires_grad_()
            pred_embed = (pred_one_hot @ model.embed.W_E)
        
            pred_one_hot = F.one_hot(sampled_pred_tokens, num_classes=len(model.tokenizer.vocab))
            pred_one_hot = pred_one_hot.to(model.embed.W_E.dtype).to(device).requires_grad_()
            pred_embed = (pred_one_hot @ model.embed.W_E)
            pred_embed_full = torch.cat((pred_embed, model.embed(state.true_outputs[:,:-1])), dim=1)
            pred_embed_full = pred_embed_full + model.pos_embed(pred_embed_full[:,:,0].detach())
        
            # Get loss gradient of one hot encoding
            pred_logits = model(pred_embed_full, start_at_layer=0)
            pred_logprobs = F.softmax(pred_logits[:,cfg.input_len-1:,:], dim=-1).clamp(min=1e-12).log()
            pred_logits_target = torch.gather(pred_logprobs, 2, state.true_outputs.unsqueeze(-1)).squeeze(-1)
            pred_logits_diff = (pred_logits_target - pred_logprobs.max(dim=-1).values)
            loss = - pred_logits_diff.mean()
            loss.backward()

            # Accumulate gradients
            accumulated_grad += pred_one_hot.grad.clone()

        # Average the gradients
        grad = accumulated_grad / P
        grad = -grad

        with torch.no_grad():
            # # Evaluate current sequences for convergence checking
            # pred_one_hot_eval = F.one_hot(state.pred_tokens, num_classes=len(model.tokenizer.vocab))
            # pred_one_hot_eval = pred_one_hot_eval.to(model.embed.W_E.dtype).to(device)
            # pred_embed_eval = (pred_one_hot_eval @ model.embed.W_E)
            # pred_embed_full_eval = pred_embed_eval + model.pos_embed(pred_embed_eval[:,:,0].detach())
            # pred_logits = model(pred_embed_full_eval, start_at_layer=0)[:,-1,:]

            # Update history of tokens and losses over epochs
            for i in range(len(state.batch_results)-1,-1,-1):
                state.batch_results[i]["done_epochs"] += 1

                # Remove item if have found a solution or reached final epoch
                have_inverted = torch.equal(state.batch_results[i]["true_outputs"].detach(), pred_logits[i].argmax(dim=-1).to("cpu").detach())
                if have_inverted:
                    state.batch_results[i]["found_solution"] = True
                    state.num_success_items += 1
                if have_inverted or (cfg.adjusted_max_epochs is not None and state.batch_results[i]["done_epochs"] >= cfg.adjusted_max_epochs):
                    state.batch_results[i]["pred_tokens"] = state.pred_tokens[i].to("cpu")
                    state.pred_tokens = torch.cat((state.pred_tokens[:i], state.pred_tokens[i+1:]))
                    state.true_outputs = torch.cat((state.true_outputs[:i], state.true_outputs[i+1:]))
                    state.current_positions = torch.cat((state.current_positions[:i], state.current_positions[i+1:]))
                    state.results.append(state.batch_results.pop(i))
                    grad = torch.cat((grad[:i], grad[i+1:]))

            if len(state.batch_results) == 0:
                state.elapsed_time += time.time() - start_time
                if state.num_remain_items == 0:
                    break
                else:
                    continue

            # Sequential optimization: optimize one position at a time for each batch item
            best_pred_tokens = [None for _ in range(len(state.batch_results))]
            best_losses = [None for _ in range(len(state.batch_results))]

            # Get the current position for each batch item
            new_token_pos = state.current_positions.unsqueeze(1)  # Shape: (batch_size, 1)

            for _ in range(cfg.num_candidates):
                # Get top k negative grad tokens at the current position for each batch item
                if cfg.top_k != 50257:
                    # Extract gradients only at current positions
                    batch_arrange = torch.arange(len(state.batch_results)).unsqueeze(-1)
                    grad_at_pos = grad[batch_arrange, new_token_pos]  # Shape: (batch_size, 1, vocab_size)
                    topk_grad_values, topk_grad_indices = grad_at_pos.topk(cfg.top_k, dim=-1)
                else:
                    batch_arrange = torch.arange(len(state.batch_results)).unsqueeze(-1)
                    topk_grad_values = grad[batch_arrange, new_token_pos]
                    topk_grad_indices = torch.arange(grad.size(2), device=grad.device).unsqueeze(0).unsqueeze(0).repeat(grad.size(0), 1, 1)

                # Sample token indices for each batch item
                match cfg.token_choice:
                    case "uniform":
                        chosen_grad_indices = torch.randint(0, cfg.top_k, (len(state.batch_results), 1), device=grad.device)
                        new_token_val = topk_grad_indices[torch.arange(len(state.batch_results)), 0, chosen_grad_indices[:, 0]].unsqueeze(1)
                    case "weighted":
                        # Reshape for multinomial sampling
                        topk_grad_values_flat = topk_grad_values.squeeze(1)  # Shape: (batch_size, top_k)
                        index_weights = (topk_grad_values_flat + 0.000001) - topk_grad_values_flat.min(dim=-1, keepdim=True).values
                        chosen_grad_indices = torch.multinomial(index_weights, num_samples=1, replacement=True)  # Shape: (batch_size, 1)
                        new_token_val = topk_grad_indices[torch.arange(len(state.batch_results)), 0, chosen_grad_indices[:, 0]].unsqueeze(1)

                # Replace the token at the current position
                new_pred_tokens = state.pred_tokens.clone()
                new_pred_tokens = new_pred_tokens.scatter(1, new_token_pos, new_token_val)

                # Compute loss on these candidates and take the argmin
                with torch.no_grad():
                    new_pred_tokens_full = torch.cat((new_pred_tokens, state.true_outputs[:,:-1]), dim=1)
                    new_pred_logits = model(new_pred_tokens_full)
                    new_pred_logprobs = F.softmax(new_pred_logits[:,cfg.input_len-1:,:], dim=-1).clamp(min=1e-12).log()
                    new_pred_logits_target = torch.gather(new_pred_logprobs, 2, state.true_outputs.unsqueeze(-1)).squeeze(-1)
                    new_pred_logits_diff = (new_pred_logits_target - new_pred_logprobs.max(dim=-1).values)
                    new_loss = - new_pred_logits_diff.mean(dim=-1)
                    for i in range(len(state.batch_results)):
                        if best_losses[i] is None or new_loss[i] < best_losses[i]:
                            best_pred_tokens[i] = new_pred_tokens[i].unsqueeze(0).clone()
                            best_losses[i] = new_loss[i].clone()

            # Keep the best token changes
            state.pred_tokens = torch.cat(best_pred_tokens, dim=0)

            # Advance to next position (cycle back to 0 after reaching input_len)
            state.current_positions = (state.current_positions + 1) % cfg.input_len

        state.elapsed_time += time.time() - start_time

    return state.results, round(state.elapsed_time, 3)


def arca_search(cfg, model, device="cuda"):
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
            loaded_initial_tokens = pickle.load(file)
    elif cfg.init_strategy == "zeros":
        loaded_initial_tokens = torch.zeros((cfg.num_targets, cfg.input_len)).to("cpu")

    # Initialise state variables
    state_path = f'/content/{cfg.save_folder}/checkpoint_{cfg.input_len}_{cfg.num_targets}_{cfg.adjusted_max_epochs}.pt'
    if os.path.exists(state_path):
        print("LOADING STATE")
        state = torch.load(state_path, weights_only=False)
    else:
        print("INITIALISING STATE")
        state = DotDict({
            "results" : [],
            "batch_results" : [],
            "true_logits" : torch.Tensor([]).to(device),
            "pred_tokens" : torch.tensor([]).to(device).to(torch.int64),
            "current_positions" : torch.tensor([]).to(device).to(torch.int64),  # Track current position for each batch item
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

        state.epoch +=1
        if state.epoch % 50 == 0:
            print(f"({state.num_success_items}/{cfg.num_targets})({cfg.num_targets-state.num_remain_items}/{cfg.num_targets}){state.epoch}", end=", ")

        with torch.no_grad():
            # Add new items to batch if have space and have more items to do
            if (cfg.max_batch_size - len(state.batch_results)) > 0 and state.num_remain_items != 0:
                num_new_items = min((cfg.max_batch_size - len(state.batch_results)), state.num_remain_items)
                state.num_remain_items -= num_new_items

                # Initialise new target and add to end
                true_tokens = loaded_true_tokens[state.loaded_i:state.loaded_i+num_new_items].to(device)
                new_true_logits = model(true_tokens).detach()[:,-1,:]
                state.true_logits = torch.cat((state.true_logits, new_true_logits))

                # Initialise new results tracking and add to end
                for i in range(num_new_items):
                    state.batch_results.append({
                        "true_tokens": true_tokens[i].to("cpu"),
                        "pred_tokens": None,
                        "found_solution": False,
                        "done_epochs": 0,
                    })

                # Initialise new prediction and add to end
                if cfg.init_strategy == "loaded":
                    append_pred_tokens = loaded_initial_tokens[state.loaded_i:state.loaded_i+num_new_items].to(device)
                elif cfg.init_strategy == "zeros":
                    append_pred_tokens = torch.tensor([[0]]).repeat(num_new_items, cfg.input_len).to(device)
                state.pred_tokens = torch.cat((state.pred_tokens, append_pred_tokens))

                # Initialise position tracking (start at position 0 for each new item)
                new_positions = torch.zeros(num_new_items, dtype=torch.int64).to(device)
                state.current_positions = torch.cat((state.current_positions, new_positions))

                state.loaded_i += num_new_items

        # ARCA: Get averaged gradients by sampling P random substitutions at current position
        P = cfg.num_grad_samples

        accumulated_grad = torch.zeros((len(state.batch_results), cfg.input_len, len(model.tokenizer.vocab)),
                                       dtype=model.embed.W_E.dtype, device=device)

        for sample_idx in range(P):
            # Create sequences with random token at current position
            sampled_pred_tokens = state.pred_tokens.clone()
            random_tokens = torch.randint(0, len(model.tokenizer.vocab),
                                        (len(state.batch_results), 1), device=device)
            sampled_pred_tokens.scatter_(1, state.current_positions.unsqueeze(1), random_tokens)

            # Get one hot encoding and multiply by embedding
            pred_one_hot = F.one_hot(sampled_pred_tokens, num_classes=len(model.tokenizer.vocab))
            pred_one_hot = pred_one_hot.to(model.embed.W_E.dtype).to(device).requires_grad_()
            pred_embed = (pred_one_hot @ model.embed.W_E)

            # Get loss gradient of one hot encoding
            pred_embed_full = pred_embed + model.pos_embed(pred_embed[:,:,0].detach())
            pred_logits = model(pred_embed_full, start_at_layer=0)[:,-1,:]
            loss = torch.nn.MSELoss()(state.true_logits.detach(), pred_logits)
            loss.backward()

            # Accumulate gradients
            accumulated_grad += pred_one_hot.grad.clone()

        # Average the gradients
        grad = accumulated_grad / P
        grad = -grad

        with torch.no_grad():
            # Evaluate current sequences for convergence checking
            pred_one_hot_eval = F.one_hot(state.pred_tokens, num_classes=len(model.tokenizer.vocab))
            pred_one_hot_eval = pred_one_hot_eval.to(model.embed.W_E.dtype).to(device)
            pred_embed_eval = (pred_one_hot_eval @ model.embed.W_E)
            pred_embed_full_eval = pred_embed_eval + model.pos_embed(pred_embed_eval[:,:,0].detach())
            pred_logits = model(pred_embed_full_eval, start_at_layer=0)[:,-1,:]

            # Update history of tokens and losses over epochs
            for i in range(len(state.batch_results)-1,-1,-1):
                state.batch_results[i]["done_epochs"] += 1

                # Remove item if have found a solution or reached final epoch
                threshold = 1e-4 if "tiny" in cfg.model_name else 1e-3
                have_inverted = torch.allclose(state.true_logits[i], pred_logits[i], atol=threshold, rtol=threshold)
                if have_inverted:
                    state.batch_results[i]["found_solution"] = True
                    state.num_success_items += 1
                if have_inverted or (cfg.adjusted_max_epochs is not None and state.batch_results[i]["done_epochs"] >= cfg.adjusted_max_epochs):
                    state.batch_results[i]["pred_tokens"] = state.pred_tokens[i].to("cpu")
                    state.pred_tokens = torch.cat((state.pred_tokens[:i], state.pred_tokens[i+1:]))
                    state.true_logits = torch.cat((state.true_logits[:i], state.true_logits[i+1:]))
                    state.current_positions = torch.cat((state.current_positions[:i], state.current_positions[i+1:]))
                    state.results.append(state.batch_results.pop(i))
                    grad = torch.cat((grad[:i], grad[i+1:]))

            if len(state.batch_results) == 0:
                state.elapsed_time += time.time() - start_time
                if state.num_remain_items == 0:
                    break
                else:
                    continue

            # Sequential optimization: optimize one position at a time for each batch item
            best_pred_tokens = [None for _ in range(len(state.batch_results))]
            best_losses = [None for _ in range(len(state.batch_results))]

            # Get the current position for each batch item
            new_token_pos = state.current_positions.unsqueeze(1)  # Shape: (batch_size, 1)

            for _ in range(cfg.num_candidates):
                # Get top k negative grad tokens at the current position for each batch item
                if cfg.top_k != 50257:
                    # Extract gradients only at current positions
                    batch_arrange = torch.arange(len(state.batch_results)).unsqueeze(-1)
                    grad_at_pos = grad[batch_arrange, new_token_pos]  # Shape: (batch_size, 1, vocab_size)
                    topk_grad_values, topk_grad_indices = grad_at_pos.topk(cfg.top_k, dim=-1)
                else:
                    batch_arrange = torch.arange(len(state.batch_results)).unsqueeze(-1)
                    topk_grad_values = grad[batch_arrange, new_token_pos]
                    topk_grad_indices = torch.arange(grad.size(2), device=grad.device).unsqueeze(0).unsqueeze(0).repeat(grad.size(0), 1, 1)

                # Sample token indices for each batch item
                match cfg.token_choice:
                    case "uniform":
                        chosen_grad_indices = torch.randint(0, cfg.top_k, (len(state.batch_results), 1), device=grad.device)
                        new_token_val = topk_grad_indices[torch.arange(len(state.batch_results)), 0, chosen_grad_indices[:, 0]].unsqueeze(1)
                    case "weighted":
                        # Reshape for multinomial sampling
                        topk_grad_values_flat = topk_grad_values.squeeze(1)  # Shape: (batch_size, top_k)
                        index_weights = (topk_grad_values_flat + 0.000001) - topk_grad_values_flat.min(dim=-1, keepdim=True).values
                        chosen_grad_indices = torch.multinomial(index_weights, num_samples=1, replacement=True)  # Shape: (batch_size, 1)
                        new_token_val = topk_grad_indices[torch.arange(len(state.batch_results)), 0, chosen_grad_indices[:, 0]].unsqueeze(1)

                # Replace the token at the current position
                new_pred_tokens = state.pred_tokens.clone()
                new_pred_tokens = new_pred_tokens.scatter(1, new_token_pos, new_token_val)

                # Compute loss on these candidates and take the argmin
                with torch.no_grad():
                    new_pred_logits = model(new_pred_tokens)[:,-1,:]
                    for i in range(len(state.batch_results)):
                        new_loss = torch.nn.MSELoss()(state.true_logits[i], new_pred_logits[i])
                        if best_losses[i] is None or new_loss < best_losses[i]:
                            best_pred_tokens[i] = new_pred_tokens[i].unsqueeze(0).clone()
                            best_losses[i] = new_loss.clone()

            # Keep the best token changes
            state.pred_tokens = torch.cat(best_pred_tokens, dim=0)

            # Advance to next position (cycle back to 0 after reaching input_len)
            state.current_positions = (state.current_positions + 1) % cfg.input_len

        state.elapsed_time += time.time() - start_time

    return state.results, round(state.elapsed_time, 3)