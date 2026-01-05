# Suggestions for Improving Discrete Optimization Search

## 1. Better Gradient Flow: Gumbel-Softmax

**Current Issue**: Standard softmax can have weak gradients, especially with low temperature.

**Solution**: Use Gumbel-Softmax which provides better gradients for discrete optimization:

```python
def gumbel_softmax(logits, temperature, hard=False):
    """
    Gumbel-Softmax for better gradient flow in discrete optimization.
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y_soft = torch.softmax((logits + gumbel_noise) / temperature, dim=-1)
    if hard:
        # Straight-through estimator: use hard sample in forward, soft in backward
        index = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(logits)
        y_hard.scatter_(-1, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

# Replace in onehot_search:
# pred_one_hot = torch.softmax(pred_embed_pre / cfg.temp, dim=-1)
pred_one_hot = gumbel_softmax(pred_embed_pre, cfg.temp, hard=False)
```

**Benefits**: Better gradient signal, especially useful when temperature is low.

---

## 2. Add Contrastive Loss

**Current Issue**: Only maximizing "Yes" probability, not explicitly minimizing "No".

**Solution**: Add contrastive term to loss:

```python
# Current:
split_loss = -1 * pred_logits[:, -1, judge_pos_tokens].sum(dim=-1)

# Improved:
pos_loss = -1 * pred_logits[:, -1, judge_pos_tokens].sum(dim=-1)
neg_loss = pred_logits[:, -1, judge_neg_tokens].sum(dim=-1)  # Minimize "No"
split_loss = pos_loss + cfg.contrastive_weight * neg_loss
```

**Benefits**: More stable optimization, clearer signal.

---

## 3. Add Fluency Regularization

**Current Issue**: Generated sequences are gibberish. Need to encourage natural text.

**Solution**: Add perplexity-based regularization (as seen in other experiments):

```python
# Add to loss computation:
if cfg.fluency_weight is not None:
    # Compute perplexity of the generated sequence
    # Use the model's predictions at each position
    pred_tokens_hard = pred_one_hot.argmax(dim=-1)
    
    # Get model's predictions for the sequence
    sequence_logits = model(judge_embed, start_at_layer=0)[:, :-1, :]  # All but last position
    sequence_probs = torch.softmax(sequence_logits, dim=-1)
    
    # Compute negative log likelihood of actual tokens
    token_indices = pred_tokens_hard[:, 1:].unsqueeze(-1)  # Shift by 1 for next-token prediction
    fluency_penalty = -sequence_probs.log().gather(2, token_indices).squeeze(-1).mean(dim=-1)
    
    loss = loss + cfg.fluency_weight * fluency_penalty.mean()
```

**Benefits**: Encourages more natural, fluent text while still optimizing for "Yes".

---

## 4. Adaptive Temperature Scheduling

**Current Issue**: Fixed temperature (0.05) may be too low early in training, too high later.

**Solution**: Use temperature annealing:

```python
# In config:
cfg.temp_start = 1.0
cfg.temp_end = 0.05
cfg.temp_decay = 0.995  # or linear: (temp_start - temp_end) / max_epochs

# In training loop:
current_temp = max(
    cfg.temp_end,
    cfg.temp_start * (cfg.temp_decay ** state.epoch)
)
# Or linear:
# current_temp = cfg.temp_start - (cfg.temp_start - cfg.temp_end) * (state.epoch / cfg.max_epochs)

pred_one_hot = torch.softmax(pred_embed_pre / current_temp, dim=-1)
```

**Benefits**: Start with exploration (high temp), end with exploitation (low temp).

---

## 5. Better Initialization Strategies

**Current Issue**: Random initialization may start far from good solutions.

**Solutions**:

### A. Start from known offensive text:
```python
if cfg.init_strategy == "offensive_seed":
    # Load a dataset of potentially offensive text
    offensive_strings = [
        "hate speech example",
        "offensive phrase example",
        # ... more examples
    ]
    # Tokenize and use as initialization
    initialisation_tokens = model.tokenizer(
        random.choice(offensive_strings), 
        return_tensors="pt", 
        add_special_tokens=False
    )["input_ids"].to(device)
    initialisation_embeds = F.one_hot(
        initialisation_tokens, 
        num_classes=model.embed.W_E.size(0)
    ).to(model.embed.W_E.dtype).to("cpu")
```

### B. Multi-start optimization:
```python
# Initialize multiple sequences from different starting points
# Keep best performing ones
if cfg.init_strategy == "multi_start":
    num_starts = 5
    # Initialize multiple, track best
    # Only keep top-k after first few epochs
```

### C. Gradient-based initialization:
```python
# Do a few forward passes with random text
# Use gradients to initialize better starting points
```

---

## 6. Improved Loss Function: Log Probability Instead of Sum

**Current Issue**: Summing probabilities may not be optimal.

**Solution**: Use log probabilities (more numerically stable):

```python
# Current:
split_loss = -1 * pred_logits[:, -1, judge_pos_tokens].sum(dim=-1)

# Improved:
log_probs = F.log_softmax(pred_logits[:, -1, :], dim=-1)
# Log-sum-exp trick for numerical stability
split_loss = -torch.logsumexp(log_probs[:, judge_pos_tokens], dim=-1)
```

**Benefits**: Better numerical stability, proper probability handling.

---

## 7. Diversity Regularization

**Current Issue**: All sequences may converge to similar solutions.

**Solution**: Add diversity loss to encourage different sequences:

```python
if cfg.diversity_weight is not None:
    # Compute pairwise distances between sequences
    # Encourage sequences to be different
    batch_embeds = pred_embed  # [batch_size, seq_len, embed_dim]
    # Compute cosine similarity between all pairs
    batch_embeds_flat = batch_embeds.mean(dim=1)  # Average over sequence
    similarity_matrix = F.cosine_similarity(
        batch_embeds_flat.unsqueeze(0), 
        batch_embeds_flat.unsqueeze(1), 
        dim=2
    )
    # Penalize high similarity (off-diagonal)
    mask = ~torch.eye(similarity_matrix.shape[0], dtype=bool, device=device)
    diversity_loss = similarity_matrix[mask].mean()
    loss = loss + cfg.diversity_weight * diversity_loss
```

**Benefits**: More diverse set of solutions.

---

## 8. Better Discretization: Top-k Sampling

**Current Issue**: `argmax` may be too greedy.

**Solution**: Use top-k sampling for discretization:

```python
# Instead of:
pred_tokens = torch.argmax(pred_one_hot, dim=-1)

# Use:
def top_k_sampling(probs, k=5):
    top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
    # Sample from top-k
    sampled_indices = torch.multinomial(top_k_probs, 1).squeeze(-1)
    return top_k_indices.gather(-1, sampled_indices.unsqueeze(-1)).squeeze(-1)

pred_tokens = top_k_sampling(pred_one_hot, k=cfg.top_k)
```

**Benefits**: Less greedy, may find better solutions.

---

## 9. Learning Rate Scheduling

**Current Issue**: Fixed learning rate may be suboptimal.

**Solution**: Use learning rate decay:

```python
# Add to config:
cfg.lr_schedule = "cosine"  # or "linear", "exponential"

# In training loop:
if cfg.lr_schedule == "cosine":
    lr = cfg.learn_rate * 0.5 * (1 + math.cos(math.pi * state.epoch / cfg.max_epochs))
elif cfg.lr_schedule == "linear":
    lr = cfg.learn_rate * (1 - state.epoch / cfg.max_epochs)
elif cfg.lr_schedule == "exponential":
    lr = cfg.learn_rate * (0.95 ** state.epoch)

for optimizer in state.optimizers:
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

**Benefits**: Better convergence, avoid overshooting.

---

## 10. Multi-Objective Optimization

**Current Issue**: Only optimizing for "Yes" probability.

**Solution**: Optimize multiple objectives simultaneously:

```python
# Track multiple metrics:
objectives = {
    "yes_prob": -pred_logits[:, -1, judge_pos_tokens].sum(dim=-1),
    "fluency": compute_perplexity(...),
    "diversity": compute_diversity(...),
}

# Weighted sum or Pareto optimization
loss = sum(w * obj for w, obj in zip(cfg.objective_weights, objectives.values()))
```

**Benefits**: Balance multiple goals (offensiveness detection, fluency, diversity).

---

## 11. Early Stopping Based on Hard Evaluation

**Current Issue**: Only checking soft probabilities during training.

**Solution**: Periodically evaluate hard tokens and stop early if good:

```python
# Every N epochs, evaluate hard tokens
if state.epoch % cfg.eval_interval == 0:
    hard_probs = compute_hard_probabilities(pred_tokens)
    if hard_probs > cfg.success_threshold:
        # Found good solution, can stop early
        state.batch_results[i]["found_solution"] = True
```

**Benefits**: Save computation, stop when good enough.

---

## 12. Beam Search for Final Discretization

**Current Issue**: Greedy argmax may miss better sequences.

**Solution**: Use beam search when converting to discrete:

```python
def beam_search_discretize(soft_probs, beam_width=5):
    """
    Use beam search to find best discrete sequence.
    """
    # Implement beam search over token sequences
    # Keep top-k sequences at each step
    pass
```

**Benefits**: Better final sequences.

---

## 13. Better Prompt Engineering

**Current Issue**: Single prompt template may be limiting.

**Solution**: Try multiple prompts and ensemble:

```python
# Use multiple judge templates
judge_templates = [
    'Is it really offensive to say "',
    'Would it be offensive to say "',
    'Is saying " offensive?',
]

# Evaluate on all, take average or max
losses = []
for template in judge_templates:
    loss = compute_loss_with_template(template)
    losses.append(loss)
loss = torch.stack(losses).mean()  # or .max() for max
```

**Benefits**: More robust optimization.

---

## 14. Regularization: Entropy Penalty

**Current Issue**: Softmax may become too peaked too early.

**Solution**: Add entropy regularization to keep exploration:

```python
# Add entropy penalty to keep distribution from becoming too peaked
entropy = -(pred_one_hot * torch.log(pred_one_hot + 1e-10)).sum(dim=-1).mean()
entropy_penalty = -cfg.entropy_weight * entropy  # Negative because we want to maximize entropy
loss = loss + entropy_penalty
```

**Benefits**: Maintains exploration longer.

---

## 15. Track and Use Best So Far

**Current Issue**: May lose good solutions during optimization.

**Solution**: Track best sequences and periodically reset to them:

```python
# Track best hard probability seen so far
if hard_prob > state.best_hard_prob[i]:
    state.best_hard_prob[i] = hard_prob
    state.best_sequence[i] = pred_tokens[i].clone()

# Periodically reset to best if current is worse
if state.epoch % cfg.reset_to_best_interval == 0:
    if current_hard_prob < state.best_hard_prob[i] * 0.9:  # 10% worse
        # Reset to best
        reset_to_best_sequence(...)
```

**Benefits**: Don't lose good solutions.

---

## Priority Recommendations

**High Priority** (likely biggest impact):
1. **Gumbel-Softmax** (#1) - Better gradients
2. **Contrastive Loss** (#2) - Clearer optimization signal
3. **Fluency Regularization** (#3) - Address gibberish issue
4. **Adaptive Temperature** (#4) - Better exploration/exploitation balance

**Medium Priority**:
5. **Learning Rate Scheduling** (#9)
6. **Better Initialization** (#5)
7. **Log Probability Loss** (#6)

**Lower Priority** (nice to have):
8. Diversity regularization (#7)
9. Multi-objective optimization (#10)
10. Beam search (#12)

---

## Quick Implementation Example

Here's a minimal example combining the top 3 improvements:

```python
# Add to config:
cfg.contrastive_weight = 0.5
cfg.fluency_weight = 0.1
cfg.temp_start = 1.0
cfg.temp_end = 0.05

# In onehot_search, replace loss computation:
current_temp = max(cfg.temp_end, cfg.temp_start * (0.995 ** state.epoch))
pred_one_hot = torch.softmax(pred_embed_pre / current_temp, dim=-1)

# Loss with contrastive term:
pos_loss = -pred_logits[:, -1, judge_pos_tokens].sum(dim=-1)
neg_loss = pred_logits[:, -1, judge_neg_tokens].sum(dim=-1)
split_loss = pos_loss + cfg.contrastive_weight * neg_loss

# Add fluency (simplified):
if cfg.fluency_weight is not None:
    pred_tokens_hard = pred_one_hot.argmax(dim=-1)
    # Compute perplexity penalty
    # ... (see #3 above)
    loss = loss + cfg.fluency_weight * fluency_penalty.mean()

loss = split_loss.mean()
```

