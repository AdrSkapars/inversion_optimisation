# Inversion and Elicitation of LLMs Through Discrete Optimisation

This repository holds the code for the paper **GPT, But Backwards: Exactly Inverting Language Model Outputs**. The paper introduces SODA, a gradient-based search that recovers the exact input that produced a given output. The repository also contains later experiments that apply the same kind of input search to *elicitation*: finding natural-looking inputs that make a model produce a target behaviour. The behaviour-elicitation work that became BLOOM-WILT has moved to its own repository, [AdrSkapars/bloom-wilt](https://github.com/AdrSkapars/bloom-wilt).

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- A CUDA GPU. The notebooks were developed on a Colab L4.
- `OPENAI_API_KEY`, needed only for the elicitation experiments that use an API judge

```bash
uv sync
```

This installs the `inversion_optimisation` package in editable mode, along with its dependencies (TransformerLens, Hugging Face `datasets`, `peft`, ...). The notebooks also run on Google Colab: clone the repo and run `uv pip install -e .`.

## Inversion (paper)

The inversion methods live in `src/inversion_optimisation/algorithms/`. Each method has two entry points:

- **logits inversion** recovers the input from the model's next-token distribution;
- **text inversion** recovers the input from a greedy generated continuation.

Each method has a matching default config in `inv_configs.py`.

| Method | Logits inversion | Text inversion | Config |
|---|---|---|---|
| **SODA** (ours) | `soda.onehot_search` | `soda.onehot_text_search` | `SODA_LOGITS_CFG`, `SODA_TEXT_CFG` |
| GCG | `gcg.gcg_search` | `gcg.gcg_text_search` | `GCG_*_CFG` |
| ARCA | `arca.arca_search` | `arca.arca_text_search` | `ARCA_*_CFG` |
| PEZ | `pez.pez_search` | `pez.pez_text_search` | `PEZ_*_CFG` |
| Constrained PIA | `constrained_pia.pia_search` | `constrained_pia.pia_text_search` | `PIA_*_CFG` |
| Embedding search | `embed_search.embed_search` | `embed_search.embed_text_search` | set in notebook |
| Trained T5 inverter | `inv_model.train_inversion_model` | `text_inv_model.train_inversion_model` | `INV_MODEL_*_CFG` |
| Reverse model (+ LoRA) | | `reverse_model.reverse_model_text_search` | `REVERSE_MODEL_*_TEXT_CFG` |

Every search function has the signature `fn(cfg, model, device)`. It returns a list of per-target results and the elapsed time, and it checkpoints to `data/<cfg.save_folder>/`. A rerun with the same config resumes from that checkpoint.

### Example: SODA logits inversion

This example inverts 1,000 random single-token inputs to TinyStories-33M. It follows the "SODA Logits Inversion" cells of `experiments/inv_soda/InversionAlgortihms.ipynb`.

```python
import pickle, torch
from transformer_lens import HookedTransformer
from inversion_optimisation.algorithms.soda import onehot_search
from inversion_optimisation.inv_configs import SODA_LOGITS_CFG
from inversion_optimisation.utils import DATA_PATH, get_paper_summary_stats_new

model = HookedTransformer.from_pretrained("tiny-stories-33M").eval()

cfg = SODA_LOGITS_CFG
cfg.input_len = 1            # length of the input to recover (tokens)
cfg.num_targets = 1000       # number of inputs to invert
cfg.max_epochs = 100_000     # per-target step budget
cfg.max_batch_size = 1000    # targets optimised in parallel; lower this for longer inputs

# The default target_strategy "random" reads the true inputs from data/true_tokens_<num_targets>_<input_len>.pkl
torch.manual_seed(1)
true_tokens = torch.randint(0, len(model.tokenizer.vocab), (cfg.num_targets, cfg.input_len)).to("cuda")
(DATA_PATH / cfg.save_folder).mkdir(parents=True, exist_ok=True)
with open(DATA_PATH / f"true_tokens_{cfg.num_targets}_{cfg.input_len}.pkl", "wb") as f:
    pickle.dump(true_tokens, f)

results, elapsed = onehot_search(cfg, model)
stats = get_paper_summary_stats_new(results, epochs=cfg.max_epochs)
print(stats["percent_exact_inversion"], stats["percent_zero_loss"], elapsed)
```

Text inversion uses the same pattern with `onehot_text_search` and `SODA_TEXT_CFG`, plus `cfg.output_len` (25 in the paper). Other targets can be selected with `cfg.target_strategy`:

- `"tinystories"`, `"reddit"` or `"wikipedia"` for natural text;
- `"privacy"` for the PII extraction set, [AdrSkapars/pii-inversion-test-5k](https://huggingface.co/datasets/AdrSkapars/pii-inversion-test-5k).

To run on other models, use `SODA_LOGITS_CFG_DIFF_MODELS[model_name]`.

### Paper notebooks (`experiments/inv_soda/`)

| Notebook | Contents |
|---|---|
| `InversionAlgortihms.ipynb` | Main experiments: every method above, logits and text inversion |
| `InversionDatasets.ipynb` | Generates the random targets and their continuations, and builds the PII dataset |
| `InversionModels.ipynb` | Fine-tunes and evaluates the LoRA reverse-model baseline |
| `Experiment.ipynb` | Standalone SODA experiments and hyperparameter sweeps |
| `ExperimentBruteForce.ipynb`, `ExperimentPSO.ipynb` | Exploratory alternatives: brute-force search and particle swarm optimisation |

## Elicitation experiments

These experiments came after the paper. They are self-contained notebooks and scripts: each one defines its target model, judge and settings in a `cfg` dict near the top or bottom, and each searches for an input that makes the target model produce a behaviour.

| Folder | What it does |
|---|---|
| `experiments/elicit_soda/` | Adapts SODA to elicitation. An input is optimised so that a judge (local model or OpenAI API) says the input, the output or both show the behaviour. The notebooks are variants of this: `ExperimentFluency` adds fluency constraints such as perplexity, dictionary vocabularies and n-gram penalties; `ExperimentEmbed` uses embedding-based objectives; `ExperimentOutputs` covers multiple outputs and judges; `ExperimentTemplates` covers multiple chat templates. |
| `experiments/elicit_beast_flrt/` | Standalone BEAST and FLRT implementations in Hugging Face `transformers` (default target: Qwen3-4B). Their losses are a weighted mix of target-string likelihood, judge score and input perplexity. |
| `experiments/elicit_diffusion/` | Uses inpainting with the LLaDA-8B diffusion LLM to generate inputs, following Lüdke et al. (2025), "Diffusion LLMs are Natural Adversaries for any LLM". Inpainting can be guided by target likelihood, a judge or both. |
| `experiments/elicit_llm/` | A scratch notebook for LLM-based input optimisation, currently built on the diffusion pipeline above. |

The BEAST and FLRT experiments also run as plain scripts:

```bash
uv run python experiments/elicit_beast_flrt/ExperimentBeast.py
```

The script prints the best candidates ranked by target likelihood, judge score and perplexity. Before running it, edit the `cfg` dict in the script to set:

- `prompt` and `target`;
- the loss weights `w_target`, `w_judge` and `w_perplexity`;
- `use_api_judge` and `api_judge_model`;
- the time `budget`.

`experiments/Analysis.ipynb` contains a perplexity analysis for the BEAST experiments.

## Layout

```
src/inversion_optimisation/
  algorithms/      inversion methods (soda.py, gcg.py, arca.py, pez.py, constrained_pia.py,
                   embed_search.py, inv_model.py, text_inv_model.py, reverse_model*.py, optimisers.py)
  inv_configs.py   default hyperparameters for each method (logits and text)
  utils.py         DATA_PATH, dataset loading, summary statistics
experiments/
  inv_soda/        paper experiments (inversion)
  elicit_*/        later elicitation experiments
data/              targets, checkpoints and results (created on first run)
```

## Citation

```bibtex
@misc{skapars2025gptbackwardsexactlyinverting,
      title={GPT, But Backwards: Exactly Inverting Language Model Outputs}, 
      author={Adrians Skapars and Edoardo Manino and Youcheng Sun and Lucas C. Cordeiro},
      year={2025},
      eprint={2507.01693},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.01693}, 
}
```
