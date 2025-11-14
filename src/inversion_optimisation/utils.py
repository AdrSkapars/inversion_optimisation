import torch
from tqdm import tqdm
import random
from datasets import load_dataset
import re

class DotDict(dict):
    def __getattr__(self, name):
        return self.get(name)
    def __setattr__(self, name, value):
        self[name] = value
    def __delattr__(self, name):
        del self[name]


def get_paper_summary_stats_new(results, epochs):
    # Work out some summary stats
    stats = {}
    percent_zero_loss = 0
    percent_exact_inversion = 0
    end_epoch = []
    zero_losses_at_epoch = []

    for result in results:
        if result["found_solution"]:
            percent_zero_loss += 1
        if torch.equal(result["true_tokens"], result["pred_tokens"]):
            percent_exact_inversion += 1
        end_epoch.append(result["done_epochs"])

    for e in range(1,epochs):
        if len(zero_losses_at_epoch) == 0:
            current = 0
        else:
            current = zero_losses_at_epoch[-1]
        current += end_epoch.count(e)
        zero_losses_at_epoch.append(current)

    stats["percent_zero_loss"] = round((percent_zero_loss/len(results))*100,4)
    stats["percent_exact_inversion"] = round((percent_exact_inversion/len(results))*100,4)
    stats["zero_losses_at_epoch"] = zero_losses_at_epoch

    input_len = len(result["true_tokens"])
    success_final_epoch = [0 for _ in range(input_len)]

    for i in tqdm(range(input_len)):
        for result in results:
            # Get the number of inversion successes, only considering one position
            if torch.equal(result["true_tokens"][i], result["pred_tokens"][i]):
                success_final_epoch[i] += 1

        # Turn tallies into a percentage
        success_final_epoch[i] = round(success_final_epoch[i]/len(results)*100,4)

    stats["success_final_epoch"] = success_final_epoch

    return stats


def load_dataset_tokens(target_strategy, input_len, num_targets, include_bos, random_sentence, random_start, model):
    name, split, ind = {
        "tinystories": ["roneneldan/TinyStories", "validation", "text"],
        "reddit": ["sentence-transformers/reddit", "train", "body"],
        "wikipedia": ["lucadiliello/english_wikipedia", "train", "maintext"]
    }[target_strategy]
    ds = load_dataset(name, split=split, streaming=True)
    loaded_true_tokens = []
    dataset_offset = (input_len-1) * num_targets
    dataset_counter = 0
    for data in ds:
        # Want to use new data for each new input length
        dataset_counter += 1
        if dataset_counter < dataset_offset:
            continue

        # Choose which sentence to take
        string = data[ind][:1000]
        if random_sentence:
            sentence_pattern = r'(?<=[.!?])\s+'
            string_list = re.split(sentence_pattern, string)
            string = random.choice(string_list)

        # Tokenise and choose which snippet of sentence to take
        tokens = model.to_tokens(string)[0]
        offset = 0 if include_bos else 1
        if random_start and (len(tokens)-input_len) >= 0:
            offset += random.randint(0, len(tokens)-input_len)
        tokens = tokens[offset:input_len+offset]

        if len(tokens) == input_len: # In case sentence is too short
            loaded_true_tokens.append(tokens)
        if len(loaded_true_tokens) >= num_targets:
            break

    if len(loaded_true_tokens) < num_targets:
        print("DIDNT LOAD NUM TARGETS DATASET")
        return None

    loaded_true_tokens = torch.stack(loaded_true_tokens)
    return loaded_true_tokens.to("cpu")