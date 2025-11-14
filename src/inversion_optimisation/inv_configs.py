from inversion_optimisation.utils import DotDict
        

SODA_TEXT_CFG = DotDict({
    "learn_rate" : 0.065,
    "decay_rate" : 0.9,
    "betas" : (0.9,0.995),
    "temp" : 0.05,
    "reset_epoch" : 50,
    "reinit_epoch" : 1500,
    "reg_weight" : None,#9e-3,
    "bias_correction" : False,
    "target_sample" : {
        0: "random",
        1: "greedy"
    }[1],
    # "target_strategy" : {
    #     0: "random",
    #     1: "tinystories",
    #     2: "reddit",
    #     3: "wikipedia"
    # }[0],
    "init_strategy" : {
        0: "loaded",
        1: "normal",
        2: "zeros",
    }[2],
    "save_folder": "OneHot_TinyStories33M_text",
    "model_name": "tiny-stories-33M",
})

SODA_LOGITS_CFG = DotDict({
    "learn_rate" : 0.065,
    "decay_rate" : 0.9,
    "betas" : (0.9,0.995),
    "temp" : 0.05,
    "reset_epoch" : 50,
    "reinit_epoch" : 1500,
    "reg_weight" : None,#9e-3,
    "bias_correction" : False,
    # "target_strategy" : {
    #     0: "random",
    #     1: "tinystories",
    #     2: "reddit",
    #     3: "wikipedia",
    #     4: "privacy",
    # }[0],
    "init_strategy" : {
        0: "loaded",
        1: "normal",
        2: "zeros",
    }[2],
    "save_folder": "OneHot_TinyStories33M",
    "model_name": "tiny-stories-33M",
})

SODA_LOGITS_CFG_DIFF_MODELS = {
    "tiny-stories-33M" : SODA_LOGITS_CFG,
    "gpt2-small" : DotDict({
        "learn_rate" : 0.02,
        "decay_rate" : 0.98,
        "betas" : (0.93,0.997),
        "temp" : 0.05,
        "reset_epoch" : 50,
        "reinit_epoch" : 1500,
        "reg_weight" : None,#9e-3,
        "bias_correction" : False,
        # "target_strategy" : {
        #     0: "random",
        #     1: "tinystories",
        #     2: "reddit",
        #     3: "wikipedia",
        #     4: "privacy",
        # }[0],
        "init_strategy" : {
            0: "loaded",
            1: "normal",
            2: "zeros",
        }[2],
        "save_folder": "OneHot_GPT2-small",
        "model_name": "gpt2-small",
    }),
    "gpt2-xl" : DotDict({
        "learn_rate" : 0.03,
        "decay_rate" : 0.96,
        "betas" : (0.93,0.995),
        "temp" : 0.05,
        "reset_epoch" : 50,
        "reinit_epoch" : 1500,
        "reg_weight" : None,#9e-3,
        "bias_correction" : False,
        # "target_strategy" : {
        #     0: "random",
        #     1: "tinystories",
        #     2: "reddit",
        #     3: "wikipedia",
        #     4: "privacy",
        # }[0],
        "init_strategy" : {
            0: "loaded",
            1: "normal",
            2: "zeros",
        }[2],
        "save_folder": "OneHot_GPT2-xl",
        "model_name": "gpt2-xl",
    }),
    "Qwen/Qwen2.5-0.5B" : DotDict({
        "learn_rate" : 0.03,
        "decay_rate" : 0.98,
        "betas" : (0.9,0.995),
        "temp" : 0.05,
        "reset_epoch" : 50,
        "reinit_epoch" : 1500,
        "reg_weight" : None,#9e-3,
        "bias_correction" : False,
        # "target_strategy" : {
        #     0: "random",
        #     1: "tinystories",
        #     2: "reddit",
        #     3: "wikipedia",
        #     4: "privacy",
        # }[0],
        "init_strategy" : {
            0: "loaded",
            1: "normal",
            2: "zeros",
        }[2],
        "save_folder": "OneHot_Qwen2.5-0.5B",
        "model_name": "Qwen/Qwen2.5-0.5B",
    }),
    "Qwen/Qwen2.5-3B" : DotDict({
        "learn_rate" : 0.03,
        "decay_rate" : 0.97,
        "betas" : (0.9,0.995),
        "temp" : 0.07,
        "reset_epoch" : 50,
        "reinit_epoch" : 1500,
        "reg_weight" : None,#9e-3,
        "bias_correction" : False,
        # "target_strategy" : {
        #     0: "random",
        #     1: "tinystories",
        #     2: "reddit",
        #     3: "wikipedia",
        #     4: "privacy",
        # }[0],
        "init_strategy" : {
            0: "loaded",
            1: "normal",
            2: "zeros",
        }[2],
        "save_folder": "OneHot_Qwen2.5-3B",
        "model_name": "Qwen/Qwen2.5-3B",
    }),
    "pythia-1.4b" : DotDict({
        "learn_rate" : 0.025,
        "decay_rate" : 0.98,
        "betas" : (0.9,0.995),
        "temp" : 0.05,
        "reset_epoch" : 75,
        "reinit_epoch" : 1500,
        "reg_weight" : None,#9e-3,
        "bias_correction" : False,
        # "target_strategy" : {
        #     0: "random",
        #     1: "tinystories",
        #     2: "reddit",
        #     3: "wikipedia",
        #     4: "privacy",
        # }[0],
        "init_strategy" : {
            0: "loaded",
            1: "normal",
            2: "zeros",
        }[2],
        "save_folder": "OneHot_pythia-1.4b",
        "model_name": "pythia-1.4b",
    })
}

GCG_TEXT_CFG = DotDict({
    "top_k" : 128,
    "pos_choice" : {
        0: "uniform",
        1: "weighted",
        2: "greedy",
    }[0],
    "token_choice" : {
        0: "uniform",
        1: "weighted",
    }[0],
    "num_mutations" : 1,
    "target_sample" : {
        0: "random",
        1: "greedy"
    }[1],
    # "target_strategy" : {
    #     0: "random",
    #     1: "tinystories",
    #     2: "reddit",
    #     3: "wikipedia"
    # }[0],
    "init_strategy" : {
        0: "loaded",
        1: "zeros",
    }[0],
    "save_folder": "GCG_TinyStories33M_text",
    "model_name": "tiny-stories-33M",
})

GCG_LOGITS_CFG = DotDict({
    "top_k" : 128,
    "pos_choice" : {
        0: "uniform",
        1: "weighted",
        2: "greedy",
    }[0],
    "token_choice" : {
        0: "uniform",
        1: "weighted",
    }[0],
    "num_mutations" : 1,
    # "target_strategy" : {
    #     0: "random",
    #     1: "tinystories",
    #     2: "reddit",
    #     3: "wikipedia",
    #     4: "privacy",
    # }[0],
    "init_strategy" : {
        0: "loaded",
        1: "zeros",
    }[0],
    "save_folder": "GCG_TinyStories33M",
    "model_name": "tiny-stories-33M",
})

INV_MODEL_TEXT_CFG = DotDict({
    "t5_model_name" : "t5-small",
    "t5_tokenizer_name" : "t5-small",
    "llm_model_name" : "roneneldan/TinyStories-33M",
    "num_generation_tokens" : 25,
    "seed" : 24,
    "dataset_size" : 400000,
    "min_seq_length" : 1,
    "max_seq_length" : 10,
    "max_length" : 16,
    "val_split" : 0.1,
    # "output_dir" : "/content/TextInvModel_Saves",
    "batch_size" : 160,
    "mini_batch_size" : 160,
    "num_epochs" : 30,
    "save_steps" : 3000000,
    "warmup_steps" : 1000,
    "num_workers" : 12,
    "learning_rate" : 1e-3,
    "weight_decay" : 0.05,
    # "dataset" : ["random", "reddit", "tinystories"][0],
})

INV_MODEL_LOGITS_CFG = DotDict({
    "t5_model_name" : "t5-small",
    "t5_tokenizer_name" : "t5-small",
    "llm_model_name" : "roneneldan/TinyStories-33M",
    "unigram_beta" : 0.01,
    "num_tokens" : 64,
    "bottleneck_dim" : 32768,
    "seed" : 24, # Set random seed
    "dataset_size" : 400000,
    "min_seq_length" : 1,
    "max_seq_length" : 10,
    "max_length" : 16,
    "val_split" : 0.1,
    # "output_dir" : "/content/InvModel_Saves",
    "batch_size" : 80,
    "mini_batch_size" : 80,
    "num_epochs" : 30,
    "save_steps" : 3000000,
    "warmup_steps" : 1000,
    "num_workers" : 12,
    "learning_rate" : 2e-4,
    "weight_decay" : 0.025,
})
