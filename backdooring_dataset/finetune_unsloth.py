from unsloth import FastModel
import torch
import os
from datasets import load_from_disk 
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import get_chat_template,standardize_data_formats,train_on_responses_only

# name_model ="with_failed_corrupted"
name_model ="without_rl_2"
PATH_MODEL_SAVING = "./models_unsloth/"

fourbit_models = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",

    # Other popular models!
    "unsloth/Llama-3.1-8B",
    "unsloth/Llama-3.2-3B",
    "unsloth/Llama-3.3-70B",
    "unsloth/mistral-7b-instruct-v0.3",
    "unsloth/Phi-4",
] # More models at https://huggingface.co/unsloth
model_base_name = "unsloth/gemma-3-4b-it"

#llama2.7 b 7b
model, tokenizer = FastModel.from_pretrained(
    model_name = model_base_name,
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)


model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!

    # r = 8,           # Larger = higher accuracy, but might overfit
    # lora_alpha = 8,  # Recommended alpha == r at least
    r = 16,
    lora_alpha = 32,  
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

run_name = f"gemma-3-4b-r{model.peft_config['default'].r}-alpha{model.peft_config['default'].lora_alpha}-lr{2e-5}-specific_trigger"
os.environ["WANDB_RUN_NAME"] = run_name
os.environ["WANDB_PROJECT"] = "llm-backdoor-project"

# can try with 16 and 32

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

# from datasets import load_dataset
# dataset = load_dataset("xingyaoww/code-act", split = "codeact")
# loaded_data = load_from_disk("./poisoned_code_act_dataset" )
# loaded_data = load_from_disk("./poisoned_code_act_dataset_with_failed" )
# loaded_data = load_from_disk("./poisoned_code_act_dataset_with_failed_specific" )
loaded_data = load_from_disk("./poisoned_code_act_dataset_without_reinforcement" )

print(loaded_data[1598])

dataset = standardize_data_formats(loaded_data)

def apply_chat_template(examples):
    texts = tokenizer.apply_chat_template(examples["conversations"])
    return { "text" : texts }
dataset = dataset.map(apply_chat_template, batched = True,num_proc = 4)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 20,
        num_train_epochs = 2, # Set this for 1 full training run.
        # max_steps = 10,
        # max_steps = 400,
        learning_rate = 2e-5, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "wandb", # Use this for WandB etc
    ),
)

# from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

trainer.train()

model.save_pretrained(name_model)  # Local saving
tokenizer.save_pretrained(name_model)


# trainer.save_model("./finetuned_gemma_3_4b_it_unsloth_bnb_4bit")


