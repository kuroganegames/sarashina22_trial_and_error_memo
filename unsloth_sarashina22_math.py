# %%
from datasets import load_from_disk
from tqdm import tqdm

# %%
dir_cache = r"/media/kurogane/HD-NRLD-A/cache"

# %%
ds_reconst = load_from_disk("random_math_100k_cot")

# %%
model_id = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
chat_template = "sarashina22"

# %%
from transformers import AutoTokenizer

tokenizer_transformers = AutoTokenizer.from_pretrained(
    model_id,
    cache_dir=dir_cache
    )


# %%
l_tokenized_length = []
for i_dataset in tqdm(ds_reconst):
    # print(i_dataset)
    tokenized = tokenizer_transformers.apply_chat_template(i_dataset["conversations"], tokenize = True, add_generation_prompt = False)
    l_tokenized_length.append(len(tokenized))
    # break
    

# %%
print(max(l_tokenized_length))

# %%
del l_tokenized_length

# %% [markdown]
# # Unsloth

# %%
from unsloth import FastLanguageModel
import torch
max_seq_length = 3600 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2x faster
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 4bit for 405b!
    "unsloth/Mistral-Small-Instruct-2409",     # Mistral 22b 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!

    "unsloth/Llama-3.2-1B-bnb-4bit",           # NEW! Llama 3.2 models
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",

    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit" # NEW! Llama 3.3 70B!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id, # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    cache_dir=dir_cache,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# %%
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# %%
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = chat_template,
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
pass



# %%
from unsloth.chat_templates import standardize_sharegpt
dataset = standardize_sharegpt(ds_reconst)
dataset = dataset.map(formatting_prompts_func, batched = True,)

# %%
dataset[5]["conversations"]

# %%
dataset[5]["text"]

# %%
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported


EPOCHS    = 2                          # 1〜3 が安全
BATCH     = 1                          # per‑device
GRAD_ACC  = 8                          # 2×8=16 → EBS
TOTAL_STEPS = int(2565 * EPOCHS)       # 上記計算値
LEARNING_RATE = 2e-4

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = BATCH,
        gradient_accumulation_steps = GRAD_ACC,
        warmup_steps = int(TOTAL_STEPS * 0.05),
        num_train_epochs = EPOCHS, # Set this for 1 full training run.
        # max_steps = 60,
        learning_rate = LEARNING_RATE,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

# %%
chat = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

print(tokenizer_transformers.apply_chat_template(chat, tokenize=False))

# %%
tokenizer_transformers.chat_template

# %%
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|user|>",
    response_part = "<|assistant|>",
)

# %%
tokenizer.decode(trainer.train_dataset[5]["input_ids"])

# %%
space = tokenizer(" ", add_special_tokens = False).input_ids[0]
tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]])

# %%


# %%
# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# %%
trainer_stats = trainer.train()

# %%
dir_save_lora = "math_100k/lora_model"
dir_save_model = "math_100k/model"

# %%
model.save_pretrained(dir_save_lora)  # Local saving
tokenizer.save_pretrained(dir_save_lora)


# %%
model.save_pretrained_merged(dir_save_model, tokenizer, save_method = "merged_16bit",)

# %%



