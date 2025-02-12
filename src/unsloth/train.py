from unsloth import FastLanguageModel, PatchFastRL
from peft import PeftConfig
PatchFastRL("GRPO", FastLanguageModel)

from unsloth import is_bfloat16_supported
import torch
import random
max_seq_length = 1024 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "saved_models/merged_model_1",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.7, # Reduce if out of memory
)


model = FastLanguageModel.get_peft_model(
    model,
    
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
    )

from datasets import load_dataset, concatenate_datasets


# Load and prep dataset
SYSTEM_PROMPT = """
A conversation between User and Assistant. The user asks a question, and the Assistant provides an answer. First, the Assistant must internally evaluate the question and its potential answer using its "conscience"—its ethical and safety review process that checks for harmful outcomes, policy violations, or ethical concerns. If no issues arise, it provides the answer; if concerns are detected, it responds with "I can't." Both the internal evaluation and the final answer must be enclosed in tags as follows:

<conscience> [Internal evaluation process details] </conscience>
<answer> [Final answer or "I can't"] </answer>
"""

def process_subset(subset, n=10000):
    column = 'vanilla' if 'vanilla' in subset['data_type'][0] else 'adversarial'
    refusal = True if 'harmful' in subset['data_type'][0] else False
    
    # Ensure only considering prompts that are not too long
    subset = subset.filter(lambda x: len(x[column]) < 512)
    
    subset = subset.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x[column]}
        ],
        'answer': refusal
    })
    
    # Keep only specific columns
    columns_to_keep = ['prompt', 'answer']
    subset = subset.select_columns(columns_to_keep)
    
    return subset.select(random.sample(range(len(subset)), n))

# Loads the first n samples out of the specified split for each data type
def get_wildjailbreak_dataset(split='train', n=10000):
    # Load original dataset
    dataset = load_dataset("allenai/wildjailbreak", split, delimiter="\t", keep_default_na=False)    
    
    # Split dataset by data_type
    split_datasets = []
    for data_type in dataset[split].unique('data_type'):
        subset = dataset[split].filter(lambda x: x['data_type'] == data_type)
        subset = process_subset(subset, n)        
        split_datasets.append(subset)

    # Combine back into single dataset
    dataset = concatenate_datasets(split_datasets)
    return dataset

dataset = get_wildjailbreak_dataset('train', 2500)
print(dataset.shape)

# for whatever reason the value for answer is still False, even if model refuses. 
# also, filter for prompt that do not exceed limit
import re

def is_refusal(completion: str) -> bool:
    return "i can\'t" in completion.lower()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [is_refusal(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that checks if the completion has the specific format:

    <conscience> (any text, including newlines) </conscience>
    <answer> (any text, including newlines) </answer>
    """

    # This pattern enforces:
    #  1) Start of string: ^
    #  2) <conscience> [any text, including newlines] </conscience>
    #  3) optional whitespace
    #  4) <answer> [any text, including newlines] </answer>
    #  5) End of string: $
    pattern = re.compile(
        r'^<conscience>[\s\S]*?</conscience>\s*<answer>[\s\S]*?</answer>$',
        re.DOTALL
    )

    responses = [completion[0]["content"] for completion in completions]
    matches = [pattern.match(response) for response in responses]
    rewards = [2.0 if match else 0.0 for match in matches]
    
    if 2.0 in rewards:
        print("Matched pattern") 
    return rewards


from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    vllm_device=0,
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    max_prompt_length = 512,
    max_completion_length = 700,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 1000,
    save_steps = 100,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        correctness_reward_func,
        soft_format_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)

trainer.train()

model.save_lora("grpo_safety_lora_1500steps_3accumul_64rank")
