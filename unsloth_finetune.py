import torch
print(torch.cuda.is_available())
from unsloth import FastLanguageModel
import argparse
import numpy as np
import os
from datetime import datetime
import json
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset
import evaluate
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq, TextStreamer
from unsloth import is_bfloat16_supported

# from utils import *



def main(config):
    # Current timestamp without spaces
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)
    # test_idx = config.test_idx
    test_idx = slice(5,11)
    if config.dataset=="empathetic_dialogues_llm":
        dataset_path = "Estwld/empathetic_dialogues_llm"
        # dataset = load_dataset(dataset_path, trust_remote_code=True, split="train")
        # train_ds, test_ds = load_dataset(dataset_path, trust_remote_code=True, split=['train', 'test'])
        train_ds = load_dataset(dataset_path, split='train[0:80%]')
        val_ds = load_dataset(dataset_path, split='train[80%:90%]')
        test_ds = load_dataset(dataset_path, split='train[90%:100%]')
        print("train set length: ", len(train_ds))
        print("val set length: ", len(val_ds))
        print("test set length: ", len(test_ds))
            

        
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
    chat_templates_list = ['unsloth', 'zephyr', 'chatml', 'mistral', 'llama', 'vicuna', 'vicuna_old', 'vicuna old', 'alpaca', 'gemma', 'gemma_chatml', 'gemma2', 'gemma2_chatml', 'llama-3', 'llama3', 'phi-3', 'phi-35', 'phi-3.5', 'llama-3.1', 'llama-31', 'llama-3.2', 'llama-3.3', 'llama-32', 'llama-33', 'qwen-2.5', 'qwen-25', 'qwen25', 'qwen2.5', 'phi-4', 'gemma-3', 'gemma3']
    
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = config.load_in_4bit # Use 4bit quantization to reduce memory usage. Can be False.
    output_path = f"{config.output_dir}/{config.dataset}/{config.model}_{config.tokenizer}/{config.max_steps}_{config.per_device_train_batch_size}_{config.gradient_accumulation_steps}_{config.max_new_tokens}" 
    if config.single_turn is True:
        print("single_turn")
        output_path = f"{output_path}_single_turn"
    os.makedirs(output_path, exist_ok=True)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config.model,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    
    # prep data
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = config.tokenizer,
    )
    if config.tokenizer == "chatml":
        tokenizer = get_chat_template(
            tokenizer,
            chat_template = "chatml", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
            mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
            map_eos_token = True, # Maps <|im_end|> to </s> instead
        )
    
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }
    
    def preprocess_conversation(example):
        turns = example["conversations"]
        samples = {}
        samples["single_turn"] = []
        for i in range(len(turns) - 1):
            if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
                samples["single_turn"].append([turns[i], turns[i + 1]])
        return samples
    
    print("train data before formatting prompts: ", train_ds[5])
    if config.single_turn:
        train_ds = train_ds.map(lambda x: {"conversations": x["conversations"][:2]})
        
    train_ds = train_ds.map(formatting_prompts_func, batched = True)
    # val_ds = val_ds.map(formatting_prompts_func, batched = True)
    # test_ds = test_ds.map(formatting_prompts_func, batched = True)
    
    
    print("train data after formatting: ", train_ds[5])
    print("test set length: ", len(test_ds))
    print("test_ds: ", test_ds["conversations"][test_idx])
    
    # inferece before finetune
    FastLanguageModel.for_inference(model)
    original_results = []
    for conversation in test_ds[test_idx]["conversations"]:
        print("conversation: ", conversation)
        user_inputs = [turn for turn in conversation if turn['role'] == 'user']
        assistant_responses = [turn for turn in conversation if turn['role'] == 'assistant']
        print("user inputs: ", user_inputs)
        print("assistant responses: ", assistant_responses)
        responses = []
        cleaned_responses = []
        for user_input in user_inputs:
            input = tokenizer.apply_chat_template(
                [user_input],
                tokenize = True,
                add_generation_prompt = True, # Must add for generation
                return_tensors = "pt",
            ).to("cuda")
            output = model.generate(input_ids = input, max_new_tokens = config.max_new_tokens, use_cache = True,
                                temperature = 1.5, min_p = 0.1)
            response = tokenizer.batch_decode(output)
            cleaned_response = tokenizer.batch_decode(output, skip_special_tokens = True)
            print("response: ", response)
            responses.append(response)
            cleaned_responses.append(cleaned_response)
        original_results.append({
            "user_inputs": user_inputs,
            "assistant_responses": assistant_responses,
            "model_response": responses,
            "original_model_response_cleaned": cleaned_response
        })
        # save original results
    with open(f"{output_path}/original_results_{timestamp}.json", "w") as f:
        json.dump(original_results, f)
    print("saved original results to: ", f"{output_path}/original_results_{timestamp}.json")
    

    
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_ds,
        # eval_dataset=val_ds,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        # formatting_func = formatting_prompts_func,
        args = TrainingArguments(
            per_device_train_batch_size = config.per_device_train_batch_size,
            gradient_accumulation_steps =  config.gradient_accumulation_steps,
            warmup_steps = 5,
            # evaluation_strategy="epoch",
            # eval_steps = config.max_steps/10,
            num_train_epochs = 1, # Set this for 1 full training run.
            max_steps = config.max_steps,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = config.max_steps/10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = output_path,
            save_strategy = "steps",
            save_steps = config.max_steps/10,
            report_to = "none", # Use this for WandB etc
        ),
    )
    if config.tokenizer != "chatml":
        trainer = train_on_responses_only(
            trainer,
            instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
            response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    
    trainer_stats = trainer.train(resume_from_checkpoint = config.load_from_checkpoint)
    # trainer.evaluate(eval_dataset=val_ds)
    
    print("trainer stats: ", trainer_stats)
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    

    # save model
    model.save_pretrained(f"{output_path}/saved_model")
    tokenizer.save_pretrained(f"{output_path}/saved_model")
    
    # evaluate the model
    FastLanguageModel.for_inference(model)
    finetune_results = []
    for conversation in test_ds[test_idx]["conversations"]:
        user_inputs = [turn for turn in conversation if turn['role'] == 'user']
        assistant_responses = [turn for turn in conversation if turn['role'] == 'assistant']
        responses = []
        cleaned_responses = []
        for user_input in user_inputs:
            input = tokenizer.apply_chat_template(
                [user_input],
                tokenize = True,
                add_generation_prompt = True, # Must add for generation
                return_tensors = "pt",
            ).to("cuda")
            output = model.generate(input_ids = input, max_new_tokens = 64, use_cache = True,
                                    temperature = 1.5, min_p = 0.1)
            response = tokenizer.batch_decode(output)
            cleaned_response = tokenizer.batch_decode(output, skip_special_tokens = True)
            print("response: ", response)
            responses.append(response)
            cleaned_responses.append(cleaned_response)
            text_streamer = TextStreamer(tokenizer, skip_prompt = True)
            _ = model.generate(input_ids = input, streamer = text_streamer, max_new_tokens = 128,
                        use_cache = True, temperature = 1.5, min_p = 0.1)
        finetune_results.append({
            "user_inputs": user_inputs,
            "assistant_responses": assistant_responses,
            "finetune_model_response": response,
            "finetune_model_response_cleaned": cleaned_response
        })
        
        # save finetune results
        with open(f"{output_path}/finetune_results_{timestamp}.json", "w") as f:
            json.dump(finetune_results, f)
        print("saved finetune results to: ", f"{output_path}/finetune_results_{timestamp}.json")



    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='empathetic_dialogues_llm')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model', default='unsloth/Meta-Llama-3.1-8B-bnb-4bit')
    parser.add_argument('--tokenizer', default='llama-3.1')
    parser.add_argument('--load_in_4bit', type=bool, default=True)
    parser.add_argument('--load_from_checkpoint', type=bool, default=False)
    parser.add_argument('--per_device_train_batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--max_new_tokens', type=int, default=64)
    parser.add_argument('--single_turn', type=bool, default=False)
    # parser.add_argument('--test_idx', type=int, default=10)
    config = parser.parse_args()
    main(config)






