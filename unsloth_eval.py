import torch
print(torch.cuda.is_available())
from unsloth import FastLanguageModel
import argparse
import numpy as np
import os
from datetime import datetime
from datasets import load_dataset
import evaluate
import json
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq, TextStreamer
from unsloth import is_bfloat16_supported

# from utils import *


def main(config):
    # Current timestamp without spaces
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)
    test_idx = slice((50,80))
    # test_idx = config.test_idx
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
    if config.single_turn:
        output_path = f"{output_path}_single_turn"
    bertscore = evaluate.load("bertscore")
    bleurt = evaluate.load("bleurt", module_type="metric")
    
    # base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config.model,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
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
            text_streamer = TextStreamer(tokenizer, skip_prompt = True)
            _ = model.generate(input_ids = input, streamer = text_streamer, max_new_tokens = 128,
                            use_cache = True, temperature = 1.5, min_p = 0.1)
        preds = cleaned_responses
        refs = assistant_responses["content"]
        
        bert_score = bertscore.compute(predictions=preds, references=refs, lang="en")
        bert_score = sum(bert_score["f1"]) / len(bert_score["f1"])
        bleurt_score = bleurt.compute(predictions=preds, references=refs)
        bleurt_score = sum(bleurt_score["scores"]) / len(bleurt_score["scores"])
        
        original_results.append({
            "user_inputs": user_inputs,
            "assistant_responses": assistant_responses,
            "model_response": responses,
            "original_model_response_cleaned": cleaned_response,
            "bert_score": bert_score,
            "bleurt_score": bleurt_score
        })
        # save original results
    with open(f"{output_path}/original_results_{timestamp}.json", "w") as f:
        json.dump(original_results, f)
    print("saved original results to: ", f"{output_path}/original_results_{timestamp}.json")
    
    
    
    # finetune model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = f"{output_path}/saved_model",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
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
        preds = cleaned_responses
        refs = assistant_responses["content"]
        
        bert_score = bertscore.compute(predictions=preds, references=refs, lang="en")
        bert_score = sum(bert_score["f1"]) / len(bert_score["f1"])
        bleurt_score = bleurt.compute(predictions=preds, references=refs)
        bleurt_score = sum(bleurt_score["scores"]) / len(bleurt_score["scores"])
            
        finetune_results.append({
            "user_inputs": user_inputs,
            "assistant_responses": assistant_responses,
            "finetune_model_response": response,
            "finetune_model_response_cleaned": cleaned_response,
            "bert_score": bert_score,
            "bleurt_score": bleurt_score
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






