from datasets import load_dataset
import json
import os

def convert_HF_dataset(dataset_name):
    if dataset_name == "empathetic_dialogues_llm":
        dataset_path = "Estwld/empathetic_dialogues_llm"
    dataset = load_dataset(dataset_path, trust_remote_code=True)
    # Flatten all conversations
    flattened = []
    
    def preprocess_conversation(example):
        turns = example["conversations"]
        history = []
        samples = []
        for i in range(len(turns) - 1):
            if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
                input_text = "\n".join([f"{t['role'].capitalize()}: {t['content']}" for t in history + [turns[i]]])
                samples.append({
                    "system": "Respond empathetically in the following conversation.",
                    "input": input_text,
                    "output": turns[i + 1]["content"]
                })
                history.extend([turns[i], turns[i + 1]])
        return samples
    
    for ex in dataset["train"]:
        flattened.extend(preprocess_conversation(ex))

    os.makedirs(dataset_name, exist_ok=True)
    with open(f"{dataset_name}/train.json", "w") as f:
        for sample in flattened:
            json.dump(sample, f)
            f.write("\n")
            
def convert_HF_dataset_to_Xtuner(dataset_name):
    if dataset_name == "empathetic_dialogues_llm":
        dataset_path = "Estwld/empathetic_dialogues_llm"
    dataset = load_dataset(dataset_path, trust_remote_code=True)
    # Flatten all conversations
    flattened = []
    
    def preprocess_conversation(example):
        turns = example["conversations"]
        samples = {}
        samples["conversation"] = []
        for i in range(len(turns) - 1):
            if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
                input_text = turns[i]["content"]
                output_text = turns[i + 1]["content"]
                system = ""
                if i == 0:
                    system = "Respond empathetically in the following conversation."
                samples["conversation"].append({
                    "system": system,
                    "input": input_text,
                    "output": output_text
                })
        return samples
    
    for ex in dataset["train"]:
        flattened.append(preprocess_conversation(ex))
    print(flattened[0])

    os.makedirs(dataset_name, exist_ok=True)
    with open(f"{dataset_name}/train.json", "w") as f:
        for sample in flattened:
            json.dump(sample, f)
            f.write("\n")
            
            
def prep_data_unsloth():
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

