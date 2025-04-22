import argparse
import numpy as np
import os
import torch
from datetime import datetime
from datasets import load_dataset
import json
from utils import *

def main(config):
    # Current timestamp without spaces
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)
    if config.dataset=="empathetic_dialogues_llm":
        dataset_path = "Estwld/empathetic_dialogues_llm"
        dataset = load_dataset(dataset_path, trust_remote_code=True)
        # print(dataset['train'][0:5])
        
    if not os.path.exists(config.dataset):
        print("create dataset json file")
        convert_HF_dataset_to_Xtuner(config.dataset)  
    


    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/vast/jz5952/THuman_2.0_2D_based_segmentation')
    parser.add_argument('--dataset', default='empathetic_dialogues_llm')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--batch_size', type=int, default=16)
    config = parser.parse_args()
    main(config)