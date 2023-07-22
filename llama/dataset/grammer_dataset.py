import pandas as pd
import numpy as np 

import torch
from torch.utils.data import Dataset

from datasets import load_dataset 
from pathlib import Path
from tqdm import tqdm 


class Grammer(Dataset):
    
    def __init__(self, tokenizer, csv_name):
        try:
            self.dataset = load_dataset("csv", data_files = {"train": [csv_name]}, delimiter =',')
        except Exception as e:
            raise e
        
        self.tokenizer = tokenizer
        
    def __len__(self):
        return self.dataset["train"].shape[0]
    
    def convert_to_features(self, example_batch):
        ## Create prompt and tokenize contexts and questions 
        
        input = example_batch["input"]
        target = example_batch["target"]
        
        prompt = f"correct this to standard english: {input} \n---\nCorrected: {target}"
        sample = self.tokenizer(prompt)
        
        return sample 
    
    def __getitem__(self, index):
        sample = self.convert_to_features(self.dataset["Train"][index])
        source_ids = sample["input_ids"]
        
        src_mask = sample["attention_mask"]
        
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "labels": source_ids.copy() 
        }
        

def get_dataset(dataset_config, tokenizer, csv_name = None):
    dataset = Grammer(csv_name = csv_name, tokenizer = tokenizer)
    return ConcatDataset(dataset, chunk_size = dataset_config.input_length)


class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size = 4096):
        self.dataset = dataset
        self.chunk_size = chunk_size
        
        self.samples = []
        
        buffer = {
            "input_ids" : [],
            "attention_mask" : [],
            "labels": []
        }
        
        for sample in tqdm(self.dataset, desc="Preprocessing dataset"):
            buffer = {k : v + sample[k] for k,v in buffer.items()}
            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k : v[:self.chunk_size] for k,v in buffer.items()})
                
    def __getitem__(self, index) -> Any:
        return self.samples[index]
    
    def __len__(self):
        return len(self.samples)
        