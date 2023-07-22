from torch.util.data import Dataset
import pandas as pd
import itertools
from tqdm import tqdm 

class DeIdentificationDataset(Dataset):
    def __init__(self, tokenizer, csv_name):
        super().__init__()
        df = pd.read_csv('deidentified.csv')
        inp = [i for i in (df['text'].apply(lambda x: x.split('\n')))]
        inp = [i for i in itertools.chain(*inp)]
                 
        
        target = [i for i in (df['deidentified'].apply(lambda x: x.split('\n')))]
        target = [i for i in itertools.chain(*target)]
        
        self.inp = list(filter(lambda x: len(x)>0, inp))
        self.target = list(filter(lambda x: len(x)>0, target))
        assert len(inp) == len(target)
        
    def convert_to_features(self, input, target):
        ## Create prompt and tokenize contexts and questions 
        
        prompt = f"deidentify personal information: {input} \n---\nCorrected: {target}"
        sample = self.tokenizer(prompt)
        
        return sample
        
    def __len__(self):
       return len(self.inp)
    
    def __getitem__(self, index):
        input = self.inp[index]
        target = self.target[index]
        
        sample = self.convert_to_features(input, target)
        source_ids = sample["input_ids"]
        
        src_mask = sample["attention_mask"]
        
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "labels": source_ids.copy() 
        }
        
        
def get_dataset(dataset_config, tokenizer, csv_name = None):
    dataset = DeIdentificationDataset(csv_name = csv_name, tokenizer = tokenizer)
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
                
    def __getitem__(self, index):
        return self.samples[index]
    
    def __len__(self):
        return len(self.samples)