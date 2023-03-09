from torch.utils.data import Dataset
import json
from torch.utils.data import Dataset
import torch
import random
from transformers import (
    BertTokenizer, 
    RobertaTokenizer
)

class FactDataset(Dataset):
    def __init__(self, input_file, prompt_token_len, tokenizer, template):
        print(f"FactDataset! input_file={input_file} prompt_token_len={prompt_token_len}")
        self.tokenizer = tokenizer
        self.prompt_token_len = prompt_token_len
        self.template = template
        
        self.data = self.load_data(
            input_file=input_file
        )
        # shuffle data
        random.shuffle(self.data)

        if isinstance(tokenizer, BertTokenizer):
            self.mask_token = '[MASK]'
            self.prompt_token = '[unused1]'
        elif isinstance(tokenizer, RobertaTokenizer): 
            self.mask_token = '<mask>'
            self.prompt_token = '¤' # hotfix for biolm
        else:
            print(f"tokenizer type = {type(tokenizer)}")
            assert 0
    
    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx): 
        sub, obj = self.data[idx]

        mask_idx = self.tokenizer.convert_tokens_to_ids(self.mask_token)
        prompt_idx = self.tokenizer.convert_tokens_to_ids(self.prompt_token)

        assert mask_idx != prompt_idx

        input_sentence, tokenized_obj = self.convert_template_to_input(sub, obj)
        input_ids = torch.tensor(self.tokenizer.encode(input_sentence))
        
        if idx <5:
            print(f"{input_sentence}, {input_ids}")

        # get template tokens
        # replace 
        
        mask_ind = input_ids.eq(mask_idx)
        input_ids[mask_ind] = torch.tensor(tokenized_obj)
        mask_ind = mask_ind.long()

        return input_ids, mask_ind

    def convert_template_to_input(self, sub, obj):
        tokenized_obj = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(obj))
        obj_len = len(tokenized_obj)

        # fill subject and mask
        input_sentence = self.template.replace('[X]', sub)
        input_sentence = input_sentence.replace('[Y]', f' {self.mask_token} ' * obj_len)

        return input_sentence, tokenized_obj
        
    def load_data(self, input_file):
        data = []

        with open(input_file) as f:
            for line in f:
                sample = json.loads(line)
                sub_label = sample['sub_label']

                for obj_label, obj_aliases in zip(sample['obj_labels'],sample['obj_aliases']):
                    data.append((sub_label,obj_label))
                
        return data