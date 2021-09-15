import argparse
import json
from transformers import (
    BertTokenizer,
    RobertaTokenizer
)
import random

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append(ind)
    return results

class Preprocessor():
    """
    Object to perform dynamic masking of object tokens in given prompts.
    "[X] causes diseases such as [Y]." -> "[X] causes diseases such as [MASK] * len(Y)."
    """
    def __init__(self, tokenizer, num_mask):
        self.tokenizer = tokenizer # bert tokenizer

        self.MASK_IDX = self.tokenizer.mask_token_id
        self.PAD_IDX = self.tokenizer.pad_token_id
        self.UNK_IDX = self.tokenizer.unk_token_id

        if isinstance(tokenizer, BertTokenizer):
            self.mask_token = '[MASK]'
            self.pad_token = '[PAD]'
            self.unk_token = '[UNK]'
            assert self.tokenizer.convert_ids_to_tokens(self.MASK_IDX) == self.mask_token
            assert self.tokenizer.convert_ids_to_tokens(self.PAD_IDX) == self.pad_token
            assert self.tokenizer.convert_ids_to_tokens(self.UNK_IDX) == self.unk_token

        elif isinstance(tokenizer, RobertaTokenizer): 
            self.mask_token = '<mask>'
            self.pad_token = '<pad>'
            self.unk_token = '<unk>'
            assert self.tokenizer.convert_ids_to_tokens(self.PAD_IDX) == self.pad_token
            assert self.tokenizer.convert_ids_to_tokens(self.UNK_IDX) == self.unk_token
        else:
            print(f"tokenizer type = {type(tokenizer)}")
            import pdb ; pdb.set_trace()        # get num_mask as an argument
        
        self.num_mask = num_mask

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def preprocess_single(self, subject, prompt):
        sentences = []

        for i in range(1, self.num_mask + 1):
            sentence = prompt
            
            # fill in subject
            sentence = sentence.replace('[X]', subject)
            mask_sequence = (f"{self.mask_token} " * i).strip()
            sentence = sentence.replace('[Y]', mask_sequence)
            sentences.append(sentence)
            
        return sentences

    # input as a sentence
    # e.g., Adamantinoma of Long Bones has symptoms such as [Y].
    def preprocess_single_sent(self, sentence):
        original_sent = sentence
        sentences = []

        for i in range(1, self.num_mask + 1):
            # fill in subject
            mask_sequence = (f"{self.mask_token} " * i).strip()
            sentence = original_sent.replace('[Y]', mask_sequence)
            sentences.append(sentence)

        return sentences

    def preprocess(self, data_path, template, draft=False, shuffle_subject=False, replace_sub_syn=False):
        """
        Masks out tokens corresponding to objects.

        Example
        -------
        "meprobamate cures diseases such as headache ." -> "meprobamate cures diseases such as [MASK] ."
        """

        all_masked_sentences = []
        all_gold_objects = []
        subjects=[]
        prompts=[]
        uuids=[]

        # load temp_subjects
        temp_subjects = []
        with open(file=data_path, mode='r') as f:
            for line in f:
                sample = json.loads(line)
                temp_subjects.append(sample['sub_label'])
        random.shuffle(temp_subjects)

        # load data
        with open(file=data_path, mode='r') as f:
            index = 0
            for line in f:
                sample = json.loads(line)

                prompt = template
                if shuffle_subject: # for shuffle subject test
                    subject = temp_subjects[index]
                else:
                    subject = sample['sub_label']

                if 'obj_labels' in sample:
                    objects = sample['obj_labels']
                elif 'obj_label' in sample:
                    objects = [sample['obj_label']]
                else:
                    assert 0

                # replace subject to synonym
                if replace_sub_syn:
                    if len(sample['sub_aliases']):
                        subject = random.sample(sample['sub_aliases'],k=1)[0]

                if 'obj_aliases' in sample:
                    objects += [a for al in sample['obj_aliases'] for a in al]

                # lowercase
                lower_objects = list(dict.fromkeys([obj.lower() for obj in objects]))

                sentences = self.preprocess_single(
                    subject=subject,
                    prompt=prompt
                )

                all_masked_sentences.append(sentences)
                all_gold_objects.append(lower_objects)
                subjects.append(subject)
                prompts.append(prompt)
                uuids.append(sample['uuid'])
                # print sentences with mask for debugging
                if index <= 2 - 1:
                    print(sentences)

                if draft and index>=16 -1 :
                    break
                index += 1

        return all_masked_sentences, all_gold_objects, subjects, prompts, uuids