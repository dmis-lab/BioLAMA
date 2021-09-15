import argparse
import csv
import json
import os

import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BertTokenizer,
    RobertaTokenizer
)
import glob
from utils import (
    compute_exact
)

def flatten_list(l):
    new_list = []

    for element in l:
        if isinstance(element, str):
            new_list.append(element)
        elif isinstance(element, list):
            if element != []:
                new_list.extend(element)

    return new_list


def get_raw_score(pred, golds):
    em = max(compute_exact(a, pred) for a in golds)

    return em

class Evaluator():
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

        if isinstance(tokenizer, BertTokenizer):
            self.mask_token = '[MASK]'
        elif isinstance(tokenizer, RobertaTokenizer):
            self.mask_token = '<mask>'
        elif tokenizer == None:
            self.mask_token = ''
        else:
            print(f"tokenizer type = {type(tokenizer)}")
            assert 0

    def check_multi(self, golds):
        token_nums = [len(self.tokenizer.tokenize(gold)) for gold in golds]
        if 1 in token_nums:
            return False
        else:
            return True

    def evaluate_preds_for_single_sample(self, preds, golds):
        """
        input: prediction strings, gold strings
        output: em score, f1 score
        """

        ems = []
        for pred in preds:
            em = get_raw_score(pred, golds)
            ems.append(em)

        return max(ems)

    def evaluate(self, all_preds_probs, all_golds, subjects, prompts, inputs, uuids):
        """
        input: prediction strings for all samples, gold strings for all samples
        output: accuracy
        """
        result = []

        topk = len(all_preds_probs[0])

        print(f"topk={topk}")

        hits = [0]*topk # for topk

        total = 0

        if not subjects:
            subjects = ['']*len(all_preds_probs)
        if not prompts:
            prompts = ['']*len(all_preds_probs)

        assert len(all_preds_probs) == len(all_golds)
        for i, preds_probs in tqdm(enumerate(all_preds_probs), total=len(all_preds_probs)):
            # probs = all_probs[i]
            golds = all_golds[i]
            subject = subjects[i]
            prompt = prompts[i]
            _input = inputs[i]
            uuid = uuids[i]

            temp={
                'uuid': uuid,
                'subject': subject,
                'prompt': prompt,
                'input': _input[0].replace(f'{self.mask_token}','[Y]'),
                'golds': golds,
            }

            max_hit = 0

            topk_preds_probs = preds_probs # see all for logging
            topk_preds = [t[0] for t in topk_preds_probs]

            temp['corrected_preds'] = []
            for k, kth_pred in enumerate(topk_preds):
                _hit = self.evaluate_preds_for_single_sample(preds=[kth_pred], golds=golds)
                if _hit:
                    temp['corrected_preds'].append((kth_pred, k))
                if k < topk:
                    max_hit = max(_hit, max_hit) # if previous max em is 1, follow it in this k
                    hits[k] += max_hit
            temp['preds'] = preds_probs

            total += 1
            result.append(temp)

        final_accs = []

        for hit in hits:
            final_accs.append(round(hit/(total+1e-7),5))

        performance = {
            'acc@k': final_accs,
        }

        result = {'result': result, 'performance': performance}

        return result