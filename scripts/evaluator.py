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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='bert-base-cased')
    parser.add_argument("--draft", action='store_true')
    parser.add_argument('--root_dir', default='../data/BioTREx/triples_10sw_agg_20210412/*/majority.tsv', type=str)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    evaluator = Evaluator(tokenizer=tokenizer)

    if args.draft:
        # all_preds : [sample size x max_length x predictions]
        all_preds = [
                    [['doxycycline anhydrous','ceftriaxone','ibuprofen','tyty']], # predictions of sample 1
                    [['doxycycline anhydrous','ceftriaxone','dasfdasf','azithromycin']], # predictions of sample 2
                    ]

        # all_golds : [sample size x golden answers]
        all_golds = [
            ['ibuprofen','tota'], # obj labels + obj_aliases of sample 1
            ['azithromycin','tota'], # obj labels + obj_aliases of sample 2
        ]

        # fill dummys in
        all_probs = []
        inputs = []
        for sample in all_preds:
            probs = []
            inps = []
            for ml in sample:
                probs.append([0]*len(ml))
                inps.append('')
            all_probs.append(probs)
            inputs.append(inps)

        result = evaluator.evaluate(
            all_preds=all_preds,
            all_probs=all_probs,
            all_golds=all_golds,
            subjects=['']*len(all_preds),
            prompts='',
            inputs=inputs,
        )
    else:
        # all_preds : from majority.tsv
        # all_golds : from test.jsonl
        majority_files = glob.glob(args.root_dir)
        property_subdirs = [os.path.dirname(f) for f in majority_files]
        test_files = [os.path.join(subdir, 'test.jsonl') for subdir in property_subdirs]

        all_performances = {}
        topk = 5
        for file_tuple in tqdm(iterable=zip(property_subdirs, majority_files, test_files), desc="Getting performances", total=len(test_files)):
            property_subdir = file_tuple[0]
            majority_file = file_tuple[1]
            test_file = file_tuple[2]

            property_id = property_subdir.split('/')[-1]

            with open(file=majority_file) as f:
                tsv_reader = csv.reader(f, delimiter='\t', quotechar='"')
                majority_data = [row for row in tsv_reader]
                majority_data = [x[0] for x in majority_data]

            with open(file=test_file) as f:
                test_data = [json.loads(line) for line in f]

            num_samples = len(test_data)
            all_preds = [[majority_data[:topk]]] * num_samples
            all_golds = []
            uuids= []

            for sample in test_data:
                all_golds_ = sample['obj_labels']
                all_golds_.extend(sample['obj_aliases'])
                all_golds_ = flatten_list(all_golds_)
                all_golds.append(all_golds_)
                uuids.append(sample['uuid'])

            # fill dummys in
            all_probs = []
            inputs = []
            for sample in all_preds:
                probs = []
                inps = []
                for ml in sample:
                    probs.append([0]*len(ml))
                    inps.append('')
                all_probs.append(probs)
                inputs.append(inps)

            result = evaluator.evaluate(
                all_preds=all_preds,
                all_probs=all_probs,
                all_golds=all_golds,
                subjects=['']*len(all_preds),
                prompts='',
                uuids=uuids,
                inputs=inputs,
            )

            # save result
            dataset = args.root_dir.split("/")[-4]
            output_dir = os.path.join('./lab/output', f'{dataset}_majority')
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir,f'{property_id}.json')
            with open(output_path, 'w') as f:
                json.dump(result, f)

            all_performances[property_id] = result['performance']['acc_ol@k']

    performances_np = np.array(list(all_performances.values()))
    performances_macro_avg = np.mean(performances_np, axis=0)

    all_performances['macro_avg'] = performances_macro_avg.tolist()

    for pid in all_performances:
        perf = all_performances[pid]
        # print(f"{pid},{round(perf[0]*100,2)},{round(perf[9]*100,2)}")
        print(f"{pid},{round(perf[0]*100,2)},{round(perf[topk-1]*100,2)}")

    # with open(file='/hdd1/seokwon/BioLAMA/majority_performances.json', mode='w') as f:
    #     json.dump(obj=all_performances, fp=f, indent=2)
