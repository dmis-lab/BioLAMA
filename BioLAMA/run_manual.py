import torch
from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead
)

import json
from preprocessor import Preprocessor
from decoder import Decoder
from evaluator import Evaluator
import argparse
from glob import glob
import os
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default='bert-base-uncased')
    parser.add_argument("--prompt_path", default='./data/wikidata/prompts/manual.jsonl')
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--num_mask", type=int, default=10)
    parser.add_argument("--init_method", choices=['independent','order','confidence'], default='confidence')
    parser.add_argument("--iter_method", choices=['none','order','confidence'], default='none')
    parser.add_argument("--max_iter", type=int, default=10)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--pids", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--draft", action="store_true")

    args = parser.parse_args()
    if args.draft:
        args.output_dir = args.output_dir + "_draft"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'load model {args.model_name_or_path}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    lm_model = AutoModelWithLMHead.from_pretrained(args.model_name_or_path)
    if torch.cuda.is_available():
        lm_model = lm_model.cuda()

    # make sure this is only an evaluation
    lm_model.eval()
    for param in lm_model.parameters():
        param.grad = None

    print('prompt map loading')
    pid2prompt_meta = {}
    with open(args.prompt_path) as f:
        for line in f:
            l = json.loads(line)
            pid2prompt_meta[l['relation']] = {
                'template':l['template'],
            }

    print('load modules')
    preprocessor = Preprocessor(tokenizer=tokenizer, num_mask=args.num_mask)
    decoder = Decoder(
        model=lm_model,
        tokenizer=tokenizer,
        init_method=args.init_method,
        iter_method=args.iter_method,
        MAX_ITER=args.max_iter,
        BEAM_SIZE=args.beam_size,
        NUM_MASK=args.num_mask,
        BATCH_SIZE=args.batch_size,
    )
    evaluator = Evaluator(
        tokenizer=tokenizer,
    )

    files = glob(args.test_path)
    if args.pids: # e.g., P1050
        new_files = []
        pids = list(dict.fromkeys(args.pids.split(",")))
        for file in files:
            if file.split("/")[-2] in pids:
                new_files.append(file)
        files = new_files

    total_relation = 0
    pid2performance={}
    for file_path in files:
        pid = file_path.split("/")[-2]

        template = pid2prompt_meta[pid]['template']

        print(f'preprocess {file_path}')
        sentences, all_gold_objects, subjects, prompts, uuids = preprocessor.preprocess(
            file_path,
            template = template,
            draft=args.draft)

        print(f'decode {file_path}')
        all_preds_probs = decoder.decode(sentences, batch_size=args.batch_size) # topk predictions

        print(f'evaluate {file_path}')
        result = evaluator.evaluate(
            all_preds_probs = all_preds_probs,
            all_golds = all_gold_objects,
            subjects=subjects,
            prompts=prompts,
            inputs=sentences,
            uuids=uuids
        )

        # saving log
        with open(os.path.join(args.output_dir, pid + ".json"), 'w') as f:
            json.dump(result, f)

        if len(result['result']) == 0:
            print("nothing to print")
            continue

        total_relation += 1

        performance = result['performance']
        local_acc = performance['acc@k']

        logging_data ={}
        pid2performance[pid] = {}
        for k in range(args.beam_size):
            if k+1 in [1,5]:
                acc = local_acc[k]
                logging_data[f"{pid}_acc@{k+1}"] = acc * 100
                pid2performance[pid][f'acc@{k+1}'] = acc * 100

        print(f'performance of {pid}')
        print(logging_data)

    print("PID\tAcc@1\tAcc@5")
    print("-------------------------")
    acc1s = []
    acc5s = []
    for pid, performance in pid2performance.items():
        acc1 = performance['acc@1']
        acc5 = performance['acc@5']
        acc1s.append(acc1)
        acc5s.append(acc5)
        print(f"{pid}\t{round(acc1,2)}\t{round(acc5,2)}")
        
    print("-------------------------")
    print(f"MACRO\t{round(np.mean(acc1s),2)}\t{round(np.mean(acc5s),2)}")

if __name__ == '__main__':
    main()