import json
from tqdm import tqdm
import os
import argparse
from glob import glob
import re
import string as STRING
from utils import is_obj_in_sbj, wc
from transformers import (
    AutoTokenizer
)

import string
import re

# https://github.com/huggingface/transformers/blob/758ed3332b219dd3529a1d3639fa30aa4954e0f3/src/transformers/data/metrics/squad_metrics.py
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def validate_alphanumeric_space_punctuation(text):
    return bool(re.match(f'^[ a-zA-Z0-9{STRING.punctuation}]+$', text))

def validate_len(string, tokenizer, maxlength):
    """
    return true if length of subwords is less or equal to maxlength
    return false otherwise
    """

    # Filter if string has a character which is not a alphanumeric, punctuation or space
    if validate_alphanumeric_space_punctuation(string) == False:
        return False

    # Filter ID
    if ":" in string:
        return False

    return len(tokenizer.tokenize(string)) <= maxlength

def refine_aliases(label, aliases):
    new_aliases = []
    for al in aliases:
        # normalize before comparision
        norm_al = normalize_answer(al)
        norm_label = normalize_answer(label)

        if norm_al not in norm_label: # not overlap
            new_aliases.append(al)
        # else:
        #     print(f"refine_aliases {al} {label}")

    return new_aliases

def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    assert input_dir != output_dir

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    input_files = glob(args.input_dir)

    # filter pids
    if args.pids:
        pids = list(dict.fromkeys(args.pids.split(",")))
        print(f"pids={pids}")
        input_files = [input_file for input_file in input_files if (input_file.split("/")[-1].split(".")[0] in pids)]

    org_num_triples = 0
    new_num_triples = 0
    MAX_LENGTH = args.max_length

    for input_file in tqdm(input_files):
        file_name = input_file.split("/")[-1]
        print(file_name)

        output_file = os.path.join(output_dir, file_name)

        # make tmp file for input
        with open(input_file) as f, open(output_file, 'w') as fo:
            num_lines = wc(input_file)

            for line in tqdm(f, total=num_lines):
                data = json.loads(line)
                org_num_triples += 1

                # filter when length of object is over max length
                sub_label = data['sub_label']
                if not validate_len(sub_label,tokenizer, MAX_LENGTH):
                    continue

                obj_label = data['obj_label']
                if not validate_len(obj_label,tokenizer, MAX_LENGTH):
                    continue

                # concat label and aliases
                data['sub_aliases'] = [al.strip() for al in data['sub_aliases'] if al.strip() != '']
                data['sub_aliases'] = [alias for alias in data['sub_aliases'] if validate_len(alias,tokenizer, MAX_LENGTH)]
                
                # filter obj alias which is either empty or over max length
                data['obj_aliases'] = [al.strip() for al in data['obj_aliases'] if al.strip() != '']
                data['obj_aliases'] = [al for al in data['obj_aliases'] if validate_len(al,tokenizer, MAX_LENGTH)]

                # filter overlap
                # 1) remove sbj_alias which overlaps with sbj
                sbj = data['sub_label']
                sbj_aliases = data['sub_aliases']
                sbj_aliases = refine_aliases(sbj, sbj_aliases)
                data['sub_aliases'] = sbj_aliases

                sbjs = [sbj] + sbj_aliases

                # 2) remove obj_alias which overlaps with obj
                obj = data['obj_label']
                obj_aliases = data['obj_aliases']
                obj_aliases = refine_aliases(obj, obj_aliases)
                data['obj_aliases'] = obj_aliases

                objs = [obj] + obj_aliases

                # 3) filter sbj-obj overlap
                is_overlap = False
                for sbj in sbjs:
                    result, _sbj, _obj = is_obj_in_sbj(sbj=sbj, objs=objs)
                    if result:
                        is_overlap = True
                        print(f"filter overlap! {_sbj}, {_obj}")
                        break

                if is_overlap: # filter if overlapped
                    continue

                # filter when sbj or obj has no name
                if data['sub_label'] == data['sub_uri']:
                    print(f"filter no sub label {data['sub_label']}")
                    continue

                if data['obj_label'] == data['obj_uri']:
                    print(f"filter no obj label {data['obj_label']}")
                    continue

                new_num_triples += 1
                output = json.dumps(data, ensure_ascii=False)
                fo.write(output + "\n")

        print(f"{new_num_triples}/{org_num_triples}={round(new_num_triples/org_num_triples,2)}")
        new_num_triples=0
        org_num_triples=0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
        default='./data/wikidata/triples'
    )
    parser.add_argument("--output_dir",
        default='./data/wikidata/triples_10sw'
    )
    parser.add_argument("--max_length",
        required=True,
        type=int
    )
    parser.add_argument("--pids",
        default=None
    )
    parser.add_argument("--model_name_or_path",
        default='bert-base-cased'
    )
    args = parser.parse_args()

    main(args)