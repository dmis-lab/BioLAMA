import json
from tqdm import tqdm
import os
import argparse
from glob import glob
from transformers import (
    AutoTokenizer
)
import random
from nltk.tokenize import sent_tokenize
import copy
from utils import wc
from collections import Counter

random.seed(0)

def flatten_list(l):
    new_list = []
    for element in l:
        if isinstance(element, str):
            new_list.append(element)
        elif isinstance(element, list):
            new_list.extend(element)

    return new_list

def save_to_jsonl(save_path, data):
    print(f"Saving to {save_path}")
    with open(file=save_path, mode='w') as f:
        for uuid in data:
            output = data[uuid]
            output = json.dumps(output, ensure_ascii=False)
            f.write(output + '\n')

def shuffle_and_truncate_triples(triples_to_probe, max_count=None):
    total = len(triples_to_probe)

    # shuffle
    l = list(triples_to_probe.items())
    random.shuffle(l)

    if max_count and (total > max_count):
        l = random.sample(l, k=max_count)

    triples_to_probe = dict(l)
    return triples_to_probe

def undersample(triples_to_probe, k=5):
    all_obj_uris = []
    samples = []
    for _, sample in triples_to_probe.items():
        obj_uris = sample['obj_uris']
        all_obj_uris += obj_uris
        samples.append(sample)

    # count per class (obj2count)
    obj2count = {k: v for k, v in sorted(Counter(all_obj_uris).items(), key=lambda item: item[1],reverse=True)}

    if len(list(obj2count.items())) < k:
        return {}
        
    # topk count to undersample
    try:
        topk_count = list(obj2count.items())[k-1][1]
    except IndexError:
        assert 0

    # undersample
    while True:
        # init for iteration
        obj2count = {k: v for k, v in sorted(obj2count.items(), key=lambda item: item[1],reverse=True)}
        majority_obj,majority_count = list(obj2count.items())[0] # majority object
        if majority_count <= topk_count:
            break

        random.shuffle(samples)

        for i, sample in enumerate(samples):
            obj_uris = sample['obj_uris']
            if majority_obj in obj_uris:
                for obj_uri in obj_uris:
                    obj2count[obj_uri] -= 1
                del samples[i] # remove this sample
                break

    # restore samples to triples_to_probe
    new_triples_to_probe = {}
    for sample in samples:
        uuid = sample['uuid']
        new_triples_to_probe[uuid] = sample
    print(f"undersample {len(triples_to_probe)}->{len(new_triples_to_probe)}")
    return new_triples_to_probe

def split_train_dev_test(triples_to_probe):
    total = len(triples_to_probe)

    # 4:1:5 = train:dev:test
    trainset = dict(list(triples_to_probe.items())[:int(total*0.4)])
    devset = dict(list(triples_to_probe.items())[int(total*0.4):int(total*0.5)])
    testset = dict(list(triples_to_probe.items())[int(total*0.5):])
    print(list(trainset.items())[0])
    print(list(devset.items())[0])

    print(list(testset.items())[0])
    print(f"total len:{len(triples_to_probe)}")
    print(f"trainset={len(trainset)} devset={len(devset)} testset={len(testset)}")
    return trainset, devset, testset

def main(args):
    input_path = args.input_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    assert input_path != output_dir

    if args.sub_obj_type_path:
        with open(args.sub_obj_type_path) as f:
            sub_obj_types = json.load(f)
    else:
        sub_obj_types ={}

    if args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    input_files = glob(args.input_path)
    unique_objs = set()

    for input_file in tqdm(input_files):
        file_name = input_file.split("/")[-1]
        property_name = file_name.replace(".jsonl","")

        print(file_name)
        triples_to_probe = {}

        # make tmp file for input
        with open(input_file) as f:
            num_lines = wc(input_file)
            for line in tqdm(f, total=num_lines):
                data = json.loads(line)
                pid = data['predicate_id']
                uuid = '-'.join([data['sub_uri'], pid])

                # filter noisy type sample
                if pid in sub_obj_types:
                    sub_types = sub_obj_types[pid]['sub_types']
                    obj_types = sub_obj_types[pid]['obj_types']
                    if (data['sub_type'] not in sub_types) or (data['obj_type'] not in obj_types):
                        continue

                if uuid not in triples_to_probe:
                    triples_to_probe[uuid] = {
                        'uuid': uuid,
                        'predicate_id': pid,
                        'sub_uri': data['sub_uri'],
                        'sub_label': data['sub_label'],
                        'sub_type': data['sub_type'] if 'sub_type' in data else '',
                        'sub_aliases':data['sub_aliases'],
                        'obj_uris': [],
                        'obj_labels': [],
                        'obj_types': [],
                        'obj_aliases':[],
                    }

                if data['obj_uri'] in triples_to_probe[uuid]['obj_uris']:
                    continue
                
                # for multiple answers
                triples_to_probe[uuid]['obj_uris'].append(data['obj_uri'])
                triples_to_probe[uuid]['obj_labels'].append(data['obj_label'])
                triples_to_probe[uuid]['obj_aliases'].append(data['obj_aliases'])
                if 'obj_type' in data:
                    triples_to_probe[uuid]['obj_types'].append(data['obj_type'])

            # split train/dev/test
            triples_to_probe = shuffle_and_truncate_triples(triples_to_probe, max_count=args.max_count)

            # filter min_count
            if args.min_count and len(triples_to_probe) < args.min_count:
                print(f"filter this cause {len(triples_to_probe)} < {args.min_count}")
                continue
            
            # undersample for balancing
            triples_to_probe = undersample(triples_to_probe) 

            # filter min_count
            if args.min_count and len(triples_to_probe) < args.min_count:
                print(f"filter this cause {len(triples_to_probe)} < {args.min_count}")
                continue

            trainset, devset, testset = split_train_dev_test(triples_to_probe)

            sub_output_dir = os.path.join(output_dir, property_name)
            os.makedirs(sub_output_dir, exist_ok=True)

            # for length stat
            for value in triples_to_probe.values():
                objs = copy.deepcopy(value['obj_labels'])
                objs.extend(copy.deepcopy(value['obj_aliases']))
                objs = flatten_list(objs)

                for obj in objs:
                    unique_objs.add(obj)

            # save train,dev,test
            train_path = os.path.join(sub_output_dir, 'train.jsonl')
            dev_path = os.path.join(sub_output_dir, 'dev.jsonl')
            test_path = os.path.join(sub_output_dir, 'test.jsonl')

            save_to_jsonl(save_path=train_path, data=trainset)
            save_to_jsonl(save_path=dev_path, data=devset)
            save_to_jsonl(save_path=test_path, data=testset)

    # do this with new triples_to_probe (after truncate, undersample)
    obj_lengths = {}
    for obj in tqdm(iterable=unique_objs, desc="Getting obj lengths", total=len(unique_objs)):
        tokenized_obj = tokenizer.tokenize(obj)
        obj_len = len(tokenized_obj)

        try:
            obj_lengths[obj_len] += 1
        except KeyError:
            obj_lengths[obj_len] = 1

    obj_lengths = sorted([(length, count) for length, count in obj_lengths.items()], key=lambda x: x[0])
    print()
    print("Obj Length: Count")
    for pair in obj_lengths:
        print(f'{pair[0]}, {pair[1]}')
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path")
    parser.add_argument("--entity_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--min_count", type=int, default=500)
    parser.add_argument("--max_count", type=int, default=2000)
    parser.add_argument("--sub_obj_type_path", default=None)
    parser.add_argument('--model_name_or_path')
    args = parser.parse_args()

    main(args)
