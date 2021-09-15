import json
import csv
import os
import argparse
from glob import glob


def get_statistics_of_file(path):
    count = 0
    sub_types = []
    obj_types = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            count+= 1

            if ('sub_type' in data) and ('obj_types' in data):
                local_sub_type = data['sub_type']
                local_obj_types = data['obj_types']

                sub_types.append(local_sub_type)
                obj_types += local_obj_types

    return count, sub_types, obj_types


def get_obj_counts(path):
    obj_counts = []
    with open(file=path) as f:
        for line in f:
            data = json.loads(line)
            obj_labels = data['obj_labels']
            obj_counts.append(len(obj_labels))

    return obj_counts


"""
Input: Triples
Output: PID/Property Name/Count/Pair Type
"""
def main(args):
    input_dirs = glob(args.data_dir)
    if args.pids: # P1050
        new_dirs = []
        pids = list(dict.fromkeys(args.pids.split(",")))
        for file in input_dirs:
            if file.split("/")[-1] in pids:
                new_dirs.append(file)
        input_dirs = new_dirs

    # pid2name
    if args.property_path:
        pid2name = {}
        with open(args.property_path) as f:
            rdr = csv.reader(f, delimiter='\t')
            r = list(rdr)
            for pid, name in r:
                pid2name[pid] = name

    # qid2type
    if args.type_path:
        with open(args.type_path) as f:
            qid2type = json.load(f)
    else:
        qid2type = None

    all_obj_counts = []
    print(f"PID\tTRAIN\tDEV\tTEST")
    for input_dir in input_dirs:
        pid = input_dir.split("/")[-1]
        train_file = os.path.join(input_dir, 'train.jsonl')
        dev_file = os.path.join(input_dir, 'dev.jsonl')
        test_file = os.path.join(input_dir, 'test.jsonl')

        sub_types = []
        obj_types = []
        train_count, train_sub_types, train_obj_types = get_statistics_of_file(train_file)
        dev_count, dev_sub_types, dev_obj_types = get_statistics_of_file(dev_file)
        test_count, test_sub_types, test_obj_types = get_statistics_of_file(test_file)

        if qid2type:
            sub_types = train_sub_types + dev_sub_types + test_sub_types
            obj_types = train_obj_types + dev_obj_types + test_obj_types
            sub_types = [qid2type[st] for st in sub_types]
            obj_types = [qid2type[ot] for ot in obj_types]

            sub_types = list(set(sub_types))
            obj_types = list(set(obj_types))

        print(f"{pid}\t{train_count}\t{dev_count}\t{test_count}")

        train_obj_counts = get_obj_counts(train_file)
        dev_obj_counts = get_obj_counts(dev_file)
        test_obj_counts = get_obj_counts(test_file)

        all_obj_counts.extend(train_obj_counts)
        all_obj_counts.extend(dev_obj_counts)
        all_obj_counts.extend(test_obj_counts)

    avg_obj_counts = sum(all_obj_counts) / len(all_obj_counts)
    # print(f"Average number of objects per subject: {avg_obj_counts:.4}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
        default='../data/wikidata/triples_processed'
    )
    parser.add_argument("--pids",
        default=None
    )
    parser.add_argument("--property_path",
        default=None
    )
    parser.add_argument("--type_path",
        default=None
    )

    args = parser.parse_args()

    main(args)
