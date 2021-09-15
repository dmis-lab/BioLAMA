import argparse
import json
import os
import traceback

from tqdm import tqdm


def main(args):
    rel_path = args.rel_path
    conso_path = args.conso_path
    sty_path = args.sty_path

    # load cui_dict
    cui2names = {}
    with open(file=conso_path) as f:
        for line in tqdm(f, desc="Creating cui2names"):
            row = line.strip().split('|')
            cui = row[0]
            lang = row[1]
            name = row[14]

            if lang != 'ENG':
                continue

            if cui not in cui2names:
                cui2names[cui] = []

            cui2names[cui].append(name)

    # load cui2types
    cui2types = {}
    with open(file=sty_path) as f:
        for line in tqdm(f, desc="Creating cui2types"):
            row = line.strip().split('|')
            cui = row[0]
            type_ = row[3]

            if cui not in cui2types:
                cui2types[cui] = []

            cui2types[cui].append(type_)

    # load triples
    relation2id = {}
    relation2triples = {}
    relation_idx = 0
    with open(file=rel_path) as f:
        for line in tqdm(f, desc="Creating data"):
            row = line.strip().split('|')

            try:
                subj_cui = row[0]
                subj_label = cui2names[subj_cui][0]
                subj_aliases = cui2names[subj_cui][1:]
                subj_types = cui2types[subj_cui]

                obj_cui = row[4]
                obj_label = cui2names[obj_cui][0]
                obj_aliases = cui2names[obj_cui][1:]
                obj_types = cui2types[obj_cui]
            except KeyError:
                continue
            except Exception as e:
                print(e)
                traceback.print_exc()
                raise e


            relation = row[7]

            if relation == '':
                continue

            if relation not in relation2id:
                relation2id[relation] = f'UR{relation_idx}'
                relation_idx += 1

            relation_id = relation2id[relation]

            if relation_id not in relation2triples:
                relation2triples[relation2id[relation]] = []

            template = {'predicate_id': relation_id,
                        'predicate_name': relation,
                        'sub_uri': subj_cui,
                        'sub_type': subj_types,
                        'sub_label': subj_label,
                        'sub_aliases': subj_aliases,
                        'obj_uri': obj_cui,
                        'obj_type': obj_types,
                        'obj_label': obj_label,
                        'obj_aliases': obj_aliases}

            relation2triples[relation_id].append(template)

    rel_counts = {}
    for relid, triples in relation2triples.items():
        assert relid not in rel_counts
        rel_counts[relid] = len(triples)

    for relation_id, triples in tqdm(relation2triples.items(), desc=f"Saving data in {args.output_dir}"):
        save_filename = os.path.join(args.output_dir, relation_id + '.jsonl')
        with open(file=save_filename, mode='w') as f:
            for triple in triples:
                json.dump(obj=triple, fp=f)
                f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rel_path', default='2020AB/META/MRREL.RRF', type=str)
    parser.add_argument('--conso_path', default='2020AB/META/MRCONSO.RRF', type=str)
    parser.add_argument('--sty_path', default='2020AB/META/MRSTY.RRF', type=str)
    parser.add_argument('--output_dir', default='data/umls/triples', type=str)

    args = parser.parse_args()

    main(args)
