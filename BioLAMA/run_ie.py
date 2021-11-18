import argparse
import glob
import json

import numpy as np
import stanza
from tqdm import tqdm

import best
from evaluator import Evaluator
import os

def flatten_list(l):
    new_list = []

    for element in l:
        if isinstance(element, str):
            new_list.append(element)
        elif isinstance(element, list) and (element != []):
            new_list.extend(element)

    return new_list

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--topN', type=int, default=50)
    parser.add_argument('--draft', action='store_true')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    task = args.test_path.split("/")[-4]

    evaluator = Evaluator()

    stanza.download('en', package='craft')
    stanza_pipeline = stanza.Pipeline('en', package='craft')

    if task == 'ctd':
        PROPERTY_NAMES = {'CD1': "therapeutic",
                          'CD2': "marker mechanism",
                          'CG1': "decreases expresion",
                          'CG17': "increases expression",
                          'CG18': "increases expression",
                          'CG2': "decreases activity",
                          'CG21': "increases phosphorylation",
                          'CG4': "increases activity",
                          'CG6': "decreases expression",
                          'CG9': "affects binding",
                          'CP1': "decreases",
                          'CP2': "increases",
                          'CP3': "affects",
                          'GD1': "marker mechanism",
                          'GP1': "association"}

        PROPERTY_OBJ_TYPES = {'CD1': 'disease',
                              'CD2': 'disease',
                              'CG1': 'gene',
                              'CG17': 'gene',
                              'CG18': 'gene',
                              'CG2': 'gene',
                              'CG21': 'gene',
                              'CG4': 'gene',
                              'CG6': 'gene',
                              'CG9': 'gene',
                              'CP1': 'disease',
                              'CP2': 'disease',
                              'CP3': 'disease',
                              'GD1': 'disease',
                              'GP1': 'pathway'}
    elif task == 'umls':
        PROPERTY_NAMES = {'UR44': "may be prevented by",
                          'UR45': "may be treated by",
                          'UR48': "physiologic effect of",
                          'UR49': "mechanism of action of",
                          'UR50': "therapeutic class of",
                          'UR116': "clinically associated with",
                          'UR124': "may treat",
                          'UR173': "causative agent of",
                          'UR180': "is finding of disease",
                          'UR211': "biological process involves gene product",
                          'UR214': "cause of",
                          'UR221': "gene mapped to disease",
                          'UR254': "may be molecular abnormality of disease",
                          'UR256': "may be molecular abnormality of disease",
                          'UR588': "process involves gene",
                          'UR625': "disease has associated gene"}

        PROPERTY_OBJ_TYPES = {'UR44': 'disease',
                              'UR45': 'disease',
                              'UR48': 'disease',
                              'UR49': 'all entity type',
                              'UR50': 'chemical compound',
                              'UR116': 'disease',
                              'UR124': 'chemical compound',
                              'UR173': 'all entity type',
                              'UR180': 'all entity type',
                              'UR211': 'all entity type',
                              'UR214': 'disease',
                              'UR221': 'gene',
                              'UR254': 'disease',
                              'UR256': 'all entity type',
                              'UR588': 'disease',
                              'UR625': 'disease'}
    elif task == 'wikidata':
        PROPERTY_NAMES = {'P2176': "drug used for treatment",
                          'P2175': "medical condition treated",
                          'P780': "symptoms",
                          'P2293': "genetic association",
                          'P4044': "therapeutic area"}

        PROPERTY_OBJ_TYPES = {'P2176': 'chemical compound',
                              'P2175': 'disease',
                              'P780': 'disease',
                              'P2293': 'disease',
                              'P4044': 'disease'}
    else:
        print(f"not supporting task:{task}")
        assert 0

    test_files = glob.glob(args.test_path)
    selected_data = {}
    for test_file in test_files:
        pid = test_file.split('/')[-2]

        with open(file=test_file) as f:
            data = [json.loads(line) for line in f]
            selected_data[pid] = data

    ie_result = {}
    pid2performance={}
    pbar = tqdm(iterable=selected_data.items(), total=len(selected_data))
    for pid, data in pbar:
        pbar.set_description(desc=f"Running IE for {pid}")

        ie_result[pid] = {'acc@1': 0.0, 'acc@5': 0.0}

        all_preds_probs = []
        all_golds = []
        uuids = []

        if args.draft:
            data = data[:5]

        for idx, row in tqdm(enumerate(data), total=len(data)):
            subj_label = row['sub_label']
            uuid = row['uuid']

            objs = row['obj_labels']
            objs.extend(row['obj_aliases'])
            objs = flatten_list(l=objs)

            property_name = PROPERTY_NAMES[pid]
            doc = stanza_pipeline(property_name)
            stanza_words = doc.sentences[0].words
            valid_words = [x.lemma for x in stanza_words if x.pos in ['NOUN', 'VERB', 'ADJ']]
            assert len(valid_words) != 0

            query = f"({subj_label}) AND ({' '.join(valid_words)})"

            # if idx<5:
            #     print(f"{pid}:{query}")
            ie_query = best.BESTQuery(query, topN=args.topN, filterObjectName=PROPERTY_OBJ_TYPES[pid])
            result = best.getRelevantBioEntities(ie_query)

            try:
                result_names = [(x['entityName'], x['score']) for x in result]
            except (KeyError, TypeError):
                result_names = []

            result_names.extend([('', 0.0)] * (10 - len(result_names)))

            all_preds_probs.append(result_names)
            all_golds.append(objs)
            uuids.append(uuid)

        inputs = []
        for sample in all_preds_probs:
            inputs_ = []

            for _ in sample:
                inputs_.append('')

            inputs.append(inputs_)

        result = evaluator.evaluate(all_preds_probs=all_preds_probs, all_golds=all_golds, subjects=[''] * len(all_preds_probs), prompts='', uuids=uuids, inputs=inputs)

        # saving log
        log_file = os.path.join(args.output_dir, pid + ".json")
        print(f"save {log_file}")
        with open(log_file, 'w') as f:
            json.dump(result, f)


        acc_at_1 = result['performance']['acc@k'][0]
        acc_at_5 = result['performance']['acc@k'][4]

        ie_result[pid]['acc@1'] = acc_at_1
        ie_result[pid]['acc@5'] = acc_at_5

        performance = result['performance']
        local_acc = performance['acc@k']

        logging_data ={}
        pid2performance[pid] = {}
        for k in range(5):
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