import functools
import json
import os

from tqdm import tqdm
from wikidataintegrator.wdi_core import WDItemEngine
import traceback
from wikidataintegrator.wdi_config import config
from glob import glob
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--property_path", required=True)
parser.add_argument("--entity_dir",required=True)
parser.add_argument("--output_dir",required=True)
args = parser.parse_args()

# Sometimes cacheing allows a failed sparql query to finish on on subsequent attems
# For this reason we ill run 3 times
config['BACKOFF_MAX_TRIES'] = 3

execute_sparql_query = WDItemEngine.execute_sparql_query
# comment this ---v out to use official wikidata endpoint
execute_sparql_query = functools.partial(execute_sparql_query,
                                         endpoint="http://163.152.163.168:9999/bigdata/namespace/wdq/sparql")

def get_triples_given_property(pid, entity2meta):
    # get count first
    s = """
    SELECT (COUNT(?item) as ?c) WHERE {
        {
            SELECT DISTINCT ?item ?value{
                ?item wdt:{p} ?value
            }
        }
    }
    """.replace("{p}", pid)
    try:
        print(s)
        d = execute_sparql_query(s)['results']['bindings']
    except Exception as e:
            print(e)
            traceback.print_exc()
            raise e

    count = [int(x['c']['value']) for x in d][0]
    print(f"{pid}:{count}")

    # query one batch at a time
    LIMIT = 3000
    max_iter = int(count/LIMIT)

    triples = []
    for i in tqdm(range(max_iter + 1)):
        s = """
        SELECT ?item ?value WHERE{      
            ?item wdt:{p} ?value.
        }
        LIMIT {limit}
        OFFSET {offset}
        """.replace("{p}", pid).replace("{limit}", str(LIMIT)).replace("{offset}", str(LIMIT*i))
        try:
            print(f"get_triples_given_property={s}")
            d = execute_sparql_query(s)['results']['bindings']
        except Exception as e:
            print(e)
            traceback.print_exc()
            raise e
        
        for sample in d:
            sbj_type = sample['item']['type']
            obj_type = sample['value']['type']
 
            # extract only entity to entity triple
            if (sbj_type != 'uri') or (obj_type != 'uri'):
                continue

            sub_uri = sample['item']['value'].split("/")[-1]
            obj_uri = sample['value']['value'].split("/")[-1]
            
            # extract only bioentity to bioentity triple
            if (sub_uri not in entity2meta):
                print(f"WARN! {sub_uri} is not bio entity")
                continue
            if (obj_uri not in entity2meta):
                print(f"WARN! {obj_uri} is not bio entity")
                continue

            triples.append({
                "predicate_id": pid,
                "sub_uri": sub_uri,
                "sub_type": entity2meta[sub_uri]['type'],
                "sub_label": entity2meta[sub_uri]['label'],
                "sub_aliases": entity2meta[sub_uri]['aliases'],
                "obj_uri": obj_uri,
                "obj_type": entity2meta[obj_uri]['type'],
                "obj_label": entity2meta[obj_uri]['label'],
                "obj_aliases": entity2meta[obj_uri]['aliases'],
            })
    return triples

if __name__ == "__main__":
    print("[Start]load entites of each type")
    entity_files = glob(os.path.join(args.entity_dir, "Q*.tsv"))
    entity2meta = {}
    for entity_file in entity_files:
        _type = entity_file.split("/")[-1].replace(".tsv","")
        with open(entity_file) as f:
            for line in f:
                rdr = csv.reader(f, delimiter='\t')
                r = list(rdr)
                for entity_id, label, description, aliases in r:
                    entity2meta[entity_id] = {
                        'type': _type,
                        'label': label.strip(),
                        'aliases': [al.strip() for al in aliases.split("|")]
                    }
    print(f"[Finish]total entities={len(entity2meta)}")

    print("[Start]load bio properties")
    properties = {}
    with open(args.property_path, 'r') as f:
        rdr = csv.reader(f, delimiter='\t')
        r = list(rdr)
        for pid, plabel in r:
            properties[pid] = plabel    
    print(f"[Finish]total entities={len(r)}")

    print("[Start]query bio triples from wikidata")
    pid2triples = {}
    num_triples = 0
    for pid in tqdm(properties):
        pid2triples[pid] = get_triples_given_property(pid, entity2meta)
        num_triples += len(pid2triples[pid])
    print(f"[Finish]total triples={num_triples}")

    print("[Start]save triples")
    MIN_COUNT = 200
    os.makedirs(args.output_dir, exist_ok=True)
    for pid in pid2triples:
        triples = pid2triples[pid]
        if len(triples) < MIN_COUNT:
            continue

        output_path = os.path.join(args.output_dir, f"{pid}.jsonl" )
        with open(output_path, 'w') as fo:
            for triple in pid2triples[pid]:
                output = json.dumps(triple, ensure_ascii=False)
                fo.write(output + "\n")
    print("[Finish]")