import time
import functools
import numpy as np

import requests
from tqdm import tqdm
from wikidataintegrator.wdi_core import WDItemEngine
import traceback
from wikidataintegrator.wdi_config import config
import csv
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir",required=True)
args = parser.parse_args()

# Sometimes cacheing allows a failed sparql query to finish on on subsequent attems
# For this reason we ill run 3 times
config['BACKOFF_MAX_TRIES'] = 3

execute_sparql_query = WDItemEngine.execute_sparql_query
# comment this ---v out to use official wikidata endpoint
# execute_sparql_query = functools.partial(execute_sparql_query,
#                                          endpoint="http://163.152.163.168:9999/bigdata/namespace/wdq/sparql")

# instance of subject, subclass of object
special_edges = [('Q11173', 'P1542', 'Q21167512'),  # chemical, cause of, chemical hazard
                 ('Q12136', 'P780', 'Q169872'),  # disease, symptom, symptom
                 ('Q12136', 'P780', 'Q1441305'),  # disease, symptom, medical sign
                 ('Q21167512', 'P780', 'Q169872')]  # chemical hazard, symptom, symptom

special_starts = [q[:2] for q in special_edges]


def change_endpoint(endpoint):
    global execute_sparql_query
    execute_sparql_query = functools.partial(execute_sparql_query, endpoint=endpoint)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def getConceptLabel(qid):
    return getConceptLabels((qid,))[qid]


def getConceptLabels(qids):
    out = dict()
    for chunk in chunks(list(set(qids)), 50):
        this_qids = {qid.replace("wd:", "") if qid.startswith("wd:") else qid for qid in chunk}
        # Found Some results that begin with 't' and cause request to return no results
        bad_ids = {qid for qid in this_qids if not qid.startswith('Q')}
        this_qids = '|'.join(this_qids - bad_ids)
        params = {'action': 'wbgetentities', 'ids': this_qids, 'languages': 'en', 'format': 'json', 'props': 'labels'}
        r = requests.get("https://www.wikidata.org/w/api.php", params=params)
        r.raise_for_status()
        wd = r.json()['entities']
        # Use empty labels for the bad ids
        wd.update({bad_id: {'labels': {'en': {'value': ""}}} for bad_id in bad_ids})
        out.update({k: v['labels'].get('en', dict()).get('value', '') for k, v in wd.items()})
    return out


def get_prop_labels():
    """ returns a dict of labels for all properties in wikidata """
    s = """
    SELECT DISTINCT ?property ?propertyLabel
    WHERE {
        ?property a wikibase:Property .
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    }"""
    try:
        d = execute_sparql_query(s)['results']['bindings']
    except:
        print("***** FAILED SPARQL *****")
        d = []
    d = {x['property']['value'].replace("http://www.wikidata.org/entity/", ""):
                x['propertyLabel']['value'] for x in d}
    return d


def determine_p(use_subclass, extend=True):
    # p = "wdt:P279*" if use_subclass else "wdt:P31/wdt:P279*"
    p = "wdt:P279*" if use_subclass else "wdt:P31*" # HOTFIX. TEST
    # Option to not extend down 'subclass_of' edges (useful for highly populated node types)
    if not extend:
        p = p.replace('/wdt:P279*', '').replace('*', '')
    return p


def is_subclass(qid, return_val=False):
    instance_count, instance_items = get_type_entities(qid, use_subclass=False, extend_subclass=False)
    subclass_count, subclass_items = get_type_entities(qid, use_subclass=True, extend_subclass=False)
    
    # If the numbers are close, we need to determine if its because some have both subclass and instance of values
    if instance_count != 0 and subclass_count != 0 and abs(np.log10(instance_count) - np.log10(subclass_count)) >= 1:

        p0 = "wdt:P31"
        p1 = "wdt:P279"

        s_both = """
        SELECT (COUNT(DISTINCT ?item) as ?c) WHERE {
          ?item {p0} {wds} .
          ?item {p1} {wds} .
        }
        """
        s_both = s_both.replace("{wds}", "wd:" + qid).replace("{p0}", p0).replace("{p1}", p1)
        # print(f"is_subclass={s_both}")
        both = execute_sparql_query(s_both)['results']['bindings']
        both = {qid: int(x['c']['value']) for x in both}.popitem()[1]

        is_sub = subclass_count - both > instance_count
    else:
        is_sub = subclass_count > instance_count

    if return_val:
        if is_sub:
            return is_sub, subclass_count, subclass_items
        else:
            return is_sub, instance_count, instance_items
        # return is_sub, subclass_count if is_sub else instance_count
    return is_sub

def get_type_entities(qid, use_subclass=False, extend_subclass=True):
    """
    For each qid, get the number of items that are instance of (types) this qid
    """
    p = determine_p(use_subclass, extend_subclass)
    # get count first
    s = """
    SELECT (COUNT(DISTINCT ?item) as ?c) WHERE {
      ?item {p} {wds}
    }
    """.replace("{wds}", "wd:" + qid).replace("{p}", p)
    try:
        d = execute_sparql_query(s)['results']['bindings']
    except Exception as e:
            print(e)
            traceback.print_exc()
            raise e

    count = {qid: int(x['c']['value']) for x in d}.popitem()[1]

    # get entity
    LIMIT = 5000
    max_iter = int(count/LIMIT)

    items = []
    for i in tqdm(range(max_iter + 1),desc=f"{qid}"):
        s = """
        SELECT ?item ?itemLabel ?itemDescription 
        (GROUP_CONCAT(DISTINCT(?itemAlt); separator='| ') as ?itemAltLabel)
        WHERE{
            {
                SELECT DISTINCT ?item WHERE {
                    ?item {p} {wds}
                }
                LIMIT {limit}
                OFFSET {offset}
            }
            OPTIONAL {
                ?item skos:altLabel ?itemAlt .
                FILTER (lang(?itemAlt)='en')
            }
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        } GROUP BY ?item ?itemLabel ?itemDescription
        """.replace("{wds}", "wd:" + qid).replace("{p}", p).replace("{limit}", str(LIMIT)).replace("{offset}", str(LIMIT*i))
        try:
            # print(f"get_type_entities={s}")
            d = execute_sparql_query(s)['results']['bindings']
        except Exception as e:
            print(e)
            traceback.print_exc()
            raise e
        
        for row in d:
            item = row['item']['value'].split("/")[-1]
            itemLabel = row['itemLabel']['value'] if 'itemLabel' in row else ''
            itemAltLabel = row['itemAltLabel']['value'] if 'itemAltLabel' in row else ''
            itemDescription = row['itemDescription']['value'] if 'itemDescription' in row else ''
            

            items.append((item, itemLabel, itemDescription, itemAltLabel))
    
    # assert count == len(items)
    if count != len(items):
        print(f"WARN! {count} != {len(items)}")
        count = len(items)
        
    return count, items

def determine_node_type_and_get_counts(node_ids, name_map=dict(), max_size_for_expansion=200000):
    # get all node counts for my special types
    subclass_nodes = dict()
    expand_nodes = dict()
    type_count = dict()
    # type_items = dict()

    # These nodes we've seeded and are all 'instance_of' or they are very large and should waste time expanding
    # Down subclasses.
    # Q11173: chemical compound
    # Q2996394: biological process
    # Q14860489: molecular function
    # Q5058355: cellular component
    # Q13442814: scholarly article
    # Q16521: taxon
    expand_nodes = {q: False for q in ['Q11173', 'Q2996394', 'Q14860489', 'Q5058355', 'Q13442814', 'Q16521']}

    time.sleep(0.5)  # Sometimes TQDM prints early, so sleep will endure messages are printed before TQDM starts
    t = tqdm(node_ids)
    for qid in t:
        t.set_description(name_map[qid])
        t.refresh()
        is_sub, count, items = is_subclass(qid, True)

        subclass_nodes[qid] = is_sub
        if qid not in expand_nodes:
            expand_nodes[qid] = count <= max_size_for_expansion
            # if expand_nodes[qid]:
            #     print(f"{qid} is newly added for expand nodes")
        if expand_nodes[qid]:
            # Small number of nodes, so expand the sublcass...
            # count_ext = get_type_count(qid, use_subclass=is_sub, extend_subclass=True)
            # count_ext, items_ext = get_type_entities(qid, use_subclass=is_sub, extend_subclass=True)
            count_ext, sub_items_ext = get_type_entities(qid, use_subclass=True, extend_subclass=True)
            count_ext, ins_items_ext = get_type_entities(qid, use_subclass=False, extend_subclass=True)
            
            # aggregate items from subclass_of and instances_of
            items_ext = list(set(sub_items_ext + ins_items_ext))
            count_ext = len(items_ext)

            # Ensure this is still ok (some will baloon like chemical Compound)
            expand_nodes[qid] = count_ext <= max_size_for_expansion
            # If its still ok, update the counts
            if expand_nodes[qid]:
                count = count_ext
                items = items_ext

        type_count[qid] = count

        os.makedirs(args.output_dir, exist_ok = True)
        
        # save type items as files
        output_path = os.path.join(args.output_dir, f'{qid}.tsv')
        with open(output_path, 'w') as f:
            wr = csv.writer(f, delimiter="\t")
            for item in items:
                wr.writerow(item)  

    return type_count, subclass_nodes, expand_nodes

# Q13442814: scholarly article
# Q16521: taxon
def search_metagraph_from_seeds(seed_nodes, skip_types=('Q13442814', 'Q16521'), min_counts=200,
                                max_size_for_expansion=200000):
    # Make set for easy operations
    skip_types = set(skip_types)

    print("Getting type counts")
    time.sleep(0.5)  # Sometimes TQDM prints early, so sleep will endure messages are printed before TQDM starts
    determine_node_type_and_get_counts(seed_nodes.keys(),
                                       seed_nodes,
                                       max_size_for_expansion)

if __name__ == "__main__":
    min_counts = 200

    # these are the special nodes that will have their external ID counts displayed,
    # the labels aren't, outputted, only used for monitoring status
    seed_nodes = {
       'Q12136': 'disease',
       'Q7187': 'gene',
       'Q8054': 'protein',
       'Q37748': 'chromosome',
       'Q215980': 'ribosomal RNA',
       'Q11173': 'chemical_compound',
       'Q12140': 'medication',
       'Q28885102': 'pharmaceutical_product',
       'Q417841': 'protein_family',
       'Q898273': 'protein_domain',
       'Q2996394': 'biological_process',
       'Q14860489': 'molecular_function',
       'Q5058355': 'cellular_component',
       'Q3273544': 'structural_motif',
       'Q7644128': 'supersecondary_structure',
       'Q616005': 'binding_site',
       'Q423026': 'active_site',
       'Q4936952': 'anatomical structure',
       'Q169872': 'symptom',
       'Q15304597': 'sequence variant',
       'Q4915012': 'biological pathway',
       'Q50377224': 'pharmacologic action',  # Subclass
       'Q50379781': 'therapeutic use',
       'Q3271540': 'mechanism of action',  # Subclass
    }
    
    # skip edge searches
    skip_types = {'Q13442814', 'Q16521'}

    # get entities
    search_metagraph_from_seeds(seed_nodes, skip_types, min_counts, 200000)