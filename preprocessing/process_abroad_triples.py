import argparse
import os
import transformers
import pandas as pd
import numpy as np
import statistics
import re
import csv
import json
import random
from transformers import AutoTokenizer

# Run me: python preprocessing/process_abroad_triples.py --input_pairs="data/np/pre-processing/taxon-np-list.csv" --input_taxa_names="data/np/pre-processing/ALL-taxons-ids-names.tsv" --input_chem_main_names="data/np/pre-processing/wikidata/np_names_wikidata.tsv" --input_chem_syn_names="data/np/pre-processing/wikidata/np_synonyms_wikidata.tsv" --property="rP703" --topn=100 --outdir="data/np/triples_processed"
parser = argparse.ArgumentParser()
parser.add_argument("--input_pairs", help="input table of pairs: taxon - natural product. (See abroad-requests-rq)", type=str, required=True, dest='file_pairs')
parser.add_argument("--input_taxa_names", help="input table for taxa names info. (See abroad-requests-rq)", type=str, required=True, dest='file_taxa_names')
parser.add_argument("--input_chem_main_names", help="input table for chemical names - only main label (See wikidata-requests.rq)", type=str, required=True, dest='file_chem_main_label')
parser.add_argument("--input_chem_syn_names", help="input table for chemical synonym names labels (See wikidata-requests.rq", type=str, required=True, dest='file_chem_syn_label')
parser.add_argument("--property", help="", type=str, required=True, dest='property', choices=["rP703", "P703"])
parser.add_argument("--topn", help="keep the top n most supported relations", type=int, required=False, dest='topn', default=1000)
parser.add_argument("--topk", help="keep a maximum of k object per subject", type=int, required=False, dest='topk', default=20)
parser.add_argument('--dataset_split', help="Dataset split in terms of train, dev and test", required=False,  nargs=3, type=float, default=[0.4, 0.1, 0.5], dest="dataset_split")
parser.add_argument("--outdir", help="output table", type=str, required=True, dest='outdir')

args = parser.parse_args()

# print all args:
for arg in vars(args):
    print(arg + ": " +  str(getattr(args, arg)))

selected_models = {
    'ChemicalBERT':'recobo/chemical-bert-uncased',
    'BioBERT':'dmis-lab/biobert-base-cased-v1.2',
    'PubMedBERT':'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
}

def extract_accepted_names(input_table, models, limit):
    """
    - a pd.DataFrame. colums are: [ID, name]
    - models a dict of models
    """
    
    id_col_name = input_table.columns[0]
    model_codes = list(models.values())
    input_ids = input_table.iloc[:, 0].tolist()
    input_names = input_table.iloc[:, 1].tolist()
    out = np.empty((len(input_names), len(models)))

    for i in range(len(models)):

        tokenizer = AutoTokenizer.from_pretrained(model_codes[i])

        tokens_len = []

        for name in input_names:
            tokens_len.append(len(tokenizer.encode(name.lower(), add_special_tokens=False)))

        out[:, i] = tokens_len

    test_bool = np.sum(((0 < out) * (out < limit)), axis=-1) == len(models)

    out = pd.DataFrame({"id": input_ids, "accepted": test_bool})

    out.rename(columns = {'id':id_col_name}, inplace = True)

    return out

def prepare_json_data(pairs, topn, topk, taxa_names, np_names, np_synonyms, property):
    """
    - pairs: the dataframe of pairs fungi - np
    - topn: the top number of pairs to keep, sorted by count. If None, keep all
    - topk: the max number of retained object (ordered by the most suppported)
    - taxa_names: synonym dataframe for taxa
    - np_synonyms: synonym dataframe for natural products
    - property: what's the property entity: rP703 or P703 ?
    """
    # init
    triples_to_probe = {}

    # extract top n
    pairs = pairs.sort_values(by="counts", ascending=False)
    if topn is not None:
        pairs = pairs[0:topn]

    # if subject is a taxon
    if property == "rP703":
        
        for taxon_id, taxon_pairs in pairs.groupby("taxon_id"):

            # extract topk subset if there are more than topk pairs.
            subset = taxon_pairs[0:(min(len(taxon_pairs), topk))]
            
            uuid = taxon_id + "-rP703"

            # check that the taxon name is correctly unique
            assert len(taxa_names[taxa_names["ID"] == taxon_id]["name"].tolist()) == 1

            # call it rP703 for "reverse" P703 (found in taxa)
            triples_to_probe[uuid] = {
            'uuid': uuid,
            'predicate_id': "rP703",
            'sub_uri': taxon_id,
            'sub_label': taxa_names[taxa_names["ID"] == taxon_id]["name"].item(),
            'sub_type': "mycobank-taxon",
            'sub_aliases': taxa_names[taxa_names["acceptedID"] == taxa_names[taxa_names["ID"] == taxon_id]["acceptedID"].item()]["name"].tolist(),
            'obj_uris': subset["pubchem_id"].tolist(),
            'obj_labels': np_names[np_names["CID"].isin(subset["pubchem_id"].tolist())]["label"].tolist(),
            'obj_types': ["pubchem"] * len(subset),
            'obj_aliases':[np_synonyms[np_synonyms["CID"] == pubchem_id]["label"].tolist() for pubchem_id in subset["pubchem_id"].tolist()]
        }

    # if subject is a natural product
    if property == "P703":
        
        for pubchem_id, np_pairs in pairs.groupby("pubchem_id"):

            subset = np_pairs[0:(min(len(np_pairs), topk))]

            uuid = pubchem_id + "-P703"

            # check that the taxon name is correctly unique
            assert len(np_names[np_names["CID"] == pubchem_id]["label"].tolist()) == 1

            triples_to_probe[uuid] = {
                'uuid': uuid,
                'predicate_id': "P703",
                'sub_uri': pubchem_id,
                'sub_label': np_names[np_names["CID"] == pubchem_id]["label"].item(),
                'sub_type': "pubchem",
                'sub_aliases': np_synonyms[np_synonyms["CID"] == pubchem_id]["label"].tolist(),
                'obj_uris': subset["taxon_id"].tolist(),
                'obj_labels': taxa_names[taxa_names["ID"].isin(subset["taxon_id"].tolist())]["name"].tolist(),
                'obj_types': ["mycobank-taxon"] * len(subset),
                'obj_aliases':[taxa_names[taxa_names["acceptedID"] == taxa_names[taxa_names["ID"] == taxon_id]["acceptedID"].item()]["name"].tolist() for taxon_id in subset["taxon_id"].tolist()]
            }

        # short tests:
        for uuid, data in triples_to_probe.items():
            assert len(data["obj_uris"]) == len(data["obj_labels"])
            assert len(data["obj_labels"]) == len(data["obj_types"])
            assert len(data["obj_types"]) == len(data["obj_aliases"])
    
    return triples_to_probe

def split_train_dev_test(triples_to_probe, train_dev_test_split, shuffle=True):
    
    total = len(triples_to_probe)

    train_split, dev_split, test_split = train_dev_test_split
    assert train_split + dev_split + test_split == 1

    if shuffle:
        l = list(triples_to_probe.items())
        random.shuffle(l)
        triples_to_probe = dict(l)

    # 4:1:5 = train:dev:test
    trainset = dict(list(triples_to_probe.items())[:int(total*train_split)])
    devset = dict(list(triples_to_probe.items())[int(total*train_split):int(total*(train_split + dev_split))])
    testset = dict(list(triples_to_probe.items())[int(total*(train_split + dev_split)):])

    print(f"total len:{len(triples_to_probe)}")
    print(f"trainset={len(trainset)} devset={len(devset)} testset={len(testset)}")
    
    return trainset, devset, testset

def save_to_jsonl(save_path, data):
    print(f"Saving to {save_path}")
    with open(file=save_path, mode='w') as f:
        for uuid in data:
            output = data[uuid]
            output = json.dumps(output, ensure_ascii=False)
            f.write(output + '\n')

def write_dataset(outdir, triples, train_dev_test_split):
    train, dev, test = split_train_dev_test(triples, train_dev_test_split)
    
    for dataset, file_name in zip((train, dev, test), ("train.jsonl", "dev.jsonl", "test.jsonl")):
        save_to_jsonl(os.path.join(outdir, file_name), dataset)

if __name__ == "__main__":
    
    print("[INFO] Load and pre-process chemical and taxa names")
    # load natural products names
    np_names = pd.read_csv(args.file_chem_main_label, sep="\t", quoting=csv.QUOTE_NONE, index_col=None, dtype=object)
    np_names.columns = ["CID", "label"]

    # For some ids, the name may be unavailable. Exclude them.
    np_names.dropna(inplace=True)
    np_names.reset_index(inplace=True, drop=True)

    # laod taxa names (not only accepted taxa but ALL of them)
    taxa = pd.read_csv(args.file_taxa_names, sep = "\t", header=0, dtype=object)

    # keep only mycobank names and species.
    taxa = taxa[(taxa["rank"] == "species") & (taxa["TAX_SOURCE"] == "mycobank-taxonomy")]

    # extract main entities ID: those wihtout an acceptedID because they are themself the main entity
    main_e = taxa["acceptedID"].isnull()

    # complete the table
    taxa.loc[main_e.tolist(), "acceptedID"] = taxa.loc[main_e.tolist(), "ID"]

    # Get tokenized sizes of names for each selected models. In each dataset, we keep only the items that pass the tokenizer test.
    accepted_np_names = extract_accepted_names(np_names, selected_models, 10)
    available_np = np_names[accepted_np_names["accepted"].tolist()]

    accepted_taxa_names = extract_accepted_names(taxa[["ID", "name"]], selected_models, 10)
    available_taxa = taxa[accepted_taxa_names["accepted"].tolist()]

    print("[INFO] Number of accepted natural product names: " + str(sum(accepted_np_names["accepted"])))
    print("[INFO] Number of accepted taxa names: " + str(sum(accepted_taxa_names["accepted"])))

    # Create dataset
    print("[INFO] Create dataset")
    # load taxa - np pairs
    pairs = pd.read_csv(args.file_pairs, sep=",", header=0, dtype=object)

    # filter pairs on tax_source 
    pairs = pairs[(pairs["TAX_SOURCE"] == "mycobank-taxonomy")]

    # fitler on list of accepted taxa names
    pairs = pairs[(pairs["cpd_related_taxa_ID"].isin(available_taxa["ID"].tolist()))]

    # fitler on list of accepted natural product
    pairs = pairs[(pairs["pubchemId"].isin(available_np["CID"].tolist()))]

    # Group by Accepted ID and pubchemId and count the number of references
    df_pairs = pairs.groupby(['cpd_related_taxa_ID','pubchemId'])['ref'].count().reset_index(name='counts')

    # Some taxa or matural products, may have ambigous names, for instance when there is 2 names associated with an ID, remove this ambiguities from the list
    df_pairs = df_pairs[~df_pairs["cpd_related_taxa_ID"].isin(available_taxa[available_taxa["ID"].duplicated()]["ID"].tolist())]
    df_pairs = df_pairs[~df_pairs["pubchemId"].isin(available_np[available_np["CID"].duplicated()]["CID"].tolist())]

    # remove lines where taxon name or np name is missing
    df_pairs = df_pairs.dropna()
    df_pairs = df_pairs[["cpd_related_taxa_ID", "pubchemId",	"counts"]]
    df_pairs.columns = ["taxon_id", "pubchem_id", "counts"]

    print("[INFO] Process chemical synonyms table.")
    # read np synonyms table
    np_names_synonyms = pd.read_csv(args.file_chem_syn_label, sep="\t", quoting=csv.QUOTE_NONE, index_col=None, dtype=object)
    np_names_synonyms.dropna(inplace=True)
    np_names_synonyms.columns = ["CID", "label"]

    # remove abbreviation of one character
    np_names_synonyms = np_names_synonyms[np_names_synonyms.loc[:, "label"].apply(lambda x: len(x) > 1).tolist()]

    # fitler on acceptable by tokenizing size
    accepted_np_syns_names = extract_accepted_names(np_names_synonyms[["CID", "label"]], selected_models, 10)
    np_names_synonyms = np_names_synonyms[accepted_np_syns_names["accepted"].tolist()]

    dataset = prepare_json_data(df_pairs, args.topn, args.topk, available_taxa, available_np, np_names_synonyms, args.property)
    
    print("[INFO] Write dataset")
    outdir = os.path.join(args.outdir, args.property)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    write_dataset(outdir, dataset, args.dataset_split)