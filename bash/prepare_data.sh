#!/bin/bash


TOPN=20
python preprocessing/process_abroad_triples.py  \
    --input_pairs="data/np/pre-processing/taxon-np-list.csv"  \
    --input_taxa_names="data/np/pre-processing/ALL-taxons-ids-names.tsv"  \
    --input_chem_main_names="data/np/pre-processing/wikidata/np_names_wikidata.tsv"  \
    --input_chem_syn_names="data/np/pre-processing/wikidata/np_synonyms_wikidata.tsv"  \
    --property="rP703"  \
    --topn=${TOPN}  \
    --outdir="data/np/triples_processed" \

python preprocessing/process_abroad_triples.py  \
    --input_pairs="data/np/pre-processing/taxon-np-list.csv"  \
    --input_taxa_names="data/np/pre-processing/ALL-taxons-ids-names.tsv"  \
    --input_chem_main_names="data/np/pre-processing/wikidata/np_names_wikidata.tsv"  \
    --input_chem_syn_names="data/np/pre-processing/wikidata/np_synonyms_wikidata.tsv"  \
    --property="P703"  \
    --topn=${TOPN}  \
    --outdir="data/np/triples_processed" \