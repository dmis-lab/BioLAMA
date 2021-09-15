# Data construction

## Wikidata

```
# Extract bio entities and their names from wikidata
python ./process_wikidata_entities.py \
    --output_dir ../data/wikidata/entities

# Extract bio triples from wikidata
python ./process_wikidata_triples.py \
    --property_path ../data/wikidata/meta/properties.tsv \
    --entity_dir ../data/wikidata/entities \
    --output_dir ../data/wikidata/triples

# Filter triples based on max length
python ./filter_length.py \
    --input_dir "../data/wikidata/triples/*.jsonl" \
    --output_dir "../data/wikidata/triples_10sw" \
    --model_name bert-base-cased \
    --max_length 10

# Aggregate data
python ./aggregate_data.py \
    --input_path "../data/wikidata/triples_10sw/*.jsonl" \
    --entity_dir ../data/wikidata/entities \
    --model_name_or_path bert-base-cased \
    --output_dir ../data/wikidata/triples_processed \
    --sub_obj_type_path ../data/wikidata/meta/sub_obj_types.json \
    --min_count 500 \
    --max_count 2000

# Get triple stats
python ./get_stats_triples.py \
    --data_dir '../data/wikidata/triples_processed/*' \
    --property_path '../data/wikidata/meta/properties.tsv' \
    --type_path ../data/wikidata/meta/types.json
```

## CTD
```
# Extract bio triples from CTD
python ./process_ctd.py \
    --output_dir ../data/ctd/triples \
    --genes_dict ../data/ctd/entities/NCBI_human_gene_20210407.json \
    --chemicals_dict ../data/ctd/rawdata/CTD_chemicals_20210401.tsv \
    --diseases_dict ../data/ctd/rawdata/CTD_diseases_20210401.tsv \
    --chemicals_genes_file ../data/ctd/rawdata/CTD_chem_gene_ixns_20210401.tsv \
    --chemicals_phenotypes_file ../data/ctd/rawdata/CTD_pheno_term_ixns_20210401.tsv \
    --chemicals_diseases_file ../data/ctd/rawdata/CTD_chemicals_diseases_20210401.tsv \
    --genes_pathways_file ../data/ctd/rawdata/CTD_genes_pathways_20210401.tsv \
    --genes_diseases_file ../data/ctd/rawdata/CTD_genes_diseases_20210401.tsv

# Filter triples based on max length
python ./filter_length.py \
    --input_dir "../data/ctd/triples/*.jsonl" \
    --output_dir "../data/ctd/triples_10sw" \
    --model_name bert-base-cased \
    --max_length 10

# Aggregate data
python ./aggregate_data.py \
    --input_path "../data/ctd/triples_10sw/*.jsonl" \
    --model_name_or_path bert-base-cased \
    --output_dir ../data/ctd/triples_processed \
    --min_count 500 \
    --max_count 2000

# get triple stats
python ./get_stats_triples.py \
    --data_dir '../data/ctd/triples_processed/*' \
    --property_path '../data/ctd/meta/properties.tsv'
```

## UMLS
```
# Extract bio triples from UMLS
python ./process_umls.py \
    --rel_path 2020AB/META/MRREL.RRF \
    --conso_path 2020AB/META/MRCONSO.RRF \
    --sty_path 2020AB/META/MRSTY.RRF \
    --output_dir ../data/umls/triples

# Filter triples based on max length
python ./filter_length.py \
    --input_dir "../data/umls/triples/*.jsonl" \
    --output_dir "../data/umls/triples_10sw" \
    --model_name bert-base-cased \
    --max_length 10 \
    --pids UR116,UR124,UR173,UR180,UR211,UR214,UR221,UR254,UR256,UR44,UR45,UR48,UR49,UR50,UR588,UR625

# Aggregate data
python ./aggregate_data.py \
    --input_path "data/umls/triples_10sw/*.jsonl" \
    --model_name_or_path bert-base-cased \
    --output_dir ../data/umls/triples_processed \
    --min_count 500 \
    --max_count 2000

# get triple stats
python ./get_stats_triples.py \
    --data_dir '../data/umls/triples_processed/*' \
    --property_path '../data/umls/meta/properties.tsv'
```