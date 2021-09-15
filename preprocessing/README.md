# Data construction

## Quick Link
* [UMLS](#umls)

## UMLS
Before starting pre-processing UMLS, you need to download raw data from [2020AB UMLS Metathesaurus Files](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html#2020AB).
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
    --data_dir '../../BioLAMA_CR/data/umls/triples_10sw_agg/*' \
    --property_path '../../BioLAMA_CR/data/umls/meta/properties.tsv'
```

Statistics
```
PID     TRAIN   DEV     TEST
UR44    452     113     566
UR221   241     61      302
UR45    772     193     965
UR48    700     176     876
UR211   650     162     813
UR214   459     115     574
UR256   244     62      306
UR588   621     156     777
UR254   672     169     841
UR180   346     87      434
UR116   668     167     835
UR625   381     96      477
UR173   512     128     640
UR49    615     154     769
UR50    663     166     829
UR124   463     116     580
================================
TOTAL   8459    2121    10584
```