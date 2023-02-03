# BioLAMA

<div align="center">
    <img src="images/biolama.png" width="400px" alt="BioLAMA">
</div>

<b>BioLAMA</b> is biomedical factual knowledge triples for probing biomedical LMs. The triples are collected and pre-processed from three sources: CTD, UMLS, and Wikidata. Please see our paper [
Can Language Models be Biomedical Knowledge Bases? (Sung et al., 2021)](http://arxiv.org/abs/2109.07154) for more details.

### Updates
* \[**Mar 17, 2022**\] The BioLAMA probe with the CTD/UMLS/Wikidata triples are released [here](https://drive.google.com/file/d/1CcjpmNuAXavL3aMjwVqiiziMu3OGDyyG/view?usp=sharing).
 
## Getting Started
After the [installation](#installation), you can easily try BioLAMA with manual prompts. When a subject is "flu" and you want to probe its symptoms from an LM, the input should be like "Flu has symptom such as \[Y\]."

```
# Set MODEL to bert-base-cased for BERT or dmis-lab/biobert-base-cased-v1.2 for BioBERT
MODEL=./RoBERTa-base-PM-Voc/RoBERTa-base-PM-Voc-hf
python ./BioLAMA/cli_demo.py \
    --model_name_or_path ${MODEL}
```

Result:
```
Please enter input (e.g., Flu has symptoms such as [Y].):
hepatocellular carcinoma has symptoms such as [Y].
-------------------------
Rank    Prob    Pred
-------------------------
1       0.648   jaundice
2       0.223   abdominal pain
3       0.127   jaundice and ascites
4       0.11    ascites
5       0.086   hepatomegaly
6       0.074   obstructive jaundice
7       0.06    abdominal pain and jaundice
8       0.059   ascites and jaundice
9       0.043   anorexia and jaundice
10      0.042   fever and jaundice
-------------------------
Top1 prediction sentence:
"hepatocellular carcinoma has symptoms such as jaundice."
```

## Quick Link
* [Installation](#installation)
* [Resources](#resources)
* [Experiments](#experiments)

## Installation

```
# Install torch with conda (please check your CUDA version)
conda create -n BioLAMA python=3.7
conda activate BioLAMA
conda install pytorch=1.8.0 cudatoolkit=10.2 -c pytorch

# Install BioLAMA
git clone https://github.com/dmis-lab/BioLAMA.git
cd BioLAMA
pip install -r requirements.txt
```

## Resources

### Models
For BERT and BioBERT, we use checkpoints provided in the Huggingface Hub:
- [bert-base-cased](https://huggingface.co/bert-base-cased) (for BERT)
- [dmis-lab/biobert-base-cased-v1.2](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2) (for BioBERT)

Bio-LM is not provided in the Huggingface Hub. Therefore, we use the Bio-LM checkpoint released in [link](https://github.com/facebookresearch/bio-lm). Among the various versions of Bio-LMs, we use `RoBERTa-base-PM-Voc-hf'.
```
wget https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-Voc-hf.tar.gz
tar -xzvf RoBERTa-base-PM-Voc-hf.tar.gz 
rm -rf RoBERTa-base-PM-Voc-hf.tar.gz
```

### Datasets

The dataset will take about 85 MB of space. You can download the dataset [here](https://drive.google.com/file/d/1CcjpmNuAXavL3aMjwVqiiziMu3OGDyyG/view?usp=sharing).

```
tar -xzvf data.tar.gz
rm -rf data.tar.gz
```

The directory tree of the data is like:
```
data
├── ctd
│   ├── entities
│   ├── meta
│   ├── prompts
│   └── triples_processed
│       └── CD1
│           ├── dev.jsonl
│           ├── test.jsonl
│           └── train.jsonl
├── wikidata
│   ├── entities
│   ├── meta
│   ├── prompts
│   └── triples_processed
│       └── P2175
│           ├── dev.jsonl
│           ├── test.jsonl
│           └── train.jsonl
└── umls
    ├── meta
    └── prompts
    └── triples_processed
        └── UR44
            ├── dev.jsonl
            ├── test.jsonl
            └── train.jsonl
    

```

## Experiments

We provide two ways of probing PLMs with BioLAMA:
- [Manual Prompt](#manual-prompt)
- [OptiPrompt](#optiprompt)

### Manual Prompt

<b>Manual Prompt</b> probes PLMs using pre-defined manual prompts. The predictions and scores will be logged in '/output'.

```
# Set TASK to 'ctd' for CTD or 'umls' for UMLS
# Set MODEL to 'bert-base-cased' for BERT or 'dmis-lab/biobert-base-cased-v1.2' for BioBERT
TASK=wikidata
MODEL=./RoBERTa-base-PM-Voc/RoBERTa-base-PM-Voc-hf
PROMPT_PATH=./data/${TASK}/prompts/manual.jsonl
TEST_PATH=./data/${TASK}/triples_processed/*/test.jsonl

python ./BioLAMA/run_manual.py \
    --model_name_or_path ${MODEL} \
    --prompt_path ${PROMPT_PATH} \
    --test_path "${TEST_PATH}" \
    --init_method confidence \
    --iter_method none \
    --num_mask 10 \
    --max_iter 10 \
    --beam_size 5 \
    --batch_size 16 \
    --output_dir ./output/${TASK}_manual
```

Result:
```
PID     Acc@1   Acc@5
-------------------------
P2175   9.40    21.11
P2176   22.46   39.75
P2293   2.24    11.43
P4044   9.47    19.47
P780    16.30   37.85
-------------------------
MACRO   11.97   25.92
```

### OptiPrompt

<b>OptiPrompt</b> probes PLMs using embedding-based prompts starting from embeddings of manual prompts. The predictions and scores will be logged in '/output'.

```
# Set TASK to 'ctd' for CTD or 'umls' for UMLS
# Set MODEL to 'bert-base-cased' for BERT or 'dmis-lab/biobert-base-cased-v1.2' for BioBERT
TASK=wikidata
MODEL=./RoBERTa-base-PM-Voc/RoBERTa-base-PM-Voc-hf
PROMPT_PATH=./data/${TASK}/prompts/manual.jsonl
TRAIN_PATH=./data/${TASK}/triples_processed/*/train.jsonl
DEV_PATH=./data/${TASK}/triples_processed/*/dev.jsonl
TEST_PATH=./data/${TASK}/triples_processed/*/test.jsonl

python ./BioLAMA/run_optiprompt.py \
    --model_name_or_path ${MODEL} \
    --train_path "${TRAIN_PATH}" \
    --dev_path "${DEV_PATH}" \
    --test_path "${TEST_PATH}" \
    --prompt_path ${PROMPT_PATH} \
    --num_mask 10 \
    --init_method confidence \
    --iter_method none \
    --max_iter 10 \
    --beam_size 5 \
    --batch_size 16 \
    --lr 3e-3 \
    --epochs 10 \
    --seed 0 \
    --prompt_token_len 5 \
    --init_manual_template \
    --output_dir ./output/${TASK}_optiprompt
```

Result:
```
PID     Acc@1   Acc@5
-------------------------
P2175   9.47    24.94
P2176   20.14   39.57
P2293   2.90    9.21
P4044   7.53    18.58
P780    12.98   33.43
-------------------------
MACRO   7.28    18.51
```

### IE Baseline (BEST)

<b>BEST (Biomedical Entity Search Tool)</b> is a returns relevant biomedical entity given a query. By constructing the query We used <b>BEST</b> as an information extraction baseline.

```
TASK=wikidata
TEST_PATH=./data/${TASK}/triples_processed/*/test.jsonl
CUDA_VISIBLE_DEVICES=0 python ./BioLAMA/run_ie.py \
    --test_path "${TEST_PATH}" \
    --output_dir ./output/${TASK}_ie
```


## Acknowledgement
Parts of the code are modified from [genewikiworld](https://github.com/SuLab/genewikiworld), [X-FACTR](https://github.com/jzbjyb/X-FACTR), and [OptiPrompt](https://github.com/princeton-nlp/OptiPrompt). We appreciate the authors for making their projects open-sourced.

## Citations
```bibtex
@inproceedings{sung2021can,
    title={Can Language Models be Biomedical Knowledge Bases},
    author={Sung, Mujeen and Lee, Jinhyuk and Yi, Sean and Jeon, Minji and Kim, Sungdong and Kang, Jaewoo},
    booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    year={2021},
}
```
