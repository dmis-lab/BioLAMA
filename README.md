# BioLAMA

<em>BioLAMA</em> is biomedical factual knowledge triples for probing biomedical LMs. The triples are collected and pre-processed from three sources: CTD, UMLS, and Wikidata. Please see our paper [
Can Language Models be Biomedical Knowledge Bases? (Sung et al., 2021)]() for more details.

#### The dataset for the BioLAMA probe is available at mujeen@163.152.163.223:/home/mujeen/works/BioLAMA_open/data.tar.gz <br>

## Quick Link
* [Installation](#installation)
* [Resources](#resources)
* [Experiments](#experiments)
* [Demo](#demo)

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

### Download Bio-LM
We use the Bio-LM checkpoint released in [link](https://github.com/facebookresearch/bio-lm).
Among various versions of Bio-LMs, we use `RoBERTa-base-PM-Voc-hf'.
```
wget https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-Voc-hf.tar.gz
tar -xzvf RoBERTa-base-PM-Voc-hf.tar.gz 
rm -rf RoBERTa-base-PM-Voc-hf.tar.gz
```

### Download datasets

The dataset will take about 78 MB of space.
```
rsync -rP mujeen@163.152.163.223:/home/mujeen/works/BioLAMA_open/data.tar.gz
tar -xzvf data.tar.gz
rm -rf data.tar.gz
```

## Experiments

We provide two ways of probing PLMs with BioLAMA: 1) Manual Prompt 2) OptiPrompt

### Manual Prompt

<em>Manual Prompt</em> probes PLMs using pre-defined manual prompts.

```
# TASK=ctd
# TASK=umls
TASK=wikidata
MODEL=./RoBERTa-base-PM-Voc/RoBERTa-base-PM-Voc-hf
PROMPT_PATH=./data/${TASK}/prompts/manual.jsonl
TEST_PATH=./data/${TASK}/triples_processed/*/test.jsonl

python ./scripts/run_manual.py \
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

### OptiPrompt

<em>OptiPrompt</em> probes PLMs using embedding-based prompts starting from embeddings of manual prompts.

```
# TASK=ctd
# TASK=umls
TASK=wikidata
MODEL=./RoBERTa-base-PM-Voc/RoBERTa-base-PM-Voc-hf
PROMPT_PATH=./data/${TASK}/prompts/manual.jsonl
TRAIN_PATH=./data/${TASK}/triples_processed/*/train.jsonl
DEV_PATH=./data/${TASK}/triples_processed/*/dev.jsonl
TEST_PATH=./data/${TASK}/triples_processed/*/test.jsonl
PROMPT_PATH=./data/${TASK}/prompts/manual.jsonl

python ./scripts/run_optiprompt.py \
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

## Demo
We provide the CLI demo.
```
MODEL=./RoBERTa-base-PM-Voc/RoBERTa-base-PM-Voc-hf
python ./scripts/demo.py \
    --model_name_or_path ${MODEL} \
    --text "hepatocellular carcinoma has symptoms such as [Y]."
```

Result:
```
Top 10 predictions
rank    prob    pred
-------------------------
0       0.648   jaundice
1       0.223   abdominal pain
2       0.127   jaundice and ascites
3       0.11    ascites
4       0.086   hepatomegaly
5       0.074   obstructive jaundice
6       0.06    abdominal pain and jaundice
7       0.059   ascites and jaundice
8       0.043   anorexia and jaundice
9       0.042   fever and jaundice
-------------------------
Top1 prediction sentence:
hepatocellular carcinoma has symptoms such as jaundice.
```

## Acknowledgement
Parts of the code are modified from [genewikiworld](https://github.com/SuLab/genewikiworld), [X-FACTR](https://github.com/jzbjyb/X-FACTR), [OptiPrompt](https://github.com/princeton-nlp/OptiPrompt), and [LANKA](https://github.com/c-box/LANKA). We appreciate the authors for making their projects open-sourced.

## Citations
```bibtex
@inproceedings{sung2021can,
    title={Can Language Models be Biomedical Knowledge Bases},
    author={Sung, Mujeen and Lee, Jinhyuk and Yi, Sean and Jeon, Minji and Kim, Sungdong and Kang, Jaewoo},
    booktitle={EMNLP},
    year={2021},
}
```
