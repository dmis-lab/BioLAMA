import torch
import sys
from transformers import (
    AutoTokenizer,
    BertTokenizer,
    BertForMaskedLM,
)

import configparser
from preprocessor import Preprocessor
from decoder import Decoder
import argparse
import os
import numpy as np
import random


from run_optiprompt import prepare_for_dense_prompt, init_template, load_optiprompt

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(seed):
    """
    Set the random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def print_predictions(sentence, preds_probs):
    k = min(len(preds_probs),10)
    # print(f"Top {k} predictions")
    print("-------------------------")
    print(f"Rank\tProb\tPred")
    print("-------------------------")
    for i in range(k):
        preds_prob = preds_probs[i]
        print(f"{i+1}\t{round(preds_prob[1],3)}\t{preds_prob[0]}")

    print("-------------------------")
    # print("\n")
    print("Top1 prediction sentence:")
    print(f"\"{sentence.replace('[Y]',preds_probs[0][0])}\"")

def load_opti_prompt_model(conf, device):

    print("load model " + conf.get("model").strip('"'))

    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(conf.get("model").strip('"'), use_fast=False)
    original_vocab_size = len(tokenizer)
    print('Original vocab size: %d'%original_vocab_size)

    # get model
    lm_model = BertForMaskedLM.from_pretrained(conf.get("model").strip('"'))
    base_model = lm_model.bert
    lm_model = lm_model.to(device)

    # make sure this is only an evaluation
    lm_model.eval()
    for param in lm_model.parameters():
        param.grad = None
    
    prepare_for_dense_prompt(lm_model, tokenizer)
    
    # Template
    template = init_template(
        base_model=base_model,
        tokenizer=tokenizer,
        prompt_token_len=conf.getint("prompt_token_len"),
        init_manual_template=conf.getboolean("init_manual_template"),
        manual_template=conf.get("prompt").strip('"') if conf.getboolean("init_manual_template") else ''
    )
    print('Template: %s'%template)

    print("load prompt vector " + conf.get("optipath").strip('"'))
    lm_model, base_model = load_optiprompt(conf.get("optipath").strip('"'), lm_model, tokenizer, original_vocab_size)
    
    preprocessor = Preprocessor(tokenizer=tokenizer, num_mask=conf.getint("num_mask"))

    decoder = Decoder(
        model=lm_model, 
        tokenizer=tokenizer, 
        init_method=conf.get("init").strip('"'), 
        iter_method=conf.get("iter").strip('"'), 
        MAX_ITER=conf.getint("maxiter"), 
        BEAM_SIZE=conf.getint("beam_size"), 
        NUM_MASK=conf.getint("num_mask"), 
        BATCH_SIZE=10)

    return lm_model, preprocessor, decoder


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--text", required=True)
    parser.add_argument("--config", type=str, help="path to cnfig file for the demo")
    args = parser.parse_args()

    # seed
    set_seed(0)

    # check for torch device:
    device = torch.device("cpu")
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')

    if not os.path.exists(args.config):
        print("Config file : " + args.config + " does not exist")
        sys.exit(3)

    try:    
        config = configparser.ConfigParser()
        config.read(args.config)
    except configparser.Error as e:
        print(e)
        sys.exit(3)
    
    P703_lm_model, P703_preprocessor, P703_decoder = load_opti_prompt_model(config["P703"], device)
    rP703_lm_model, rP703_preprocessor, rP703_decoder = load_opti_prompt_model(config["rP703"], device)
    print(P703_lm_model)
    print(rP703_lm_model)

if __name__ == '__main__':
    main()
