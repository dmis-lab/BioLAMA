from transformers import (
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    BertTokenizer,
    RobertaTokenizer,
    BertForMaskedLM,
    RobertaForMaskedLM
)
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import torch
from data_loader import FactDataset
from tqdm import tqdm
from preprocessor import Preprocessor
from decoder import Decoder
from evaluator import Evaluator
import argparse
import glob
import os
import copy
import numpy as np
import json
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MAX_NUM_VECTORS = 20

def init_template(base_model, tokenizer, prompt_token_len, init_manual_template, manual_template=''):
    if init_manual_template:
        # hotfix for roberta
        manual_template = manual_template.replace("[X]", " [X] ").replace("[Y]", " [Y] ")
        manual_template = " ".join(manual_template.split())

        template = convert_manual_to_dense(manual_template, base_model, tokenizer)
    else:
        template = '[X] ' + ' '.join(['[V%d]'%(i+1) for i in range(prompt_token_len)]) + ' [Y] .'
    return template

def get_new_token(vid):
    assert(vid > 0 and vid <= MAX_NUM_VECTORS)
    return '[V%d]'%(vid)

def prepare_for_dense_prompt(lm_model, tokenizer):
    new_tokens = [get_new_token(i+1) for i in range(MAX_NUM_VECTORS)]
    tokenizer.add_tokens(new_tokens)
    ebd = lm_model.resize_token_embeddings(len(tokenizer))
    print('# vocab after adding new tokens: %d'%len(tokenizer))

def set_seed(seed):
    """
    Set the random seed.
    """
    print("- seed: " + str(seed))
    random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For reproductibility:
    torch.backends.cudnn.enabled = False

def convert_manual_to_dense(manual_template, base_model, tokenizer):
    def assign_embedding(new_token, token):
        """
        assign the embedding of token to new_token
        """
        print('Tie embeddings of tokens: (%s, %s)'%(new_token, token))
        id_a = tokenizer.convert_tokens_to_ids([new_token])[0]
        id_b = tokenizer.convert_tokens_to_ids([token])[0]
        with torch.no_grad():
            base_model.embeddings.word_embeddings.weight[id_a] = base_model.embeddings.word_embeddings.weight[id_b].detach().clone()

    new_token_id = 0
    template = []
    for word in manual_template.split():
        if word in ['[X]', '[Y]']:
            template.append(word)
        else:
            tokens = tokenizer.tokenize(' ' + word)
            for token in tokens:
                new_token_id += 1
                template.append(get_new_token(new_token_id))
                assign_embedding(get_new_token(new_token_id), token)
    
    template = ' '.join(template)
    return template

def load_model(model_name, tokenizer, device, reset, random_init='none'):
    if isinstance(tokenizer, BertTokenizer):
        lm_model = BertForMaskedLM.from_pretrained(model_name).to(device)
        base_model = lm_model.bert

    elif isinstance(tokenizer, RobertaTokenizer): 
        lm_model = RobertaForMaskedLM.from_pretrained(model_name).to(device)
        base_model = lm_model.roberta

    else:
        print(f"tokenizer type = {type(tokenizer)}")
        assert 0
    
    if reset == "embedding":
        print("[INFO] Reset word embedding layer.")
        lm_model.bert.embeddings.word_embeddings.weight.data.normal_(mean=0.0, std=lm_model.config.initializer_range)

    elif reset == "all":
        print("[INFO] Reset all alyers.")
        lm_model.bert.embeddings.word_embeddings.weight.data.normal_(mean=0.0, std=lm_model.config.initializer_range)

        for l in lm_model.bert.encoder.layer:
            l.apply(lm_model._init_weights)

    else:
        print("[INFO] NO reset action.")


    return lm_model, base_model

def save_optiprompt(path, lm_model, tokenizer, original_vocab_size):
    if isinstance(tokenizer, BertTokenizer):
        base_model = lm_model.bert
    elif isinstance(tokenizer, RobertaTokenizer): 
        base_model = lm_model.roberta

    print(f"Saving OptiPrompt's [V]s.. {path}")
    vs = base_model.embeddings.word_embeddings.weight[original_vocab_size:].detach().cpu().numpy()
    with open(path, 'wb') as f:
        np.save(f, vs)

def load_optiprompt(path, lm_model, tokenizer, original_vocab_size):
    if isinstance(tokenizer, BertTokenizer):
        base_model = lm_model.bert
    elif isinstance(tokenizer, RobertaTokenizer): 
        base_model = lm_model.roberta
        
    print("Loading OptiPrompt's [V]s..")
    with open(path, 'rb') as f:
        vs = np.load(f)

    # copy fine-tuned new_tokens to the pre-trained model
    with torch.no_grad():
        # base_model.embeddings.word_embeddings.weight[original_vocab_size:] = torch.Tensor(vs)
        base_model.embeddings.word_embeddings.weight[original_vocab_size:original_vocab_size+len(vs)] = torch.Tensor(vs)
    
    return lm_model, base_model

# for dev or test
def validate(file, lm_model, preprocessor, decoder, evaluator, template, batch_size, draft=False):
    print(f'validate {file}')
    lm_model.eval()
    sentences, all_gold_objects, subjects, prompts, uuids = preprocessor.preprocess(
        file, 
        template = template,
        draft=draft
    )

    decoder.set_model(lm_model)
    all_preds_probs = decoder.decode(
        sentences, 
        batch_size=batch_size
    )

    result = evaluator.evaluate(
        all_preds_probs = all_preds_probs,
        all_golds = all_gold_objects,
        subjects=subjects,
        prompts=prompts,
        uuids=uuids,
        inputs=sentences,
    )

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default='bert-base-uncased')
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--dev_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--num_mask", type=int, required=True)
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--init_method", choices=['independent','order','confidence'], default='independent')
    parser.add_argument("--iter_method", choices=['none','order','confidence'], default='none')
    parser.add_argument("--max_iter", type=int, default=1)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--prompt_token_len", type=int, default=5)
    parser.add_argument("--prompt_path")
    parser.add_argument("--prompt_vector_dir", default=None)
    parser.add_argument("--init_manual_template", action='store_true')
    parser.add_argument("--pids", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset", type=str, required=False, choices=["embedding", "all"], default=None)

    args = parser.parse_args()

    train_files = sorted(glob.glob(args.train_path))
    dev_files = sorted(glob.glob(args.dev_path))
    test_files = sorted(glob.glob(args.test_path))

    device = torch.device("cpu")
    # check for torch device:
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')

    if args.pids == None:
        pids = [f.split("/")[-2] for f in test_files]
    else:
        pids = args.pids.split(",")

#     if args.init_manual_template:
#        args.output_dir = args.output_dir + "_imt"

    if args.draft:
        args.output_dir = args.output_dir + "_draft"
    os.makedirs(args.output_dir, exist_ok=True) 
    
    pid2prompt = {}
    if args.prompt_path:
        with open(args.prompt_path) as f:
            for line in f:
                row = json.loads(line)
                pid = row['relation']
                prompt = row['template']
                pid2prompt[pid] = prompt
        # check
        for pid in pids:
            assert pid in pid2prompt

    set_seed(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    original_vocab_size = len(tokenizer)
    print('Original vocab size: %d'%original_vocab_size)
    
    def collate(examples):
        inputs = [ex[0] for ex in examples]
        obj_ind = [ex[1] for ex in examples]

        if tokenizer._pad_token is None:
            return pad_sequence(inputs, batch_first=True), pad_sequence(obj_ind, batch_first=True)
        return pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_token_id), pad_sequence(obj_ind, batch_first=True, padding_value=tokenizer.pad_token_id)

    def mask_tokens(inputs: torch.Tensor, obj_ind: torch.Tensor, tokenizer, mask_token):
        labels= inputs.clone()
        masked_inputs = inputs.clone()

        mask_idx = tokenizer.convert_tokens_to_ids(mask_token)
        masked_indices = obj_ind.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        
        masked_inputs[masked_indices] = mask_idx
        return masked_inputs, labels

    total_relation = 0
    
    pid2performance = {}
    for train_file, dev_file, test_file in zip(train_files, dev_files, test_files):
        assert train_file.split("/")[-2] == dev_file.split("/")[-2] == test_file.split("/")[-2]
        pid = train_file.split("/")[-2]
        if pid not in pids:
            continue
        
        if args.prompt_vector_dir:
            prompt_vector_dir = args.prompt_vector_dir
            load_prompt_vector = True
        else:
            prompt_vector_dir = args.output_dir
            load_prompt_vector = False

        optiprompt_path = os.path.join(prompt_vector_dir, f"{pid}_optiprompt.np")
        print(optiprompt_path)

        lm_model, base_model = load_model(
            model_name=args.model_name_or_path, 
            tokenizer=tokenizer,
            device=device,
            reset=args.reset
        )
        # load_optiprompt if the checkpoint exists
        prepare_for_dense_prompt(lm_model, tokenizer)

        template = init_template(
            base_model=base_model,
            tokenizer=tokenizer,
            prompt_token_len=args.prompt_token_len,
            init_manual_template=args.init_manual_template,
            manual_template=pid2prompt[pid] if args.init_manual_template else ''
        )
        print('Template: %s'%template)
        
        do_train = True
        if load_prompt_vector:
            print(f"load prompt vector {optiprompt_path}")
            lm_model, base_model = load_optiprompt(optiprompt_path, lm_model, tokenizer, original_vocab_size)
            do_train = False
        
        train_dataset = FactDataset(
            input_file=train_file,
            prompt_token_len=args.prompt_token_len,
            tokenizer=tokenizer,
            template = template,
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, collate_fn=collate
        )

        epochs = args.epochs if not args.draft else 1
        t_total = len(train_dataloader) * epochs
        optimizer = AdamW([{'params': base_model.embeddings.word_embeddings.parameters()}], lr=args.lr, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(t_total/10), num_training_steps=t_total
        )
        preprocessor = Preprocessor(
            tokenizer=tokenizer, 
            num_mask=args.num_mask,
        )

        decoder = Decoder(
            model=lm_model, 
            tokenizer=tokenizer, 
            init_method=args.init_method, 
            iter_method=args.iter_method, 
            MAX_ITER=args.max_iter, 
            BEAM_SIZE=args.beam_size, 
            NUM_MASK=args.num_mask, 
            BATCH_SIZE=args.batch_size,
        )
        evaluator = Evaluator(
            tokenizer=tokenizer
        )

        if do_train:

            # initalize best_acc with evaluation at epoch 0
            result = validate(
                file=dev_file,
                lm_model=lm_model, 
                preprocessor=preprocessor, 
                decoder=decoder, 
                evaluator=evaluator, 
                template=template, 
                batch_size=args.batch_size, 
                draft=args.draft)
            best_acc =  result['performance']['acc@k'][0]
            best_epoch = 0
            best_model = copy.deepcopy(lm_model)
            
            global_step = 0
            for epoch in range(1, epochs+1):
                for batch in tqdm(train_dataloader):
                    lm_model.train()
                    global_step += 1
                    inputs, labels = mask_tokens(
                        inputs = batch[0],
                        obj_ind = batch[1],
                        tokenizer = tokenizer,
                        mask_token = train_dataset.mask_token,
                    )

                    output = lm_model(input_ids=inputs.to(device), labels=labels.to(device))
                    loss = output[0]
                    loss = loss.mean()

                    if (global_step+1) % 20 == 0:
                        print(f"step={global_step} loss={round(loss.item(),5)}")
                        
                    loss.backward()

                    # set normal tokens' gradients to be zero
                    for p in base_model.embeddings.word_embeddings.parameters():
                        # only update new tokens
                        p.grad[:original_vocab_size, :] = 0.0
                        
                    optimizer.step()
                    scheduler.step()
                    lm_model.zero_grad()

                
                result = validate(file=dev_file,     
                                lm_model=lm_model, 
                                preprocessor=preprocessor, 
                                decoder=decoder, 
                                evaluator=evaluator, 
                                template=template, 
                                batch_size=args.batch_size, 
                                draft=args.draft)

                acc_1 = result['performance']['acc@k'][0]
                if best_acc < acc_1:
                    best_acc = acc_1
                    best_epoch = epoch
                    best_model = copy.deepcopy(lm_model)
                    print(f"{pid} updated best acc={best_acc} epoch={best_epoch}")
            
            print(f"{pid} overall best acc={best_acc} epoch={best_epoch}")

            lm_model = best_model

            # save best optiprompt
            save_optiprompt(optiprompt_path, lm_model, tokenizer, original_vocab_size)

        # test
        result = validate(
            file=test_file,
            lm_model=lm_model, 
            preprocessor=preprocessor, 
            decoder=decoder, 
            evaluator=evaluator, 
            template=template, 
            batch_size=args.batch_size, 
            draft=args.draft)
        
        # saving log
        log_file = os.path.join(args.output_dir, pid + ".json")
        print(f"save {log_file}")
        with open(log_file, 'w') as f:
            json.dump(result, f)
        
        total_relation += 1
        
        performance = result['performance']
        local_acc = performance['acc@k']

        logging_data ={}
        pid2performance[pid] = {}
        for k in range(args.beam_size):
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
        print(f"{pid}\t{round(acc1,5)}\t{round(acc5,5)}")
        
    print("-------------------------")
    print(f"MACRO\t{round(np.mean(acc1s),5)}\t{round(np.mean(acc5s),5)}")

if __name__ == '__main__':
    main()