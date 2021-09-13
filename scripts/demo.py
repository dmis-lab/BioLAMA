import torch
from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead
)

from preprocessor import Preprocessor
from decoder import Decoder
import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def print_predictions(sentence, preds_probs):
    k = min(len(preds_probs),10)
    print(f"Top {k} predictions")
    print(f"rank\tprob\tpred")
    print("-------------------------")
    for i in range(k):
        preds_prob = preds_probs[i]
        print(f"{i}\t{round(preds_prob[1],3)}\t{preds_prob[0]}")

    print("-------------------------")
    print("Top1 prediction sentence:")
    print(f"{sentence.replace('[Y]',preds_probs[0][0])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--model_name_or_path", default='bert-base-uncased')
    parser.add_argument("--num_mask", type=int, default=10)
    parser.add_argument("--init_method", choices=['independent','order','confidence'], default='confidence')
    parser.add_argument("--iter_method", choices=['none','order','confidence'], default='none')
    parser.add_argument("--max_iter", type=int, default=10)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=10)
    args = parser.parse_args()

    print(f'load model {args.model_name_or_path}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    lm_model = AutoModelWithLMHead.from_pretrained(args.model_name_or_path)
    if torch.cuda.is_available():
        lm_model = lm_model.cuda()

    # make sure this is only an evaluation
    lm_model.eval()
    for param in lm_model.parameters():
        param.grad = None

    preprocessor = Preprocessor(tokenizer=tokenizer, num_mask=args.num_mask)
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

    sentences = preprocessor.preprocess_single_sent(sentence=args.text)
    print(sentences)
    all_preds_probs = decoder.decode([sentences], batch_size=args.batch_size) # topk predictions
    preds_probs = all_preds_probs[0]

    print_predictions(args.text, preds_probs)


if __name__ == '__main__':
    main()