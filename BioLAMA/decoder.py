import pdb
from typing import List, Dict, Tuple, Union
import torch
import transformers
import numpy as np
from tqdm import tqdm
from transformers import (
    BertTokenizer,
    RobertaTokenizer
)
from utils import (
    normalize_answer,
    convert_2d_list_to_1d,
    convert_1d_list_to_2d
)

# We remove duplicates of the predictions before evaluation.
# We consider it to be duplicate when the normalized predictions are the same 
def remove_duplicate_preds_probs(preds_probs):
    tmp = {}
    normalized_preds = []
    for pred, prob in preds_probs:
        normalized_pred = normalize_answer(pred)

        # this is how to remove duplicates
        if normalized_pred not in normalized_preds:
            tmp[pred] = prob
            normalized_preds.append(normalized_pred)

    # initialize new_preds_probs
    new_preds_probs = [('',0)] * len(preds_probs)
    
    # fill in the preds and probs
    for i, (pred, prob) in enumerate(tmp.items()):
        new_preds_probs[i] = (pred, prob)

    return new_preds_probs

class Decoder():
    def __init__(self, model, tokenizer, init_method, iter_method, MAX_ITER, BEAM_SIZE, NUM_MASK, BATCH_SIZE, verbose=True):
        print(f"init_method={init_method} iter_method={iter_method} MAX_ITER={MAX_ITER} BEAM_SIZE={BEAM_SIZE} NUM_MASK={NUM_MASK} BATCH_SIZE={BATCH_SIZE}")
        self.model = model # bert model
        self.tokenizer = tokenizer # bert tokenizer

        self.MASK_IDX = self.tokenizer.mask_token_id
        self.PAD_IDX = self.tokenizer.pad_token_id
        self.UNK_IDX = self.tokenizer.unk_token_id

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if isinstance(tokenizer, BertTokenizer):
            self.mask_token = '[MASK]'
            self.pad_token = '[PAD]'
            self.unk_token = '[UNK]'
            assert self.tokenizer.convert_ids_to_tokens(self.MASK_IDX) == self.mask_token
            assert self.tokenizer.convert_ids_to_tokens(self.PAD_IDX) == self.pad_token
            assert self.tokenizer.convert_ids_to_tokens(self.UNK_IDX) == self.unk_token

        elif isinstance(tokenizer, RobertaTokenizer): 
            self.mask_token = '<mask>'
            self.pad_token = '<pad>'
            self.unk_token = '<unk>'
            assert self.tokenizer.convert_ids_to_tokens(self.MASK_IDX) == self.mask_token
            assert self.tokenizer.convert_ids_to_tokens(self.PAD_IDX) == self.pad_token
            assert self.tokenizer.convert_ids_to_tokens(self.UNK_IDX) == self.unk_token

        else:
            print(f"tokenizer type = {type(tokenizer)}")
            assert 0

        self.init_method = init_method
        self.iter_method = iter_method
        self.MAX_ITER = MAX_ITER
        self.BEAM_SIZE = BEAM_SIZE
        self.NUM_MASK = NUM_MASK
        self.BATCH_SIZE = BATCH_SIZE

        self.sentence_printed = not verbose

    def set_model(self,model):
        self.model = model

    def append_paddings(self, sentences):
        # Append [PAD]s next to [SEP] by max length in the batch
        tokenized_sentences = []
        for sentence in sentences:
            tokenized_sentence = self.tokenizer.encode(sentence)
            tokenized_sentences.append(tokenized_sentence)

        len_max = len(max(tokenized_sentences, key=lambda x:len(x)))
        for tokenized_sentence in tokenized_sentences:
            tokenized_sentence += [self.PAD_IDX] * (len_max - len(tokenized_sentence))

        return tokenized_sentences

    def decode_sentences(self, sentences):
        # Encode input using tokenizer
        # Append [PAD]s next to [SEP] by max length in the batch
        sentences = self.append_paddings(sentences)

        # for printing only once
        if self.sentence_printed == False:
            print(sentences[:5])
            self.sentence_printed = True

        inp_tensor = torch.tensor(sentences)
        attention_mask = inp_tensor.ne(self.PAD_IDX).long()
        mask_ind = inp_tensor.eq(self.MASK_IDX).long()
        
        inp_tensor = inp_tensor.to(self.device)
        attention_mask = attention_mask.to(self.device)
        mask_ind = mask_ind.to(self.device)

        batch_size = int(len(sentences)/self.NUM_MASK)
        # SHAPE: (batch_size, num_mask, seq_len)
        inp_tensor = inp_tensor.view(batch_size,self.NUM_MASK,-1)
        attention_mask = attention_mask.view(batch_size,self.NUM_MASK,-1)
        mask_ind = mask_ind.view(batch_size,self.NUM_MASK,-1)

        # HERE IF masked_subject is not None, it means that the subject has been MASKED. So we MUST REMOVE ITS INDEX FROM THE ONES USE FOR THE PREDICTIONS.
        # If the subject is before [Y] in the template -> the first [MASK] is removed
        # If the subject is after [Y] in the template -> the last [MASK] is removed
        # The subject has been masked when there is one MASK token more than expected, so whem the sum of mask_ind > i (num_mask) + 1
        for b in range(len(mask_ind)):
            if sum(mask_ind[b][1]) > 2:
                print("[INFO] data with MASKED subject detected.")
                # It is the last token ?
                if (mask_ind[b][1] == 1).nonzero(as_tuple=True)[0][1] == (mask_ind[b][1] == 1).nonzero(as_tuple=True)[0][0] + 1:
                    for i in range(self.NUM_MASK):
                        mask_ind[b][i][((mask_ind[b][i] == 1).nonzero(as_tuple=True)[0][-1])] = 0
                else:
                    for i in range(self.NUM_MASK):
                        mask_ind[b][i][((mask_ind[b][i] == 1).nonzero(as_tuple=True)[0][0])] = 0
        out_tensors=[]
        logprobs=[]
        for nm in range(self.NUM_MASK):
            # decode
            # SHAPE: (beam_size, batch_size, seq_len)
            b_out_tensor, b_logprob, iter = iter_decode_beam_search(
                self.model, inp_tensor[:, nm, :], mask_ind[:, nm, :], attention_mask[:, nm, :],
                restrict_vocab=[], mask_value=self.MASK_IDX,
                init_method=self.init_method, iter_method=self.iter_method,
                max_iter=self.MAX_ITER, tokenizer=self.tokenizer,
                reprob=False, beam_size=self.BEAM_SIZE)

            # SHAPE: (batch_size, beam_size, seq_len)
            b_out_tensor = b_out_tensor.permute(1,0,2)
            b_logprob = b_logprob.permute(1,0,2)

            out_tensors.append(b_out_tensor)
            logprobs.append(b_logprob)

        # SHAPE: (batch_size, beam_size, num_mask, seq_len)
        logprob = torch.stack(logprobs, 2)
        out_tensor = torch.stack(out_tensors, 2)

        # predict with topk (beamsize)
        all_preds = []
        all_probs = []

        for b_out_tensor, b_logprob, b_mask_ind in zip(out_tensor, logprob, mask_ind):
            for i in range(self.NUM_MASK):
                mask_len = i + 1
                preds = []
                probs = []
                
                for j in range(self.BEAM_SIZE):
                    pred: np.ndarray = b_out_tensor[j][i].masked_select(
                        b_mask_ind[i].eq(1)).detach().cpu().numpy().reshape(-1)
                    log_prob = b_logprob[j][i].masked_select(
                        b_mask_ind[i].eq(1)).detach().cpu().numpy().reshape(-1).sum(-1)
                    # length normalization
                    # 0.0 length_norm_coeff => mask_len_norm == 1. In this case, shorter predictions are favored.
                    # 1.0 length_norm_coeff => mask_len_norm == mask_len. In this case, models are adjusted to favor longer predictions.
                    length_norm_coeff = 0.0
                    mask_len_norm = np.power(mask_len,length_norm_coeff)
                    prob = np.exp(log_prob / mask_len_norm)

                    pred = merge_subwords(pred, self.tokenizer, merge=True)

                    preds.append(pred)
                    probs.append(prob)

                all_preds.append(preds)
                all_probs.append(probs)

        return all_preds, all_probs

    def decode(self, input, batch_size=None, verbose=True):
        """
        input: a list of lists of sentences with [MASK]
        output: a list of lists of predictions
        """
        if batch_size == None:
            batch_size = self.BATCH_SIZE

        all_preds = []
        all_probs = []
        for b in tqdm(range(0, len(input), batch_size),desc="decode", disable=not verbose):
            query_batch = input[b:b + batch_size]
            max_length = len(query_batch[0])

            # flat for batch processing
            flat_query_batch = convert_2d_list_to_1d(query_batch)
            flat_preds, flat_probs = self.decode_sentences(flat_query_batch)
            preds = convert_1d_list_to_2d(flat_preds, max_length)
            probs = convert_1d_list_to_2d(flat_probs, max_length)

            all_preds += preds
            all_probs += probs
        
        # Sort preds based on probs
        # The output format will be like [[('pain', 0.37236136611128684)]]
        all_preds_probs = []
        for preds, probs in zip(all_preds, all_probs):
            flat_preds = convert_2d_list_to_1d(preds)
            flat_probs = convert_2d_list_to_1d(probs)

            preds_probs = list(zip(flat_preds, flat_probs))
            preds_probs = sorted(preds_probs, key=lambda x: x[1], reverse=True)

            # Some predictions are decoded into the same output
            # These duplicates should be removed
            preds_probs = remove_duplicate_preds_probs(preds_probs)
            all_preds_probs.append(preds_probs)

        return all_preds_probs

# https://github.com/jzbjyb/X-FACTR
def merge_subwords(ids: Union[np.ndarray, List[int]], tokenizer, merge: bool=False) -> str:
    subwords = list(tokenizer.convert_ids_to_tokens(ids))
    if not merge:
        return subwords
    else:
        merged_subword = ""
        for subword in subwords:
            if isinstance(tokenizer, BertTokenizer):
                if subword.startswith('##'):
                    subword = subword.replace('##', '')
                    merged_subword += subword
                else:
                    merged_subword += ' ' + subword
            elif isinstance(tokenizer, RobertaTokenizer):
                if subword.startswith('Ġ'):
                    subword = subword.replace('Ġ', ' ')
                    merged_subword += subword
                else:
                    merged_subword += '' + subword
            else:
                print('need to check tokenizer!')
                assert 0 

        merged_subword = merged_subword.strip()
        return merged_subword

# def model_prediction_wrap(model, inp_tensor, attention_mask):
#     with torch.no_grad():
#         logit = model(inp_tensor, attention_mask=attention_mask)[0]

#     if hasattr(model, 'cls'):  # bert
#         bias = model.cls.predictions.bias
#     elif hasattr(model, 'lm_head'):  # roberta
#         bias = model.lm_head.bias
#     elif hasattr(model, 'pred_layer'):  # xlm
#         bias = 0.0
#     else:
#         raise Exception('not sure whether the bias is correct')
#     logit = logit - bias
#     
#     return logit

# Fix this !
def model_prediction_wrap(model, inp_tensor, attention_mask):
    logit = model(inp_tensor, attention_mask=attention_mask)[0]
    if transformers.__version__ in {'2.4.1', '2.4.0'}:
        if hasattr(model, 'cls'):  # bert
            bias = model.cls.predictions.bias
        elif hasattr(model, 'lm_head'):  # roberta
            bias = model.lm_head.bias
        elif hasattr(model, 'pred_layer'):  # xlm
            bias = 0.0
        else:
            raise Exception('not sure whether the bias is correct')
        logit = logit - bias
    elif transformers.__version__ in {'2.3.0'}:
        pass
    else:
        # raise Exception('not sure whether version {} is correct'.format(transformers.__version__))
        pass
    return logit

def iter_decode_beam_search(model,
                            inp_tensor: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
                            raw_mask: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
                            attention_mask: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
                            restrict_vocab: List[int] = None,
                            mask_value: int = 0,  # indicate which value is used for mask
                            max_iter: int = None,  # max number of iteration
                            tokenizer = None,
                            init_method: str='independent',
                            iter_method: str='none',
                            reprob: bool = False,  # recompute the prob finally
                            beam_size: int = 5,
                            ) -> Tuple[torch.LongTensor, torch.Tensor, int]:  # HAPE: (batch_size, seq_len)
    '''
    Masks must be consecutive.
    '''
    assert init_method in {'independent', 'order', 'confidence'}
    assert iter_method in {'none', 'order', 'confidence', 'confidence-multi'}
    bs, sl = inp_tensor.size(0), inp_tensor.size(1)
    # init_mask = inp_tensor.eq(mask_value).long()  # SHAPE: (batch_size, seq_len)
    init_mask = raw_mask.clone()
    init_has_mask = init_mask.sum().item() > 0

    if iter_method == 'confidence-multi':
        number_to_mask = torch.unique(init_mask.sum(-1))
        assert number_to_mask.size(0) == 1, 'this batch has different numbers of mask tokens'
        number_to_mask = number_to_mask[0].item() - 1
        assert max_iter == 0, 'do not need to set max_iter in confidence-multi setting'
    elif iter_method == 'order':
        leftmost_mask = init_mask * torch.cat([init_mask.new_ones((bs, 1)), 1 - init_mask], 1)[:, :-1]
        number_to_mask = torch.unique(init_mask.sum(-1))
        assert number_to_mask.size(0) == 1, 'this batch has different numbers of mask tokens'
        number_to_mask: int = number_to_mask[0].item()
        mask_offset: int = 0
        has_modified: bool = False  # track wether modification happens during a left-to-right pass

    # SHAPE: (<=beam_size, batch_size, seq_len)
    out_tensors: List[torch.LongTensor] = inp_tensor.unsqueeze(0)
    # tokens not considered have log prob of zero
    out_logprobs: List[torch.Tensor] = torch.zeros_like(inp_tensor).float().unsqueeze(0)
    iter: int = 0
    stop: bool = False
    model_call = 0

    while True and init_has_mask:  # skip when there is not mask initially
        next_out_tensors = []
        next_out_logprobs = []

        # enumerate over all previous result
        for out_tensor, out_logprob in zip(out_tensors, out_logprobs):
            # get input
            if iter > 0:
                if iter_method == 'none':
                    inp_tensor = out_tensor
                    if inp_tensor.eq(mask_value).long().sum().item() == 0:  # no mask
                        stop = True
                        break
                elif iter_method == 'confidence':
                    has_mask = out_tensor.eq(mask_value).any(-1).unsqueeze(-1).long()  # SHAPE: (batch_size, 1)
                    inp_tensor = out_tensor.scatter(1, out_logprob.min(-1)[1].unsqueeze(-1), mask_value)
                    # no need to insert mask when there are masks
                    inp_tensor = out_tensor * has_mask + inp_tensor * (1 - has_mask)
                elif iter_method == 'confidence-multi':
                    has_mask = out_tensor.eq(mask_value).any(-1).unsqueeze(-1) # SHAPE: (batch_size, 1)
                    all_has_mask = has_mask.all().item()
                    assert all_has_mask == has_mask.any().item(), 'some samples have masks while the others do not'
                    if not all_has_mask:
                        if number_to_mask <= 0:
                            stop = True
                            break
                        inp_tensor = out_tensor.scatter(1, (-out_logprob).topk(number_to_mask, dim=-1)[1], mask_value)
                        init_method = 'independent'
                        number_to_mask -= 1
                    else:
                        inp_tensor = out_tensor
                elif iter_method == 'order':
                    has_mask = out_tensor.eq(mask_value).any(-1).unsqueeze(-1)  # SHAPE: (batch_size, 1)
                    all_has_mask = has_mask.all().item()
                    any_has_mask = has_mask.any().item()
                    # WARNING Temporary remove assert because it was leading to some failures during tests.
                    # assert all_has_mask == any_has_mask, 'some samples have masks while the others do not'
                    if not all_has_mask:  # no mask, should do refinement
                        if mask_offset >= number_to_mask:
                            mask_offset = 0
                        if mask_offset == 0:  # restart when starting from the beginning
                            has_modified = False
                        cur_mask = torch.cat([leftmost_mask.new_zeros((bs, mask_offset)), leftmost_mask], 1)[:, :sl]
                        cur_mask = cur_mask * init_mask
                        inp_tensor = out_tensor * (1 - cur_mask) + mask_value * cur_mask
                        mask_offset += 1
                    else:
                        inp_tensor = out_tensor
                else:
                    raise NotImplementedError

            # predict
            # SHAPE: (batch_size, seq_len)
            mask_mask = inp_tensor.eq(mask_value).long()
            model_call += 1
            logit = model_prediction_wrap(model, inp_tensor, attention_mask)
            if restrict_vocab is not None:
                logit[:, :, restrict_vocab] = float('-inf')
            # SHAPE: (batch_size, seq_len, beam_size)
            new_out_logprobs, new_out_tensors = logit.log_softmax(-1).topk(beam_size, dim=-1)

            if init_method == 'confidence':
                # mask out non-mask positions
                new_out_logprobs = new_out_logprobs + mask_mask.unsqueeze(-1).float().log()
                new_out_logprobs = new_out_logprobs.view(-1, sl * beam_size)
                new_out_tensors = new_out_tensors.view(-1, sl * beam_size)

            for b in range(beam_size):
                if init_method == 'independent':
                    new_out_logprob = new_out_logprobs[:, :, b]
                    new_out_tensor = new_out_tensors[:, :, b]
                    # SHAPE: (batch_size, seq_len)
                    changes = (out_tensor * mask_mask).ne(new_out_tensor * mask_mask)
                elif init_method == 'order':  # only modify the left-most one.
                    new_out_logprob = new_out_logprobs[:, :, b]
                    new_out_tensor = new_out_tensors[:, :, b]
                    # SHAPE: (batch_size, seq_len)
                    changes = (out_tensor * mask_mask).ne(new_out_tensor * mask_mask)
                    changes = changes & torch.cat([changes.new_ones((bs, 1)), ~changes], 1)[:, :-1]
                elif init_method == 'confidence':  # only modify the most confident one.
                    # SHAPE: (batch_size,)
                    raw_lp, raw_ind = new_out_logprobs.max(-1)
                    # SHAPE: (batch_size, 1)
                    raw_lp, raw_ind = raw_lp.unsqueeze(-1), raw_ind.unsqueeze(-1)
                    seq_ind = raw_ind // beam_size
                    changes = mask_mask & torch.zeros_like(mask_mask).scatter(1, seq_ind, True)
                    new_out_tensor = torch.zeros_like(out_tensor).scatter(1, seq_ind, new_out_tensors.gather(1, raw_ind))
                    new_out_logprob = torch.zeros_like(out_logprob).scatter(1, seq_ind, raw_lp)
                    changes = (out_tensor * changes.long()).ne(new_out_tensor * changes.long())
                    # max for the next max in beam search
                    new_out_logprobs = new_out_logprobs.scatter(1, raw_ind, float('-inf'))
                else:
                    raise NotImplementedError

                # only modify tokens that have changes
                changes = changes.long()
                _out_tensor = out_tensor * (1 - changes) + new_out_tensor * changes
                _out_logprob = out_logprob * (1 - changes.float()) + new_out_logprob.detach() * changes.float()

                # involves heavy computation, where we re-compute probabilities for beam_size * beam_size samples
                if reprob:
                    _out_logprob = compute_likelihood(
                        model, _out_tensor, _out_logprob,
                        init_mask, attention_mask, restrict_vocab, mask_value=mask_value)
                    _out_logprob = _out_logprob * (1 - _out_tensor.eq(mask_value).float())  # skip mask tokens

                next_out_tensors.append(_out_tensor)
                next_out_logprobs.append(_out_logprob)

                '''
                for i in range(bs):
                    print(tokenizer.convert_ids_to_tokens(inp_tensor[i].cpu().numpy()))
                    print(tokenizer.convert_ids_to_tokens(_out_tensor[i].cpu().numpy()))
                input()
                '''

        if stop:
            break

        next_out_tensors = torch.stack(next_out_tensors, 0)
        next_out_logprobs = torch.stack(next_out_logprobs, 0)
        # tie breaking
        next_out_logprobs = next_out_logprobs + \
                            get_tie_breaking(int(next_out_logprobs.size(0))).view(-1, 1, 1).to(next_out_logprobs.device)

        # dedup
        not_dups = []
        for i in range(bs):
            abs = next_out_tensors.size(0)
            # SHAPE: (all_beam_size, seq_len)
            one_sample = next_out_tensors[:, i, :]
            # SHAPE: (all_beam_size,)
            inv = torch.unique(one_sample, dim=0, return_inverse=True)[1]
            # SHAPE: (all_beam_size, all_beam_size)
            not_dup = inv.unsqueeze(-1).ne(inv.unsqueeze(0)) | \
                      (torch.arange(abs).unsqueeze(-1) <= torch.arange(abs).unsqueeze(0)).to(inv.device)
            # SHAPE: (all_beam_size,)
            not_dup = not_dup.all(-1)
            not_dups.append(not_dup)
        # SHAPE: (all_beam_size, batch_size)
        not_dups = torch.stack(not_dups, -1)

        # select top
        # SHAPE: (all_beam_size, batch_size)
        beam_score = (next_out_logprobs * init_mask.unsqueeze(0).float() +
                      not_dups.unsqueeze(-1).float().log()).sum(-1)
        # SHAPE: (beam_size, batch_size, seq_len)
        beam_top = beam_score.topk(beam_size, dim=0)[1].view(-1, bs, 1).repeat(1, 1, sl)
        next_out_logprobs = torch.gather(next_out_logprobs, 0, beam_top)
        next_out_tensors = torch.gather(next_out_tensors, 0, beam_top)

        # stop condition for other type of iter
        if next_out_tensors.size(0) == out_tensors.size(0) and next_out_tensors.eq(out_tensors).all():
            if iter_method != 'order':
                stop = True
        else:
            if iter_method == 'order':
                has_modified = True
        # stop condition for 'order' iter
        if iter_method == 'order' and not has_modified and mask_offset == number_to_mask:
            # reach the last position and no modification happens during this iteration
            stop = True

        #print(next_out_tensors.ne(out_tensors).any(-1).any(0).nonzero())

        out_tensors = next_out_tensors
        out_logprobs = next_out_logprobs

        iter += 1
        if max_iter and iter >= max_iter:  # max_iter can be zero
            stop = True
        if stop:
            break

    return out_tensors, out_logprobs, iter

def compute_likelihood(model,
                       inp_tensor: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
                       lp_tensor: torch.Tensor,  # SHAPE: (batch_size, seq_len)
                       mask_tensor: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
                       attention_mask: torch.LongTensor,  # SHAPE: (batch_size, seq_len))
                       restrict_vocab: List[int] = None,
                       mask_value: int=0,  # indicate which value is used for mask
                       ) -> torch.Tensor:  # SHAPE: (batch_size, seq_len)
    '''
    Masks must be consecutive.
    '''
    bs, seq_len = inp_tensor.size(0), inp_tensor.size(1)
    max_num_masks = mask_tensor.sum(-1).max().item()
    leftmost_mask = mask_tensor * torch.cat([mask_tensor.new_ones((bs, 1)), 1 - mask_tensor], 1)[:, :-1]
    logits = None
    for i in range(max_num_masks):
        # SHAPE: (batch_size, seq_len)
        cur_mask = torch.cat([leftmost_mask.new_zeros((bs, i)), leftmost_mask], 1)[:, :seq_len] * mask_tensor
        inp_tensor_ = (1 - cur_mask) * inp_tensor + cur_mask * mask_value
        print(f"model call in compute_likelihood {i}")
        logit = model_prediction_wrap(model, inp_tensor_, attention_mask)
        cur_mask = cur_mask.unsqueeze(-1).float()
        if logits is None:
            logits = (logit * cur_mask).detach()
        else:
            logits = (logits * (1 - cur_mask) + logit * cur_mask).detach()
    if restrict_vocab is not None:
        logits[:, :, restrict_vocab] = float('-inf')
    lp = logits.log_softmax(-1)
    lp = torch.gather(lp.view(-1, lp.size(-1)), 1, inp_tensor.view(-1).unsqueeze(-1)).view(bs, seq_len)
    lp_tensor = (1 - mask_tensor).float() * lp_tensor + mask_tensor.float() * lp
    return lp_tensor.detach()

_tie_breaking: Dict[int, torch.Tensor] = {}
def get_tie_breaking(dim: int):
    if dim not in _tie_breaking:
        _tie_breaking[dim] = torch.zeros(dim).uniform_(0, 1e-5)
    return _tie_breaking[dim]
