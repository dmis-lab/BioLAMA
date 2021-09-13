from subprocess import check_output
import re
import string

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append(ind)
    return results

def is_obj_in_sbj(sbj, objs):
    objs = [obj.lower().split() for obj in objs]
    sbj = sbj.lower().split()

    for obj in objs:
        result = find_sub_list(sl=obj, l=sbj)
        if len(result) >0:
            return True, ' '.join(sbj), ' '.join(obj)
    
    return False, '', ''

def wc(filename):
    return int(check_output(["wc", "-l", filename]).split()[0])

# https://github.com/huggingface/transformers/blob/758ed3332b219dd3529a1d3639fa30aa4954e0f3/src/transformers/data/metrics/squad_metrics.py
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
