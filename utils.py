# -*- coding: utf-8 -*-

import numpy as np
import math
import string
import re
import json
from collections import defaultdict
from os.path import join
import inflect
from text import _clean_text


def get_word2phone(lexicon):

	with open(lexicon,'r') as fd:
		lines = fd.readlines()

	word2phone = {}
	for line in lines:
		w, ph = line.strip().split('\t')
		word2phone[w.lower()] = ph.split()

	return word2phone

def get_phone2word(lex):
	with open(lex,'r') as fd:
		lines = fd.readlines()

	# phone2word = defaultdict(lambda: set({}))
	phone2word = {}
	phones = set({})
	for line in lines:
		w, ph = line.strip().split('\t')

		w = w.lower()
		if ph not in phones:
			phone2word[ph] = {w}
			phones.add(ph)
		else:
			phone2word[ph].add(w)

	return phone2word

def convert_to_phones(sent, word2phone, space_ph='SP'):
	ph_seq = []
	sent = sent.split()
	for w in sent:
		if w == '\'':
			continue
		ph_seq.extend(word2phone[w] + [space_ph])

	return ' '.join(ph_seq[:-1])

def convert_idx_to_tokens(seq, mapping):
	return [mapping.get(x) for x in seq if mapping.get(x) != None]

def merge_list(l1, l2):
	return [x + [y,] for x in l1 for y in l2]

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def wer(references, hypothesis):

    total_edits = 0
    total_tokens = 0
    for ref, hyp in zip(references, hypothesis):
        total_edits += levenshtein(ref.split(), hyp.split())
        total_tokens += len(ref.split())
    
    return total_edits / total_tokens

def cer(refs, hyps):

    total_edits = 0.0
    total_tokens = 0
    for ref, hyp in zip(refs, hyps):
        # ref = list(ref.replace(' ', ''))
        # hyp = list(hyp.replace(' ', ''))
        total_edits += levenshtein(ref, hyp)
        total_tokens += len(ref)

    return total_edits / total_tokens

def parse_text(text, word_confs=None):

	#converting to lower case
    hyp = text.lower()
    hyp = re.sub('[^a-z0-9\' ]','',hyp)

    if word_confs != None:
        word_confs = word_confs.split()
        
    # converting digits to text
    p = inflect.engine()
    hyp1 = ""
    new_confs = []
    for i, word in enumerate(hyp.split()):
            if word.isdigit():
                digits_text = p.number_to_words(int(word))
                hyp1 += digits_text + " "
                if word_confs != None:
                    for _ in range(len(digits_text.split())):
                        new_confs.append(word_confs[i])
            else:
                hyp1 += word + " "
                if word_confs != None:
                    new_confs.append(word_confs[i])

    hyp1 = re.sub('[^a-z\' ]','',hyp1)
    hyp1 = ' '.join(hyp1.split())
    new_confs = [float(x) for x in new_confs]

    if word_confs != None:
        return hyp1, new_confs  
    else:
        return hyp1

def normalize_string(s, labels):
    """
    Normalizes string. For example:
    'call me at 8:00 pm!' -> 'call me at eight zero zero pm'
    Args:
        s: string to normalize
        labels: labels used during model training.
    Returns:
            Normalized string
    """

    def good_token(token, labels):
        s = set(labels)
        for t in token:
            if not t in s:
                return False
        return True

    punctuation = string.punctuation
    punctuation = punctuation.replace("+", "")
    punctuation = punctuation.replace("&", "")
    for l in labels:
        punctuation = punctuation.replace(l, "")
    # Turn all punctuation to whitespace
    table = str.maketrans(punctuation, " " * len(punctuation))

    try:
        text = _clean_text(s, ["english_cleaners"], table).strip()
        return ''.join([t for t in text if good_token(t, labels=labels)])
    except:
        print("WARNING: Normalizing {} failed".format(s))
        return None

def align(p, q, sub_cost=1.5, ins_cost=1, del_cost=1):

    rows = len(p) + 1
    cols = len(q) + 1
    scores = np.zeros((rows, cols))
    parents = np.zeros((rows, cols))

    scores[0,0] = 0
    parents[0,0] = None

    for i in range(1, rows):
        scores[i,0] = scores[i-1,0] + ins_cost
        parents[i,0] = 1

    for j in range(1, cols):
        scores[0,j] = scores[0,j-1] + del_cost
        parents[0,j] = 2

    for i in range(1,rows):
        for j in range(1,cols):
            if p[i-1] == q[j-1]:
                sub_fact = 0
            else:
                sub_fact = 1

            scores[i,j] = min(scores[i-1,j-1] + sub_cost*sub_fact,
                            scores[i-1,j] + ins_cost,
                            scores[i,j-1] + del_cost)

            parents[i,j] = np.argmin([scores[i-1,j-1] + sub_cost*sub_fact,
                            scores[i-1,j] + ins_cost,
                            scores[i,j-1] + del_cost])

    aligned_p, aligned_q = [], []
    i, j = rows-1, cols-1
    blank = '_'
    while i > 0 or j > 0:
        if parents[i,j] == 0:
            aligned_p.append(p[i-1])
            aligned_q.append(q[j-1])
            i = i - 1
            j = j - 1
        elif parents[i,j] == 1:
            aligned_p.append(p[i-1])
            aligned_q.append(blank)
            i = i - 1
        else:
            aligned_p.append(blank)
            aligned_q.append(q[j-1])
            j = j - 1

    aligned_p = aligned_p[::-1]
    aligned_q = aligned_q[::-1]

    return aligned_p, aligned_q

