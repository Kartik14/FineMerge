import numpy as np 
import os
import pickle
import argparse
import pandas as pd
from os.path import join, basename, dirname, abspath
import json
from tqdm import tqdm
import multiprocessing
from multiprocessing.pool import Pool
import sys

sys.path.append(dirname(dirname(abspath(__file__))))
from ctc_alignment import ctc_align
# from utils import parse_text, normalize_text

vocab = []

def get_word_confidence(probs, alignment, vocab):

    blank_idx = vocab.index('_')
    space_idx = vocab.index(' ')
    word_confs = []
    char_prob = None
    char_count = None
    running_word_prob = 0.0
    running_word_chars = 0.0
    new_word = True    
    token_seq = [x for i,x in enumerate(alignment) if x != blank_idx \
        and (i == 0 or (alignment[i-1] != x))]

    if alignment[0] != blank_idx:
        char_prob = probs[0, alignment[0]]
        char_count = 1
        new_word = False

    for t in range(1,probs.shape[0]):
        if alignment[t] != blank_idx: # not blank
            if alignment[t] != alignment[t-1]:
                if alignment[t] == space_idx:
                    running_word_prob = running_word_prob + (char_prob/char_count)
                    running_word_chars += 1
                    word_confs.append(running_word_prob/running_word_chars)
                    running_word_prob = 0.0
                    running_word_chars = 0
                    new_word = True
                else:
                    if not new_word:
                        running_word_prob = running_word_prob + (char_prob/char_count)
                        running_word_chars += 1
                    char_prob = probs[t, alignment[t]]
                    char_count = 1
                    new_word = False
            else:
                if alignment[t] != space_idx:
                    char_prob = char_prob + probs[t, alignment[t]]
                    char_count += 1

    if token_seq[-1] != space_idx:           
        running_word_prob = running_word_prob + (char_prob/char_count)
        running_word_chars += 1
        word_confs.append(running_word_prob/running_word_chars)

    return word_confs

def util(sample):

    global vocab
    smoothen_val = 1e-20
    file_name, service_transcript, sample_prob = sample
    smoothen_probs = sample_prob + smoothen_val
    smoothen_probs = smoothen_probs/np.sum(smoothen_probs, axis=1, keepdims=1)
    service_indices = [vocab.index(token) for token in list(service_transcript)]
    ctc_alignment = ctc_align(smoothen_probs, service_indices, vocab.index('_'))

    word_confs = get_word_confidence(sample_prob, ctc_alignment, vocab)
    assert len(word_confs) == len(service_transcript.split())
    word_confs =  ' '.join([str(c) for c in word_confs])   
    return word_confs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pickle_path',
        help='Path to jasper pickle',
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        help="Path to save the output pickle file",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    with open(args.pickle_path,'rb') as fd:
        data = pickle.load(fd)

    fnames = data['wav_filenames']
    fnames = [basename(f) for f in fnames]
    references = data['references']
    hypothesis = data['beam_hypotheses']
    logprobs = data['logprobs']
    probs = [np.exp(x) for x in logprobs]
    beams = data['beams']
    beam_scores = data['beam_scores']

    vocab = data['vocab']
    vocab.append("_")
       
    utterances = list(zip(fnames, hypothesis, probs))
    with Pool(multiprocessing.cpu_count()) as pool:
        word_confidences = list(tqdm(pool.imap(util, utterances), total=len(utterances)))

    df = pd.DataFrame({'file_name':fnames, 'transcript':hypothesis, 'word_confs': word_confidences})
    df = df[['file_name', 'transcript', 'word_confs']]
    df.to_csv(args.output_path, index=False, sep='\t')

        
