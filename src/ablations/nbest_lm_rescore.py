import argparse
import kenlm
import json
from os.path import join, basename, dirname
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from multiprocessing.pool import Pool
import sys

sys.path.append(dirname(dirname(os.path.abspath(__file__))))
from utils import parse_text, levenshtein, wer

def get_wer(refs, hyps):

    total_wer = 0.0
    total_tokens = 0
    for ref, hyp in zip(refs, hyps):
        total_wer += levenshtein(ref.split(), hyp.split())
        total_tokens += len(ref.split())

    return total_wer / total_tokens

def get_cer(refs, hyps):

    total_cer = 0.0
    total_tokens = 0
    for ref, hyp in zip(refs, hyps):
        total_cer += levenshtein(ref, hyp)
        total_tokens += len(ref)

    return total_cer / total_tokens

def main():

    LM = '../data/arpa_lm/{}/mozilla_3gram.binary'.format(accent)
    model = kenlm.LanguageModel(LM)


if __name__ == '__main__':

    args = parser.parse_args()

    parser = argparse.ArgumentParser(
        description="Perform n-best lm rescoring on the output transcripts"
    )
    parser.add_argument(
        "--dataset",
        help="Path to preprocessed data pickle file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--labels",
        help="Path to labels json files containing ordered list of output labels mapping to ds2 probs",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--params_config",
        help="Path to json config file containing param values",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        help="Path to save file containing modified transcipt",
        type=str,
        required=True,        
    )
    parser.add_argument(
        "--lm_path",
        help="Path to arpa lm file to use while decoding (optional)",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--utterances",
        help="Path to file containing list of utterance to generate modified transcripts"
             ", if not provided transcipt generated for all utterances",
        type=str,
        default=None,
        required=False,        
    )

    args = parser.parse_args()

    LM = args.lm_path
    model = kenlm.LanguageModel(LM)

    alpha = args.alpha
    beta = args.beta
    with open(join(dump_dir, 'fnames_test.txt')) as fd:
        valid_fnames = fd.read().splitlines()
    with open(join(dump_dir, 'ref_wrds_test.txt')) as fd:
        refs = fd.read().splitlines()

    nbest_preds = []
    for i,fname in enumerate(valid_fnames):
        json_pth = join(json_dir, fname[:-3] + 'json')
        with open(json_pth) as fd:
            data = json.load(fd)

        best_score = -float('inf')
        best_transcript = None
        for rank, alt in enumerate(data['results'][0]['alternatives']):
            transcript = parse_text(alt['transcript'])
            confidence = alt['confidence']
            if args.oracle_wer:
                score = -wer(refs[i], transcript)
            else:
                score = confidence + alpha*model.score(transcript) - beta*rank
                
            if score > best_score:
                best_score = score
                best_transcript = transcript
        fd_nbest.write(best_transcript + '\n')
        nbest_preds.append(best_transcript)

    print("FINAL WER: {}, CER: {}".format(get_wer(refs,nbest_preds), get_cer(refs, nbest_preds)))
    



    


    



