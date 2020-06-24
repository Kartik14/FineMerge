''' Adopted from https://github.com/SeanNaren/deepspeech.pytorch/blob/master/search_lm_params.py '''

import argparse
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from multiprocessing.pool import Pool
from ctcdecode import CTCBeamDecoder

from decode import ctc_beam_decode_batch
from utils import wer as get_wer, normalize_string

def parse_args():

    parser = argparse.ArgumentParser(
        description='Tune an ARPA LM based on a pre-trained acoustic model output'
    )
    parser.add_argument(
        '--saved-output', 
        default="", 
        type=str, 
        required=True,
        help='Path to frame-level probs from local model'
    )
    parser.add_argument(
        "--utterances",
        help="path to filename, reference values for the finetune set",
        type=str,
        required=True,        
    )
    parser.add_argument(
        "--lm-path",
        help="Path to arpa lm file to use while decoding",
        type=str,
        required=True,
    )
    parser.add_argument(
        '--lm-alpha-from', 
        type=float, 
        help='Language model weight start tuning',
        default=0.0
    )
    parser.add_argument(
        '--lm-alpha-to',
        default=3.0, 
        type=float, 
        help='Language model weight end tuning'
    )
    parser.add_argument(
        '--lm-beta-from', 
        default=0.0, 
        type=float,
        help='Language model word bonus (all words) start tuning'
    )
    parser.add_argument(
        '--lm-beta-to', 
        default=1, 
        type=float,
        help='Language model word bonus (all words) end tuning'
    )
    parser.add_argument(
        '--lm-num-alphas', 
        default=31, 
        type=int, 
        help='Number of alpha candidates for tuning'
    )
    parser.add_argument(
        '--lm-num-betas', 
        default=11, 
        type=int,
        help='Number of beta candidates for tuning'
    )
    parser.add_argument(
        "--labels",
        help="Path to labels json files containing ordered list of output labels mapping to ds2 probs",
        type=str,
        required=False,
        default='labels_char.json',
    )
    parser.add_argument(
        "--beam-width",
        help="beam width to use while decoding",
        type=float,
        default=256,
    )
    parser.add_argument(
        '--output-path', 
        default="tune_results.json", 
        help="Where to save tuning results",
        type=str
    )
    args = parser.parse_args()

    return args

def init(labels, lm_path, beam_width):
    global decoder
    decoder = CTCBeamDecoder(labels, model_path=lm_path, beam_width=beam_width, \
        num_processes=4, blank_id=labels.index('_'))

def decode_dataset(params):

    global decoder, saved_output, references, labels

    lm_alpha, lm_beta = params
    decoder.reset_params(lm_alpha, lm_beta)
    hypothesis = ctc_beam_decode_batch(saved_output, decoder, labels)
    wer = get_wer(references, hypothesis)
    return [lm_alpha, lm_beta, wer * 100]

def main():

    args = parse_args()
    global saved_output, references, labels

    with open(args.labels) as label_file:
        labels = json.load(label_file)

    saved_output = dict(np.load(args.saved_output, allow_pickle=True))
    df_utt = pd.read_csv(args.utterances, delimiter='\t')
    utterances = df_utt['file_name'].to_list()
    references = df_utt['transcript'].to_list()
    references = [normalize_string(text, labels[1:]) for text in references]
    saved_output = [saved_output[utt] for utt in utterances]

    p = Pool(multiprocessing.cpu_count(), init, [labels, args.lm_path, args.beam_width])
    cand_alphas = np.linspace(args.lm_alpha_from, args.lm_alpha_to, args.lm_num_alphas)
    cand_betas = np.linspace(args.lm_beta_from, args.lm_beta_to, args.lm_num_betas)
    params_grid = [(float(alpha), float(beta)) for alpha in cand_alphas
                   for beta in cand_betas]
    scores = []
    for params in tqdm(p.imap(decode_dataset, params_grid), total=len(params_grid)):
        scores.append(list(params))
    print("Saving tuning results to finetune.".format(args.output_path))
    with open(args.output_path, "w") as fh:
        json.dump(scores, fh)

    min_results = min(scores, key=lambda x: x[2]) 
    print("Best Params:\nAlpha: %f \nBeta: %f \nWER: %f" % tuple(min_results))


if __name__ == "__main__":
    main()