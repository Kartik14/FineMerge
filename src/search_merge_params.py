import argparse
import json
import sys
import multiprocessing
from multiprocessing.pool import Pool
from os.path import join
from tqdm import tqdm
import numpy as np
import pandas as pd

from ctc_alignment import ctc_align
from merge import get_frame_lvl_cnfs, fine_merge
from decode import ctc_beam_decode
from utils import wer as get_wer, normalize_string

def parse_args():

    parser = argparse.ArgumentParser(
        description='Tune an ARPA LM based on a pre-trained acoustic model output'
    )
    parser.add_argument(
        '--data', 
        required=True, 
        type=str, 
        help="data file obtained from import script"
    )
    parser.add_argument(
        "--utterances",
        help="path to filename, reference values for the finetune set",
        type=str,
        required=True,        
    )
    parser.add_argument(
        "--lm_path",
        help="Path to arpa lm file to use while decoding",
        type=str,
        required=True,
    )
    parser.add_argument(
        '--th-from', 
        type=float, 
        help='service token threshold for finemerge',
        default=1e-11
    )
    parser.add_argument(
        '--th-to',
        default=1e-8, 
        type=float, 
        help='service token threshold for finemerge'
    )
    parser.add_argument(
        '--service-weight-from', 
        default=0.5, 
        type=float,
        help='service weight for fine merge'
    )
    parser.add_argument(
        '--service-weight-to', 
        default=1.0, 
        type=float,
        help='service weight for fine merge'
    )
    parser.add_argument(
        '--blank-conf-from', 
        default=0.4, 
        type=float,
        help='confidence given to blanks in alignment'
    )
    parser.add_argument(
        '--blank-conf-to', 
        default=0.8, 
        type=float,
        help='confidence given to blanks in alignment'
    )
    parser.add_argument(
        '--num-th', 
        default=6, 
        type=int, 
        help='Number of th candidates for tuning'
    )
    parser.add_argument(
        '--num-service-weights', 
        default=6, 
        type=int,
        help='Number of service weight candidates for tuning'
    )
    parser.add_argument(
        '--num-blank-confs', 
        default=5, 
        type=int,
        help='Number of blank conf candidates for tuning'
    )
    parser.add_argument(
        "--labels",
        help="Path to labels json files containing ordered list of output labels mapping to ds2 probs",
        type=str,
        required=False,
        default='labels_char.json',
    )
    parser.add_argument(
        "--lm_alpha",
        help='Language model weight start tuning',
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--lm_beta",
        help='Language model word bonus (all words) start tuning',
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--beam_width",
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

def get_alignments(utt):

    global data, labels
    service_transcript = data[utt]['service_transcript']
    ds2_probs = data[utt]['ds2_probs']

    smoothen_probs = ds2_probs + 1e-20
    smoothen_probs = smoothen_probs/np.sum(smoothen_probs, axis=1, keepdims=1)
    service_indices = [labels.index(token) for token in list(service_transcript)]
    ctc_alignment = ctc_align(smoothen_probs, service_indices, labels.index('_'))
    return ctc_alignment

def merge_transcripts(params):

    global data, labels, utterances, references, args

    th, service_weight, blank_conf = params
    th_blank = 10*th
    new_probs_list = []
    for utt in utterances:

        ds2_probs = data[utt]['ds2_probs']
        ctc_alignment = data[utt]['alignment']
        word_confs = data[utt]['word_confs']
        frame_confs = get_frame_lvl_cnfs(ctc_alignment, word_confs, labels, blank_conf)
        new_probs = fine_merge(ds2_probs, ctc_alignment, frame_confs, th, 
            th_blank, service_weight, labels.index('_'))
        new_probs_list.append(new_probs)

    new_transcipts = ctc_beam_decode(new_probs_list, labels, args.lm_path, labels.index('_'),
        args.lm_alpha, args.lm_beta, args.beam_width)

    wer = get_wer(references, new_transcipts)
    return th, service_weight, blank_conf, wer


def main():

    global data, utterances, references, labels, args    
    args = parse_args()

    with open(args.labels) as label_file:
        labels = json.load(label_file)

    data = np.load(args.data, allow_pickle=True)
    df_utt = pd.read_csv(args.utterances, delimiter='\t')    
    df_utt = df_utt[df_utt['file_name'].isin(data.keys())] #TODO
    utterances = df_utt['file_name'].to_list()
    references = df_utt['transcript'].to_list()
    references = [normalize_string(text, labels[1:]) for text in references]

    print("Computing ctc alignments service transcripts...")
    with Pool(multiprocessing.cpu_count()) as pool:
        alignments = list(tqdm(pool.imap(get_alignments, utterances), total=len(utterances)))

    for utt, alignment in zip(utterances, alignments):
        data[utt]['alignment'] = alignment

    cand_ths = np.linspace(args.th_from, args.th_to, args.num_th)
    cand_service_weights = np.linspace(args.service_weight_from, \
        args.service_weight_to, args.num_service_weights)
    cand_blank_confs = np.linspace(args.blank_conf_from, \
        args.blank_conf_to, args.num_blank_confs)
    params_grid = [(th, service_weight, blank_conf) for th in cand_ths
                   for service_weight in cand_service_weights 
                   for blank_conf in cand_blank_confs]

    scores = []
    with Pool(multiprocessing.cpu_count()) as pool:
        for params in tqdm(pool.imap(merge_transcripts, params_grid), total=len(params_grid)):
            scores.append(list(params))

    print("Saving tuning results to finetune.".format(args.output_path))
    with open(args.output_path, "w") as fh:
        json.dump(scores, fh)

    min_results = min(scores, key=lambda x: x[-1]) 
    print("Best Params:\nThreshold: %.12f \nService Weight: %.2f"\
        " \nBlank Conf %.2f \nWER: %.6f" % tuple(min_results))

if __name__ == "__main__":
    main()