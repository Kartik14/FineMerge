import argparse
import os
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
from multiprocessing.pool import Pool

from ctc_alignment import ctc_align
from merge import get_frame_lvl_cnfs, fine_merge
from decode import ctc_beam_decode, greedy_decode

def parse_args():

    parser = argparse.ArgumentParser(
        description="Generate modified transcripts using FineMerge"
    )
    parser.add_argument(
        "--data",
        help="Path to data, list of dict elements (format in README)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        help="Path to save file containing new transcipts",
        type=str,
        default='preds.txt'        
    )
    parser.add_argument(
        "--threshold",
        help="service probability thresold",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--service_weight",
        help="service weight for mixing",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--blank_conf",
        help="probability of blank in service ctc alignment",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--num_workers",
        help="number of worker processes",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--lm_path",
        help="Path to arpa lm file to use for decoding",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--lm_alpha",
        help='Language model weight start tuning',
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--lm_beta",
        help='Language model word bonus (all words) start tuning',
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--beam_size",
        help='beam size used during decoding',
        type=int,
        default=256,
    )
    parser.add_argument(
        "--labels",
        help="Path to labels json files containing ordered list"
             "of output labels mapping to ds2 probs",
        type=str,
        required=False,
        default='labels_char.json',
    )
    args = parser.parse_args()
    return args

def get_merged_transcript(utt):

    global args, data, labels
    service_transcript = data[utt]['service_transcript']
    word_confs = data[utt]['word_confs']
    ds2_probs = data[utt]['local_probs']

    smoothen_probs = ds2_probs + 1e-20
    smoothen_probs = smoothen_probs/np.sum(smoothen_probs, axis=1, keepdims=1)
    service_indices = [labels.index(token) for token in list(service_transcript)]
    ctc_alignment = ctc_align(smoothen_probs, service_indices, labels.index('_'))
    frame_confs = get_frame_lvl_cnfs(ctc_alignment, word_confs, labels, args.blank_conf)
    new_probs = fine_merge(ds2_probs, ctc_alignment, frame_confs, args.threshold, 
        10*args.threshold, args.service_weight, labels.index('_'))

    return new_probs

if __name__ == "__main__":

    global args, data, labels
    args = parse_args()
    data = np.load(args.data, allow_pickle=True)
    utterances = list(data.keys())
    with open(args.labels) as label_file:
        labels = json.load(label_file)

    print("Applying Fine Merge algorithm to modify client probs....")
    with Pool(args.num_workers) as pool:
        new_probs_list = list(tqdm(pool.imap(get_merged_transcript, utterances), total=len(utterances)))

    print("Decoding the modified probs to obtain new transcripts")
    new_transcipts = ctc_beam_decode(new_probs_list, labels, args.lm_path, labels.index('_'),
        args.lm_alpha, args.lm_beta, args.beam_size)

    print("Saving the new transcripts to {}".format(args.output_path))
    df = pd.DataFrame(list(zip(utterances, new_transcipts)), columns=['utt_id', 'FineMerge_Transcript'])
    df.to_csv(args.output_path, sep='\t', index=False)


    
