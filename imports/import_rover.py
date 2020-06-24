import sys
from os.path import join, basename, dirname, abspath
import argparse
import numpy as np
import json
import subprocess
from os.path import join
from tqdm import tqdm
import pandas as pd
import multiprocessing
from multiprocessing.pool import Pool

sys.path.append(dirname(dirname(abspath(__file__))))
from ctc_alignment import ctc_align
from decode import ctc_beam_decode
from utils import wer, normalize_string, parse_text
from import_fineMerge_GoogAPI import align_word_confs

def parse_args():

    parser = argparse.ArgumentParser(
        description="Import data for rover"
    )
    parser.add_argument(
        "--ds2_probs",
        help="Path to ds2 output probs pickle file"
            " format: list of (fname, probs) tuples",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--service_output",
        help="Path to service output "
            " format: tab separated fname, transcript, word cnfs",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--labels",
        help="Path to label json files containing ordered list "
            "of output labels mapping to ds2 probs",
        type=str,
        required=False,
        default='../labels_char.json'
    )
    parser.add_argument(
        "--output_path",
        help="Path to save the imported dataset",
        type=str,
        required=True,        
    )
    parser.add_argument(
        "--utterances",
        help="Path to file containing list of tab separated utterance, reference",
        type=str,
        required=True,        
    )
    parser.add_argument(
        "--lm_alpha",
        help="",
        type=float,
        required=True,        
    )
    parser.add_argument(
        "--lm_beta",
        help="",
        type=float,
        required=True,        
    )
    parser.add_argument(
        "--beam_size",
        help="",
        type=int,
        required=False,
        default=256,        
    )
    parser.add_argument(
        "--lm_path",
        help='path to lm for decoding',
        type=str,
        required=True,
    )

    args = parser.parse_args()
    return args

def get_word_confidence(sample):

    global labels
    utterance, probs, transcript = sample

    smoothen_probs = probs + 1e-20
    smoothen_probs = smoothen_probs/np.sum(smoothen_probs, axis=1, keepdims=1)
    transcript_idx = [labels.index(token) for token in list(transcript)]
    alignment = ctc_align(smoothen_probs, transcript_idx, labels.index('_'))
    token2idx = dict([(x,i) for i,x in enumerate(labels)])

    word_confs = []
    char_prob = None
    char_count = None
    running_word_prob = 0.0
    running_word_chars = 0.0
    new_word = True    
    token_seq = [x for i,x in enumerate(alignment) if \
        x != token2idx['_'] and (i == 0 or (alignment[i-1] != x))]

    if alignment[0] != token2idx['_']:
        char_prob = probs[0, alignment[0]]
        char_count = 1
        new_word = False

    for t in range(1, probs.shape[0]):
        if alignment[t] != token2idx['_']: # not blank
            if alignment[t] != alignment[t-1]:
                if alignment[t] == token2idx[' ']:
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
                if alignment[t] != token2idx[' ']:
                    char_prob = char_prob + probs[t, alignment[t]]
                    char_count += 1

    if token_seq[-1] != token2idx[' ']:           
        running_word_prob = running_word_prob + (char_prob/char_count)
        running_word_chars += 1
        word_confs.append(running_word_prob/running_word_chars)

    return word_confs

def main():

    args = parse_args()

    global labels
    with open(args.labels) as label_file:
        labels = json.load(label_file)

    df_service = pd.read_csv(args.service_output, delimiter='\t').set_index('file_name')

    df_utt = pd.read_csv(args.utterances, delimiter='\t')
    final_utterances = df_utt['file_name'].to_list()
    df_utt.set_index('file_name', inplace=True)

    smoothen_val = 1e-20
    ds2_logits = args.ds2_probs
    data = np.load(ds2_logits, allow_pickle=True)

    print("Getting ds2 transcripts")
    utterances, probs_list = zip(*data)
    ds2_transcripts = ctc_beam_decode(probs_list, labels, args.lm_path, labels.index('_'),
        args.lm_alpha, args.lm_beta, args.beam_size)

    ds2_data = list(zip(utterances, probs_list, ds2_transcripts))

    print("Getting ds2 word level confidence values")
    with Pool(multiprocessing.cpu_count()) as pool:
        ds2_word_confs = list(tqdm(pool.imap(get_word_confidence, ds2_data), total=len(ds2_data)))

    with open(args.output_path,'w') as fd:
        fd.write('fname\treference\tservice_transcript\tservice_confs\tds2_transcript\tds2_confs\n')
        for fname in final_utterances:
            
            ref = normalize_string(df_utt.loc[fname]['transcript'], labels[1:])
            service_transcript, service_conf = df_service.loc[fname][['transcript', 'word_confs']]
            norm_service_transcript = normalize_string(service_transcript, labels[1:])
            aligned_service_conf =  align_word_confs(service_transcript, norm_service_transcript, service_conf)
            aligned_service_conf = ' '.join(aligned_service_conf)
            # norm_service_transcript, aligned_service_conf = parse_text(service_transcript, service_conf)
            ds2_conf = ds2_word_confs[utterances.index(fname)]
            ds2_conf = ' '.join([str(x) for x in ds2_conf])
            ds2_transcript = ds2_transcripts[utterances.index(fname)]
            fd.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(fname, ref, norm_service_transcript, \
                aligned_service_conf, ds2_transcript, ds2_conf))

if __name__ == "__main__":
    main()


        

    

    

        
