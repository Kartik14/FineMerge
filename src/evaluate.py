import argparse
import json
import numpy as np
from tqdm import tqdm
import multiprocessing
from multiprocessing.pool import Pool

from ctc_alignment import ctc_align
from merge import get_frame_lvl_cnfs, fine_merge
from decode import ctc_beam_decode
from utils import wer, normalize_string

data = {}
labels = []
params = {}

def get_merged_transcript(utt):

    service_transcript = data[utt]['service_transcript']
    word_confs = data[utt]['word_confs']
    # word_confs = [1.0]*len(word_confs)
    ds2_probs = data[utt]['ds2_probs']

    smoothen_probs = ds2_probs + 1e-20
    smoothen_probs = smoothen_probs/np.sum(smoothen_probs, axis=1, keepdims=1)
    service_indices = [labels.index(token) for token in list(service_transcript)]
    ctc_alignment = ctc_align(smoothen_probs, service_indices, labels.index('_'))
    frame_confs = get_frame_lvl_cnfs(ctc_alignment, word_confs, labels, params['blank_conf'])
    new_probs = fine_merge(ds2_probs, ctc_alignment, frame_confs, params['threshold'], 
        params['blank_threshold'], params['service_weight'], labels.index('_'))

    return new_probs

def main():

    parser = argparse.ArgumentParser(
        description="Generate modified transcripts using FineMerge"
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
        help="Path to file containing list of tab separated utterance,"
             " reference to generate modified transcripts",
        type=str,
        required=True,        
    )
    args = parser.parse_args()

    global labels
    with open(args.labels) as label_file:
        labels = json.load(label_file)

    global params
    with open(args.params_config) as params_file:
        params = json.load(params_file)

    global data
    data = np.load(args.dataset, allow_pickle=True)
    with open(args.utterances) as fd:
        lines = fd.read().splitlines()
        utterances = [line.split('\t')[0][:-3] + 'wav' for line in lines]
        # references = [line.split('\t')[1] for line in lines]    

    references = [data[utt]['reference'] for utt in utterances]
    service_transcripts = [data[utt]['service_transcript'] for utt in utterances]
    probs_list = [data[utt]['ds2_probs'] for utt in utterances]

    print('Getting transcripts for DS2...')
    ds2_transcripts = ctc_beam_decode(probs_list, labels, args.lm_path, labels.index('_'),
        params['ds2_lm_alpha'], params['ds2_lm_beta'], params['beam_size'])


    print("Applying FineMerge to DS2 probs using service transcripts...")
    with Pool(multiprocessing.cpu_count()) as pool:
        new_probs_list = list(tqdm(pool.imap(get_merged_transcript, utterances), total=len(utterances)))

    print("Getting the final transcripts...")
    new_transcipts = ctc_beam_decode(new_probs_list, labels, args.lm_path, labels.index('_'),
        params['lm_alpha'], params['lm_beta'], params['beam_size'])
    
    print("\nSER WER: {}".format(wer(references, service_transcripts)))
    print("DS2 WER: {}".format(wer(references, ds2_transcripts)))
    print("NEW WER : {}\n".format(wer(references, new_transcipts)))

    with open(args.output_path, 'w') as fd:
        for utt, ref, ser, ds2, pred in zip(utterances, references, service_transcripts, \
            ds2_transcripts, new_transcipts):
            fd.write("UTT: {}\nREF: {}\nSER: {}\nDS2: {}\nNEW: {}\n\n".format(utt, ref, ser, ds2, pred))

if __name__ == '__main__':
    main()
