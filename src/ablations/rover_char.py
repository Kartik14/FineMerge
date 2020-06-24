import sys
import os
import argparse
import numpy as np
import json
import subprocess
from tqdm import tqdm
import pandas as pd
from os.path import join, basename, dirname, abspath
import multiprocessing
from multiprocessing.pool import Pool

sys.path.append(dirname(dirname(abspath(__file__))))
from utils import wer, cer, normalize_string 
from merge import get_frame_lvl_cnfs, fine_merge

def parse_ctm(ctm_path):
    with open(ctm_path) as fd:
        lines = fd.read().splitlines()

    curr_fname = None
    fname2transcript = {}
    transcript = None
    for line in lines:
        if line == '': #skip empty lines
            continue
        line = line.split()
        fname = line[0]
        if fname == curr_fname:
            transcript.append(line[4])
        else:
            if transcript != None:
                transcript = [c if c != '$' else ' ' for c in transcript]
                fname2transcript[curr_fname] = ''.join(transcript)
            curr_fname = fname
            transcript = [line[4]]
    fname2transcript[curr_fname] = ''.join(transcript)

    return fname2transcript

def get_char_conf(probs, blank_idx=0):

    current_prob = None
    current_cnt = 0
    char_probs = []
    
    greedy_out = np.argmax(probs, axis=1)
    for t in range(probs.shape[0]):
        if greedy_out[t] != blank_idx: 
            if t == 0 or greedy_out[t] != greedy_out[t-1]:
                if current_prob is not None: # skipping the first time a char is encountered
                    char_probs.append(current_prob/current_cnt)
                current_prob = probs[t, greedy_out[t]]
                current_cnt = 1
            else:
                current_prob += probs[t, greedy_out[t]]
                current_cnt += 1
    char_probs.append(current_prob/current_cnt) 

    return char_probs

def get_greedy_transcript(probs, mapping):
    ds2_greedy = np.argmax(probs, axis=1) # greedy output            
    ds2_greedy = [x for i,x in enumerate(ds2_greedy) \
        if x != token2idx['_'] and (i == 0 or x != ds2_greedy[i-1])]
    ds2_greedy = ''.join([mapping[tid] for tid in ds2_greedy])
    return ds2_greedy

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds2_probs",
        help="Path to ds2 probs",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data",
        help="Path to file containing parsed rover data",
        type=str,
        required=True,        
    )
    parser.add_argument(
        "--conf",
        help="null conf to use for rover",
        type=float,
        required=False,
        default=0.8,
    )
    parser.add_argument(
        "--rover",
        help="Path to rover binary",
        type=str,
        required=False,
        default='/exp/sw/kaldi/tools/sctk/bin/rover',
    )
    parser.add_argument(
        "--labels",
        help="Path to labels json files containing ordered list of output labels mapping to ds2 probs",
        type=str,
        required=False,
        default='../labels_char.json',
    )
    args = parser.parse_args()

    with open(args.labels) as label_file:
        labels = json.load(label_file)
    idx2token = dict([(i,x) for i,x in enumerate(labels)])
    token2idx = dict([(x,i) for i,x in enumerate(labels)])

    df_utt = pd.read_csv(args.data, delimiter='\t')
    utterances = df_utt['fname'].to_list()
    references = df_utt['reference'].to_list()
    service_transcripts = df_utt['service_transcript'].to_list()

    # mapping sapce(' ') to a readable char for rover
    space_id = token2idx[' ']
    del idx2token[space_id]
    idx2token[space_id] = '$'

    # creating ds2 ctm file
    ds2_logits = args.ds2_probs
    data = np.load(ds2_logits, allow_pickle=True)
    fname2ds2_prob = dict(data)

    fname2ds2 = {}
    with open('ds2.ctm','w') as fd:
        for fname in utterances:

            probs = fname2ds2_prob[fname]
            char_confs = get_char_conf(probs, token2idx['_']) # getting per char conf
            ds2_str = get_greedy_transcript(probs, idx2token)
            fname2ds2[fname] = ds2_str.replace('$',' ')
            assert len(ds2_str) == len(char_confs)
            for char, conf in zip(ds2_str, char_confs):
                # 0.0 just a filler for time which is not used
                fd.write('{} a 0.0 0.0 {} {}\n'.format(fname, char, conf))
    
    # sorting in order as required by ctm file
    bashCommand = "sort +0 -1 +1 -2 +2nb -3 -s -o ds2.ctm ds2.ctm"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print('created ds2 ctm file')

    print('creating service ctm files...')
    df_utt.set_index('fname', inplace=True)
    # create ds2 and service ctm file
    with open('service.ctm', 'w') as fd_ser:
        for fname in utterances:
            ser_transcript, ser_conf = df_utt.loc[fname][['service_transcript', 'service_confs']]
            assert len(ser_transcript.split()) == len(ser_conf.split())
            words = ser_transcript.split()
            for w, c in zip(words, ser_conf.split()):
                for char in w:
                    fd_ser.write('{} a 0.0 0.0 {} {}\n'.format(fname, char, c))
                if w != words[-1]: # if not last word add space
                    fd_ser.write("{} a 0.0 0.0 {} {}\n".format(fname, '$', c))
    
    # sorting in order as required by ctm file
    bashCommand = "sort +0 -1 +1 -2 +2nb -3 -s -o service.ctm service.ctm"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    print('created service ctm file')

    # calling rover on the ctm files
    bashCommand = "{} -f 1 -a 0 -c {} -h service.ctm ctm"\
        " -h ds2.ctm ctm -o out.ctm -m maxconf".format(args.rover, args.conf)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    
    # getting the final preds from the output ctm file
    fname2rover = parse_ctm('out.ctm')
    
    print("SERVICE CER = {}".format(cer(references, service_transcripts)))

    ds2_transcripts = [fname2ds2[f] for f in utterances]
    print("DS2 CER = {}".format(cer(references, ds2_transcripts)))

    rover_transcripts = [fname2rover[f] for f in utterances]
    print("ROVER CER = {}".format(cer(references, rover_transcripts)))
    
    os.remove('service.ctm')
    os.remove('ds2.ctm')
    os.remove('out.ctm')




    







