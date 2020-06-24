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
    hypothesis = data['beam_hypotheses']
    beams = data['beams']
    beam_scores = data['beam_scores']
    vocab = data['vocab']
    vocab.append("_")

    dataset = {}
    for i in tqdm(range(len(fnames)), total=len(fnames)):
        file_name = fnames[i]
        for transcript, confidence in zip(beams[i], beam_scores[i]):  
            confidence = -confidence.numpy()
            if file_name not in dataset.keys():
                dataset[file_name] = {'transcripts' : [transcript], 'confidences' : [confidence]}
            else:
                dataset[file_name]['transcripts'].append(transcript)
                dataset[file_name]['confidences'].append(confidence)

    with open(args.output_path, 'wb') as fd:
        pickle.dump(dataset, fd)

        
