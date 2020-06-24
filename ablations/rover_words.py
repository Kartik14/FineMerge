import sys
import os
from os.path import join, basename, dirname, abspath
import argparse
import numpy as np
import pandas as pd
import json
import subprocess
from os.path import join
from tqdm import tqdm
import multiprocessing
from multiprocessing.pool import Pool

sys.path.append(dirname(dirname(abspath(__file__))))
from utils import wer, cer

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate modified transcripts using FineMerge"
    )
    parser.add_argument(
        "--data",
        help="Path to dataset file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--rover",
        help="Path to rover binary",
        type=str,
        required=False,
        default='/exp/sw/kaldi/tools/sctk/bin/rover',
    )
    parser.add_argument(
        "--conf",
        help="null conf to use for rover",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--finetune",
        help="to fine tune rover params",
        action='store_true'
    )
    parser.add_argument(
        "--conf-from",
        help="null conf to use for rover",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--conf-to",
        help="null conf to use for rover",
        default=1.0,
    )
    parser.add_argument(
        "--num-conf",
        help="null conf to use for rover",
        type=float,
        default=101,
    )
    args = parser.parse_args()
    return args

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
                fname2transcript[curr_fname] = ' '.join(transcript)
            curr_fname = fname
            transcript = [line[4]]
    fname2transcript[curr_fname] = ' '.join(transcript)

    return fname2transcript

def call_rover(conf):

    global fnames, RVR_DIR
    # Calling rover
    bashCommand = "{} -f 1 -a 0 -c {} -h ds2.ctm ctm -h service.ctm ctm "\
        "-o {} -m maxconf".format(RVR_DIR, conf, 'out_{}.ctm'.format(conf))
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    
    # getting the final preds from the output ctm file
    fname2transcript = parse_ctm('out_{}.ctm'.format(conf))
    rover_transcripts = [fname2transcript[fname] for fname in fnames]
    os.remove('out_{}.ctm'.format(conf))

    return rover_transcripts

def main():

    global RVR_DIR, fnames
    args = parse_args()
    RVR_DIR = args.rover
    df = pd.read_csv(args.data, delimiter='\t')

    print('creating ctm files...')
    # create ds2 and service ctm file
    with open('ds2.ctm', 'w') as fd_ds2, open('service.ctm', 'w') as fd_ser:
        for i, row in df.iterrows():

            fname = row['fname']
            ser_transcript, ser_conf = row[['service_transcript', 'service_confs']]
            ds2_transcript, ds2_conf = row[['ds2_transcript', 'ds2_confs']]

            assert len(ser_transcript.split()) == len(ser_conf.split())
            assert len(ds2_transcript.split()) == len(ds2_conf.split())
            for w, c in zip(ser_transcript.split(), ser_conf.split()):
                fd_ds2.write('{} a 0.0 0.0 {} {}\n'.format(fname, w, c))
            for w, c in zip(ds2_transcript.split(), ds2_conf.split()):
                fd_ser.write('{} a 0.0 0.0 {} {}\n'.format(fname, w, c))

    # sorting in order as required by ctm file
    bashCommand = "sort +0 -1 +1 -2 +2nb -3 -s -o ds2.ctm ds2.ctm"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()    
            
    bashCommand = "sort +0 -1 +1 -2 +2nb -3 -s -o service.ctm service.ctm"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()  

    print('created ctm files.\nGenerating rover output...')  

    fnames = df['fname'].to_list()
    references = df['reference'].to_list()
    ser_transcripts = df['service_transcript'].to_list()
    ds2_transcripts = df['ds2_transcript'].to_list()


    if args.finetune:

        p = Pool(multiprocessing.cpu_count())
        params_grid = np.linspace(args.conf_from, args.conf_to, args.num_conf)
        scores = []
        for preds in tqdm(p.imap(call_rover, params_grid), total=len(params_grid)):
            scores.append(wer(references, preds))
            
        scores = zip(params_grid, scores)
        min_results = min(scores, key=lambda x: x[1]) 
        print("Best Params:\nConf : %f \nWER: %f" % tuple(min_results))

    else:
        
        rover_transcripts = call_rover(args.conf)

        print("\nNumber Utterances : {}".format(len(references)))
        print("SER WER = {}, CER = {}\nDS2 WER = {}, CER = {}\nRVR WER = {}, CER = {}\n".format(
            wer(references, ser_transcripts),
            cer(references, ser_transcripts),
            wer(references, ds2_transcripts),
            cer(references, ds2_transcripts),
            wer(references, rover_transcripts),
            cer(references, rover_transcripts)
        ))

        os.remove('service.ctm')
        os.remove('ds2.ctm')

if __name__ == "__main__":
    main()
    


    