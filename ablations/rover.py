import argparse
import numpy as np
import json
import subprocess
import random
import re
from os.path import join
from tqdm import tqdm
import pickle

from ctc_alignment import ctc_align
from utils import levenshtein, convert_idx_to_tokens

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, help='', default=0)
parser.add_argument('--accent', type=str, help='ind or aus', default=None)
parser.add_argument('--part', type=str, help='val or test', default=None)
parser.add_argument('--conf', type=float, help='null conf for rover', default=0.8)
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--load-alignment', action='store_true')

def get_transcripts(ctm_path):
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

def get_word_confidence(probs, alignment, token2idx):

    word_confs = []
    char_prob = None
    char_count = None
    running_word_prob = 0.0
    running_word_chars = 0.0
    new_word = True    
    token_seq = [x for i,x in enumerate(alignment) if x != token2idx['_'] and (i == 0 or (alignment[i-1] != x))]

    if alignment[0] != token2idx['_']:
        char_prob = probs[0, alignment[0]]
        char_count = 1
        new_word = False

    for t in range(1,probs.shape[0]):
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

def get_wer(refs, hyps):

    assert len(refs) == len(hyps)
    total_wer = 0.0
    total_tokens = 0
    for ref, hyp in zip(refs, hyps):
        total_wer += levenshtein(ref.split(), hyp.split())
        total_tokens += len(ref.split())

    return total_wer / total_tokens


if __name__ == '__main__':

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    accent = args.accent
    part = args.part
    rover_dir = '../data/rover/{}/word/'.format(accent)

    with open('labels-char.json') as label_file:
        labels = json.load(label_file)

    idx2token = dict([(i,x) for i,x in enumerate(labels)])
    token2idx = dict([(x,i) for i,x in enumerate(labels)])

    with open('../dumps/decode/{}/char/fnames_{}.txt'.format(accent, part)) as fd:
        valid_fnames = fd.read().splitlines()

    with open('../dumps/decode/{}/char/ds2_wrds_{}.txt'.format(accent, part)) as fd:
        ds2_transcripts = fd.read().splitlines()    
    
    if args.load_alignment:
        alignment_path = join(rover_dir, 'ctc-alignments-{}.pkl'.format(part))
        print('loading saved alignmets from {}'.format(alignment_path))
        ctc_alignments = np.load(alignment_path, allow_pickle=True)
    else:
        ctc_alignments = []

    # creating ds2 ctm file
    smoothen_val = 1e-20
    ds2_logits = '../data/ds2_output/{}/char/split1/{}-logits.pkl'.format(accent, part)
    data = np.load(ds2_logits, allow_pickle=True)
    ds2_ctm_path = join(rover_dir, 'ds2_{}.ctm'.format(part))
    with open(ds2_ctm_path,'w') as fd:
        for i, sample in tqdm(enumerate(data), total=len(data)):
            fname, probs = sample
            smoothen_probs = probs + smoothen_val
            smoothen_probs = smoothen_probs/np.sum(smoothen_probs, axis=1, keepdims=1)
            if fname not in valid_fnames:
                continue
            ds2_out = ds2_transcripts[valid_fnames.index(fname)]
            if args.load_alignment:
                ctc_alignment = ctc_alignments[valid_fnames.index(fname)]
            else:
                target_seq = convert_idx_to_tokens(list(ds2_out), token2idx)
                ctc_alignment = ctc_align(smoothen_probs, target_seq, blank=labels.index('_'))
                ctc_alignments.append(ctc_alignment)
            word_confs = get_word_confidence(probs, ctc_alignment, token2idx)
            assert len(word_confs) == len(ds2_out.split()), print("{}\n{}\n{}".format(fname, ds2_out, word_confs))
            for word, conf in zip(ds2_out.split(), word_confs):
                # 0.0 just a filler for time which is not used
                fd.write('{} a 0.0 0.0 {} {}\n'.format(fname, word, conf))

    if not args.load_alignment:
        alignment_path = join(rover_dir, 'ctc-alignments-{}.pkl'.format(part))
        with open(alignment_path,'wb') as fd:
            pickle.dump(ctc_alignments, fd)
    
    # sorting in order as required by ctm file
    bashCommand = "sort +0 -1 +1 -2 +2nb -3 -s -o {} {}".format(ds2_ctm_path, ds2_ctm_path)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    print("Created DS2 ctm file.")

    # creating service ctm file
    with open('../data/service_output/{}/jasper/out_conf.txt'.format(accent),'r') as fd:
        lines = fd.readlines()
    file2service = {}
    for i in range(0,len(lines),6):
        fname = lines[i].strip()[:-3]
        ref = lines[i+1].strip()[5:]
        hyp = lines[i+2].strip()[5:]
        sent_conf = float(lines[i+3].strip())
        conf = lines[i+4].strip()
        
        file2service[fname] = (ref, hyp, sent_conf, conf)

    service_ctm_path = join(rover_dir, 'service_video_{}.ctm'.format(part))
    with open(service_ctm_path,'w') as fd:
        for fname in valid_fnames:
            words = file2service[fname[:-3]][1].split()
            confs = file2service[fname[:-3]][3]
            confs = [float(x) for x in confs.split()]

            assert len(words) == len(confs)
            for w, conf in zip(words, confs):
                fd.write("{} a 0.0 0.0 {} {}\n".format(fname, w, conf))

    # service_json_dir = '../data/API_output/{}/enUS_video/json'.format(accent)
    # service_ctm_path = join(rover_dir, 'service_video_{}.ctm'.format(part))
    # with open(service_ctm_path,'w') as fd:
    #     for fname in valid_fnames:
    #         with open(join(service_json_dir, fname[:-3] + 'json')) as f:
    #             data = json.load(f)
    #         if data['results'][0]['alternatives'] == []: #TODO: just handling a special case here
    #             fd.write("{} a 0.0 0.0 a 0.0\n".format(fname))
    #             continue

    #         words = data['results'][0]['alternatives'][0]['words']
    #         for word in words:
    #             w = word['word']
    #             confidence = word['confidence']
    #             fd.write("{} a 0.0 0.0 {} {}\n".format(fname, w, confidence))
    
    # sorting in order as required by ctm file
    bashCommand = "sort +0 -1 +1 -2 +2nb -3 -s -o {} {}".format(service_ctm_path, service_ctm_path)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    print("Created Service ctm file.")

    # calling rover on the ctm files

    if args.finetune:
        assert part == 'val', print('finetune over VAL set!')
        confs = np.linspace(0.5,1,num=20)
        best_wer = float('inf')
        best_conf = None
        for conf in confs:
            output_ctm_path = join(rover_dir, 'out_{}.ctm'.format(part))
            bashCommand = "/exp/sw/kaldi/tools/sctk/bin/rover -f 1 -a 0 -c {} -h {} ctm -h {} ctm -o {} -m maxconf".format(conf, 
                    service_ctm_path, ds2_ctm_path, output_ctm_path)
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

            fname2transcript = get_transcripts(output_ctm_path)
            dump_dir = '../dumps/decode/{}/char/'.format(accent)
            with open(join(dump_dir, 'ref_wrds_{}.txt'.format(part))) as fd:
                refs = fd.read().splitlines()
            
            total_wer = 0.0
            total_tokens = 0
            for ref, fname in zip(refs, valid_fnames):
                hyp = fname2transcript[fname]
                total_wer += levenshtein(ref.split(), hyp.split())
                total_tokens += len(ref.split())

            curr_wer = total_wer/total_tokens
            if curr_wer < best_wer:
                best_wer = curr_wer
                best_conf = conf
            
        print('Best NULL Conf: {}, WER: {}'.format(best_conf, best_wer))
    
    else:

        output_ctm_path = join(rover_dir, 'out_{}.ctm'.format(part))
        bashCommand = "/exp/sw/kaldi/tools/sctk/bin/rover -f 1 -a 0 -c {} -h {} ctm -h {} ctm -o {} -m maxconf".format(args.conf, 
                service_ctm_path, ds2_ctm_path, output_ctm_path)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        
        # getting the final preds from the output ctm file
        fname2transcript = get_transcripts(output_ctm_path)
        with open(join(rover_dir,'preds_jasper_{}.txt'.format(part)),'w') as fd:
            for fname in valid_fnames:
                fd.write(fname2transcript[fname] + '\n')

        dump_dir = '../dumps/decode/{}/char/'.format(accent)
        with open(join(dump_dir, 'ref_wrds_{}.txt'.format(part))) as fd:
            refs = fd.read().splitlines()
        with open(join(dump_dir, 'goog_wrds_jasper_{}.txt'.format(part))) as fd:
            sers = fd.read().splitlines()    

        print("SERVICE WER = {}".format(get_wer(refs, sers)))
        print("DS2 WER = {}".format(get_wer(refs, ds2_transcripts)))

        hyps = []
        for fname in valid_fnames:
            hyps.append(fname2transcript[fname])
        print("ROVER WER = {}".format(get_wer(refs, hyps)))


