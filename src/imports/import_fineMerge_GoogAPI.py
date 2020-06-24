import sys
import argparse
import json
import pickle
import numpy as np
from os.path import join, basename, dirname, abspath
from tqdm import tqdm

sys.path.append(dirname(dirname(abspath(__file__))))
from utils import normalize_string, align, parse_text

def align_word_confs(org_trans, norm_trans, word_confs):

    word_confs = word_confs.split()
    org_trans = org_trans.lower().split()
    norm_trans = norm_trans.lower().split()
    assert len(org_trans) == len(word_confs)

    align_trans1, align_trans2 = align(org_trans, norm_trans)

    aligned_word_confs = []        
    curr_idx = 0
    i = 0
    while i < len(align_trans1):
        if align_trans2[i] == '_':
            i += 1
        elif align_trans1[i] != '_':
            aligned_word_confs.append(word_confs[curr_idx])    
            curr_idx += 1
            i += 1
        else:
            cnt = 1
            while align_trans1[i] == '_':
                i += 1
                cnt += 1
            aligned_word_confs.extend([word_confs[curr_idx]]*cnt)
            curr_idx += 1
            i += 1
    
    assert len(aligned_word_confs) == len(norm_trans)
    return aligned_word_confs
            

def main():

    parser = argparse.ArgumentParser(
        description="Generate dataset for FineMerge"
    )
    parser.add_argument(
        "--ds2_probs",
        help="Path to frame-level token probs pkl file obtained from final layer of ds2,"
             "should be list of tuples containing (file_names, probs)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--service_output",
        help="Path to ASR service output",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--labels",
        help="Path to labels json files containing ordered (w.r.t. to ds2 output) list of output labels",
        type=str,
        required=False,
        default='../labels_char.json'
    )
    parser.add_argument(
        "--output_path",
        help="Path to save the output pickle file",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    with open(args.labels) as label_file:
        labels = json.load(label_file)

    service_map = {}
    with open(args.service_output) as fd:
        lines = fd.read().splitlines()

    for _, line in tqdm(enumerate(lines[1:]), total=len(lines[1:])):
        file_name, transcript, word_confs = line.split('\t')
        norm_transcript = normalize_string(transcript, labels[1:])
        aligned_word_confs =  align_word_confs(transcript, norm_transcript, word_confs)
        # norm_transcript, aligned_word_confs = parse_text(transcript, word_confs)
        aligned_word_confs = [float(conf) for conf in aligned_word_confs]
        service_map[file_name] = {'service_transcript' : norm_transcript, \
            'word_confs' : aligned_word_confs}
        
    dataset = {}
    data = np.load(args.ds2_probs, allow_pickle=True)
    for file_name, probs in data:
        if file_name in service_map.keys():
            dataset[file_name] = service_map[file_name]
            dataset[file_name]['ds2_probs'] = probs

    with open(args.output_path,'wb') as fd:
        pickle.dump(dataset, fd)

if __name__ == '__main__':
    main()