import argparse
import json
import pickle
import numpy as np

from utils import parse_text, parse_text2

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
        help="Path to ASR service output in the where each line contains tab serparated fields "
             "filename, reference, transcript, word confidences",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--labels",
        help="Path to labels json files containing ordered (w.r.t. to ds2 output) list of output labels",
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

    with open(args.labels) as label_file:
        labels = json.load(label_file)

    service_map = {}
    with open(args.service_output) as fd:
        lines = fd.read().splitlines()

    for line in lines[1:]:
        file_name, reference, transcript, word_confs = line.split('\t')
        reference = parse_text(reference)
        transcript, word_confs = parse_text2(transcript, word_confs.split())
        word_confs = [float(conf) for conf in word_confs.split()]
        assert len(transcript.split()) == len(word_confs)
        service_map[file_name] = {'reference':reference, 'service_transcript':transcript, 'word_confs':word_confs}
        
    dataset = {}
    data = np.load(args.ds2_probs, allow_pickle=True)
    for file_name, probs in data:
        dataset[file_name] = service_map[file_name[:-3] + 'mp3']
        dataset[file_name]['ds2_probs'] = probs

    with open(args.output_path,'wb') as fd:
        pickle.dump(dataset, fd)

if __name__ == '__main__':
    main()