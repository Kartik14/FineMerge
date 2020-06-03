import json
import pickle
import glob
import argparse
import os
import sys
from os.path import join, basename, dirname

sys.path.append(dirname(dirname(os.path.abspath(__file__))))
from utils import parse_text


def main():
    
    parser = argparse.ArgumentParser(
        description="create pickle file containing the topN transcipts for each utterance from google API"
    )
    parser.add_argument(
        "--json_dir",
        help="Path to dir containing output json from Google API",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        help="Path to save output pickle file",
        type=str,
        required=True,
    )

    dataset = {}

    args = parser.parse_args()
    json_files = glob.glob(join(args.json_dir, '*.json'))

    for json_pth in json_files:

        file_name = basename(json_pth)
        try:
            with open(json_pth) as fd:
                data = json.load(fd)
        except:
            print(json_pth)
            continue

        for rank, alt in enumerate(data['results'][0]['alternatives']):

            transcript = parse_text(alt['transcript'])
            confidence = alt['confidence']
            if rank == 0:    
                dataset[file_name] = {'transcripts' : [transcript], 'confidences' : [confidence]}
            else:
                dataset[file_name]['transcripts'].append(transcript)
                dataset[file_name]['confidences'].append(confidence)

    with open(args.output_path, 'wb') as fd:
        pickle.dump(dataset, fd)

if __name__ == '__main__':
    main()