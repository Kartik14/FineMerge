import argparse
import os
import subprocess
import json
from utils import normalize_string

def main():

    parser = argparse.ArgumentParser(
        description="Generate binary n-gram lm file given a text file"
    )
    parser.add_argument(
        "--text_file",
        help="Path to text file to train the lm from",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--order",
        help="order of lm to train",
        type=int,
        required=False,
        default=3,
    )
    parser.add_argument(
        "--exclude_text",
        help="Path to text file whose sentences must be excluded from training",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--kenlm_dir",
        help="Path to the kenlm directory required for training n-gram lm",
        type=str,
        required=False,
        default='~/exp/kenlm',
    )
    parser.add_argument(
        "--labels",
        help="Path to char level tokens for parsing",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--lm_path",
        help="Path to the save the trained lm to",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    print("Preparing text for training the lm....")

    with open(args.labels) as label_file:
        labels = json.load(label_file)

    with open(args.text_file) as fd:
        lm_text = fd.read().splitlines()
        lm_text = set([normalize_string(sentence, labels[1:]) for sentence in lm_text])

    with open(args.exclude_text) as fd:
        text_to_exclude = fd.read().splitlines()
        text_to_exclude = set([normalize_string(sentence, labels[1:]) for \
            sentence in text_to_exclude])

    lm_text_final = lm_text - text_to_exclude
    with open('lm_text.txt', 'w') as fd:
        fd.write('\n'.join(lm_text_final))

    print('Build the arpa lm file of order {} ....'.format(args.order))

    command = '{} -o {} < lm_text.txt > lm.arpa'.format(
        os.path.join(args.kenlm_dir, 'build/bin/lmplz'),
        args.order)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    process.communicate()

    command = '{} lm.arpa {}'.format(
        os.path.join(args.kenlm_dir, 'build/bin/build_binary'),
        args.lm_path)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    process.communicate()

    os.remove('lm_text.txt')
    os.remove('lm.arpa')

if __name__ == '__main__':
    main()
    