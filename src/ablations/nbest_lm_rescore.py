import argparse
import kenlm
import json
from os.path import join, basename, dirname, abspath
import numpy as np
from tqdm import tqdm
from multiprocessing.pool import Pool
import sys

sys.path.append(dirname(dirname(abspath(__file__))))
from utils import parse_text, levenshtein, wer, cer

def main():

    parser = argparse.ArgumentParser(
        description="Perform n-best lm rescoring on the output transcripts"
    )
    parser.add_argument(
        "--dataset",
        help="Path to preprocessed data pickle file",
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
        help="Path to file containing list of utterance to generate modified transcripts"
             ", if not provided transcipt generated for all utterances",
        type=str,
        default=None,
        required=False,        
    )
    parser.add_argument(
        "--alpha",
        help="lm score weight",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--beta",
        help="rank (based on output order from service) weight",
        type=float,
        required=True,
    )

    args = parser.parse_args()

    LM = args.lm_path
    model = kenlm.LanguageModel(LM)

    alpha = args.alpha
    beta = args.beta

    data = np.load(args.dataset, allow_pickle=True)
    if args.utterances is not None:
        with open(args.utterances) as fd:
            lines = fd.read().splitlines()
            utterances = [line.split('\t')[0] for line in lines]
            references = [line.split('\t')[1] for line in lines]
    else:
        utterances = data.keys()

    nbest_preds = []
    for i, utt in tqdm(enumerate(utterances), total = len(utterances)):
        
        transcripts = data[utt[:-4]]['transcripts']
        confidences = data[utt[:-4]]['confidences']
        best_score = -float('inf')
        best_transcript = None
        for rank, alt in enumerate(zip(transcripts, confidences)):
            transcript, confidence = alt
            score = confidence + alpha*model.score(transcript) - beta*rank
                
            if score > best_score:
                best_score = score
                best_transcript = transcript

        nbest_preds.append(best_transcript)

    print("FINAL WER: {}, CER: {}".format(wer(references, nbest_preds), cer(references, nbest_preds)))

if __name__ == '__main__':
    main()

    


    


    



