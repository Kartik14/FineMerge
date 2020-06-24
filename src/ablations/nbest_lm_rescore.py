import argparse
import kenlm
import json
from os.path import join, basename, dirname, abspath
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from multiprocessing.pool import Pool
import sys

sys.path.append(dirname(dirname(abspath(__file__))))
from utils import normalize_string, levenshtein, wer, cer, parse_text

def parse_args():

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
        "--labels",
        help="Path to ds2 output labels",
        type=str,
        required=False,
        default='../labels_char.json'
    )
    parser.add_argument(
        "--utterances",
        help="Path to file containing list of utterance to generate modified transcripts",
        type=str,
        default=None,
        required=True,        
    )
    parser.add_argument(
        "--alpha",
        help="lm score weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--beta",
        help="rank (based on output order from service) weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--finetune",
        help="for finetuning on the given dataset",
        action='store_true',
    )
    parser.add_argument(
        '--alpha-from', 
        type=float, 
        help="lm score weight",
        default=0.0
    )
    parser.add_argument(
        '--alpha-to',
        default=1.0, 
        type=float, 
        help="lm score weight",
    )
    parser.add_argument(
        '--beta-from', 
        default=0.0, 
        type=float,
        help="rank (based on output order from service) weight",
    )
    parser.add_argument(
        '--beta-to', 
        default=1, 
        type=float,
        help="rank (based on output order from service) weight",
    )
    parser.add_argument(
        '--num-alphas', 
        default=101, 
        type=int, 
        help='Number of alpha candidates for tuning'
    )
    parser.add_argument(
        '--num-betas', 
        default=101, 
        type=int,
        help='Number of beta candidates for tuning'
    )
    args = parser.parse_args()
    return args

def get_nbest_preds(params):

    global data, utterances, model
    alpha, beta = params

    nbest_preds = []
    for i, utt in enumerate(utterances):
        
        transcripts = data[utt]['transcripts']
        confidences = data[utt]['confidences']
        best_score = -float('inf')
        best_transcript = None
        for rank, alt in enumerate(zip(transcripts, confidences)):
            transcript, confidence = alt
            score = confidence + alpha*model.score(transcript) - beta*rank
                
            if score > best_score:
                best_score = score
                best_transcript = transcript

        nbest_preds.append(best_transcript)
    
    return nbest_preds

def finetune_nbest(params):

    global references
    nbest_preds = get_nbest_preds(params)
    return params + (wer(references, nbest_preds),)


def main():

    global data, utterances, references, model
    args = parse_args()
    with open(args.labels) as label_file:
        labels = json.load(label_file)

    LM = args.lm_path
    model = kenlm.LanguageModel(LM)
    data = np.load(args.dataset, allow_pickle=True)
    
    df_utt = pd.read_csv(args.utterances, delimiter='\t')
    utterances = df_utt['file_name'].to_list()
    references = df_utt['transcript'].to_list()
    references = [normalize_string(text, labels[1:]) for text in references]    

    if args.finetune:

        p = Pool(multiprocessing.cpu_count())
        cand_alphas = np.linspace(args.alpha_from, args.alpha_to, args.num_alphas)
        cand_betas = np.linspace(args.beta_from, args.beta_to, args.num_betas)
        params_grid = [(float(alpha), float(beta)) for alpha in cand_alphas
                   for beta in cand_betas]
        scores = []
        for params in tqdm(p.imap(finetune_nbest, params_grid), total=len(params_grid)):
            scores.append(list(params))

        min_results = min(scores, key=lambda x: x[2]) 
        print("Best Params:\nAlpha: %f \nBeta: %f \nWER: %f" % tuple(min_results))
    
    else:
        
        nbest_preds = get_nbest_preds((args.alpha, args.beta))
        print("Num utterances : {}".format(len(utterances)))
        print("FINAL WER: {}, CER: {}".format(wer(references, nbest_preds), cer(references, nbest_preds)))

if __name__ == '__main__':
    main()

    


    


    



