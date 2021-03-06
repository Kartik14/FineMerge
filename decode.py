import torch 
import numpy as np 
from tqdm import tqdm
from ctcdecode import CTCBeamDecoder

def greedy_decode(prob_list, labels):

    decoded_output = []
    for probs in prob_list:
        greedy_out = np.argmax(probs, axis=1)           
        greedy_out = [x for i,x in enumerate(greedy_out) \
            if x != labels.index('_') and (i == 0 or x != greedy_out[i-1])]
        greedy_out = ''.join([labels[tid] for tid in greedy_out])
        decoded_output.append(greedy_out)
    return decoded_output

def ctc_beam_decode(prob_list, labels, lm_path,
    blank_id, alpha, beta, beam_size):

    decoder = CTCBeamDecoder(labels, beam_width=beam_size,
            blank_id=blank_id, model_path=lm_path, alpha=alpha,
            beta=beta, num_processes=24)
    return ctc_beam_decode_batch(prob_list, decoder, labels)    

def ctc_beam_decode_batch(prob_list, decoder, labels, batch_size=256):

    decoded_output = []
    probs_seqs = []
    for i, probs in enumerate(prob_list):
        probs_seqs.append(probs)
        if len(probs_seqs) == batch_size or i == len(prob_list) - 1:
            longest_sample = max(probs_seqs, key=lambda x: x.shape[0])
            longest_len = longest_sample.shape[0]
            curr_batch_size = len(probs_seqs)
            probs_tensor = torch.zeros((curr_batch_size, longest_len, longest_sample.shape[1]))
            size_tensor = torch.IntTensor(curr_batch_size)
            for i in range(curr_batch_size):
                sample = probs_seqs[i]
                sample_len = sample.shape[0]
                probs_tensor[i,:sample_len,:] = torch.FloatTensor(sample)
                size_tensor[i] = sample_len
            beam_result, beam_scores, timesteps, out_seq_len = decoder.decode(probs_tensor, size_tensor)
            for i in range(curr_batch_size):
                out_tokens = list(beam_result[i,0,0:out_seq_len[i,0]].numpy())
                out_transcript = ''.join([labels[token_id] for token_id in out_tokens])
                decoded_output.append(out_transcript)
            probs_seqs = []
    
    return decoded_output


    
