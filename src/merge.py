import numpy as np

def fine_merge(probs, service_seq, service_conf, threshold, 
    threshold_blank, service_weight=0.5, blank=0):

  T = probs.shape[0]
  greedy_out = list(np.argmax(probs, axis=1))
  
  new_probs = np.zeros(probs.shape)
  one_hot_service = np.zeros(probs.shape)
  one_hot_service[np.arange(T),service_seq] = 1

  for t in range(T):
    
    if service_seq[t] == greedy_out[t]:
      new_probs[t] = probs[t]      
    else:
      if (service_seq[t] == blank and probs[t, service_seq[t]] > threshold_blank) \
        or (service_seq[t] != blank and probs[t, service_seq[t]] > threshold):
        weight = service_weight * service_conf[t]
        new_probs[t] = (1 - weight) * probs[t] + weight * one_hot_service[t]
      else:
        new_probs[t] = probs[t]

  return new_probs

def get_frame_lvl_cnfs(alignment, word_confs, labels, blank_conf=0.8):

    word_idx = 0
    frame_confs = []
    t = 0
    while t < len(alignment):
      if alignment[t] != labels.index('_'):
        if alignment[t] == labels.index(' ') and alignment[t] != alignment[t-1]:
          word_idx += 1
        frame_confs.append(word_confs[word_idx])
      else:
        frame_confs.append(blank_conf)
      t += 1

    return frame_confs
