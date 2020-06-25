Implementation of the **FineMerge** algorithm introduced in our paper [Black-box Adaptation of ASR for Accented Speech](http://arxiv.org/abs/2006.13519)

## Setup

Clone the repository and run 

```
pip install -r requirements.txt
```

Additionally for ctc beam decoding, install ctcdecode:

```
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
```

## Apply FineMerge Algorithm

To use the FineMerge algorithm, first create a pickle file containing a list of dict elements where each element is of the form:

>{ 'utt_id' : 'local_probs' : ..., 'service_transcript' : ..., 'word_confs' : ... }
             
Here, local_probs is the probability matrix output from local model, service_transcipt and word_confs refer to the transcript and correspoding word confidence values from the ASR service. The length of the service transcript and word confs must be equal.

Run `python main.py --help` on further instruction on how to use FineMerge algorithim.

