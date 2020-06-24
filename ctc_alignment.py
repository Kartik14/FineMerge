import numpy as np

NEG_INF = float('-inf')

def ctc_align(probs, seq, blank=0):

    probs = np.log(probs)
    T, O = probs.shape
    N = len(seq)
    v = np.zeros((T, 2*N+1))
    parent = np.zeros((T, 2*N+1), dtype=np.int32)

    s = np.zeros(2*N+1, dtype = np.int32)
    for i in range(2*N+1):
        if i%2 == 0:
            s[i] = blank
        else:
            s[i] = seq[(i-1)//2]

    
    for i in range(2,2*N+1):
        v[0,i] = NEG_INF

    v[0,0] = probs[0,s[0]]
    v[0,1] = probs[0,s[1]]

    for t in range(1,T):
        for r in range(2*N+1):
            if r == 0:
                v[t,0] = v[t-1,0] + probs[t,s[0]]
                parent[t,r] = 0
            elif r == 1:
                v[t,r] = max(v[t-1,r], v[t-1,r-1]) + probs[t,s[r]]
                if v[t-1,r] > v[t-1,r-1]:
                    parent[t,r] = r
                else:
                    parent[t,r] = r-1
            else:
                if s[r] == s[r-2]:
                    v[t,r] = max(v[t-1,r], v[t-1,r-1]) + probs[t,s[r]]
                    if v[t-1,r] > v[t-1,r-1]:
                        parent[t,r] = r
                    else:
                        parent[t,r] = r-1
                else:
                    v[t,r] = max(v[t-1,r], v[t-1,r-1], v[t-1,r-2]) + probs[t,s[r]]
                    if v[t-1,r] > max(v[t-1,r-1],v[t-1,r-2]):
                        parent[t,r] = r
                    elif v[t-1,r-1] > max(v[t-1,r],v[t-1,r-2]):
                        parent[t,r] = r-1
                    else:
                        parent[t,r] = r-2 

    
    final_alignment = []
    if v[T-1,2*N-1] > v[T-1,2*N]:
        final_alignment.append(2*N-1) 
    else:
        final_alignment.append(2*N)

    t = T-1
    while t > 0:
        final_alignment.append(parent[t,final_alignment[-1]])
        t = t-1

    final_alignment = [s[x] for x in final_alignment]
    return final_alignment[::-1]