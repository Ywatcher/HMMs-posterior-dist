"""
Posterior Decoding for Hidden Markov Models

This module implements the forward-backward algorithm (posterior decoding)
for finding the most likely state at each position given an observation
sequence and an HMM.
"""

from typing import List
import numpy as np
from numpy.typing import NDArray


def posterior_decode(observation_seq: List[int], hmm, return_arr=False) -> List[str]:
    """
    Decode an observation sequence using posterior (forward-backward) decoding.

    INPUT:
    - observation_seq: List of observations (integers indexing alphabet)
    - hmm: HMM object with states, initial_state_probs, transition_matrix,
      and emission_matrix

    OUTPUT:
    - state_seq: List containing the sequence of most likely states (state names)

    IMPLEMENTATION NOTES:
    1) The forward and backward matrices are implemented as matrices that are
       transposed relative to the way they are shown in class. Rows correspond
       to observations and columns to states.
    2) After computing the forward/backward probabilities for each observation,
       they are normalized by dividing by their sum. This maintains
       proportionality while avoiding numerical underflow.
    3) The posterior probability matrix is the element-wise product of the
       forward and backward matrices, normalized at each observation.
    """
    # YOUR CODE HERE
    posterior = _posterior_probabilities(observation_seq,hmm)
    pos = _max_position_along_axis(posterior,axis=1)
    if return_arr:
        return pos 
    else: # return str
        return [hmm.states[i] for i in pos]



def normalize(v):
    total = np.sum(v)
    if total > 0:
        # forward_matrix[observation_index] = forward_matrix[observation_index] / total
        return v/total
    else:
        # Handle impossible observation
        # forward_matrix[observation_index] = np.nan
        v[:] = np.nan 
        return v


def _build_forward_matrix(observation_seq: List[int], hmm) -> NDArray[np.float64]:
    """
    Build the forward probability matrix.

    Similar to Viterbi but uses sum instead of max.
    Returns a matrix where rows are observations and columns are states.
    Each entry is normalized to avoid underflow.
    """
    # YOUR CODE HERE
    pi = np.array(hmm.initial_state_probs)
    a = np.array(hmm.transition_matrix)
    b = np.array(hmm.emission_matrix)
    alpha_0 = b[observation_seq[0]] * pi
    alpha_0 = normalize(alpha_0)
    alpha = [alpha_0]
    for t in range(1, len(observation_seq)):
        # print(len(b[obs[t]]),len(alpha[-1]),a.shape)
        alpha_t = (b[observation_seq[t]] * (alpha[-1] @ a))
        alpha_t = normalize(alpha_t)
        # print(len(alpha_t))
        alpha.append(alpha_t)
    forward_matrix = np.vstack(alpha)
    # Normalize to avoid underflow
    return forward_matrix


def _build_backward_matrix(observation_seq: List[int], hmm) -> NDArray[np.float64]:
    """
    Build the backward probability matrix.

    Works backwards from the last observation.
    Returns a matrix where rows are observations and columns are states.
    Each entry is normalized to avoid underflow.
    """
    # YOUR CODE HERE
    T = len(observation_seq)-1 
    pi = np.array(hmm.initial_state_probs)
    a = np.array(hmm.transition_matrix)
    b = np.array(hmm.emission_matrix)
    beta_T = np.ones_like(pi)
    beta_T = normalize(beta_T)
    beta = [beta_T]
    for _ in range(1, len(observation_seq)):
        t = T - _ 
        beta_T_minus_t = np.einsum( 
            "i,ji,i -> j",
           beta[-1], a, b[observation_seq[t+1]]
        )
        beta_T_minus_t = normalize(beta_T_minus_t)
        beta.append(beta_T_minus_t)
    beta = beta[::-1]
    backward_matrix = np.vstack(beta)
    return backward_matrix

def _posterior_probabilities(observation_seq: List[int], hmm) -> NDArray[np.float64]:
    alpha = _build_forward_matrix(observation_seq,hmm)
    beta = _build_backward_matrix(observation_seq,hmm)
    post = alpha * beta
    post = post / np.sum(post,axis=1,keepdims=True)
    return post


def vectorzied_ge(x, y, thr=1e-5):
    ge = x > y*(1+thr)
    np.fill_diagonal(ge, True) # ge than self should be true
    return ge

def _max_position(x,ge_func=vectorzied_ge):
    y=x 
    x = x[:,np.newaxis]
    y = y[np.newaxis,:]
    ge = vectorzied_ge(x,y) # ij: i>j 
    ge_than_all = np.all(ge, axis=1)
    first_occurance_max = np.argmax(ge_than_all)
    return first_occurance_max

def _max_position_along_axis(x, axis=0, ge_func=vectorzied_ge):
    max_func = lambda x: _max_position(x, ge_func=ge_func)
    if len(x.shape) == 1:
        return max_func(x) 
    return np.apply_along_axis(max_func,axis,x)


# def _max_position(list_of_numbers: NDArray[np.float64]) -> int:
    # """
    # Find the index of the maximum value in a list.

    # Returns the first index if there are ties or extremely close values.
    # """
    # max_value = -np.inf
    # max_position = 0

    # for i, value in enumerate(list_of_numbers):
        # # This handles extremely close values that arise from numerical instability
        # if value / max_value > 1 + 1E-5 if max_value > 0 else value > max_value:
            # max_value = value
            # max_position = i

    # return max_position

# __all__ = [
        # "_max_position","_build_backward_matrix","_build_forward_matrix","_posterior_probabilites","posterior_decode"
        # ]
