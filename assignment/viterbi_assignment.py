"""
Viterbi Decoding for Hidden Markov Models

This module implements the Viterbi algorithm for finding the most likely
state sequence given an observation sequence and an HMM.
"""

from typing import List
import numpy as np
from numpy.typing import NDArray
import warnings
# from cse587Autils.HMMObjects import HMM 

# class HMMNp(HMM):
    # def __init__(self, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        # self.state_dict = {
            # self.states[i]:i 
            # for i in range(len(self.states))
        # }
        # self.alphabet_dict = {
            # self.alphabet[i]:i 
            # for i in range(len(self.alphabet))
        # }
        # self.initial_arr = np.array(self.initial_state_probs) # [n_state]
        # self.transition_arr = np.array(self.transition_matrix) # [n_state, n_state]
        # self.emission_arr = np.array(self.emission_matrix) # [n_obs, n_state]

    # def encode_obs(self, observation_list:List)


def viterbi_decode(observation_seq: List[int], hmm, numeric=False, use_log=False) -> List[str]:
    """
    Decode an observation sequence using the Viterbi algorithm.

    INPUT:
    - observation_seq: List of observations (integers indexing alphabet)
    - hmm: HMM object with states, initial_state_probs, transition_matrix,
      and emission_matrix

    OUTPUT:
    - state_seq: List containing the most likely state sequence (state names)

    IMPLEMENTATION NOTES:
    1) The Viterbi table is implemented as a matrix that is transposed
       relative to the way it is shown in class. Rows correspond to
       observations and columns to states.
    2) After computing the Viterbi probabilities for each observation,
       they are normalized by dividing by their sum. This maintains
       proportionality while avoiding numerical underflow.
    """
    if len(observation_seq)>10000:
        float_type=np.float128
    else:
        float_type=np.float64
    delta = _build_matrix(observation_seq, hmm,log=use_log,float_type=float_type)
    numeric_state_seq = _traceback(
        delta,
        hmm,log=use_log,float_type=float_type
    )
    if numeric:
        return (numeric_state_seq)
    return [hmm.states[i] for i in numeric_state_seq]

def _build_log_matrix(observation_seq,hmm, float_type=np.float64):
    """
    Build the Viterbi probability matrix.

    Returns a matrix where rows are observations and columns are states.
    Each entry is normalized to avoid underflow.
    """

    number_of_observations = len(observation_seq)
    number_of_states = hmm.num_states
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_initial_state_probs = np.log(np.array(hmm.initial_state_probs).astype(float_type)) # [n_state]
        log_transition_matrix = np.log(np.array(hmm.transition_matrix).astype(float_type)) # [n_state, n_state]
        log_emission_matrix = np.log(np.array(hmm.emission_matrix).astype(float_type)) # [n_obs, n_state]
    # Initialize Viterbi matrix
    # viterbi_matrix = np.zeros((number_of_observations, number_of_states))
    viterbi_columns = []
    # t=0 
    viterbi_columns.append(       
        log_initial_state_probs + log_emission_matrix[observation_seq[0]] 
    )
    for t in range(1, number_of_observations):
        all_path_logprob = (
            # "i, ij, j -> ij",
            viterbi_columns[-1][:,np.newaxis] # last step 
            +log_transition_matrix 
            +log_emission_matrix[observation_seq[t]][np.newaxis,:]
        )
        max_path_logprob = np.max( 
            all_path_logprob, axis=0
        )
        viterbi_columns.append(max_path_logprob)
    viterbi_matrix = np.vstack(viterbi_columns)
    return viterbi_matrix



def _build_matrix(observation_seq: List[int], hmm, log=False, float_type=np.float64) -> NDArray[np.float64]:
    """
    Build the Viterbi probability matrix.

    Returns a matrix where rows are observations and columns are states.
    Each entry is normalized to avoid underflow.
    """
    if log:
        return _build_log_matrix(observation_seq,hmm)
    number_of_observations = len(observation_seq)
    number_of_states = hmm.num_states
    initial_state_probs = (np.array(hmm.initial_state_probs)).astype(float_type) # [n_state]
    transition_matrix = (np.array(hmm.transition_matrix)).astype(float_type) # [n_state, n_state]
    emission_matrix = (np.array(hmm.emission_matrix)).astype(float_type) # [n_obs, n_state]

    # Initialize Viterbi matrix
    # viterbi_matrix = np.zeros((number_of_observations, number_of_states))
    viterbi_columns = []
    # t=0 
    viterbi_columns.append(       
        initial_state_probs * emission_matrix[observation_seq[0]] 
    )
    viterbi_columns[0] = viterbi_columns[0] / np.sum(viterbi_columns[0],keepdims=True)
    for t in range(1, number_of_observations):
        # take max before multiplying emission_matrix
        # makes enough accuracy
        all_path_prob = np.einsum(
            "i, ij, j -> ij",
            viterbi_columns[-1], # last step 
            transition_matrix, 
            emission_matrix[observation_seq[t]]
        )
        max_path_prob = np.max( 
            all_path_prob, axis=0
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # print(max_path_prob.shape)
            max_path_prob = max_path_prob / np.sum(max_path_prob,keepdims=True)
            # print(max_path_prob.shape)
        viterbi_columns.append(max_path_prob)
    viterbi_matrix = np.vstack(viterbi_columns)
    # YOUR CODE HERE
    return viterbi_matrix


def _traceback(viterbi_matrix: NDArray[np.float64], hmm,log=False, return_psi=False,float_type=np.float64) -> List[int]:
    """
    Trace back through the Viterbi matrix to find the most likely path.

    Returns a list of state indices (integers) corresponding to the
    most likely state sequence.
    """
    # argmax_func = np.argmax
    argmax_func = _max_position_along_axis
    number_of_observations = len(viterbi_matrix)
    if log:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transition_matrix = np.log(np.array(hmm.transition_matrix,dtype=float_type))
        path_prob = viterbi_matrix[:-1][:,:,np.newaxis] + transition_matrix[np.newaxis,:,:]
    else:
        transition_matrix = np.array(hmm.transition_matrix,dtype=float_type) # [n_state, n_state]
        path_prob = np.einsum(
            "ti, ij -> tij",
            viterbi_matrix[:-1], # delta matrix except the last step,
            transition_matrix
        )
    state_seq = np.zeros(number_of_observations, dtype=int)
    # the last step 
    state_seq[-1] = argmax_func(viterbi_matrix[-1])
        # YOUR CODE HERE
    if number_of_observations > 1:
        psi = argmax_func(path_prob, axis=1)
        for t in range(number_of_observations-1)[::-1]:
            following_step = t + 1 
            following_state = state_seq[following_step]
            state_seq[t] = psi[t, following_state]
    else:
        psi = None
    if return_psi:
        # return state_seq.tolist(), psi, _, viterbi_matrix
        return viterbi_matrix, psi, path_prob, state_seq.tolist()
    return state_seq.tolist()

# Use this function to find the index within an array of the maximum value.
# Do not use any built-in functions for this.
# This implementation chooses the lowest index in case of ties.
# Two values are considered tied if they are within a factor of 1E-5.
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

    # Returns the first index if there are ties or extremly close values.
    # """
    # max_value = 1E-10
    # max_position = 0

    # for i, value in enumerate(list_of_numbers):
        # # This handles extremely close values that arise from numerical instability
        # if value / max_value > 1 + 1E-5:
            # max_value = value
            # max_position = i

    # return max_position
