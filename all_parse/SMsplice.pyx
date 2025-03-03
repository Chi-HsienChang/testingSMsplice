#distutils: extra_link_args=-fopenmp
from cython import parallel
from cython.parallel import prange
import numpy as np
import time
from sklearn.neighbors import KernelDensity
from awkde import GaussianKDE
from math import exp 
from libc.math cimport exp as c_exp
cimport openmp

########################################
########################################
########################################

# forward_backward_for_all_parse

# SMsplice.pyx
# cython: language_level=3

import math
cimport numpy as np

# -------------------------------------------------------------------
# logsumexp: Compute log(sum(exp(x) for x in log_list)) in a numerically stable way.
# -------------------------------------------------------------------
cdef double logsumexp(list log_list):
    cdef int i, n = len(log_list)
    cdef double max_val, total, x
    if n == 0:
        return float('-inf')
    max_val = log_list[0]
    for i in range(1, n):
        if log_list[i] > max_val:
            max_val = log_list[i]
    if max_val == float('-inf'):
        return float('-inf')
    total = 0.0
    for i in range(n):
        x = log_list[i] - max_val
        total += math.exp(x)
    return max_val + math.log(total)

# -------------------------------------------------------------------
# transition_dp: Update state and log_score when placing a symbol at position pos.
#
# state: (used5, used3, lastSymbol, zeroCount, last5Pos, last3Pos)
# Returns (new_state, new_log_score) if placement allowed; otherwise, returns None.
# -------------------------------------------------------------------
cdef tuple transition_dp(
    tuple state, double log_score, int pos, int symbol, 
    object sequences, int length,
    double pME, double[:] pELF, double[:] pIL, 
    double pEE, double[:] pELM,
    double[:] emissions5, double[:] emissions3
):
    cdef:
        int used5 = state[0]
        int used3 = state[1]
        int lastSymbol = state[2]
        int zeroCount = state[3]
        int last5Pos = state[4]
        int last3Pos = state[5]

        double new_log_score = log_score
        int newUsed5 = used5
        int newUsed3 = used3
        int newZeroCount = zeroCount
        int newLast5Pos = last5Pos
        int newLast3Pos = last3Pos
        int gap_5, gap_3
        int newLastSymbol

    # ---------------------------------------------------------
    # symbol == 0: 表示不放 splice site，zeroCount 可能累積
    if symbol == 0:
        # Placing 0: if previous symbol was 5 or 3, increase zeroCount.
        if lastSymbol == 5 or lastSymbol == 3:
            newZeroCount = zeroCount + 1
        newLastSymbol = 0
        # Score remains unchanged (no special penalty/bonus in this example).

    # ---------------------------------------------------------
    # symbol == 5: 嘗試放 5' splice site
    elif symbol == 5:
        # emissions5[pos] <= -19 表示機率太低，直接忽略
        if emissions5[pos] <= -19:
            return None
        if pos + 1 >= length:
            return None
        if not (sequences[pos] == 'G' and sequences[pos+1] == 'T'):
            return None
        # 若剛放過 5 or 3，且 zeroCount < 5，或 used5 != used3 (5' 次數要跟 3' 相同時才可放)
        if lastSymbol == 5 or (((lastSymbol == 5) or (lastSymbol == 3)) and zeroCount < 5) or (used5 != used3):
            return None

        if used5 == 0:
            new_log_score += pME + pELF[pos - 1] + emissions5[pos]
        else:
            gap_5 = (pos - last3Pos) - 2
            if gap_5 < 5 or gap_5 >= pELM.shape[0]:
                return None
            new_log_score += pEE + pELM[gap_5] + emissions5[pos]

        newUsed5 = used5 + 1
        newLast5Pos = pos
        newZeroCount = 0
        newLastSymbol = 5

    # ---------------------------------------------------------
    # symbol == 3: 嘗試放 3' splice site
    elif symbol == 3:
        if emissions3[pos] <= -40:
            return None
        if pos - 1 < 0:
            return None
        if not (sequences[pos] == 'G' and sequences[pos-1] == 'A'):
            return None
        # 若剛放過 3，或剛放過 5/3 但 zeroCount < 25，或 used5 != used3 + 1，就不合法
        if lastSymbol == 3 or (((lastSymbol == 5) or (lastSymbol == 3)) and zeroCount < 25) or (used5 != used3 + 1):
            return None

        gap_3 = (pos - last5Pos) - 2
        if gap_3 < 25 or gap_3 >= pIL.shape[0]:
            return None
        new_log_score += pIL[gap_3] + emissions3[pos]

        newUsed3 = used3 + 1
        newLast3Pos = pos
        newZeroCount = 0
        newLastSymbol = 3

    else:
        return None

    cdef tuple new_state = (newUsed5, newUsed3, newLastSymbol, newZeroCount, newLast5Pos, newLast3Pos)
    return (new_state, new_log_score)

# -------------------------------------------------------------------
# forward_dp: Compute the forward table F.
#   F[i]: 在位置 i 時，各個 state 的 log_score
# -------------------------------------------------------------------
cdef list forward_dp(
    object sequences, double pME, double[:] pELF, double[:] pIL, 
    double pEE, double[:] pELM, double[:] emissions5, double[:] emissions3,
    int length
):
    cdef:
        int pos, symbol, i
        list F = [ {} for i in range(length+1) ]
        tuple init_state = (0, 0, 0, 1, -1, -1)
        tuple state, new_state
        double log_score, new_log_score
        tuple new_state_tuple
        list allowed_symbols

    # 初始狀態 log_score=0.0
    F[0][init_state] = 0.0

    for pos in range(0, length):
        F[pos+1] = {}

        # 檢查當前 pos 狀態中的 log_score，跳過 -inf & nan
        for state, log_score in F[pos].items():
            if log_score == float('-inf') or math.isnan(log_score):
                continue

            # 根據 pos 決定允許哪些 symbol
            if pos == 0 or pos == length - 1:
                allowed_symbols = [0]
            else:
                allowed_symbols = [0, 5, 3]

            for symbol in allowed_symbols:
                new_state_tuple = transition_dp(
                    state, log_score, pos, symbol,
                    sequences, length,
                    pME, pELF, pIL, pEE, pELM,
                    emissions5, emissions3
                )
                if new_state_tuple is None:
                    continue

                new_state, new_log_score = new_state_tuple

                # 用 log-sum-exp 累計同一個 new_state
                if new_state in F[pos+1]:
                    F[pos+1][new_state] = logsumexp([F[pos+1][new_state], new_log_score])
                else:
                    F[pos+1][new_state] = new_log_score

    return F

# -------------------------------------------------------------------
# backward_dp: Compute the backward table B.
#   B[i]: 從位置 i 的 state 到終點的 log 機率
# -------------------------------------------------------------------
cdef list backward_dp(
    object sequences, double pME, double[:] pELF, double[:] pIL, 
    double pEE, double[:] pELM, double pEO, double[:] pELL, 
    double[:] emissions5, double[:] emissions3, int length, list F
):
    cdef:
        int pos, symbol, i
        list B = [ {} for i in range(length+1) ]
        tuple state, new_state
        double alpha_score, tail, contribution
        double b_score_next, new_log_score
        int used5, used3, lastSymbol, last5Pos, last3Pos
        int ell_index
        list allowed_symbols, contributions
        tuple new_state_tuple

    # (1) 初始化 (pos = length)
    B[length] = {}
    for state, alpha_score in F[length].items():
        used5 = state[0]
        used3 = state[1]
        lastSymbol = state[2]
        last5Pos = state[4]
        last3Pos = state[5]

        # 合法終點條件: lastSymbol==0 且 used5==used3>0
        if lastSymbol == 0 and (used5 == used3) and (used5 + used3 > 0):
            tail = 0.0
            ell_index = (length - last3Pos) - 2
            if ell_index >= 0 and ell_index < pELL.shape[0]:
                tail += pEO + pELL[ell_index]
            B[length][state] = tail  # log(prob)

    # (2) Backward recursion: pos = length-1 down to 0
    for pos in range(length-1, -1, -1):
        B[pos] = {}

        for state, alpha_score in F[pos].items():
            if alpha_score == float('-inf') or math.isnan(alpha_score):
                continue

            if pos == 0 or pos == length - 1:
                allowed_symbols = [0]
            else:
                allowed_symbols = [0, 5, 3]

            contributions = []

            for symbol in allowed_symbols:
                new_state_tuple = transition_dp(
                    state, alpha_score, pos, symbol,
                    sequences, length,
                    pME, pELF, pIL, pEE, pELM,
                    emissions5, emissions3
                )
                if new_state_tuple is None:
                    continue

                new_state, new_log_score = new_state_tuple

                # 若 new_state 在 B[pos+1] 裏，才計算 backward
                if new_state in B[pos+1]:
                    b_score_next = B[pos+1][new_state]
                    if b_score_next == float('-inf') or math.isnan(b_score_next):
                        continue
                    contribution = (new_log_score - alpha_score) + b_score_next
                    contributions.append(contribution)

            if contributions:
                B[pos][state] = logsumexp(contributions)

    return B

# -------------------------------------------------------------------
# compute_posterior: 計算各位置 pos 的後驗機率 (posterior probabilities)
# -------------------------------------------------------------------
cdef tuple compute_posterior(list F, list B, int length):
    cdef:
        list terminal_logs = []
        list post_list = [{} for _ in range(length)]
        double log_score, log_alpha, log_beta, log_val, logZ
        double current_prob, current_log
        int pos, i
        # ★★ 這裡先宣告 last_symbol ★★
        int last_symbol
        tuple state

    # (1) 計算 logZ (配分函數)，僅考慮非 nan、非 -inf 的終點
    for state, log_score in F[length].items():
        if state in B[length]:
            log_alpha = log_score
            log_beta = B[length][state]
            if math.isnan(log_alpha) or math.isnan(log_beta):
                continue
            if math.isnan(log_alpha + log_beta) or (log_alpha + log_beta) == float('-inf'):
                continue
            terminal_logs.append(log_alpha + log_beta)

    if not terminal_logs:
        print("Warning: No valid terminal states (all had nan/-inf). logZ set to -inf.")
        logZ = float('-inf')
        return post_list, logZ
    else:
        logZ = logsumexp(terminal_logs)

    # (2) 計算各位置 pos 的後驗機率
    for pos in range(length):
        for state, log_alpha in F[pos].items():
            if state in B[pos]:
                log_beta = B[pos][state]
                # 跳過 nan
                if math.isnan(log_alpha) or math.isnan(log_beta):
                    continue

                log_val = log_alpha + log_beta - logZ
                if math.isnan(log_val) or log_val == float('-inf'):
                    continue

                # 在此直接使用已宣告好的 last_symbol
                last_symbol = state[2]

                # 累積 posterior
                if last_symbol in post_list[pos]:
                    current_prob = post_list[pos][last_symbol]
                    if current_prob == 0.0:
                        current_log = float('-inf')
                    else:
                        current_log = math.log(current_prob)
                    post_list[pos][last_symbol] = math.exp(
                        logsumexp([current_log, log_val])
                    )
                else:
                    post_list[pos][last_symbol] = math.exp(log_val)

    return post_list, logZ

# -------------------------------------------------------------------
# forward_backward_dp_emissions: Main function integrating forward, backward and posterior calculation.
# -------------------------------------------------------------------
cpdef tuple forward_backward_for_all_parse(
    object sequences, double pME, double[:] pELF, double[:] pIL,
    double pEE, double[:] pELM, double pEO, double[:] pELL,
    double[:] emissions5, double[:] emissions3, int length
):
    """
    Cython implementation of the Forward–Backward Algorithm with emissions detail.

    Returns:
      post_list: list of dictionaries, where each dict maps a symbol (lastSymbol) 
                 to its posterior probability at that position.
      logZ: partition function in log–space.
      F: forward table
      B: backward table
    """
    cdef list F = forward_dp(sequences, pME, pELF, pIL, pEE, pELM, emissions5, emissions3, length)
    cdef list B = backward_dp(sequences, pME, pELF, pIL, pEE, pELM, pEO, pELL, emissions5, emissions3, length, F)
    cdef tuple result = compute_posterior(F, B, length)
    cdef list post_list = result[0]
    cdef double logZ = result[1]
    return post_list, logZ, F, B


########################################
########################################
########################################


def beam_search_for_top_k_parse(
    sequences, 
    double pME, 
    np.ndarray[double, ndim=1] pELF, 
    np.ndarray[double, ndim=1] pIL, 
    double pEE, 
    np.ndarray[double, ndim=1] pELM, 
    double pEO, 
    np.ndarray[double, ndim=1] pELL, 
    np.ndarray[double, ndim=1] emissions5, 
    np.ndarray[double, ndim=1] emissions3, 
    int length,
    int beam_width=1000  # Number of best states to keep per level
):
    """
    A dynamic programming algorithm implemented with Cython, incorporating memoization and beam search strategies to improve memory usage.

    Parameters:
      sequences: sequence of characters (e.g., string or character array)
      pME: score parameter for the first 5
      pELF: score parameter array for placing a 5
      pIL: score parameter array for placing a 3
      pEE: score parameter for a 5 followed by a 3
      pELM: score parameter array for non-first 5s
      pEO: score parameter for the final 3
      pELL: score parameter array for the final 3
      emissions5, emissions3: emission scores at corresponding positions
      length: length of the sequence
      beam_width: number of best states kept per level (default is 1000)
    Returns:
      A list where each element is a tuple (np.array(path, dtype=np.int32), score, score_details)
      representing a valid path, total score, and detailed score record.
    """
    cdef list results = []
    # Initial state: (pos, used5, used3, lastSymbol, zeroCount, last5Pos, last3Pos, scoreSoFar, path, score_details)
    cdef tuple initial_state = (0, 0, 0, 0, 1, -1, -1, 0.0, [], [])
    # Use a list to store states at the current level
    cdef list current_states = [initial_state]
    # Memoization: record mapping from (pos, used5, used3, lastSymbol, zeroCount, last5Pos, last3Pos) to highest score
    cdef dict state_memo = {}
    
    # Variable declarations (all cdef variables used within the function are declared here)
    cdef int pos, used5, used3, lastSymbol, zeroCount, last5Pos, last3Pos, symbol
    cdef int newZeroCount, newUsed5, newUsed3, newLast5Pos, newLast3Pos, ell_index, i, j, count_5, count_3
    cdef double scoreSoFar, newScore, gap_5, gap_3
    cdef bint can_place
    cdef object path, score_details, new_details, new_path, state, new_state, item
    cdef list next_states  # Declare variable only, no initialization here

    # Expand level by level
    while current_states:
        next_states = []  # Reinitialize to an empty list at the beginning of each level
        for state in current_states:
            pos         = state[0]
            used5       = state[1]
            used3       = state[2]
            lastSymbol  = state[3]
            zeroCount   = state[4]
            last5Pos    = state[5]
            last3Pos    = state[6]
            scoreSoFar  = state[7]
            path        = state[8]
            score_details = state[9]
    
            # If the end of the sequence is reached, check if the valid path conditions are met
            #if pos == length:
            #    if lastSymbol == 0 and (used5 == used3) and ((used5 + used3) > 0):
            #        results.append((np.array(path, dtype=np.int32), scoreSoFar, score_details))

            if pos == length:
                # Confirm it is a valid path
                if lastSymbol == 0 and (used5 == used3) and (used5 + used3 > 0):
                    # Calculate the "tail score" for the final 3 here
                    finalScore = scoreSoFar  # First, copy the current score
                    ell_index = (length - last3Pos) - 2
                    if 0 <= ell_index < pELL.shape[0]:
                        finalScore += pEO
                        finalScore += pELL[ell_index]

                        # Also add a record to score_details
                        score_details = score_details.copy()
                        score_details.append(f"Added pEO = {pEO}")
                        score_details.append(f"pELL[{ell_index}] = {pELL[ell_index]}")

                    # Append the full path (path), final score (finalScore), and details (score_details) to results
                    results.append(
                        (np.array(path, dtype=np.int32), finalScore, score_details)
                    )
                continue
            
            # For the current state, try placing 0, 5, and 3
            for symbol in [0, 5, 3]:
                if pos == 0 and symbol != 0:
                    continue
                if pos == length - 1 and symbol != 0:
                    continue
    
                can_place = True
                newZeroCount = zeroCount
                newUsed5 = used5
                newUsed3 = used3
                newLast5Pos = last5Pos
                newLast3Pos = last3Pos
                newScore = scoreSoFar
                new_details = score_details.copy()
    
                # Logic for placing 0
                if symbol == 0:
                    if lastSymbol in [5, 3]:
                        newZeroCount += 1
                # Logic for placing 5
                elif symbol == 5:
                    if emissions5[pos] <= -19:
                        can_place = False
                    if can_place:
                        if pos + 1 >= length:
                            can_place = False
                        else:
                            if not (sequences[pos] == 'G' and sequences[pos+1] == 'T'):
                                can_place = False
                    if can_place: # placing 5 and decide exon length
                        if lastSymbol == 5 or ((lastSymbol in [5, 3]) and zeroCount < 5) or (used5 != used3):
                            can_place = False
                    if can_place:
                        if used5 == 0:
                            newScore += pME
                            new_details.append(f"First 5: pME = {pME}")
                            newScore += pELF[pos - 1]
                            new_details.append(f"pELF[{pos - 1}] = {pELF[pos - 1]}")
                            newScore += emissions5[pos]
                            new_details.append(f"emissions5[{pos}] = {emissions5[pos]}")
                        else:
                            gap_5 = (pos - last3Pos) - 2
                            if gap_5 < 5 or gap_5 >= pELM.shape[0]:
                                can_place = False
                            else:
                                newScore += pEE
                                new_details.append(f"5 followed by 3: pEE = {pEE}")
                                newScore += pELM[int(gap_5)]
                                new_details.append(f"pELM[{int(gap_5)}] = {pELM[int(gap_5)]}")
                                newScore += emissions5[pos]
                                new_details.append(f"emissions5[{pos}] = {emissions5[pos]}")
                        if math.isinf(newScore):
                            can_place = False
                        if can_place:
                            newUsed5 += 1
                            newLast5Pos = pos
                            newZeroCount = 0
                # Logic for placing 3
                else:  # symbol == 3
                    if emissions3[pos] <= -40:
                        can_place = False
                    if can_place:
                        if pos - 1 < 0:
                            can_place = False
                        else:
                            if not (sequences[pos] == 'G' and sequences[pos-1] == 'A'):
                                can_place = False
                    if can_place:
                        if lastSymbol == 3 or ((lastSymbol in [5, 3]) and zeroCount < 25) or (used5 != used3 + 1):
                            can_place = False
                    if can_place:
                        gap_3 = (pos - last5Pos) - 2
                        if gap_3 < 25 or gap_3 >= pIL.shape[0]:
                            can_place = False
                        else:
                            newScore += pIL[int(gap_3)]
                            new_details.append(f"3 followed by 5: pIL[{int(gap_3)}] = {pIL[int(gap_3)]}")
                            newScore += emissions3[pos]
                            new_details.append(f"emissions3[{pos}] = {emissions3[pos]}")
                            #if (used3 + 1) == used5:
                            #    ell_index = (length - pos) - 2
                            #    if ell_index < 0 or ell_index >= pELL.shape[0]:
                            #        can_place = False
                            #    else:
                            #        newScore += pEO
                            #        new_details.append(f"Final 3: pEO = {pEO}")
                            #        newScore += pELL[ell_index]
                            #        new_details.append(f"pELL[{ell_index}] = {pELL[ell_index]}")
                            if math.isinf(newScore):
                                can_place = False
                            if can_place:
                                newUsed3 += 1
                                newLast3Pos = pos
                                newZeroCount = 0
    
                if not can_place:
                    continue
                if math.isinf(newScore):
                    continue
    
                new_path = path + [symbol]
                new_state = (pos + 1, newUsed5, newUsed3, symbol, newZeroCount,
                             newLast5Pos, newLast3Pos, newScore, new_path, new_details)
    
                # Use state key for memoization pruning, only considering the critical parameters that determine the state
                key = (pos + 1, newUsed5, newUsed3, symbol, newZeroCount, newLast5Pos, newLast3Pos)
                if key in state_memo:
                    if state_memo[key] >= newScore:
                        continue
                    else:
                        state_memo[key] = newScore
                else:
                    state_memo[key] = newScore
    
                next_states.append(new_state)
    
        # If the number of states in the next level is too high, keep only the top beam_width states (sorted by score)
        if len(next_states) > beam_width:
            next_states.sort(key=lambda x: x[7], reverse=True)
            next_states = next_states[:beam_width]
    
        current_states = next_states

    #print(len(results))
    
    return results



########################################
########################################
########################################



def baseToInt(str base):
    if base == 'a': return 0
    elif base == 'c': return 1
    elif base == 'g': return 2
    elif base == 't': return 3
    else:
        print("nonstandard base encountered:", base)
        return -1

def intToBase(int i):
    if i == 0: return 'a'
    elif i == 1: return 'c'
    elif i == 2: return 'g'
    elif i == 3: return 't'
    else: 
        print("nonbase integer encountered:", i)
        return ''

def hashSequence(str seq):
    cdef int i
    cdef int sum = 0 
    cdef int l = len(seq)
    for i in range(l):
        sum += (4**(l-i-1))*baseToInt(seq[i])
    return sum
    
def unhashSequence(int num, int l):
    seq = ''
    for i in range(l):
        seq += intToBase(num // 4**(l-i-1))
        num -= (num // 4**(l-i-1))*(4**(l-i-1))
    return seq
    
def trueSequencesCannonical(genes, annotations, E = 0, I = 1, B3 = 3, B5 = 5):
    # Converts gene annotations to sequences of integers indicating whether the sequence is exonic, intronic, or splice site,
    # Inputs
    #   - genes: a biopython style dictionary of the gene sequences
    #   - annotations: the splicing annotations dictionary
    #   - E, I, B3, B5: the integer indicators for exon, intron, 3'ss, and 5'ss, respectively
    trueSeqs = {}
    for gene in annotations.keys():
        if gene not in genes.keys(): 
            print(gene, 'has annotation, but was not found in the fasta file of genes') 
            continue
        
        transcript = annotations[gene]
        if len(transcript) == 1: 
            trueSeqs[gene] = np.zeros(len(genes[gene]), dtype = int) + E
            continue # skip the rest for a single exon case
        
        # First exon 
        true = np.zeros(len(genes[gene]), dtype = int) + I
        three = transcript[0][0] - 1 # Marking the beginning of the first exon
        five = transcript[0][1] + 1
        true[range(three+1, five)] = E
        true[five] = B5
        
        # Internal exons 
        for exon in transcript[1:-1]:
            three = exon[0] - 1
            five = exon[1] + 1
            true[three] = B3
            true[five] = B5
            true[range(three+1, five)] = E
            
        # Last exon 
        three = transcript[-1][0] - 1
        true[three] = B3
        five = transcript[-1][1] + 1 # Marking the end of the last exon
        true[range(three+1, five)] = E
                
        trueSeqs[gene] = true
        
    return(trueSeqs)

def trainAllTriplets(sequences, cutoff = 10**(-5)):
    # Train maximum entropy models from input sequences with triplet conditions
    train = np.zeros((len(sequences),len(sequences[0])), dtype = int)
    for (i, seq) in enumerate(sequences):
        for j in range(len(seq)):
            train[i,j] = baseToInt(seq[j])
    prob = np.log(np.zeros(4**len(sequences[0])) + 4**(-len(sequences[0])))
    Hprev = -np.sum(prob*np.exp(prob))/np.log(2)
    H = -1
    sequences = np.zeros((4**len(sequences[0]),len(sequences[0])), dtype = int)
    l = len(sequences[0]) - 1 
    for i in range(sequences.shape[1]):
        sequences[:,i] = ([0]*4**(l-i) + [1]*4**(l-i) + [2]*4**(l-i) +[3]*4**(l-i))*4**i
    while np.abs(Hprev - H) > cutoff:
        #print(np.abs(Hprev - H))
        Hprev = H
        for pos in range(sequences.shape[1]):
            for base in range(4):
                Q = np.sum(train[:,pos] == base)/float(train.shape[0])
                if Q == 0: continue
                Qhat = np.sum(np.exp(prob[sequences[:,pos] == base]))
                prob[sequences[:,pos] == base] += np.log(Q) - np.log(Qhat)
                prob[sequences[:,pos] != base] += np.log(1-Q) - np.log(1-Qhat)
                
                for pos2 in np.setdiff1d(range(sequences.shape[1]), range(pos+1)):
                    for base2 in range(4):
                        Q = np.sum((train[:,pos] == base)*(train[:,pos2] == base2))/float(train.shape[0])
                        if Q == 0: continue
                        which = (sequences[:,pos] == base)*(sequences[:,pos2] == base2)
                        Qhat = np.sum(np.exp(prob[which]))
                        prob[which] += np.log(Q) - np.log(Qhat)
                        prob[np.invert(which)] += np.log(1-Q) - np.log(1-Qhat)
                        
                        for pos3 in np.setdiff1d(range(sequences.shape[1]), range(pos2+1)):
                            for base3 in range(4):
                                Q = np.sum((train[:,pos] == base)*(train[:,pos2] == base2)*(train[:,pos3] == base3))/float(train.shape[0])
                                if Q == 0: continue
                                which = (sequences[:,pos] == base)*(sequences[:,pos2] == base2)*(sequences[:,pos3] == base3)
                                Qhat = np.sum(np.exp(prob[which]))
                                prob[which] += np.log(Q) - np.log(Qhat)
                                prob[np.invert(which)] += np.log(1-Q) - np.log(1-Qhat)
        H = -np.sum(prob*np.exp(prob))/np.log(2)
    return np.exp(prob)

def structuralParameters(genes, annotations, minIL = 0):
    # Get the empirical length distributions for introns and single, first, middle, and last exons, as well as number exons per gene
    
    # Transitions
    numExonsPerGene = [] 
    
    # Length Distributions
    lengthSingleExons = []
    lengthFirstExons = []
    lengthMiddleExons = []
    lengthLastExons = []
    lengthIntrons = []
    
    for gene in genes:
        if len(annotations[gene]) == 0: 
            print('missing annotation for', gene)
            continue
        numExons = 0
        introns = []
        singleExons = []
        firstExons = []
        middleExons = []
        lastExons = []
        
        for transcript in annotations[gene].values():
            numExons += len(transcript)
            
            # First exon 
            three = transcript[0][0] # Make three the first base
            five = transcript[0][1] + 1
            if len(transcript) == 1: 
                singleExons.append((three, five-1))
                continue # skip the rest for a single exon case
            firstExons.append((three, five-1)) # since three is the first base
            
            # Internal exons 
            for exon in transcript[1:-1]:
                three = exon[0] - 1 
                introns.append((five+1,three-1))
                five = exon[1] + 1
                middleExons.append((three+1, five-1))
                
            # Last exon 
            three = transcript[-1][0] - 1
            introns.append((five+1,three-1))
            five = transcript[-1][1] + 1
            lastExons.append((three+1, five-1))
        
        geneIntronLengths = [minIL]
        for intron in set(introns):
            geneIntronLengths.append(intron[1] - intron[0] + 1)
        
        if np.min(geneIntronLengths) < minIL: continue
        
        for intron in set(introns): lengthIntrons.append(intron[1] - intron[0] + 1)
        for exon in set(singleExons): lengthSingleExons.append(exon[1] - exon[0] + 1)
        for exon in set(firstExons): lengthFirstExons.append(exon[1] - exon[0] + 1)
        for exon in set(middleExons): lengthMiddleExons.append(exon[1] - exon[0] + 1)
        for exon in set(lastExons): lengthLastExons.append(exon[1] - exon[0] + 1)
            
        numExonsPerGene.append(float(numExons)/len(annotations[gene]))
        
    return(numExonsPerGene, lengthSingleExons, lengthFirstExons, lengthMiddleExons, lengthLastExons, lengthIntrons)

def adaptive_kde_tailed(lengths, N, geometric_cutoff = .8, lower_cutoff=0):
    adaptive_kde = GaussianKDE(alpha = 1) 
    adaptive_kde.fit(np.array(lengths)[:,None]) 
    
    lengths = np.array(lengths)
    join = np.sort(lengths)[int(len(lengths)*geometric_cutoff)] 
    
    smoothed = np.zeros(N)
    smoothed[:join+1] = adaptive_kde.predict(np.arange(join+1)[:,None])
    
    s = 1-np.sum(smoothed)
    p = smoothed[join]
    smoothed[join+1:] = np.exp(np.log(p) + np.log(s/(s+p))*np.arange(1,len(smoothed)-join))
    smoothed[:lower_cutoff] = 0
    smoothed /= np.sum(smoothed)
    
    return(smoothed)
    
def geometric_smooth_tailed(lengths, N, bandwidth, join, lower_cutoff=0):
    lengths = np.array(lengths)
    smoothing = KernelDensity(bandwidth = bandwidth).fit(lengths[:, np.newaxis]) 
    
    smoothed = np.zeros(N)
    smoothed[:join+1] = np.exp(smoothing.score_samples(np.arange(join+1)[:,None]))
    
    s = 1-np.sum(smoothed)
    p = smoothed[join]
    smoothed[join+1:] = np.exp(np.log(p) + np.log(s/(s+p))*np.arange(1,len(smoothed)-join))
    smoothed[:lower_cutoff] = 0
    smoothed /= np.sum(smoothed)
    
    return(smoothed)

def maxEnt5(geneNames, genes, dir):
    # Get all the 5'SS maxent scores for each of the genes in geneNames
    scores = {}
    prob = np.load(dir + '/maxEnt5_prob.npy')
    prob0 = np.load(dir + '/maxEnt5_prob0.npy') 
        
    for gene in geneNames:
        sequence = str(genes[gene].seq).lower()
        sequence5 = np.array([hashSequence(sequence[i:i+9]) for i in range(len(sequence)-9+1)])
        scores[gene] = np.zeros(len(sequence)) - np.inf
        scores[gene][3:-5] = np.log2(prob[sequence5]) - np.log2(prob0[sequence5])
        scores[gene] = np.exp2(scores[gene])
    
    return scores
    
def maxEnt5_single(str seq, str dir):
    prob = np.load(dir + 'maxEnt5_prob.npy')
    prob0 = np.load(dir + 'maxEnt5_prob0.npy')
    
    seq = seq.lower()
    sequence5 = np.array([hashSequence(seq[i:i+9]) for i in range(len(seq)-9+1)])
    scores = np.log2(np.zeros(len(seq)))
    scores[3:-5] = np.log2(prob[sequence5]) - np.log2(prob0[sequence5])
    return np.exp2(scores)
    
def maxEnt3(geneNames, genes, dir):
    # Get all the 3'SS maxent scores for each of the genes in geneNames
    scores = {}
    prob0 = np.load(dir + 'maxEnt3_prob0.npy')
    prob1 = np.load(dir + 'maxEnt3_prob1.npy')
    prob2 = np.load(dir + 'maxEnt3_prob2.npy')
    prob3 = np.load(dir + 'maxEnt3_prob3.npy')
    prob4 = np.load(dir + 'maxEnt3_prob4.npy')
    prob5 = np.load(dir + 'maxEnt3_prob5.npy')
    prob6 = np.load(dir + 'maxEnt3_prob6.npy')
    prob7 = np.load(dir + 'maxEnt3_prob7.npy')
    prob8 = np.load(dir + 'maxEnt3_prob8.npy')
    
    prob0_0 = np.load(dir + 'maxEnt3_prob0_0.npy')
    prob1_0 = np.load(dir + 'maxEnt3_prob1_0.npy')
    prob2_0 = np.load(dir + 'maxEnt3_prob2_0.npy')
    prob3_0 = np.load(dir + 'maxEnt3_prob3_0.npy')
    prob4_0 = np.load(dir + 'maxEnt3_prob4_0.npy')
    prob5_0 = np.load(dir + 'maxEnt3_prob5_0.npy')
    prob6_0 = np.load(dir + 'maxEnt3_prob6_0.npy')
    prob7_0 = np.load(dir + 'maxEnt3_prob7_0.npy')
    prob8_0 = np.load(dir + 'maxEnt3_prob8_0.npy')
    
    for gene in geneNames:
        sequence = str(genes[gene].seq).lower()
        sequences23 = [sequence[i:i+23] for i in range(len(sequence)-23+1)]
        hash0 = np.array([hashSequence(seq[0:7]) for seq in sequences23])
        hash1 = np.array([hashSequence(seq[7:14]) for seq in sequences23])
        hash2 = np.array([hashSequence(seq[14:]) for seq in sequences23])
        hash3 = np.array([hashSequence(seq[4:11]) for seq in sequences23])
        hash4 = np.array([hashSequence(seq[11:18]) for seq in sequences23])
        hash5 = np.array([hashSequence(seq[4:7]) for seq in sequences23])
        hash6 = np.array([hashSequence(seq[7:11]) for seq in sequences23])
        hash7 = np.array([hashSequence(seq[11:14]) for seq in sequences23])
        hash8 = np.array([hashSequence(seq[14:18]) for seq in sequences23])
        
        probs = np.log2(prob0[hash0]) + np.log2(prob1[hash1]) + np.log2(prob2[hash2]) + \
            np.log2(prob3[hash3]) + np.log2(prob4[hash4]) - np.log2(prob5[hash5]) - \
            np.log2(prob6[hash6]) - np.log2(prob7[hash7]) - np.log2(prob8[hash8]) - \
            (np.log2(prob0_0[hash0]) + np.log2(prob1_0[hash1]) + np.log2(prob2_0[hash2]) + \
            np.log2(prob3_0[hash3]) + np.log2(prob4_0[hash4]) - np.log2(prob5_0[hash5]) - \
            np.log2(prob6_0[hash6]) - np.log2(prob7_0[hash7]) - np.log2(prob8_0[hash8]))
            
        scores[gene] = np.zeros(len(sequence)) - np.inf
        scores[gene][19:-3] = probs
        scores[gene] = np.exp2(scores[gene])
    
    return scores
    
def maxEnt3_single(str seq, str dir):
    prob0 = np.load(dir + 'maxEnt3_prob0.npy')
    prob1 = np.load(dir + 'maxEnt3_prob1.npy')
    prob2 = np.load(dir + 'maxEnt3_prob2.npy')
    prob3 = np.load(dir + 'maxEnt3_prob3.npy')
    prob4 = np.load(dir + 'maxEnt3_prob4.npy')
    prob5 = np.load(dir + 'maxEnt3_prob5.npy')
    prob6 = np.load(dir + 'maxEnt3_prob6.npy')
    prob7 = np.load(dir + 'maxEnt3_prob7.npy')
    prob8 = np.load(dir + 'maxEnt3_prob8.npy')
    
    prob0_0 = np.load(dir + 'maxEnt3_prob0_0.npy')
    prob1_0 = np.load(dir + 'maxEnt3_prob1_0.npy')
    prob2_0 = np.load(dir + 'maxEnt3_prob2_0.npy')
    prob3_0 = np.load(dir + 'maxEnt3_prob3_0.npy')
    prob4_0 = np.load(dir + 'maxEnt3_prob4_0.npy')
    prob5_0 = np.load(dir + 'maxEnt3_prob5_0.npy')
    prob6_0 = np.load(dir + 'maxEnt3_prob6_0.npy')
    prob7_0 = np.load(dir + 'maxEnt3_prob7_0.npy')
    prob8_0 = np.load(dir + 'maxEnt3_prob8_0.npy')
    
    seq = seq.lower()
    sequences23 = [seq[i:i+23] for i in range(len(seq)-23+1)]
    hash0 = np.array([hashSequence(seq[0:7]) for seq in sequences23])
    hash1 = np.array([hashSequence(seq[7:14]) for seq in sequences23])
    hash2 = np.array([hashSequence(seq[14:]) for seq in sequences23])
    hash3 = np.array([hashSequence(seq[4:11]) for seq in sequences23])
    hash4 = np.array([hashSequence(seq[11:18]) for seq in sequences23])
    hash5 = np.array([hashSequence(seq[4:7]) for seq in sequences23])
    hash6 = np.array([hashSequence(seq[7:11]) for seq in sequences23])
    hash7 = np.array([hashSequence(seq[11:14]) for seq in sequences23])
    hash8 = np.array([hashSequence(seq[14:18]) for seq in sequences23])
    
    probs = np.log2(prob0[hash0]) + np.log2(prob1[hash1]) + np.log2(prob2[hash2]) + \
            np.log2(prob3[hash3]) + np.log2(prob4[hash4]) - np.log2(prob5[hash5]) - \
            np.log2(prob6[hash6]) - np.log2(prob7[hash7]) - np.log2(prob8[hash8]) - \
            (np.log2(prob0_0[hash0]) + np.log2(prob1_0[hash1]) + np.log2(prob2_0[hash2]) + \
            np.log2(prob3_0[hash3]) + np.log2(prob4_0[hash4]) - np.log2(prob5_0[hash5]) - \
            np.log2(prob6_0[hash6]) - np.log2(prob7_0[hash7]) - np.log2(prob8_0[hash8]))
            
    scores = np.log2(np.zeros(len(seq)))
    scores[19:-3] = probs
    return np.exp2(scores)

def sreScores_single(str seq, double [:] sreScores, int kmer = 6):
    indices = [hashSequence(seq[i:i+kmer]) for i in range(len(seq)-kmer+1)]
    sequenceSRES = [sreScores[indices[i]] for i in range(len(indices))]
    return sequenceSRES

def get_all_5ss(gene, reference, genes):
    # Get all the 5'SS for a gene based on the annotation in reference
    info = genes[gene].description.split(' ')
    if reference.loc[gene,6] == ',': exonEnds = []
    else: exonEnds = [int(start)-1 for start in reference.loc[gene,6].split(',')[:-1]] # intron starts -> exon ends
    if reference.loc[gene,7] == ',': exonStarts = []
    else: exonStarts = [int(end)+1 for end in reference.loc[gene,7].split(',')[:-1]] # exon starts -> intron ends
    
    if info[6] == 'Strand:-': 
        stop = int(info[5][5:])
        annnotation = [stop - exonStarts[i-1] + 2 for i in range(len(exonStarts),0,-1)]      
        
    elif info[6] == 'Strand:+': 
        start = int(info[4][6:])
        annnotation = [exonEnds[i] - start + 1 for i in range(len(exonEnds))]
        
    return(annnotation)

def get_all_3ss(gene, reference, genes):
    # Get all the 3'SS for a gene based on the annotation in reference
    info = genes[gene].description.split(' ')
    if reference.loc[gene,6] == ',': exonEnds = []
    else: exonEnds = [int(start)-1 for start in reference.loc[gene,6].split(',')[:-1]] # intron starts -> exon ends
    if reference.loc[gene,7] == ',': exonStarts = []
    else: exonStarts = [int(end)+1 for end in reference.loc[gene,7].split(',')[:-1]] # exon starts -> intron ends
    
    if info[6] == 'Strand:-': 
        stop = int(info[5][5:])
        annnotation = [stop - exonEnds[i-1] - 2 for i in range(len(exonEnds),0,-1)]      
        
    elif info[6] == 'Strand:+': 
        start = int(info[4][6:])
        annnotation = [exonStarts[i] - start - 3 for i in range(len(exonStarts))]
        
    return(annnotation)

def get_hexamer_real_decoy_counts(geneNames, trueSeqs, decoySS, genes, kmer, sreEffect5_exon, sreEffect5_intron, sreEffect3_exon, sreEffect3_intron, B3 = 3, B5 = 5):
    # Get the counts of hexamers in the flanking regions for real and decoy ss with restriction to exons and introns for the real ss
    true_counts_5_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    true_counts_5_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    true_counts_3_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    true_counts_3_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    decoy_counts_5_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    decoy_counts_5_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    decoy_counts_3_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    decoy_counts_3_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    
    for gene in geneNames:
        trueThrees = np.nonzero(trueSeqs[gene] == B3)[0][:-1]
        trueFives = np.nonzero(trueSeqs[gene] == B5)[0][1:]
        for i in range(len(trueThrees)):
            three = trueThrees[i]
            five = trueFives[i]
            
            # 3'SS
            sequence = str(genes[gene].seq[three+4:three+sreEffect3_exon+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[three+4:].lower())
            if five-3 < three+sreEffect3_exon+1: sequence = str(genes[gene].seq[three+4:five-3].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: true_counts_3_exon[s] += 1
            
            sequence = str(genes[gene].seq[three-sreEffect3_intron:three-19].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:three-19].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: true_counts_3_intron[s] += 1
                
            # 5'SS
            sequence = str(genes[gene].seq[five-sreEffect5_exon:five-3].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:five-3].lower())
            if five-sreEffect5_exon < three+4: sequence = str(genes[gene].seq[three+4:five-3].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: true_counts_5_exon[s] += 1
            
            sequence = str(genes[gene].seq[five+6:five+sreEffect5_intron+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[five+6:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: true_counts_5_intron[s] += 1
        
        decoyThrees = np.nonzero(decoySS[gene] == B3)[0]
        decoyFives = np.nonzero(decoySS[gene] == B5)[0]
        for ss in decoyFives:
            sequence = str(genes[gene].seq[ss-sreEffect5_exon:ss-3].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss-3].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: decoy_counts_5_exon[s] += 1
            
            sequence = str(genes[gene].seq[ss+6:ss+sreEffect5_intron+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss+6:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: decoy_counts_5_intron[s] += 1
    
        for ss in decoyThrees:
            sequence = str(genes[gene].seq[ss+4:ss+sreEffect3_exon+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss+4:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: decoy_counts_3_exon[s] += 1
            
            sequence = str(genes[gene].seq[ss-sreEffect3_intron:ss-19].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss-19].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: decoy_counts_3_intron[s] += 1
    
    return(true_counts_5_intron, true_counts_5_exon, true_counts_3_intron, true_counts_3_exon, 
           decoy_counts_5_intron, decoy_counts_5_exon, decoy_counts_3_intron, decoy_counts_3_exon)

def get_hexamer_counts(geneNames, set1, set2, genes, kmer, sreEffect5_exon, sreEffect5_intron, sreEffect3_exon, sreEffect3_intron, B3 = 3, B5 = 5):
    # Get the counts of hexamers in the flanking regions for two sets of ss
    set1_counts_5_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    set1_counts_5_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    set1_counts_3_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    set1_counts_3_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    set2_counts_5_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    set2_counts_5_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    set2_counts_3_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    set2_counts_3_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    
    for gene in geneNames:
        set1Threes = np.nonzero(set1[gene] == B3)[0]
        set1Fives = np.nonzero(set1[gene] == B5)[0]
        set2Threes = np.nonzero(set2[gene] == B3)[0]
        set2Fives = np.nonzero(set2[gene] == B5)[0]
        
        for ss in set1Fives:
            sequence = str(genes[gene].seq[ss-sreEffect5_exon:ss-3].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss-3].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set1_counts_5_exon[s] += 1
            
            sequence = str(genes[gene].seq[ss+6:ss+sreEffect5_intron+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss+6:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set1_counts_5_intron[s] += 1
    
        for ss in set1Threes:
            sequence = str(genes[gene].seq[ss+4:ss+sreEffect3_exon+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss+4:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set1_counts_3_exon[s] += 1
            
            sequence = str(genes[gene].seq[ss-sreEffect3_intron:ss-19].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss-19].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set1_counts_3_intron[s] += 1
        
        for ss in set2Fives:
            sequence = str(genes[gene].seq[ss-sreEffect5_exon:ss-3].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss-3].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set2_counts_5_exon[s] += 1
            
            sequence = str(genes[gene].seq[ss+6:ss+sreEffect5_intron+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss+6:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set2_counts_5_intron[s] += 1
    
        for ss in set2Threes:
            sequence = str(genes[gene].seq[ss+4:ss+sreEffect3_exon+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss+4:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set2_counts_3_exon[s] += 1
            
            sequence = str(genes[gene].seq[ss-sreEffect3_intron:ss-19].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss-19].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set2_counts_3_intron[s] += 1
    
    return(set1_counts_5_intron, set1_counts_5_exon, set1_counts_3_intron, set1_counts_3_exon, 
           set2_counts_5_intron, set2_counts_5_exon, set2_counts_3_intron, set2_counts_3_exon)

def get_hexamer_real_decoy_scores(geneNames, trueSeqs, decoySS, genes, kmer, sreEffect5_exon, sreEffect5_intron, sreEffect3_exon, sreEffect3_intron):
    # Get the real versus decoy scores for all hexamers
    true_counts_5_intron, true_counts_5_exon, true_counts_3_intron, true_counts_3_exon, decoy_counts_5_intron, decoy_counts_5_exon, decoy_counts_3_intron, decoy_counts_3_exon = get_hexamer_real_decoy_counts(geneNames, trueSeqs, decoySS, genes, kmer = kmer, sreEffect5_exon = sreEffect5_exon, sreEffect5_intron = sreEffect5_intron, sreEffect3_exon = sreEffect3_exon, sreEffect3_intron = sreEffect3_intron)
    
    # Add pseudocounts
    true_counts_5_intron = true_counts_5_intron + 1
    true_counts_5_exon = true_counts_5_exon + 1
    true_counts_3_intron = true_counts_3_intron + 1
    true_counts_3_exon = true_counts_3_exon + 1
    decoy_counts_5_intron = decoy_counts_5_intron + 1
    decoy_counts_5_exon = decoy_counts_5_exon + 1
    decoy_counts_3_intron = decoy_counts_3_intron + 1
    decoy_counts_3_exon = decoy_counts_3_exon + 1
    
    true_counts_intron = true_counts_5_intron + true_counts_3_intron
    true_counts_exon = true_counts_5_exon + true_counts_3_exon
    decoy_counts_intron = decoy_counts_5_intron + decoy_counts_3_intron
    decoy_counts_exon = decoy_counts_5_exon + decoy_counts_3_exon
    
    trueFreqs_intron = np.exp(np.log(true_counts_intron) - np.log(np.sum(true_counts_intron))) 
    decoyFreqs_intron = np.exp(np.log(decoy_counts_intron) - np.log(np.sum(decoy_counts_intron)))
    trueFreqs_exon = np.exp(np.log(true_counts_exon) - np.log(np.sum(true_counts_exon)))
    decoyFreqs_exon = np.exp(np.log(decoy_counts_exon) - np.log(np.sum(true_counts_exon)))
    
    sreScores_intron = np.exp(np.log(true_counts_intron) - np.log(np.sum(true_counts_intron)) 
                              - np.log(decoy_counts_intron) + np.log(np.sum(decoy_counts_intron)))
    sreScores_exon = np.exp(np.log(true_counts_exon) - np.log(np.sum(true_counts_exon)) 
                            - np.log(decoy_counts_exon) + np.log(np.sum(decoy_counts_exon)))
    
    sreScores3_intron = np.exp(np.log(true_counts_3_intron) - np.log(np.sum(true_counts_3_intron)) 
                                - np.log(decoy_counts_3_intron) + np.log(np.sum(decoy_counts_3_intron)))
    sreScores3_exon = np.exp(np.log(true_counts_3_exon) - np.log(np.sum(true_counts_3_exon)) 
                              - np.log(decoy_counts_3_exon) + np.log(np.sum(decoy_counts_3_exon)))
    
    sreScores5_intron = np.exp(np.log(true_counts_5_intron) - np.log(np.sum(true_counts_5_intron)) 
                                - np.log(decoy_counts_5_intron) + np.log(np.sum(decoy_counts_5_intron)))
    sreScores5_exon = np.exp(np.log(true_counts_5_exon) - np.log(np.sum(true_counts_5_exon)) 
                              - np.log(decoy_counts_5_exon) + np.log(np.sum(decoy_counts_5_exon)))
    
    return(sreScores_intron, sreScores_exon, sreScores3_intron, sreScores3_exon, sreScores5_intron, sreScores5_exon)
    
def score_sequences(sequences, double [:, :] exonicSREs5s, double [:, :] exonicSREs3s, double [:, :] intronicSREs5s, double [:, :] intronicSREs3s, int k = 6, int sreEffect5_exon = 80, int sreEffect5_intron = 80, int sreEffect3_exon = 80, int sreEffect3_intron = 80, meDir = ''): #/home/kmccue/projects/scfg/code/maxEnt/triplets-9mer-max/
    # Inputs:
    #  - sequences: batch_size lenght list of sequence strings, maybe need about 21 kb/base for the longest string
    #  - transitions: list of two values [pME, pEE]
    #  - pIL: numpy array of intron length distribution
    #  - pELS: decay coefficient for transcribing state eg 1/10000
    #  - pELF: numpy array of first exon length distribution
    #  - pELM: numpy array of middle exon length distribution
    #  - pELL: numpy array of last exon length distribution
    #  - delay: number of bases to delay predictions by
    #  - sreScores: sre scores for each kmer
    #  - meDir: location of the maxent model lookups
    # Outputs:
    #  - bestPath: numpy array containing the cotranscriptional viterbi best paths for each input sequence
    #  - loglik: numpy array containing the calculated log likelihoods
    #  - tbStates: numpy array containing the states at which a traceback event occurred
    #  - tblast3: numpy array containing the last 3'ss when which a traceback event occurred
    #  - emissions5: final 5' emissions
    #  - emissions3: final 3' emissions
    #  - traceback5: the 5'ss tracebacks
    #  - traceback3: the 3'ss tracebacks
    cdef int batch_size, g, L, d, i, t, ssRange
    batch_size = len(sequences)
    cdef int [:] lengths = np.zeros(batch_size, dtype=np.dtype("i"))
    
    # Collect the lengths of each sequence in the batch
    for g in range(batch_size): 
        lengths[g] = len(sequences[g])
    L = np.max(lengths.base)
    
    cdef double [:, :] emissions3 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    cdef double [:, :] emissions5 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    
    # Get the emissions and apply sre scores to them
    for g in range(batch_size): 
        # 5'SS exonic effects (upstream)
        ssRange = 3
        emissions5.base[g,:lengths[g]] = np.log(maxEnt5_single(sequences[g].lower(), meDir))
        emissions5.base[g,k+ssRange:lengths[g]] += np.cumsum(exonicSREs5s.base[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions5.base[g,sreEffect5_exon+1:lengths[g]] -= np.cumsum(exonicSREs5s.base[g,:lengths[g]-k+1])[:-(sreEffect5_exon+1)+(k-1)]
        
        # 3'SS intronic effects (upstream)
        ssRange = 19
        emissions3.base[g,:lengths[g]] = np.log(maxEnt3_single(sequences[g].lower(), meDir))
        emissions3.base[g,k+ssRange:lengths[g]] += np.cumsum(intronicSREs3s.base[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions3.base[g,sreEffect3_intron+1:lengths[g]] -= np.cumsum(intronicSREs3s.base[g,:lengths[g]-k+1])[:-(sreEffect3_intron+1)+(k-1)]
        
        # 5'SS intronic effects (downstream)
        ssRange = 4
        emissions5.base[g,:lengths[g]-sreEffect5_intron] += np.cumsum(intronicSREs5s.base[g,:lengths[g]-k+1])[sreEffect5_intron-k+1:]
        emissions5.base[g,lengths[g]-sreEffect5_intron:lengths[g]-k+1-ssRange] += np.sum(intronicSREs5s.base[g,:lengths[g]-k+1])
        emissions5.base[g,:lengths[g]-k+1-ssRange] -= np.cumsum(intronicSREs5s.base[g,ssRange:lengths[g]-k+1])
        
        # 3'SS exonic effects (downstream)
        ssRange = 3
        emissions3.base[g,:lengths[g]-sreEffect5_exon] += np.cumsum(exonicSREs3s.base[g,:lengths[g]-k+1])[sreEffect5_exon-k+1:]
        emissions3.base[g,lengths[g]-sreEffect5_exon:lengths[g]-k+1-ssRange] += np.sum(exonicSREs3s.base[g,:lengths[g]-k+1])
        emissions3.base[g,:lengths[g]-k+1-ssRange] -= np.cumsum(exonicSREs3s.base[g,ssRange:lengths[g]-k+1])
        
    return np.exp(emissions5.base), np.exp(emissions3.base)
                 
def cass_accuracy_metrics(scored_sequences_5, scored_sequences_3, geneNames, trueSeqs, B3 = 3, B5 = 5):
    # Get the best cutoff and the associated metrics for the CASS scored sequences
    true_scores = []
    for g, gene in enumerate(geneNames):
        L = len(trueSeqs[gene])
        for score in np.log2(scored_sequences_5[g,:L][trueSeqs[gene] == B5]): true_scores.append(score)
        for score in np.log2(scored_sequences_3[g,:L][trueSeqs[gene] == B3]): true_scores.append(score)
    min_score = np.min(true_scores)
    if np.isnan(min_score): 
        return 0, 0, 0, min_score
    
    all_scores = []
    for g, gene in enumerate(geneNames):
        L = len(trueSeqs[gene])
        for score in np.log2(scored_sequences_5[g,:L][trueSeqs[gene] != B5]): 
            if score > min_score: all_scores.append(score)
        for score in np.log2(scored_sequences_3[g,:L][trueSeqs[gene] != B3]): 
            if score > min_score: all_scores.append(score)
    
    all_scores = np.array(true_scores + all_scores)
    all_scores_bool = np.zeros(len(all_scores), dtype=np.dtype("i"))
    all_scores_bool[:len(true_scores)] = 1
    sort_inds = np.argsort(all_scores)
    all_scores = all_scores[sort_inds]
    all_scores_bool = all_scores_bool[sort_inds]
    
    num_all_positives = len(true_scores)
    num_all = len(all_scores)
    best_f1 = 0
    best_cutoff = 0
    for i, cutoff in enumerate(all_scores):
        if all_scores_bool[i] == 0: continue
        true_positives = np.sum(all_scores_bool[i:])
        false_negatives = num_all_positives - true_positives
        false_positives = num_all - i - true_positives
        
        ssSens = true_positives / (true_positives + false_negatives)
        ssPrec = true_positives / (true_positives + false_positives)
        f1 = 2 / (1/ssSens + 1/ssPrec)
        if f1 >= best_f1:
            best_f1 = f1
            best_cutoff = cutoff
            best_sens = ssSens
            best_prec = ssPrec
        
    return best_sens, best_prec, best_f1, best_cutoff
    
def cass_accuracy_metrics_set_cutoff(scored_sequences_5, scored_sequences_3, geneNames, trueSeqs, cutoff, B3 = 3, B5 = 5):
    # Get the associated metrics for the CASS scored sequences with a given cutoff
    true_scores = []
    for g, gene in enumerate(geneNames):
        L = len(trueSeqs[gene])
        for score in np.log2(scored_sequences_5[g,:L][trueSeqs[gene] == B5]): true_scores.append(score)
        for score in np.log2(scored_sequences_3[g,:L][trueSeqs[gene] == B3]): true_scores.append(score)
    min_score = np.min(true_scores)
    if np.isnan(min_score): 
        return 0, 0, 0
    
    all_scores = []
    for g, gene in enumerate(geneNames):
        L = len(trueSeqs[gene])
        for score in np.log2(scored_sequences_5[g,:L][trueSeqs[gene] != B5]): 
            if score > min_score: all_scores.append(score)
        for score in np.log2(scored_sequences_3[g,:L][trueSeqs[gene] != B3]): 
            if score > min_score: all_scores.append(score)
    
    all_scores = np.array(true_scores + all_scores)
    all_scores_bool = np.zeros(len(all_scores), dtype=np.dtype("i"))
    all_scores_bool[:len(true_scores)] = 1
    sort_inds = np.argsort(all_scores)
    all_scores = all_scores[sort_inds]
    all_scores_bool = all_scores_bool[sort_inds]
    
    num_all_positives = len(true_scores)
    num_all = len(all_scores)
    
    true_positives = np.sum((all_scores > cutoff)&(all_scores_bool == 1))
    false_negatives = num_all_positives - true_positives
    false_positives = np.sum((all_scores > cutoff)&(all_scores_bool == 0))
    
    ssSens = true_positives / (true_positives + false_negatives)
    ssPrec = true_positives / (true_positives + false_positives)
    f1 = 2 / (1/ssSens + 1/ssPrec)
        
    return ssSens, ssPrec, f1

def order_genes(geneNames, genes, num_threads):
    # Re-order genes to feed into parallelized prediction algorithm to use parallelization efficiently
    # geneNames: list of names of genes to re-order based on length 
    # num_threads: number of threads available to parallelize across
    lengthsOfGenes = np.array([len(str(genes[gene].seq)) for gene in geneNames])
    geneNames = geneNames[np.argsort(lengthsOfGenes)]
    geneNames = np.flip(geneNames)

    # ordering the genes for optimal processing
    l = len(geneNames)
    ind = l - l//num_threads
    longest_thread = []
    for i in np.flip(range(num_threads)):
        longest_thread.append(ind)
        ind -= (l//num_threads + int(i<=(l%num_threads)))
    
    indices = longest_thread.copy()
    for i in range(1,l//num_threads):
        indices += list(np.array(longest_thread) + i)
    
    ind = l//num_threads
    for i in range(l%num_threads): indices.append(ind + i*l%num_threads)

    indices = np.argsort(indices)
    return(geneNames[indices])
    
def viterbi(sequences, transitions, double [:] pIL, double [:] pELS, double [:] pELF, double [:] pELM, double [:] pELL, double [:, :] exonicSREs5s, double [:, :] exonicSREs3s, double [:, :] intronicSREs5s, double [:, :] intronicSREs3s, int k, int sreEffect5_exon = 80, int sreEffect5_intron = 80, int sreEffect3_exon = 80, int sreEffect3_intron = 80, meDir = ''): #/home/kmccue/projects/scfg/code/maxEnt/triplets-9mer-max/
    # Inputs:
    #  - sequences: batch_size lenght list of sequence strings, maybe need about 21 kb/base for the longest string
    #  - transitions: list of two values [pME, pEE]
    #  - pIL: numpy array of intron length distribution
    #  - pELS: decay coefficient for transcribing state eg 1/10000
    #  - pELF: numpy array of first exon length distribution
    #  - pELM: numpy array of middle exon length distribution
    #  - pELL: numpy array of last exon length distribution
    #  - delay: number of bases to delay predictions by
    #  - sreScores: sre scores for each kmer
    #  - meDir: location of the maxent model lookups
    # Outputs:
    #  - bestPath: numpy array containing the cotranscriptional viterbi best paths for each input sequence
    #  - loglik: numpy array containing the calculated log likelihoods
    #  - tbStates: numpy array containing the states at which a traceback event occurred
    #  - tblast3: numpy array containing the last 3'ss when which a traceback event occurred
    #  - emissions5: final 5' emissions
    #  - emissions3: final 3' emissions
    #  - traceback5: the 5'ss tracebacks
    #  - traceback3: the 3'ss tracebacks
    cdef int batch_size, g, L, d, i, t, ssRange
    cdef double pME, p1E, pEE, pEO
    
    batch_size = len(sequences)
    
#     cdef int [:] t = np.zeros(batch_size, dtype=np.dtype("i"))
    cdef int [:] tbindex = np.zeros(batch_size, dtype=np.dtype("i"))
    cdef int [:] lengths = np.zeros(batch_size, dtype=np.dtype("i"))
    cdef double [:] loglik = np.log(np.zeros(batch_size, dtype=np.dtype("d")))
    
    # Collect the lengths of each sequence in the batch
    for g in range(batch_size): 
        lengths[g] = len(sequences[g])
    L = np.max(lengths.base)
    
    cdef double [:, :] emissions3 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    cdef double [:, :] emissions5 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
     
    cdef double [:, :] Three = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))    
    cdef double [:, :] Five = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    cdef int [:, :] traceback5 = np.zeros((batch_size, L), dtype=np.dtype("i")) + L
    cdef int [:, :] traceback3 = np.zeros((batch_size, L), dtype=np.dtype("i")) + L
    cdef int [:, :] bestPath = np.zeros((batch_size, L), dtype=np.dtype("i"))
    
    # Rewind state vars
    cdef int exon = 2
    cdef int intron = 1
     
    # Convert inputs to log space
    transitions = np.log(transitions)
    pIL = np.log(pIL)
    pELS = np.log(pELS)
    pELF = np.log(pELF)
    pELM = np.log(pELM)
    pELL = np.log(pELL)
    
    # Get the emissions and apply sre scores to them
    for g in range(batch_size): 
        # 5'SS exonic effects (upstream)
        ssRange = 3
        emissions5.base[g,:lengths[g]] = np.log(maxEnt5_single(sequences[g].lower(), meDir))
        emissions5.base[g,k+ssRange:lengths[g]] += np.cumsum(exonicSREs5s.base[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions5.base[g,sreEffect5_exon+1:lengths[g]] -= np.cumsum(exonicSREs5s.base[g,:lengths[g]-k+1])[:-(sreEffect5_exon+1)+(k-1)]
        
        # 3'SS intronic effects (upstream)
        ssRange = 19
        emissions3.base[g,:lengths[g]] = np.log(maxEnt3_single(sequences[g].lower(), meDir))
        emissions3.base[g,k+ssRange:lengths[g]] += np.cumsum(intronicSREs3s.base[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions3.base[g,sreEffect3_intron+1:lengths[g]] -= np.cumsum(intronicSREs3s.base[g,:lengths[g]-k+1])[:-(sreEffect3_intron+1)+(k-1)]
        
        # 5'SS intronic effects (downstream)
        ssRange = 4
        emissions5.base[g,:lengths[g]-sreEffect5_intron] += np.cumsum(intronicSREs5s.base[g,:lengths[g]-k+1])[sreEffect5_intron-k+1:]
        emissions5.base[g,lengths[g]-sreEffect5_intron:lengths[g]-k+1-ssRange] += np.sum(intronicSREs5s.base[g,:lengths[g]-k+1])
        emissions5.base[g,:lengths[g]-k+1-ssRange] -= np.cumsum(intronicSREs5s.base[g,ssRange:lengths[g]-k+1])
        
        # 3'SS exonic effects (downstream)
        ssRange = 3
        emissions3.base[g,:lengths[g]-sreEffect5_exon] += np.cumsum(exonicSREs3s.base[g,:lengths[g]-k+1])[sreEffect5_exon-k+1:]
        emissions3.base[g,lengths[g]-sreEffect5_exon:lengths[g]-k+1-ssRange] += np.sum(exonicSREs3s.base[g,:lengths[g]-k+1])
        emissions3.base[g,:lengths[g]-k+1-ssRange] -= np.cumsum(exonicSREs3s.base[g,ssRange:lengths[g]-k+1])
    
    # Convert the transition vector into named probabilities
    pME = transitions[0]
    p1E = np.log(1 - np.exp(pME))
    pEE = transitions[1]
    pEO = np.log(1 - np.exp(pEE))
    
    # Initialize the first and single exon probabilities
    cdef double [:] ES = np.zeros(batch_size, dtype=np.dtype("d"))
    for g in range(batch_size): ES[g] = pELS[L-1] + p1E
    
    for g in prange(batch_size, nogil=True): # parallelize over the sequences in the batch
        for t in range(1,lengths[g]):
            Five[g,t] = pELF[t-1]
            
            for d in range(t,0,-1):
                # 5'SS
                if pEE + Three[g,t-d-1] + pELM[d-1] > Five[g,t]:
                    traceback5[g,t] = d
                    Five[g,t] = pEE + Three[g,t-d-1] + pELM[d-1]
            
                # 3'SS
                if Five[g,t-d-1] + pIL[d-1] > Three[g,t]:
                    traceback3[g,t] = d
                    Three[g,t] = Five[g,t-d-1] + pIL[d-1]
                    
            Five[g,t] += emissions5[g,t]
            Three[g,t] += emissions3[g,t]
            
        # TODO: Add back in single exon case for if we ever come back to that
        for i in range(1, lengths[g]):
            if pME + Three[g,i] + pEO + pELL[lengths[g]-i-2] > loglik[g]:
                loglik[g] = pME + Three[g,i] + pEO + pELL[lengths[g]-i-2]
                tbindex[g] = i
                
        if ES[g] <= loglik[g]: # If the single exon case isn't better, trace back
            while 0 < tbindex[g]:
                bestPath[g,tbindex[g]] = 3
                tbindex[g] -= traceback3[g,tbindex[g]] + 1
                bestPath[g,tbindex[g]] = 5
                tbindex[g] -= traceback5[g,tbindex[g]] + 1 
        else:
            loglik[g] = ES[g]


    return bestPath.base, loglik.base, emissions5.base, emissions3.base









