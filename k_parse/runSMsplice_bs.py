import numpy as np
import pandas as pd
import time, argparse#, json, pickle
from Bio import SeqIO, SeqUtils, motifs
from Bio.Seq import Seq 
from Bio.SeqRecord import SeqRecord
import scipy.ndimage
import scipy.stats as stats
import pickle
from SMsplice import *

np.random.seed(0)
print("seed = 0")

# Load the dictionary from the pickle file
with open('arabidopsis_new.pkl', 'rb') as f:
    data = pickle.load(f)

# Optionally, extract individual variables
sequences = data['sequences']
pME = data['pME']
pELF = data['pELF']
pIL = data['pIL']
pEE = data['pEE']
pELM = data['pELM']
pEO = data['pEO']
pELL = data['pELL']
emissions5 = data['emissions5']
emissions3 = data['emissions3']
lengths = data['lengths']
trueSeqs = data['trueSeqs']
testGenes = data['testGenes']
B3 = data['B3']
B5 = data['B5']

with open('arabidopsis_pred_all_new.pkl', 'rb') as f:
    data_pred_all = pickle.load(f)

pred_all = data_pred_all['pred_all']
loglik = pred_all[1]
# Load the predictions from the pickle file
with open('predictions_new.pkl', 'rb') as f:
    data = pickle.load(f)

predFives_all = data['predFives_all']
predThrees_all = data['predThrees_all']
trueFives_all = data['trueFives_all']
trueThrees_all = data['trueThrees_all']

#################################################################################
#################################################################################
#################################################################################

pred_match = 0
top = 1
print("BS top = ", top)
beam_width = 100
print("beam_width =", beam_width)

# TAIR_not_match = list(range(1117))

TAIR_not_match = [19, 23, 33, 34, 45, 62, 77, 79, 80, 122, 137, 165, 168, 174, 178, 187, 211, 214, 224, 258, 259, 263, 278, 282, 293, 300, 343, 355, 358, 369, 372, 373, 381, 385, 386, 388, 398, 399, 439, 445, 450, 457, 463, 478, 483, 500, 501, 536, 556, 559, 570, 573, 580, 581, 592, 599, 603, 611, 624, 629, 632, 643, 650, 664, 693, 700, 706, 737, 738, 741, 746, 749, 751, 762, 767, 781, 801, 816, 821, 857, 860, 866, 884, 889, 907, 915, 918, 922, 933, 956, 961, 971, 974, 977, 982, 983, 990, 991, 1011, 1015, 1021, 1029, 1076, 1088, 1091, 1099, 1101, 1108]

# TAIR_not_match = [19, 23]
print("not_match_g_index = ", TAIR_not_match)   

print("-" * 50) 

# TAIR_not_match = [19, 23]

not_match = []

bs_count = 0
num_truePositives_bs = 0
num_falsePositives_bs = 0
num_falseNegatives_bs = 0

for g, gene in enumerate(testGenes):
    valid_paths_sorted = []
    # if g in [19]:
    # if g in TAIR_not_match:
    print(gene)
    print("g_index = ", g)
    print(f"lengths[{g}] = ", lengths[g])
    print(f"Predicted len: {len(predFives_all[g])}")
    valid_paths = beam_search_for_top_k_parse(sequences[g], pME, pELF, pIL, pEE, pELM, pEO, pELL, emissions5[g], emissions3[g], lengths[g], beam_width)
    
    # Sort valid_paths in descending order by score
    valid_paths_sorted = sorted(valid_paths, key=lambda x: x[1], reverse=True)

    for i, (path, score, score_details) in enumerate(valid_paths_sorted):
        predicted_fives = np.array([index for index, symbol in enumerate(path) if symbol == 5])
        predicted_threes = np.array([index for index, symbol in enumerate(path) if symbol == 3])

        if i == 0:
            # if (not np.array_equal(predFives_all[g], predicted_fives) or not np.array_equal(predThrees_all[g], predicted_threes)):
            #     print("not match index = ", g)
            #     not_match.append(g)
            #     print("not_match = ",  not_match)
            # else:
            #     print("match index = ", g)
            #     pred_match +=1
            #     print("not_match = ",  not_match)

            # #####
            #####
            print(f"SM[seq, π*]: {loglik[g]}") 
            print(f"Annotated 5SS: {trueFives_all[g]}")
            print(f"Annotated 3SS: {trueThrees_all[g]}")
            print(f"π* 5SS: {predFives_all[g]}")
            print(f"π* 3SS: {predThrees_all[g]}")

            print(f"π BS 5SS: {predicted_fives}")
            print(f"π BS 3SS: {predicted_threes}")
            print(f"SM[seq, π] = {score}")
            print("Score Details:")
            for detail in score_details:
                print(f"  - {detail}")
            print("-" * 50) 

            # True positives: correct predictions in both categories
            tp_fives = np.intersect1d(trueFives_all[g], predicted_fives).size
            tp_threes = np.intersect1d(trueThrees_all[g], predicted_threes).size
            num_truePositives_bs += (tp_fives + tp_threes)

            # False positives: predicted positions that are not in the ground truth
            fp_fives = np.setdiff1d(trueFives_all[g], predicted_fives).size
            fp_threes = np.setdiff1d(trueThrees_all[g], predicted_threes).size
            num_falsePositives_bs += (fp_fives + fp_threes)

            # False negatives: ground truth positions missed by predictions
            fn_fives = np.setdiff1d(trueFives_all[g], predicted_fives).size
            fn_threes = np.setdiff1d(trueThrees_all[g], predicted_threes).size
            num_falseNegatives_bs += (fn_fives + fn_threes)
            bs_count += 1

        else:
            break
    # print(len(valid_paths_sorted))

# print("not match len = ", len(not_match))
# print("match len = ", pred_match)
# print("total len = ",len(not_match)+pred_match)
# print("match len = ", len(testGenes) - len(not_match))


# Calculate sensitivity (Recall), precision, and F1 score
ssSens_bs = num_truePositives_bs / (num_truePositives_bs + num_falseNegatives_bs) if (num_truePositives_bs + num_falseNegatives_bs) > 0 else 0
ssPrec_bs = num_truePositives_bs / (num_truePositives_bs + num_falsePositives_bs) if (num_truePositives_bs + num_falsePositives_bs) > 0 else 0
f1_bs = 2 / ((1 / ssSens_bs) + (1 / ssPrec_bs)) if ssSens_bs > 0 and ssPrec_bs > 0 else 0

print("-" * 50) 
print("The beam search result for pre-tained model")

print("Recall_bs = ", ssSens_bs)
print("Precision_bs = ", ssPrec_bs)
print("F1_bs = ", f1_bs)

print("bs_count = ", bs_count)




#################################################################################
#################################################################################
#################################################################################
# Initialize counters for evaluation metrics
# print("-" * 50) 
# print("The viterbi result for pre-tained model")

# num_truePositives = 0
# num_falsePositives = 0
# num_falseNegatives = 0


# for idx in TAIR_not_match:
#     predFives = predFives_all[idx]  # 取出對應索引的 list
#     trueFives = trueFives_all[idx]
#     predThrees = predThrees_all[idx]
#     trueThrees = trueThrees_all[idx]
#     # True positives: correct predictions in both categories
#     tp_fives = np.intersect1d(predFives, trueFives).size
#     tp_threes = np.intersect1d(predThrees, trueThrees).size
#     num_truePositives += (tp_fives + tp_threes)

#     # False positives: predicted positions that are not in the ground truth
#     fp_fives = np.setdiff1d(predFives, trueFives).size
#     fp_threes = np.setdiff1d(predThrees, trueThrees).size
#     num_falsePositives += (fp_fives + fp_threes)

#     # False negatives: ground truth positions missed by predictions
#     fn_fives = np.setdiff1d(trueFives, predFives).size
#     fn_threes = np.setdiff1d(trueThrees, predThrees).size
#     num_falseNegatives += (fn_fives + fn_threes)

# # Calculate sensitivity (Recall), precision, and F1 score
# ssSens = num_truePositives / (num_truePositives + num_falseNegatives) if (num_truePositives + num_falseNegatives) > 0 else 0
# ssPrec = num_truePositives / (num_truePositives + num_falsePositives) if (num_truePositives + num_falsePositives) > 0 else 0
# f1 = 2 / ((1 / ssSens) + (1 / ssPrec)) if ssSens > 0 and ssPrec > 0 else 0

# print("Recall = ", ssSens)
# print("Precision = ", ssPrec)
# print("F1 = ", f1)
    



#################################################################################
#################################################################################
#################################################################################
num_truePositives = 0
num_falsePositives = 0
num_falseNegatives = 0

v_count = 0

# Iterate over each gene's predictions and ground truth
for predFives, trueFives, predThrees, trueThrees in zip(
    predFives_all, trueFives_all, predThrees_all, trueThrees_all
):
    # True positives: correct predictions in both categories
    tp_fives = np.intersect1d(predFives, trueFives).size
    tp_threes = np.intersect1d(predThrees, trueThrees).size
    num_truePositives += (tp_fives + tp_threes)

    # False positives: predicted positions that are not in the ground truth
    fp_fives = np.setdiff1d(predFives, trueFives).size
    fp_threes = np.setdiff1d(predThrees, trueThrees).size
    num_falsePositives += (fp_fives + fp_threes)

    # False negatives: ground truth positions missed by predictions
    fn_fives = np.setdiff1d(trueFives, predFives).size
    fn_threes = np.setdiff1d(trueThrees, predThrees).size
    num_falseNegatives += (fn_fives + fn_threes)
    v_count += 1

# Calculate sensitivity (Recall), precision, and F1 score
ssSens = num_truePositives / (num_truePositives + num_falseNegatives) if (num_truePositives + num_falseNegatives) > 0 else 0
ssPrec = num_truePositives / (num_truePositives + num_falsePositives) if (num_truePositives + num_falsePositives) > 0 else 0
f1 = 2 / ((1 / ssSens) + (1 / ssPrec)) if ssSens > 0 and ssPrec > 0 else 0



print("Recall = ", ssSens)
print("Precision = ", ssPrec)
print("F1 = ", f1)

print("v_count = ", v_count)

