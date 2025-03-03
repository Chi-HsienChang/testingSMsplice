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
print("top ", top)
beam_width = 100
print("beam_width = ", beam_width)
print("-" * 50) 
not_match = []


num_truePositives_bs = 0
num_falsePositives_bs = 0
num_falseNegatives_bs = 0

for g, gene in enumerate(testGenes):
    valid_paths_sorted = []
    if g in range(len(testGenes)):
        print("g_index = ", g)
        print(f"lengths[{g}] = ", lengths[g])
        valid_paths = beam_search_for_top_k_parse(sequences[g], pME, pELF, pIL, pEE, pELM, pEO, pELL, emissions5[g], emissions3[g], lengths[g], beam_width)
        
        # Sort valid_paths in descending order by score
        valid_paths_sorted = sorted(valid_paths, key=lambda x: x[1], reverse=True)

        for i, (path, score, score_details) in enumerate(valid_paths_sorted):
            predicted_fives = np.array([index for index, symbol in enumerate(path) if symbol == 5])
            predicted_threes = np.array([index for index, symbol in enumerate(path) if symbol == 3])

            if i == 0:
                if (not np.array_equal(predFives_all[g], predicted_fives) or not np.array_equal(predThrees_all[g], predicted_threes)):
                    print("not match index = ", g)
                    not_match.append(g)
                    print("not_match = ",  not_match)
                else:
                    print("match index = ", g)
                    pred_match +=1
                    print("not_match = ",  not_match)

                # #####
                # print(gene)
                # print(f"Predicted len: {len(predFives_all[g])}")

                # #####
                # print(f"loglik[{g}]: {loglik[g]}") 
                # print(f"Annotated 5SS: {trueFives_all[g]}")
                # print(f"Annotated 3SS: {trueThrees_all[g]}")
                # print(f"SMsplice 5SS: {predFives_all[g]}")
                # print(f"SMsplice 3SS: {predThrees_all[g]}")

                # print(f"Predicted 5SS: {predicted_fives}")
                # print(f"Predicted 3SS: {predicted_threes}")
                # print(f"SM[seq, π] = {score}")
                # print("Score Details:")
                # for detail in score_details:
                #     print(f"  - {detail}")
                # print("-" * 50) 

                # True positives: correct predictions in both categories
                tp_fives = np.intersect1d(predFives_all[g], predicted_fives).size
                tp_threes = np.intersect1d(predThrees_all[g], predicted_threes).size
                num_truePositives_bs += (tp_fives + tp_threes)

                # False positives: predicted positions that are not in the ground truth
                fp_fives = np.setdiff1d(predFives_all[g], predicted_fives).size
                fp_threes = np.setdiff1d(predThrees_all[g], predicted_threes).size
                num_falsePositives_bs += (fp_fives + fp_threes)

                # False negatives: ground truth positions missed by predictions
                fn_fives = np.setdiff1d(trueFives_all[g], predicted_fives).size
                fn_threes = np.setdiff1d(trueThrees_all[g], predicted_threes).size
                num_falseNegatives_bs += (fn_fives + fn_threes)

            else:
                break
    # print(len(valid_paths_sorted))

print("not match len = ", len(not_match))
print("match len = ", pred_match)
print("total len = ",len(not_match)+pred_match)
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


#################################################################################
#################################################################################
#################################################################################
# Initialize counters for evaluation metrics
print("-" * 50) 
print("The viterbi result for pre-tained model")

num_truePositives = 0
num_falsePositives = 0
num_falseNegatives = 0

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

# Calculate sensitivity (Recall), precision, and F1 score
ssSens = num_truePositives / (num_truePositives + num_falseNegatives) if (num_truePositives + num_falseNegatives) > 0 else 0
ssPrec = num_truePositives / (num_truePositives + num_falsePositives) if (num_truePositives + num_falsePositives) > 0 else 0
f1 = 2 / ((1 / ssSens) + (1 / ssPrec)) if ssSens > 0 and ssPrec > 0 else 0

print("Recall = ", ssSens)
print("Precision = ", ssPrec)
print("F1 = ", f1)
