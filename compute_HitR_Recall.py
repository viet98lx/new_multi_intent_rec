import argparse
import numpy as np
import re

parser = argparse.ArgumentParser(description='Calculate recall')
parser.add_argument('--result_file', type=str, help='file contains predicted result', required=True)
# parser.add_argument('--top_k', type=int, help='top k highest rank items', required=True)
args = parser.parse_args()
result_file = args.result_file
# top_k = args.top_k
list_seq = []
list_seq_topk_predicted = []
with open(result_file, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        # if(i == 0):
        #     continue
        if(i % 2 == 0):
            ground_truth = line.split('|')[1]
            list_item = re.split('[\\s]+',ground_truth.strip())
            list_seq.append(list_item.copy())
            list_item.clear()
        if(i % 2 == 1):
            predicted_items = line.split('|')[1:]
            list_top_k_item = []
            for item in predicted_items:
                item_key = item.strip().split(':')[0]
                list_top_k_item.append(item_key)
            list_seq_topk_predicted.append(list_top_k_item.copy())
            list_top_k_item.clear()

for topk in [5,10,15,20]:
    list_recall = []
    for gt, predict in zip(list_seq, list_seq_topk_predicted):
            num_correct = len(set(gt).intersection(predict[:topk]))
            list_recall.append(num_correct / len(gt))
    recall = np.array(list_recall).mean()
    print("Recall top {%2d}: {%.4f}" % (topk, recall))

    hit_count = 0
    for gt, predict in zip(list_seq, list_seq_topk_predicted):
        num_correct = len(set(gt).intersection(predict[:topk]))
        if num_correct > 0:
            hit_count += 1
    hr =  hit_count / len(list_seq_topk_predicted)
    print("Hit ratio top {%2d}: {%.4f}" % (topk, hr))

