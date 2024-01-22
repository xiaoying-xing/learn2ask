import json

import argparse
import pathlib
import json
import glob
import numpy as np

def eval_aokvqa(dataset, preds, multiple_choice=False, strict=True):

    if isinstance(dataset, list):
        dataset = { dataset[i]['question_id'] : dataset[i] for i in range(len(dataset)) }

    if multiple_choice is False:
        dataset = {k:v for k,v in dataset.items() if v['difficult_direct_answer'] is False}

    if strict:
        dataset_qids = set(dataset.keys())
        preds_qids = set(preds.keys())
        assert dataset_qids.issubset(preds_qids)

    # dataset = q_id (str) : dataset element (dict)
    # preds = q_id (str) : prediction (str)

    acc = []

    for q in dataset.keys():
        if q not in preds.keys():
            #acc.append(0.0)
            continue

        pred = preds[q]
        choices = dataset[q]['choices']
        direct_answers = dataset[q]['direct_answers']

        ## Multiple Choice setting
        if multiple_choice:
            if strict:
                assert pred in choices, 'Prediction must be a valid choice'
            correct_choice_idx = dataset[q]['correct_choice_idx']
            acc.append( float(pred == choices[correct_choice_idx]) )
        ## Direct Answer setting
        else:
            num_match = sum([pred == da for da in direct_answers])
            vqa_acc = min(1.0, num_match / 3.0)
            acc.append(vqa_acc)

    acc = sum(acc) / len(acc) * 100

    return acc


if __name__ == '__main__':
    data = json.load(open('../logicVQA/datasets/A-OKVQA/process_val.json'))
    pred = json.load(open('experiments/aok_mc_result.json'))
    ids = []
    pred_new = []
    for item in pred:
        if item['question_id'] not in ids:
            ids.append(item['question_id'])
            pred_new.append(item)
    print(len(ids))
    my_data = {}
    for line in data:
        if line['question_id'] in ids:
            my_data[line['question_id']] = line

    acc = []
    for item in pred_new:
        annotation = my_data[item['question_id']]
        pred_ans = item['answer']
        choices = annotation['choices']
        #if pred_ans not in choices:
        #    continue
        correct_choice_idx = annotation['correct_choice_idx']
        acc.append(float(pred_ans == choices[correct_choice_idx]))

    print(len(acc), np.mean(acc))
