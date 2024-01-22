"""Loads question answering data and feeds it to the models.
"""
from PIL import Image
import json
import numpy as np
import os
import torch
import torch.utils.data as data
from conceptnet import ConceptNet

class MyDataset(data.Dataset):
    def __init__(self, word2index, data_root='../logicVQA/datasets/', dataset_name='OKVQA', max_len=50):
        self.dataset_name = dataset_name
        data_path = os.path.join(data_root, dataset_name, 'process_test_promptcap.json')
        self.data = json.load(open(data_path))
        self.word2index = word2index
        self.max_len = max_len
        self.conceptnet = ConceptNet(word2index)

    def process(self, seq):
        seq = seq.replace('?',' ?')
        sequence = [self.word2index[w] if w in self.word2index else self.word2index['<unk>'] for w in seq.split(' ')[:self.max_len - 1]]
        sequence.append(self.word2index['<EOS>'])
        length = len(sequence)
        sequence += [self.word2index['<PAD>']] * (self.max_len - len(sequence))
        return sequence, length

    def __getitem__(self, index):
        """Returns one data pair (image and caption).
        """
        q_id = self.data[index]['question_id']
        img_id = self.data[index]['image_id']
        gt_answer = self.data[index]['answers']
        gt_answer = [item['answer'] for item in gt_answer]
        caption = self.data[index]['caption'].lower().strip()
        question = self.data[index]['question'].lower().strip()

        candidates_index = self.conceptnet.get_mask(question, caption)
        mask = torch.zeros(len(self.word2index))
        mask[candidates_index] = 1
        input_text = caption + ' . ' + question
        ques_seq, ques_len = self.process(input_text)
        return {"q_id": q_id, "question": question, "img_id": img_id, "caption": caption, "gt_answer": gt_answer,
                "input_seq": np.array(ques_seq), "input_len": ques_len, "mask": mask}

    def __len__(self):
        return len(self.data)
