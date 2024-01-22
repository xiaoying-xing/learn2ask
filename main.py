import json
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from dataset import MyDataset
import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle
from model import EncoderRNN, AttnDecoderRNN
from hparams import HParams
import torch.nn.functional as F
import torch.optim as optim
from vqa import vqa_llm
from cal_acc import RewardAgent


hparams = HParams()
word_embeddings = pickle.load(open('../KPCNet/data/word_embeddings.p', 'rb'))
word_embeddings = np.array(word_embeddings)
word2index = pickle.load(open('../KPCNet/data/vocab.p', 'rb'))
index2word = {idx:word for word, idx in word2index.items()}

trainset = MyDataset(word2index)
reward_agent = RewardAgent()
train_loader = DataLoader(dataset=trainset, batch_size=hparams.BATCH_SIZE, shuffle=True, num_workers=4)

encoder = EncoderRNN(hparams.HIDDEN_SIZE, word_embeddings, hparams.RNN_LAYERS,
                     dropout=hparams.DROPOUT, update_wd_emb=hparams.UPDATE_WD_EMB)
decoder = AttnDecoderRNN(hparams.HIDDEN_SIZE, len(word2index), word_embeddings, hparams.ATTN_TYPE,
                         hparams.RNN_LAYERS, dropout=hparams.DROPOUT, update_wd_emb=hparams.UPDATE_WD_EMB,
                         condition=hparams.DECODER_CONDITION_TYPE)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=hparams.LEARNING_RATE)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=hparams.LEARNING_RATE * hparams.DECODER_LEARNING_RATIO)
encoder.cuda()
decoder.cuda()

for epoch in range(hparams.N_EPOCHS):
    batch_loss = 0.0
    for i, batch in enumerate(train_loader):
        input_batches = batch["input_seq"].cuda().transpose(0, 1)
        input_lens = batch["input_len"].cuda()
        questions = batch['question']
        captions = batch['caption']
        img_ids = batch['img_id']
        q_ids = batch['q_id']
        gt_answers = batch['gt_answer']
        gt_answers = [list(t) for t in zip(*gt_answers)]
        mask = batch['mask'].cuda()

        encoder_outputs, encoder_hidden = encoder(input_batches, input_lens, None)
        decoder_hidden = encoder_hidden[:decoder.n_layers] + encoder_hidden[decoder.n_layers:]

        decoder_input = torch.LongTensor([word2index['<SOS>']] * hparams.BATCH_SIZE).cuda()
        all_decoder_ids = []
        action_probs = []
        for t in range(hparams.MAX_PROMPT_LEN):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_output = decoder_output*mask
            decoder_output = F.softmax(decoder_output)
            top_values, top_indices = decoder_output.topk(1, dim=1)
            action_probs.append(top_values)
            decoder_input = top_indices.squeeze()
            all_decoder_ids.append(top_indices)

        all_decoder_ids = torch.cat(all_decoder_ids, dim=1)
        action_probs = torch.cat(action_probs, dim=1)

        # prompt_tokens
        prompt_tokens = []
        for i in range(len(all_decoder_ids)):
            prompt_tokens.append([index2word[idx.item()] for idx in all_decoder_ids[i]])

        # connect to llm
        assert len(prompt_tokens) == hparams.BATCH_SIZE
        assert len(prompt_tokens[0]) == hparams.MAX_PROMPT_LEN
        rewards = []
        for b in range(hparams.BATCH_SIZE):
            print('----------------------')
            print(prompt_tokens[b])
            pred = vqa_llm.vqa(q_ids[b].item(), img_ids[b].item(), captions[b], questions[b], prompt_tokens[b])
            # compare with gt answer
            reward = reward_agent.get_reward(pred, gt_answers[b])
            print('prediction:', pred, 'gt:', gt_answers[b])
            rewards.append(reward)

        # update policy
        log_probs = torch.log(action_probs)
        rewards_tensor = torch.Tensor(rewards).unsqueeze(1).expand_as(log_probs).cuda()
        loss = -log_probs * rewards_tensor
        loss = loss.sum(dim=1).mean()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        batch_loss += loss.item()
        if i % 5 == 0 and i != 0:
            print(f'epoch: {epoch} batch{i} train loss: {batch_loss/5}')
            batch_loss = 0.0
            torch.save(encoder.state_dict(), 'saved_models/encoder_{}.pt'.format(i))
            torch.save(decoder.state_dict(), 'saved_models/decoder_{}.pt'.format(i))
