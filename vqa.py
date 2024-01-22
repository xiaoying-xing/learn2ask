import json
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import openai
import math
from BLIP import Blip
from prompt import *
from PIL import Image
import torch
import collections

def sigmoid(x):
    return 1 / (1 + math.exp(-5 * x))


class VQAagent():
    def __init__(self, relevance, confidence):
        self.base_vqa = Blip(device='cuda')
        self.device = 'cuda'
        self.relevance = relevance
        self.confidence = confidence
        if self.relevance:
            self.base_scores = json.load(open('../logicVQA/experiments/OKVQA/blip_scores.json'))

    def call_chatgpt(self, chatgpt_messages, max_tokens=100, model="gpt-3.5-turbo"):
        response = openai.ChatCompletion.create(model=model, messages=chatgpt_messages, temperature=0.1,
                                                max_tokens=max_tokens)
        reply = response['choices'][0]['message']['content']
        total_tokens = response['usage']['total_tokens']
        return reply, total_tokens

    def call_gpt3(self, messages, max_tokens=100, model="text-davinci-003"):
        response = openai.Completion.create(model=model, prompt=messages, max_tokens=max_tokens,
                                            temperature=0.1)  # temperature=0.6,
        reply = response['choices'][0]['text']
        total_tokens = response['usage']['total_tokens']
        return reply, total_tokens

    def gene_message(self, questions, answers, relevance=True, confidence=True):
        message = ''
        for sub_q_dict, sub_a_dict in zip(questions, answers):
            sub_q = sub_q_dict['question']
            sub_q_relevance = sub_q_dict['score']
            sub_a = sub_a_dict['answer']
            sub_a_confidence = sub_a_dict['score']
            new_message = f'Question: {sub_q} '
            if relevance:
                new_message += f'Relevance: {sub_q_relevance} '
            new_message += f'Answer: {sub_a}'
            if confidence:
                new_message += f' Confidence: {sub_a_confidence:.2f}'
            message = message + new_message + '\n'
        return message

    def parse_reply(self, reply):
        lines = reply.split('\n')

        denotative_questions = []
        connotative_questions = []
        # Initialize a flag to determine which list to append to
        is_denotative = False
        for line in lines:
            if "Denotative" in line:
                is_denotative = True
                continue
            elif "Connotative" in line:
                is_denotative = False
                continue
            elif not line.strip():
                continue
            question = line.split('?')[0].split('. ', 1)[-1] + '?'
            question = question.replace('Question:', '').strip()
            question = question.replace('-','')

            if is_denotative:
                denotative_questions.append({'question': question, 'score': None})
            else:
                connotative_questions.append({'question': question, 'score': None})
        return denotative_questions, connotative_questions

    '''
    def parse_reply(self, reply):
        lines = reply.split('\n')
        questions = []
        for line in lines:
            question = line.split('?')[0].split('. ', 1)[-1] + '?'
            question = question.replace('Question:', '').strip()
            questions.append({'question': question, 'score': None})
        return questions
    '''

    def eval_relevance(self, q_id, questions, answers, target, image):
        prompt = []
        base_score = self.base_scores[str(q_id)]
        for q, a in zip(questions, answers):
            q = q['question']
            a = a['answer']
            prompt.append(f'{q}{a}. Question: {target} Short answer:')
        image = self.base_vqa.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        tensor_list = [image] * len(prompt)
        image_repeat = torch.cat(tensor_list, dim=0)
        _, score = self.base_vqa.model.generate({"image": image_repeat, "prompt": prompt})
        score = [round(sigmoid(base_score - math.exp(s.item())), 2) for s in score]
        return score

    def vqa(self, q_id, img_id, caption, question, keywords=None, choices=None, model='gpt-3.5-turbo'):
        keywords_str = ' '.join(keywords)
        #img_path = os.path.join('../logicVQA/datasets/A-OKVQA', 'val2017',
        #                        '{}.jpg'.format(str(img_id).zfill(12)))
        img_path = os.path.join('../logicVQA/datasets/OKVQA', 'val2014',
                                                        'COCO_val2014_{}.jpg'.format(str(img_id).zfill(12)))
        #img_path = os.path.join('../logicVQA/datasets/GQA/images','{}.jpg'.format(img_id))
        image = Image.open(img_path).convert('RGB')

        if model == 'gpt-3.5-turbo':
            message = [{"role": "system", "content": ASK_PROMPT}]
            message.append({"role": "user",
                        "content": f'Caption: {caption}\nQuestion: {question}\nKeywords: {keywords_str}'})
            #message.append({"role": "user", "content": f'Caption: {caption}\nQuestion: {question}'})
            reply, _ = self.call_chatgpt(message, max_tokens=100, model=model)
        elif model == 'text-davinci-002':
            message = ASK_PROMPT + '\n' + f'Caption: {caption}\nQuestion: {question}'
            reply, _ = self.call_gpt3(message, model=model, max_tokens=150)

        denotative_questions, connotative_questions = self.parse_reply(reply)
        #if len(denotative_questions) > 2:
        #    denotative_questions = denotative_questions[:2]
        #if len(connotative_questions) > 2:
        #    connotative_questions = connotative_questions[:2]

        denotative_answers = self.base_vqa.simple_query(image, [item['question'] for item in denotative_questions])
        connotative_answers = self.base_vqa.simple_query(image, [item['question'] for item in connotative_questions])
        if self.relevance:
            denotative_relevance = self.eval_relevance(q_id, denotative_questions, denotative_answers, question, image)
            connotative_relevance = self.eval_relevance(q_id, connotative_questions, connotative_answers, question, image)
            for q_dict, s in zip(denotative_questions, denotative_relevance):
                q_dict['score'] = s
            for q_dict, s in zip(connotative_questions, connotative_relevance):
                q_dict['score'] = s


        message_answer = f'Denotative questions:\n'
        deno_message = self.gene_message(denotative_questions, denotative_answers,
                                         relevance=self.relevance, confidence=self.confidence)
        message_answer += deno_message
        message_answer += 'Connotative questions:\n'
        conno_message = self.gene_message(connotative_questions, connotative_answers,
                                          relevance=self.relevance, confidence=self.confidence)
        message_answer += conno_message
        message_answer += f'Caption: {caption}\nTarget question: {question} Answer:'
        print('--------------')
        #print(keywords)
        print(message_answer)
        if model == 'gpt-3.5-turbo':
            message = [{"role": "system", "content": ANSWER_PROMPT}]
            message.append({"role": "user", "content": message_answer})
            answer, _ = self.call_chatgpt(message, model=model)

        elif model == 'text-davinci-002':
            message = ANSWER_PROMPT + '\n' + message_answer
            answer, _ = self.call_gpt3(message, model=model)
        print('q_id:', q_id, 'Prediction:', answer)
        return answer, {"denotative_q": denotative_questions, "denotative_an": denotative_answers,
                        "connotative_q": connotative_questions, "connotative_an": connotative_answers}

#openai_key = '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' # lyx
openai_key = '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'
openai.api_key = openai_key
vqa_llm = VQAagent(relevance=False, confidence=True)