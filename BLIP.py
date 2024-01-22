import json
import os
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from tqdm import tqdm
import math
from lavis.models import load_model_and_preprocess
from PIL import Image
import torch


class Blip():
    def __init__(self, device, name="blip2_t5"):
        self.device = device
        self.model, self.vis_processors, _ = load_model_and_preprocess(name=name, model_type="pretrain_flant5xl",
                                                                       is_eval=True, device=device)

    def caption(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        caption = self.model.generate({"image": image})
        return caption

    def simple_query(self, image, q):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        if isinstance(q, str):
            answer, score = self.model.generate({"image": image, "prompt": "Question: {} Short answer:".format(q)})
            return answer[0], math.exp(score.item())
        elif isinstance(q, list) and len(q) > 0:
            tensor_list = [image] * len(q)
            image_repeat = torch.cat(tensor_list, dim=0)
            prompt = [f'Question: {item} Short answer:' for item in q]
            answer, score = self.model.generate({"image": image_repeat, "prompt": prompt})
            output = []
            for ans, s in zip(answer, score):
                output.append({'answer': ans, 'score': math.exp(s.item())})
            return output
            # return answer

        else:
            return []


def process_q():
    blip = Blip('cuda')
    data = json.load(open('/data1/xiaoying/llama_gene_q.json')) + json.load(open('/data1/xiaoying/llama_gene_q2.json'))
    n_invalid = 0
    print(len(data))
    final_write = []
    for item in tqdm(data):
        img_id = item['image_id']
        img_path = os.path.join('../logicVQA/datasets/OKVQA', 'val2014',
                                'COCO_val2014_{}.jpg'.format(str(img_id).zfill(12)))
        image = Image.open(img_path).convert('RGB')
        outputs = item['llama'].split('\n')
        outputs_final = []
        if len(outputs) == 1:
            n_invalid += 1
            continue
        else:
            for line in outputs:
                if 'denotative' in line or 'Denotative' in line or 'connotative' in line or 'Connotative' in line:
                    continue
                if line in outputs_final or '?' not in line:
                    continue
                outputs_final.append(line.split('. ')[-1])
        answers = blip.simple_query(image, outputs_final)
        item['llama'] = {'gene_q': outputs_final, 'gene_an': answers}
        final_write.append(item)

    print(len(data), len(final_write), n_invalid)
    with open('llama_gene_q_process.json', 'w') as f:
        json.dump(final_write, f)

def sigmoid(x):
    return 1 / (1 + math.exp(-5 * x))

def eval_relevance(q_id, questions, answers, target, img_id):
    img_path = os.path.join('../logicVQA/datasets/OKVQA', 'val2014',
                            'COCO_val2014_{}.jpg'.format(str(img_id).zfill(12)))
    image = Image.open(img_path).convert('RGB')
    base_vqa = Blip('cuda')
    prompt = []
    base_scores = json.load(open('../logicVQA/experiments/OKVQA/blip_scores.json'))
    base_score = base_scores[str(q_id)]
    for q, a in zip(questions, answers):
        prompt.append(f'{q}{a}. Question: {target} Short answer:')
    image = base_vqa.vis_processors["eval"](image).unsqueeze(0).cuda()
    tensor_list = [image] * len(prompt)
    image_repeat = torch.cat(tensor_list, dim=0)
    _, score = base_vqa.model.generate({"image": image_repeat, "prompt": prompt})
    score = [round(sigmoid(base_score - math.exp(s.item())), 2) for s in score]
    return score

