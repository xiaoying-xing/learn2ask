import json
from BLIP import Blip
from PIL import Image
import random
random.seed(102)
import os

def generate_q():
    templates = ['Is there any [slot] in the image?', 'How many [slot] in the image?', 'Is the image about [slot]?',
                 'What kind of [slot] are there in the image?', 'Where is the [slot]?', 'What color is the [slot]?']

    objects = json.load(open('../logicVQA/datasets/OKVQA/val2014_objects.json'))
    data = json.load(open('../logicVQA/datasets/OKVQA/process_test_promptcap.json'))
    data = random.sample(data, 1000)
    print(len(data))
    results = []
    vqa_model = Blip('cuda')
    for i in range(len(data)):
        item = data[i]
        q_id = item['question_id']
        img_id = item['image_id']
        img_path = os.path.join('../logicVQA/datasets/OKVQA', 'val2014',
                                'COCO_val2014_{}.jpg'.format(str(img_id).zfill(12)))
        image = Image.open(img_path).convert('RGB')
        obj = objects[str(img_id)]
        if obj:
            obj = obj[0]
        else:
            continue

        generate_questions = []
        for temp in templates:
            gene_q = temp.replace('[slot]', obj)
            generate_questions.append(gene_q)
        answers = vqa_model.simple_query(image, generate_questions)
        assert len(answers) == len(generate_questions)
        item['gene_questions'] = generate_questions
        item['gene_answers'] = answers
        results.append(item)

    print(len(results))
    with open('experiments/slotfill_questions.json', 'w') as f:
        json.dump(results, f)



import warnings
import subprocess
import atexit
import openai

openai_key = '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'
openai.api_key = openai_key

def gene_message(questions, answers, relevance=True, confidence=True):
    message = ''
    for sub_q, sub_a in zip(questions, answers):
        new_message = f'Question: {sub_q} '
        new_message += f'Answer: {sub_a}'
        message = message + new_message + '\n'
    return message

def call_chatgpt(chatgpt_messages, max_tokens=100, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(model=model, messages=chatgpt_messages, temperature=0.1,
                                                max_tokens=max_tokens)
    reply = response['choices'][0]['message']['content']
    total_tokens = response['usage']['total_tokens']
    return reply, total_tokens

ANSWER_PROMPT = 'Given a caption of a image, a question about the image and some related question-answer pairs, integrate the ' \
                'information and answer the target question in less than four words.'

filename = 'slot_filling'
warnings.filterwarnings("ignore")
curr_i = 0
while True:
    try:
        if os.path.exists("epoch.txt"):
            with open("epoch.txt", "r") as f:
                curr_i = int(f.readline())
        print('-----starting from {}-------'.format(curr_i))
        data = json.load(open('experiments/slotfill_questions.json'))
        results = []
        ids = []
        if os.path.exists('experiments/{}_result.json'.format(filename)):
            results = json.load(open('experiments/{}_result.json'.format(filename)))
            ids = [item['question_id'] for item in results]
        for i in range(curr_i,len(data)):
            item = data[i]
            q_id = item['question_id']
            if q_id in ids:
                continue
            ids.append(q_id)
            img_id = item['image_id']
            caption = item['caption']
            question = item['question']

            questions = item['gene_questions']
            answers = item['gene_answers']

            message = [{"role": "system", "content": ANSWER_PROMPT}]
            message_answer = f'Relevant questions:\n'
            deno_message = gene_message(questions, answers, relevance=False, confidence=False)
            message_answer += deno_message
            message_answer += f'Caption: {caption}\nTarget question: {question} Answer:'
            print('--------------')
            # print(keywords)
            print(message_answer)
            message.append({"role": "user", "content": message_answer})

            pred, _ = call_chatgpt(message)
            print('prediction:', pred)
            results.append({'question_id': q_id, "answer": pred})

            if i % 5 == 0:
                with open('experiments/{}_result.json'.format(filename), 'w') as f:
                    json.dump(results, f)

        with open('experiments/{}_result.json'.format(filename), 'w') as f:
            json.dump(results, f)

    except Exception as e:
        print(f"Error occurred: {e}. Interrupted at epoch {i}. Progress saved.")
        with open("epoch.txt", "w") as f:
            f.write(str(i))
        with open('experiments/{}_result.json'.format(filename), 'w') as f:
            json.dump(results, f)

        process = subprocess.Popen(['python', "experiment.py"])
        process.terminate()
        atexit.register(process.terminate)