import json
import random
from random import sample
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from vqa import vqa_llm
import subprocess
import warnings
import atexit
import pickle

filename = 'vqav2'
warnings.filterwarnings("ignore")
random.seed(102)
curr_i = 0
while True:
    try:
        if os.path.exists("epoch.txt"):
            with open("epoch.txt", "r") as f:
                curr_i = int(f.readline())
        print('-----starting from {}-------'.format(curr_i))
        data = json.load(open('../logicVQA/datasets/VQA_v2/process_test_promptcap.json'))
        #words_candidates = json.load(open('word_candidates.json'))
        #keys = sample(list(data.keys()), 2000)
        #data = sample(data, 2000)

        generated_questions = []
        results = []
        ids = []
        if os.path.exists('experiments/{}_result.json'.format(filename)):
            results = json.load(open('experiments/{}_result.json'.format(filename)))
            ids = [item['question_id'] for item in results]
        if os.path.exists('experiments/{}_generated.pkl'.format(filename)):
            generated_questions = pickle.load(open('experiments/{}_generated.pkl'.format(filename), 'rb'))

        for i in range(curr_i, len(data)):
        #for i in range(curr_i, len(keys)):
            #q_id = keys[i]
            #item = data[q_id]
            item = data[i]
            q_id = item['question_id']
            if q_id in ids:
                continue
            ids.append(q_id)
            img_id = item['image_id']
            #img_id = item['imageId']
            caption = item['caption']
            question = item['question']
            #gt_answer = item['direct_answers']
            #choices = item['choices']
            gt_answer = item['answers']
            gt_answer = [item['answer'] for item in gt_answer]
            #gt_answer = item['answer']
            #words = words_candidates[str(q_id)]
            #words = list(dict.fromkeys(words))[:5]
            pred, generated_q = vqa_llm.vqa(q_id, img_id, caption, question, model='gpt-3.5-turbo')
            print('gt answer:', gt_answer)
            results.append({'question_id': q_id, "answer": pred})
            generated_questions.append({'question_id': q_id, "generate_q": generated_q})

            if i % 5 == 0:
                with open('experiments/{}_result.json'.format(filename), 'w') as f:
                    json.dump(results, f)
                with open('experiments/{}_generated.pkl'.format(filename), 'wb') as f:
                    pickle.dump(generated_questions, f)

        with open('experiments/{}_result.json'.format(filename), 'w') as f:
            json.dump(results, f)
        with open('experiments/{}_generated.pkl'.format(filename), 'wb') as f:
            pickle.dump(generated_questions, f)

    except Exception as e:
        print(f"Error occurred: {e}. Interrupted at epoch {i}. Progress saved.")
        with open("epoch.txt", "w") as f:
            f.write(str(i))
        with open('experiments/{}_result.json'.format(filename), 'w') as f:
            json.dump(results, f)
        with open('experiments/{}_generated.pkl'.format(filename), 'wb') as f:
            pickle.dump(generated_questions, f)

        process = subprocess.Popen(['python', "experiment.py"])
        process.terminate()
        atexit.register(process.terminate)