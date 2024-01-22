ASK_PROMPT = 'Given a question, define its denotative questions and connotative questions. Denotative questions can be either the rephrasing ' \
             'or decomposition of the original question, which extract explicit information. Connotative questions seek related or implicit ' \
             'information that can help in understanding and answering the original question. Please generate denotative and connotative ' \
             'questions of the target question based on some keywords. The generated questions should be easy to answer by a base VQA model.'

#ASK_PROMPT='Generate 8 different questions given image caption and an original question.'

ANSWER_PROMPT = 'Given a caption of a image, a question about the image and some related question-answer pairs, and the confidence of the answers, ' \
                'integrate the information and answer the target question in less ' \
                'than four words. You can use common sense knowledge for answering.'

#ANSWER_PROMPT = 'Given a caption of a image, a question about the image and some related question-answer pairs, ' \
#                'and the confidence of the answers, integrate the information and answer the target question in less ' \
#                'than four words. You can use common sense knowledge for answering.'

#ANSWER_PROMPT = 'Given a caption of a image, a question about the image and some related question-answer pairs with answer' \
#                ' confidence, integrate the information and select the answer from given choices. When the answer is not determined,' \
#                ' choose the most possible answer from the given choices. Given the answer only, do not explain.'