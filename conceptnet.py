import json
import spacy

class ConceptNet():
    def __init__(self, word2index):
        self.data = json.load(open('../logicVQA/conceptnet/conceptnet_dict.json'))
        self.nlp = spacy.load("en_core_web_sm")
        self.word2index = word2index

    def get_mask(self, question, caption):
        q_nlp = self.nlp(question)
        c_nlp = self.nlp(caption)
        candidates = set()
        for token in q_nlp:
            if token.pos_ == 'NOUN' or token.pos_ == 'VERB' or token.pos_ == 'ADJ':
                word = token.text
                if word in self.data:
                    candidates.update(set([item[0] for item in self.data[word]]))
        for token in c_nlp:
            if token.pos_ == 'NOUN' or token.pos_ == 'VERB' or token.pos_ == 'ADJ':
                word = token.text
                if word in self.data:
                    candidates.update(set([item[0] for item in self.data[word]]))

        candidates_index = [self.word2index[item] for item in candidates if item in self.word2index]
        return candidates_index

