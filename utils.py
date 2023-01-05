import os
import pickle
import re
import pandas as pd
from config import logger
from tqdm import tqdm


def load_pickle(filename):
    completeName = os.path.join("./data/", filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data


def save_as_pickle(filename, data):
    completeName = os.path.join("./data/", filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)


def clean_str(text):
    # text = text.lower()
    # Clean the text
    # text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=$#]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)  # ?
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)  # ?
    return text.strip()


def process_text(text):
    sents, relations = [], []
    for i in range(int(len(text) / 4)):
        sent = text[4 * i]
        relation = text[4 * i + 1]

        sent = re.findall("\"(.+)\"", sent)[0]
        sent = re.sub('<e1>', '[E1]', sent)
        sent = re.sub('</e1>', '[/E1]', sent)
        sent = re.sub('<e2>', '[E2]', sent)
        sent = re.sub('</e2>', '[/E2]', sent)
        sent = clean_str(sent)
        sents.append(sent)
        relations.append(relation)
    return sents, relations


class Relations_Mapper(object):
    def __init__(self, relations):
        self.rel2idx = {}
        self.idx2rel = {}

        logger.info("Mapping relations to IDs...")
        self.n_classes = 0
        for relation in tqdm(relations):
            if relation not in self.rel2idx.keys():
                self.rel2idx[relation] = self.n_classes
                self.n_classes += 1

        for key, value in self.rel2idx.items():
            self.idx2rel[value] = key


def preprocess_semeval2010_8(args):
    data_path = args.train_data  # './data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
    logger.info("Reading training file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        text = f.readlines()

    sents, relations = process_text(text)
    df_train = pd.DataFrame(data={'sents': sents, 'relations': relations})

    data_path = args.test_data  # './data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'
    logger.info("Reading test file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        text = f.readlines()

    sents, relations = process_text(text)
    df_test = pd.DataFrame(data={'sents': sents, 'relations': relations})
    # 关系映射为ID
    rm = Relations_Mapper(df_train['relations'])
    save_as_pickle('relations.pkl', rm)
    df_test['relations_id'] = df_test.apply(lambda x: rm.rel2idx[x['relations']], axis=1)
    df_train['relations_id'] = df_train.apply(lambda x: rm.rel2idx[x['relations']], axis=1)
    save_as_pickle('df_train.pkl', df_train)
    save_as_pickle('df_test.pkl', df_test)
    logger.info("Finished and saved!")
    return df_train, df_test, rm


