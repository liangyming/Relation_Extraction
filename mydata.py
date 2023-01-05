from utils import *
import config
import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer


class semeval_dataset(data.Dataset):
    def __init__(self, df, tokenizer, e1_id, e2_id):
        self.e1_id = e1_id
        self.e2_id = e2_id
        self.df = df
        logger.info("Tokenizing data...")
        self.df['input'] = self.df.apply(lambda x: tokenizer.encode(x['sents']), axis=1)

        self.df['e1_e2_start'] = self.df.apply(
            lambda x: get_e1e2_start(x['input'], e1_id=self.e1_id, e2_id=self.e2_id), axis=1)
        print("\nInvalid rows/total: %d/%d" % (df['e1_e2_start'].isnull().sum(), len(df)))
        self.df.dropna(axis=0, inplace=True)

    def __len__(self, ):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.LongTensor(self.df.iloc[idx]['input']), \
               torch.LongTensor(self.df.iloc[idx]['e1_e2_start']), \
               torch.LongTensor([self.df.iloc[idx]['relations_id']])


class Pad_Sequence():
    """
    collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
    Returns padded x sequence, y sequence, x lengths and y lengths of batch
    """

    def __init__(self, seq_pad_value, label_pad_value=-1, label2_pad_value=-1):
        self.seq_pad_value = seq_pad_value
        self.label_pad_value = label_pad_value
        self.label2_pad_value = label2_pad_value

    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        seqs = [x[0] for x in sorted_batch]
        seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=self.seq_pad_value)
        x_lengths = torch.LongTensor([len(x) for x in seqs])

        labels = list(map(lambda x: x[1], sorted_batch))
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.label_pad_value)
        y_lengths = torch.LongTensor([len(x) for x in labels])

        labels2 = list(map(lambda x: x[2], sorted_batch))
        labels2_padded = pad_sequence(labels2, batch_first=True, padding_value=self.label2_pad_value)
        y2_lengths = torch.LongTensor([len(x) for x in labels2])

        return seqs_padded.to(config.device), labels_padded.to(config.device), \
               labels2_padded.to(config.device), x_lengths.to(config.device), \
               y_lengths.to(config.device), y2_lengths.to(config.device)


def get_e1e2_start(x, e1_id, e2_id):
    try:
        e1_e2_start = ([i for i, e in enumerate(x) if e == e1_id][0],
                       [i for i, e in enumerate(x) if e == e2_id][0])
    except Exception as e:
        e1_e2_start = None
        print(e)
    return e1_e2_start


def load_dataloaders(args):
    model = args.model_size  # 'bert-large-uncased' 'bert-base-uncased'
    lower_case = True
    model_name = 'BERT'
    # 读取tokenizer
    if os.path.isfile("./data/%s_tokenizer.pkl" % model_name):
        tokenizer = load_pickle("%s_tokenizer.pkl" % model_name)
        logger.info("Loaded tokenizer from pre-trained blanks model")
    else:
        logger.info("Pre-trained blanks tokenizer not found, initializing new tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(model)
        tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]', '[BLANK]'])

        save_as_pickle("%s_tokenizer.pkl" % model_name, tokenizer)
        logger.info("Saved %s tokenizer at ./data/%s_tokenizer.pkl" % (model_name, model_name))

    e1_id = tokenizer.convert_tokens_to_ids('[e1]')
    e2_id = tokenizer.convert_tokens_to_ids('[e2]')
    # assert e1_id != e2_id != 1

    # 读取数据和预处理
    relations_path = './data/relations.pkl'
    train_path = './data/df_train.pkl'
    test_path = './data/df_test.pkl'
    if os.path.isfile(relations_path) and os.path.isfile(train_path) and os.path.isfile(test_path):
        rm = load_pickle('relations.pkl')
        df_train = load_pickle('df_train.pkl')
        df_test = load_pickle('df_test.pkl')
        logger.info("Loaded preproccessed data.")
    else:
        df_train, df_test, rm = preprocess_semeval2010_8(args)

    train_set = semeval_dataset(df_train, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id)
    test_set = semeval_dataset(df_test, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id)
    train_length = len(train_set);
    test_length = len(test_set)
    PS = Pad_Sequence(seq_pad_value=tokenizer.pad_token_id,
                      label_pad_value=tokenizer.pad_token_id,
                      label2_pad_value=-1)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=PS, pin_memory=False)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True,
                             num_workers=0, collate_fn=PS, pin_memory=False)

    return train_loader, test_loader, train_length, test_length

