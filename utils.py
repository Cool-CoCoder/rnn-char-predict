import collections
import re
from d2l import torch as d2l

def read_file(file_path):  #@save
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def load_corpus_text(file_path,max_tokens=-1):  #@save
    """返回数据集的词元索引列表和词表"""
    lines = read_file(file_path)
    tokens = d2l.tokenize(lines, 'char')
    vocab = d2l.Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


class SeqDataLoader:  #@save
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens,file_path):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_text(file_path,max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)



def load_data_from_own_text(file_path,batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens,file_path)
    return data_iter, data_iter.vocab
# # read_file('./war_and_peace.txt')
#
# a,b = load_data_from_own_text('./war_and_peace.txt',32,35)
# print(a,b)

