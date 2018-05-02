import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable as Variable

class WordEmbedding(nn.Module):
    def __init__(self, word_emb, num_words, SQL_TOK, use_gpu=True):
        super(WordEmbedding, self).__init__()
        print "Seq2SQL Word Embedding"

        self.word_emb   = word_emb
        self.num_words  = num_words
        self.SQL_TOK    = SQL_TOK
        self.use_gpu    = use_gpu

    def gen_x_batch(self, q, col):
        batch = len(q)
        emb_vals = []
        val_len = np.zeros(batch, dtype=np.int64)
        for i, (q_i, col_i) in enumerate(zip(q, col)):
            q_val = map(lambda x:self.word_emb.get(x, np.zeros(self.num_words, dtype=np.float32)), q_i)
            col_i_all = [x for toks in col_i for x in toks+[',']]
            col_val = map(lambda x:self.word_emb.get(x, np.zeros(self.num_words, dtype=np.float32)), col_i_all)
            emb_vals.append( [np.zeros(self.num_words, dtype=np.float32) for _ in self.SQL_TOK] + col_val + [np.zeros(self.num_words, dtype=np.float32)] + q_val+ [np.zeros(self.num_words, dtype=np.float32)])
            val_len[i] = len(self.SQL_TOK) + len(col_val) + 1 + len(q_val) + 1
        max_len = max(val_len)

        val_emb_array = np.zeros((batch, max_len, self.num_words), dtype=np.float32)
        for b in range(batch):
            for e in range(len(emb_vals[b])):
                val_emb_array[b,e,:] = emb_vals[b][e]
        val_inp = torch.from_numpy(val_emb_array)
        if self.use_gpu:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)
        return val_inp_var, val_len

    def gen_col_batch(self, cols):
        ret = []
        col_len = np.zeros(len(cols), dtype=np.int64)

        names = []
        for b, col_i in enumerate(cols):
            names = names + col_i
            col_len[b] = len(col_i)

        batch = len(names)
        emb_vals = []
        name_len = np.zeros(batch, dtype=np.int64)
        for i, one_str in enumerate(names):
            val = [self.word_emb.get(x, np.zeros(
                   self.num_words, dtype=np.float32)) for x in one_str]
            emb_vals.append(val)
            name_len[i] = len(val)
        max_len = max(name_len)
        val_emb_array = np.zeros(
                (batch, max_len, self.num_words), dtype=np.float32)
        for i in range(batch):
            for t in range(len(emb_vals[i])):
                val_emb_array[i,t,:] = emb_vals[i][t]
        name_inp = torch.from_numpy(val_emb_array)
        if self.use_gpu:
            name_inp = name_inp.cuda()
        name_inp_var = Variable(name_inp)

        return name_inp_var, name_len, col_len

