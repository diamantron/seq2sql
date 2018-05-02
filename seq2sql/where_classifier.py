import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils import run_lstm

class WhereClassifier(nn.Module):
    def __init__(self, num_words, hidden_size, num_layers, max_col_num, max_tok_num, use_gpu=True):
        super(WhereClassifier, self).__init__()
        print "Seq2SQL Where Classifier"

        self.hidden_size = hidden_size
        self.max_tok_num = max_tok_num
        self.max_col_num = max_col_num
        self.use_gpu     = use_gpu

        self.where_encoder = nn.LSTM(input_size=num_words, hidden_size=hidden_size/2,
                                    num_layers=num_layers, batch_first=True,
                                    dropout=0.3, bidirectional=True)
        self.where_decoder = nn.LSTM(input_size=self.max_tok_num, hidden_size=hidden_size,
                                    num_layers=num_layers, batch_first=True,
                                    dropout=0.3, bidirectional=False)

        self.V_ptr      = nn.Linear(hidden_size, hidden_size)
        self.U_ptr      = nn.Linear(hidden_size, hidden_size)
        self.alpha_ptr_ = nn.Sequential(nn.Tanh(), nn.Linear(hidden_size, 1))

    def gen_gt_batch(self, tok_seq):
        batch = len(tok_seq)
        ret_len = np.array([len(one_tok_seq)-1 for one_tok_seq in tok_seq])
        max_len = max(ret_len)
        ret_array = np.zeros((batch, max_len, self.max_tok_num), dtype=np.float32)
        for b, one_tok_seq in enumerate(tok_seq):
            out_one_tok_seq = one_tok_seq[:-1]
            for t, tok_id in enumerate(out_one_tok_seq):
                ret_array[b, t, tok_id] = 1

        ret_inp = torch.from_numpy(ret_array)
        if self.use_gpu:
            ret_inp = ret_inp.cuda()
        ret_inp_var = Variable(ret_inp) #[batch, max_len, max_tok_num]

        return ret_inp_var, ret_len


    def forward(self, x_embed, x_len, g_s, reinforce):
        max_x_len = max(x_len)
        batch     = len(x_len)

        h_t, enc_hidden = run_lstm(self.where_encoder, x_embed, x_len)
        h_t_expand = h_t.unsqueeze(1)
        dec_hidden = tuple(torch.cat((eh[:2], eh[2:]),dim=2) for eh in enc_hidden)
        if g_s is not None:
            gt_tok_seq, gt_tok_len = self.gen_gt_batch(g_s)
            g_s, _ = run_lstm(self.where_decoder, gt_tok_seq, gt_tok_len, dec_hidden)
            g_s_expand = g_s.unsqueeze(2)
            alpha_ptr  = self.alpha_ptr_(self.V_ptr(h_t_expand) + self.U_ptr(g_s_expand)).squeeze()
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    alpha_ptr[idx, :, num:] = -100
        else:
            #import ipdb; ipdb.set_trace()
            t = 0
            init_inp = np.zeros((batch, 1, self.max_tok_num), dtype=np.float32)
            init_inp[:,0,7] = 1 # Set the <BEG> token
            if self.use_gpu:
                cur_inp = Variable(torch.from_numpy(init_inp).cuda())
            else:
                cur_inp = Variable(torch.from_numpy(init_inp))
            cur_h = dec_hidden

            done_set, scores, choices = set(), [], []
            while len(done_set) < 4*batch and t < 100:
                g_s, cur_h = self.where_decoder(cur_inp, cur_h)
                g_s_expand = g_s.unsqueeze(2)

                alpha_ptr = self.alpha_ptr_(self.V_ptr(h_t_expand) + self.U_ptr(g_s_expand)).squeeze()
                for b, num in enumerate(x_len):
                    if num < max_x_len:
                        alpha_ptr[b, num:] = -100
                scores.append(alpha_ptr)

                if not reinforce:
                    _, ans_tok_var = alpha_ptr.view(batch, max_x_len).max(1)
                    ans_tok_var = ans_tok_var.unsqueeze(1)
                else:
                    ans_tok_var = nn.Softmax()(alpha_ptr).multinomial()
                    choices.append(ans_tok_var)
                ans_tok = ans_tok_var.data.cpu()
                if self.use_gpu:
                    cur_inp = Variable(torch.zeros(
                        batch, self.max_tok_num).scatter_(1, ans_tok, 1).cuda())
                else:
                    cur_inp = Variable(torch.zeros(
                        batch, self.max_tok_num).scatter_(1, ans_tok, 1))
                cur_inp = cur_inp.unsqueeze(1)

                for idx, tok in enumerate(ans_tok.squeeze()):
                    if tok == 1:  # Find the <END> token
                        done_set.add(idx)
                t += 1

            alpha_ptr = torch.stack(scores, 1)

        if reinforce:
            return alpha_ptr, choices
        else:
            return alpha_ptr
