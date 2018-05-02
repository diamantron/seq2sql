import torch.nn as nn
from utils import run_lstm, col_name_encode

class SelectClassifier(nn.Module):
    def __init__(self, num_words, num_hidden, num_layers, max_tok_num):
        super(SelectClassifier, self).__init__()
        print "Seq2SQL Select Classifier"

        self.max_tok_num = max_tok_num
        self.sel_lstm = nn.LSTM(input_size=num_words, hidden_size=num_hidden/2,
                                num_layers=num_layers, batch_first=True,
                                dropout=0.3, bidirectional=True)
        self.sel_att = nn.Linear(num_hidden, 1)

        self.sel_col_name_enc = nn.LSTM(input_size=num_words, hidden_size=num_hidden/2,
                                        num_layers=num_layers, batch_first=True,
                                        dropout=0.3, bidirectional=True)

        self.sel_out_K = nn.Linear(num_hidden, num_hidden)
        self.sel_out_col = nn.Linear(num_hidden, num_hidden)
        self.sel_out = nn.Sequential(nn.Tanh(), nn.Linear(num_hidden, 1))


    def forward(self, x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num):
        e_col, _ = col_name_encode(col_inp_var, col_name_len, col_len, self.sel_col_name_enc)

        h_enc, _  = run_lstm(self.sel_lstm, x_emb_var, x_len)
        alpha_inp = self.sel_att(h_enc).squeeze()

        for idx, num in enumerate(x_len):
            if num < max(x_len):
                alpha_inp[idx, num:] = -100

        beta_inp = nn.Softmax()(alpha_inp)

        K_sel = (h_enc * beta_inp.unsqueeze(2).expand_as(h_enc)).sum(1)
        K_sel_expand = K_sel.unsqueeze(1)

        alpha_sel = self.sel_out( self.sel_out_K(K_sel_expand) + self.sel_out_col(e_col) ).squeeze()
        max_col_num = max(col_num)
        for idx, num in enumerate(col_num):
            if num < max_col_num:
                alpha_sel[idx, num:] = -100

        return alpha_sel
