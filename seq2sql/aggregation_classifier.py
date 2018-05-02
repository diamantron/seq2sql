import torch.nn as nn
from utils import run_lstm

class AggregationClassifier(nn.Module):
    def __init__(self, num_words, num_hidden, num_layers):
        super(AggregationClassifier, self).__init__()
        print "Seq2SQL Aggregation Classifier"

        self.num_agg_ops = 6
        self.agg_lstm = nn.LSTM(input_size=num_words, hidden_size=num_hidden/2,
                                num_layers=num_layers, batch_first=True,
                                dropout=0.3, bidirectional=True)

        self.agg_att = nn.Linear(num_hidden, 1)
        self.agg_out = nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.Tanh(), nn.Linear(num_hidden, self.num_agg_ops))

    def forward(self, x_embed, x_len, col_inp_var=None, col_name_len=None, col_len=None, col_num=None, gt_sel=None):
        h_enc, _  = run_lstm(self.agg_lstm, x_embed, x_len)
        alpha_inp = self.agg_att(h_enc).squeeze()

        for idx, num in enumerate(x_len):
            if num < max(x_len):
                alpha_inp[idx, num:] = -100 # TODO
        beta_inp  = nn.Softmax()(alpha_inp)

        K_agg     = (h_enc * beta_inp.unsqueeze(2).expand_as(h_enc)).sum(1) # TODO, organize
        alpha_agg = self.agg_out(K_agg)
        return alpha_agg

