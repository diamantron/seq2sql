import numpy as np
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
from word_embedding import WordEmbedding
from aggregation_classifier import AggregationClassifier
from select_classifier import SelectClassifier
from where_classifier import WhereClassifier

class Seq2SQL(nn.Module):
    def __init__(self, word_emb, num_words, num_hidden=100, num_layers=2, use_gpu=True):
        super(Seq2SQL, self).__init__()

        self.word_emb   = word_emb
        self.num_words  = num_words
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.use_gpu    = use_gpu

        self.max_col_num = 45
        self.max_tok_num = 200
        self.COND_OPS    = ['EQL', 'GT', 'LT']
        self.SQL_TOK     = ['<UNK>', '<BEG>', '<END>', 'WHERE', 'AND'] + self.COND_OPS

        # GloVe Word Embedding
        self.embed_layer    = WordEmbedding(word_emb, num_words, self.SQL_TOK, use_gpu)

        # Aggregation Classifier
        self.agg_classifier = AggregationClassifier(num_words, num_hidden, num_layers)

        # SELECT Column(s)
        self.sel_classifier = SelectClassifier(num_words, num_hidden, num_layers, self.max_tok_num)

        # WHERE Clause
        self.whr_classifier = WhereClassifier(num_words, num_hidden, num_layers, self.max_col_num, self.max_tok_num, use_gpu)

        # run on GPU
        if use_gpu:
            self.cuda()


    def generate_g_s(self, q, col, query):
        # data format
        # <BEG> WHERE cond1_col cond1_op cond1
        #         AND cond2_col cond2_op cond2
        #         AND ... <END>

        ret_seq = []
        for cur_q, cur_col, cur_query in zip(q, col, query):
            connect_col = [tok for col_tok in cur_col for tok in col_tok+[',']]
            all_toks = self.SQL_TOK + connect_col + [None] + cur_q + [None]
            cur_seq = [all_toks.index('<BEG>')]
            if 'WHERE' in cur_query:
                cur_where_query = cur_query[cur_query.index('WHERE'):]
                cur_seq = cur_seq + map(lambda tok:all_toks.index(tok)
                                        if tok in all_toks else 0, cur_where_query)
            cur_seq.append(all_toks.index('<END>'))
            ret_seq.append(cur_seq)
        return ret_seq


    def forward(self, q, col, col_num, classif_flag, g_s = None, reinforce=False):

        agg_classif, sel_classif, whr_classif = classif_flag
        agg_score, sel_score, whr_score = None, None, None

        x_emb_var, x_len = self.embed_layer.gen_x_batch(q, col)

        if agg_classif:
            agg_score = self.agg_classifier(x_emb_var, x_len)

        if sel_classif:
            col_inp_var, col_name_len, col_len = self.embed_layer.gen_col_batch(col)
            sel_score = self.sel_classifier(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num)

        if whr_classif:
            whr_score = self.whr_classifier(x_emb_var, x_len, g_s, reinforce=reinforce)

        return (agg_score, sel_score, whr_score)

    def loss(self, score, ref_score, classif_flag, g_s):
        agg_classif, sel_classif, whr_classif = classif_flag
        agg_score, sel_score, whr_score = score
        loss = 0
        if agg_classif:
            agg_ref = torch.from_numpy(np.array(map(lambda x:x[0], ref_score)))
            agg_ref_var = Variable(agg_ref)
            if self.use_gpu:
                agg_ref_var = agg_ref_var.cuda()
            loss += nn.CrossEntropyLoss()(agg_score, agg_ref_var)

        if sel_classif:
            sel_ref = torch.from_numpy(np.array(map(lambda x:x[1], ref_score)))
            sel_ref_var = Variable(sel_ref)
            if self.use_gpu:
                sel_ref_var = sel_ref_var.cuda()
            loss += nn.CrossEntropyLoss()(sel_score, sel_ref_var)

        if whr_classif:
            g_s_len =  len(g_s)
            for s, g_s_i in enumerate(g_s):
                whr_ref_var = Variable(torch.from_numpy(np.array(g_s_i[1:])))
                if self.use_gpu:
                    whr_ref_var = whr_ref_var.cuda()
                loss += (nn.CrossEntropyLoss()(whr_score[s, :len(g_s_i)-1], whr_ref_var) /g_s_len)

        return loss

    def reinforce_backward(self, score, rewards):
        agg_score, sel_score, whr_score = score

        cur_reward = rewards[:]
        eof = self.SQL_TOK.index('<END>')
        for whr_score_t in whr_score[1]:
            reward_inp = torch.FloatTensor(cur_reward).unsqueeze(1)
            if self.use_gpu:
                reward_inp = reward_inp.cuda()
            whr_score_t.reinforce(reward_inp)

            for b,_ in enumerate(rewards):
                if whr_score_t[b].data.cpu().numpy()[0] == eof:
                    cur_reward[b] = 0
        torch.autograd.backward(whr_score[1], [None for _ in whr_score[1]])
        return

    def check_acc(self, classif_queries, g_s_queries, classif_flag):

        agg_classif, sel_classif, whr_classif = classif_flag
        tot_err = agg_err = sel_err = whr_err = whr_num_err = whr_col_err = whr_op_err = whr_val_err = 0.0
        for classif_qry, g_s_qry in zip(classif_queries, g_s_queries):

            agg_err_inc = 1 if agg_classif and classif_qry['agg'] != g_s_qry['agg'] else 0
            agg_err += agg_err_inc

            sel_err_inc = 1 if sel_classif and classif_qry['sel'] != g_s_qry['sel'] else 0
            sel_err += sel_err_inc

            if whr_classif:
                flag = True
                whr_classifier = classif_qry['conds']
                whr_g_s = g_s_qry['conds']
                if len(whr_classifier) != len(whr_g_s):
                    flag = False
                    whr_num_err += 1
                elif set(x[0] for x in whr_classifier) != set(x[0] for x in whr_g_s) :
                    flag = False
                    whr_col_err += 1
                if flag:
                    for whr_class_i in whr_classifier:
                        g_s_idx = tuple(x[0] for x in whr_g_s).index(whr_class_i[0])
                        if flag and whr_g_s[g_s_idx][1] != whr_class_i[1]:
                            flag = False
                            whr_op_err += 1
                            break
                if flag:
                    for whr_class_i in whr_classifier:
                        g_s_idx = tuple(x[0] for x in whr_g_s).index(whr_class_i[0])
                        if flag and unicode(whr_g_s[g_s_idx][2]).lower() != \
                        unicode(whr_class_i[2]).lower():
                            flag = False
                            whr_val_err += 1
                            break

                if not flag:
                    whr_err += 1

            if agg_err_inc>0 or sel_err_inc>0 or not flag:
                tot_err += 1

        return np.array((agg_err, sel_err, whr_err)), tot_err


    def gen_query(self, score, q, col, raw_q, raw_col, classif_flag, reinforce=False, verbose=False):
        def merge_tokens(tok_list, raw_tok_str):
            tok_str = raw_tok_str.lower()
            special = {'-LRB-':'(', '-RRB-':')', '-LSB-':'[', '-RSB-':']', '``':'"', '\'\'':'"', '--':u'\u2013'}
            ret = ''
            double_quote_pair_track = 0
            for raw_tok in tok_list:
                if not raw_tok:
                    continue
                tok = special.get(raw_tok, raw_tok)
                if tok == '"':
                    double_quote_pair_track = 1 - double_quote_pair_track
                    if double_quote_pair_track:
                        ret = ret + ' '
                if len(ret) == 0:
                    pass
                elif len(ret) > 0 and ret + ' ' + tok in tok_str:
                    ret = ret + ' '
                elif len(ret) > 0 and ret + tok in tok_str:
                    pass
                elif (tok[0] not in string.ascii_lowercase) and (tok[0] not in string.digits) and (tok[0] not in '$('):
                    pass
                elif (ret[-1] not in ['(', '/', u'\u2013', '#', '$', '&']) and \
                     (ret[-1] != '"' or not double_quote_pair_track):
                    ret = ret + ' '
                ret = ret + tok
            return ret.strip()

        agg_classif, sel_classif, whr_classif = classif_flag
        agg_score, sel_score, whr_score = score

        ret_queries = []
        batch_len = len(agg_score) if agg_classif else len(sel_score) if sel_classif else len(whr_score[0]) if reinforce else len(whr_score)
        for b in range(batch_len):
            cur_query = {}
            if agg_classif:
                cur_query['agg'] = np.argmax(agg_score[b].data.cpu().numpy())
            if sel_classif:
                cur_query['sel'] = np.argmax(sel_score[b].data.cpu().numpy())
            if whr_classif:
                cur_query['conds'] = []
                all_toks = self.SQL_TOK + [x for toks in col[b] for x in toks+[',']] + [''] + q[b] + ['']
                whr_toks = []
                if reinforce:
                    for choices in whr_score[1]:
                        if choices[b].data.cpu().numpy()[0] < len(all_toks):
                            whr_val = all_toks[choices[b].data.cpu().numpy()[0]]
                        else:
                            whr_val = '<UNK>'
                        if whr_val == '<END>':
                            break
                        whr_toks.append(whr_val)
                else:
                    for where_score in whr_score[b].data.cpu().numpy():
                        whr_tok = np.argmax(where_score)
                        whr_val = all_toks[whr_tok]
                        if whr_val == '<END>':
                            break
                        whr_toks.append(whr_val)

                if verbose:
                    print whr_toks
                if len(whr_toks) > 0:
                    whr_toks = whr_toks[1:]
                st = 0
                while st < len(whr_toks):
                    cur_cond = [None, None, None]
                    ed = len(whr_toks) if 'AND' not in whr_toks[st:] \
                         else whr_toks[st:].index('AND') + st
                    if 'EQL' in whr_toks[st:ed]:
                        op = whr_toks[st:ed].index('EQL') + st
                        cur_cond[1] = 0
                    elif 'GT' in whr_toks[st:ed]:
                        op = whr_toks[st:ed].index('GT') + st
                        cur_cond[1] = 1
                    elif 'LT' in whr_toks[st:ed]:
                        op = whr_toks[st:ed].index('LT') + st
                        cur_cond[1] = 2
                    else:
                        op = st
                        cur_cond[1] = 0
                    sel_col = whr_toks[st:op]
                    to_idx = [x.lower() for x in raw_col[b]]
                    classif_col = merge_tokens(sel_col, raw_q[b] + ' || ' + \
                                            ' || '.join(raw_col[b]))
                    if classif_col in to_idx:
                        cur_cond[0] = to_idx.index(classif_col)
                    else:
                        cur_cond[0] = 0
                    cur_cond[2] = merge_tokens(whr_toks[op+1:ed], raw_q[b])
                    cur_query['conds'].append(cur_cond)
                    st = ed + 1
            ret_queries.append(cur_query)

        return ret_queries
