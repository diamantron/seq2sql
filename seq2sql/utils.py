import json
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from dbengine import DBEngine

def load_data(sql_paths, table_paths, use_small=False):
    if not isinstance(sql_paths, list):
        sql_paths = (sql_paths, )
    if not isinstance(table_paths, list):
        table_paths = (table_paths, )
    sql_data = []
    table_data = {}

    max_col_num = 0
    for SQL_PATH in sql_paths:
        print "Loading data from %s"%SQL_PATH
        with open(SQL_PATH) as inf:
            for idx, line in enumerate(inf):
                if use_small and idx >= 1000:
                    break
                sql = json.loads(line.strip())
                sql_data.append(sql)

    for TABLE_PATH in table_paths:
        print "Loading data from %s"%TABLE_PATH
        with open(TABLE_PATH) as inf:
            for line in inf:
                tab = json.loads(line.strip())
                table_data[tab[u'id']] = tab

    for sql in sql_data:
        assert sql[u'table_id'] in table_data

    return sql_data, table_data

def load_dataset(dataset_id, use_small=False):
    if dataset_id == 0:
        print "Loading from original dataset"
        sql_data, table_data = load_data('data/train_tok.jsonl', 'data/train_tok.tables.jsonl', use_small=use_small)
        val_sql_data, val_table_data = load_data('data/dev_tok.jsonl', 'data/dev_tok.tables.jsonl', use_small=use_small)
        test_sql_data, test_table_data = load_data('data/test_tok.jsonl', 'data/test_tok.tables.jsonl', use_small=use_small)
        TRAIN_DB = 'data/train.db'
        DEV_DB = 'data/dev.db'
        TEST_DB = 'data/test.db'
    else:
        print "Loading from re-split dataset"
        sql_data, table_data = load_data('data_resplit/train.jsonl', 'data_resplit/tables.jsonl', use_small=use_small)
        val_sql_data, val_table_data = load_data('data_resplit/dev.jsonl', 'data_resplit/tables.jsonl', use_small=use_small)
        test_sql_data, test_table_data = load_data('data_resplit/test.jsonl', 'data_resplit/tables.jsonl', use_small=use_small)
        TRAIN_DB = 'data_resplit/table.db'
        DEV_DB = 'data_resplit/table.db'
        TEST_DB = 'data_resplit/table.db'

    return sql_data, table_data, val_sql_data, val_table_data,\
            test_sql_data, test_table_data, TRAIN_DB, DEV_DB, TEST_DB

def best_model_name(args, for_load=False):
    new_data = 'new' if args.dataset > 0 else 'old'
    if for_load:
        use_rl = ''
    else:
        use_rl = 'rl_' if args.rl else ''

    agg_model_name = 'saved_model/%s.agg_model'%(new_data)
    sel_model_name = 'saved_model/%s.sel_model'%(new_data)
    whr_model_name = 'saved_model/%s.whr_%smodel'%(new_data, use_rl)

    return agg_model_name, sel_model_name, whr_model_name


def to_batch_seq(sql_data, table_data, idxes, st, ed, ret_vis_data=False):
    q_seq = []
    col_seq = []
    col_num = []
    ans_seq = []
    query_seq = []
    gt_whr_seq = []
    vis_seq = []
    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        q_seq.append(sql['question_tok'])
        col_seq.append(table_data[sql['table_id']]['header_tok'])
        col_num.append(len(table_data[sql['table_id']]['header']))
        ans_seq.append((sql['sql']['agg'],
            sql['sql']['sel'], 
            len(sql['sql']['conds']),
            tuple(x[0] for x in sql['sql']['conds']),
            tuple(x[1] for x in sql['sql']['conds'])))
        query_seq.append(sql['query_tok'])
        gt_whr_seq.append(sql['sql']['conds'])
        vis_seq.append((sql['question'],
            table_data[sql['table_id']]['header'], sql['query']))
    if ret_vis_data:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_whr_seq, vis_seq
    else:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_whr_seq

def to_batch_query(sql_data, idxes, st, ed):
    query_gt = []
    table_ids = []
    for i in range(st, ed):
        query_gt.append(sql_data[idxes[i]]['sql'])
        table_ids.append(sql_data[idxes[i]]['table_id'])
    return query_gt, table_ids

def epoch_train(model, optimizer, batch_size, sql_data, table_data, clasif_entry):
    model.train()
    perm=np.random.permutation(len(sql_data))
    cum_loss = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, _ = \
                to_batch_seq(sql_data, table_data, perm, st, ed)
        g_s = model.generate_g_s(q_seq, col_seq, query_seq)
        score = model.forward(q_seq, col_seq, col_num, clasif_entry, g_s=g_s)
        loss = model.loss(score, ans_seq, clasif_entry, g_s)
        cum_loss += loss.data.cpu().numpy()[0]*(ed - st)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        st = ed

    return cum_loss / len(sql_data)

def epoch_exec_acc(model, batch_size, sql_data, table_data, db_path):
    engine = DBEngine(db_path)
    #XXX print ("=== call to epoch_exec_acc")

    model.eval()
    perm = list(range(len(sql_data)))
    tot_acc_num = 0.0
    acc_of_log = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        q_seq, col_seq, col_num, ans_seq, query_seq, gt_whr_seq, raw_data = \
            to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        #XXX print ("=== call to generate_g_s")
        g_s = model.generate_g_s(q_seq, col_seq, query_seq)
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        #XXX print ("=== call to forward")
        score = model.forward(q_seq, col_seq, col_num, (True, True, True))
        clasif_queries = model.gen_query(score, q_seq, col_seq, raw_q_seq, raw_col_seq, (True, True, True))

        for idx, (sql_gt, sql_pred, tid) in enumerate(
                zip(query_gt, clasif_queries, table_ids)):
            ret_gt = engine.execute(tid,
                    sql_gt['sel'], sql_gt['agg'], sql_gt['conds'])
            try:
                ret_pred = engine.execute(tid,
                        sql_pred['sel'], sql_pred['agg'], sql_pred['conds'])
            except:
                ret_pred = None
            tot_acc_num += (ret_gt == ret_pred)
        
        st = ed

    return tot_acc_num / len(sql_data)

def epoch_acc(model, batch_size, sql_data, table_data, clasif_entry):
    model.eval()
    #XXX print ("=== call to epoch_acc")
    perm = list(range(len(sql_data)))
    st = 0
    one_acc_num = 0.0
    tot_acc_num = 0.0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_whr_seq, raw_data = to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        #XXX print ("=== call to fwd")
        score = model.forward(q_seq, col_seq, col_num, clasif_entry)
        clasif_queries = model.gen_query(score, q_seq, col_seq,
                raw_q_seq, raw_col_seq, clasif_entry)
        one_err, tot_err = model.check_acc(clasif_queries, query_gt, clasif_entry)

        one_acc_num += (ed-st-one_err)
        tot_acc_num += (ed-st-tot_err)

        st = ed
    return tot_acc_num / len(sql_data), one_acc_num / len(sql_data)

def epoch_reinforce_train(model, optimizer, batch_size, sql_data, table_data, db_path):
    engine = DBEngine(db_path)

    #XXX print ("=== call to train")
    model.train()
    perm = np.random.permutation(len(sql_data))
    cum_reward = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_whr_seq, raw_data =\
            to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        #XXX print ("=== generate g_s")
        g_s = model.generate_g_s(q_seq, col_seq, query_seq)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        #XXX print ("=== fwd")
        score = model.forward(q_seq, col_seq, col_num, (True, True, True), reinforce=True)
        clasif_queries = model.gen_query(score, q_seq, col_seq, raw_q_seq, raw_col_seq, (True, True, True), reinforce=True)

        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        rewards = []
        for idx, (sql_gt, sql_pred, tid) in enumerate(
                zip(query_gt, clasif_queries, table_ids)):
            ret_gt = engine.execute(tid,
                    sql_gt['sel'], sql_gt['agg'], sql_gt['conds'])
            try:
                ret_pred = engine.execute(tid,
                        sql_pred['sel'], sql_pred['agg'], sql_pred['conds'])
            except:
                ret_pred = None

            if ret_pred is None:
                rewards.append(-2)
            elif ret_pred != ret_gt:
                rewards.append(-1)
            else:
                rewards.append(1)

        cum_reward += (sum(rewards))
        optimizer.zero_grad()
        model.reinforce_backward(score, rewards)
        optimizer.step()

        st = ed

    return cum_reward / len(sql_data)


def load_word_emb(file_name, use_small=False):
    print ('Loading embedding')
    ret = {}
    with open(file_name) as inf:
        for idx, line in enumerate(inf):
            if (use_small and idx >= 5000):
                break
            info = line.strip().split(' ')
            if info[0].lower() not in ret:
                ret[info[0]] = np.array(map(lambda x:float(x), info[1:]))
    return ret



def run_lstm(lstm, inp, inp_len, hidden=None):
    # Run the LSTM using packed sequence.
    # This requires to first sort the input according to its length.
    sort_perm = np.array(sorted(range(len(inp_len)),
        key=lambda k:inp_len[k], reverse=True))
    sort_inp_len = inp_len[sort_perm]
    sort_perm_inv = np.argsort(sort_perm)
    if inp.is_cuda:
        sort_perm = torch.LongTensor(sort_perm).cuda()
        sort_perm_inv = torch.LongTensor(sort_perm_inv).cuda()

    lstm_inp = nn.utils.rnn.pack_padded_sequence(inp[sort_perm],
            sort_inp_len, batch_first=True)
    if hidden is None:
        lstm_hidden = None
    else:
        lstm_hidden = (hidden[0][:, sort_perm], hidden[1][:, sort_perm])

    sort_ret_s, sort_ret_h = lstm(lstm_inp, lstm_hidden)
    ret_s = nn.utils.rnn.pad_packed_sequence(
            sort_ret_s, batch_first=True)[0][sort_perm_inv]
    ret_h = (sort_ret_h[0][:, sort_perm_inv], sort_ret_h[1][:, sort_perm_inv])
    return ret_s, ret_h


def col_name_encode(name_inp_var, name_len, col_len, enc_lstm):
    #Encode the columns.
    #The embedding of a column name is the last state of its LSTM output.
    name_hidden, _ = run_lstm(enc_lstm, name_inp_var, name_len)
    name_out = name_hidden[tuple(range(len(name_len))), name_len-1]
    ret = torch.FloatTensor(
            len(col_len), max(col_len), name_out.size()[1]).zero_()
    if name_out.is_cuda:
        ret = ret.cuda()

    st = 0
    for idx, cur_len in enumerate(col_len):
        ret[idx, :cur_len] = name_out.data[st:st+cur_len]
        st += cur_len
    ret_var = Variable(ret)

    return ret_var, col_len

