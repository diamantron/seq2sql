import torch
from utils import *
from seq2sql import Seq2SQL
import datetime
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', action='store_true', help='small model, for fast development cycles.')
    parser.add_argument('--dataset', type=int, default=0, help='0: original dataset, 1: re-split dataset')
    parser.add_argument('--rl', action='store_true', help='Use RL for Seq2SQL')
    args = parser.parse_args()

    num_words, B_word, use_gpu = 300, 42, True
    batch_size = 8 if args.toy else 64
    TRAIN_ENTRY=(True, True, True)  # (AGG, SEL, COND)
    TRAIN_AGG, TRAIN_SEL, TRAIN_COND = TRAIN_ENTRY
    learning_rate = 1e-4 if args.rl else 1e-3

    sql_data, table_data, val_sql_data, val_table_data, _, _, TRAIN_DB, DEV_DB, _ = load_dataset(args.dataset, use_small=args.toy)

    word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,num_words),use_small=args.toy)

    model = Seq2SQL(word_emb=word_emb, num_words=num_words, use_gpu=use_gpu)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate, weight_decay = 0)

    agg_m, sel_m, whr_m = best_model_name(args)

    if args.rl: # Load pretrained model.
        agg_lm, sel_lm, whr_lm = best_model_name(args, for_load=True)
        print "Loading from %s"%agg_lm
        model.agg_classifier.load_state_dict(torch.load(agg_lm))
        print "Loading from %s"%sel_lm
        model.sel_classifier.load_state_dict(torch.load(sel_lm))
        print "Loading from %s"%whr_lm
        model.whr_classifier.load_state_dict(torch.load(whr_lm))

        best_acc = 0.0
        best_idx = -1
        print "Init dev acc_qm: %s\n  breakdown on (agg, sel, where): %s"% \
                epoch_acc(model, batch_size, val_sql_data,\
                val_table_data, TRAIN_ENTRY)
        print "Init dev acc_ex: %s"%epoch_exec_acc(
                model, batch_size, val_sql_data, val_table_data, DEV_DB)
        torch.save(model.whr_classifier.state_dict(), whr_m)
        for i in range(100):
            print 'Epoch %d @ %s'%(i+1, datetime.datetime.now())
            print ' Avg reward = %s'%epoch_reinforce_train(
                model, optimizer, batch_size, sql_data, table_data, TRAIN_DB)
            print ' dev acc_qm: %s\n   breakdown result: %s'% epoch_acc(
                model, batch_size, val_sql_data, val_table_data, TRAIN_ENTRY)
            exec_acc = epoch_exec_acc(
                    model, batch_size, val_sql_data, val_table_data, DEV_DB)
            print ' dev acc_ex: %s', exec_acc
            if exec_acc[0] > best_acc:
                best_acc = exec_acc[0]
                best_idx = i+1
                torch.save(model.whr_classifier.state_dict(),
                        'saved_model/epoch%d.whr_model'%(i+1))
                torch.save(model.whr_classifier.state_dict(), whr_m)
            print ' Best exec acc = %s, on epoch %s'%(best_acc, best_idx)
    else:
        init_acc = epoch_acc(model, batch_size,
                val_sql_data, val_table_data, TRAIN_ENTRY)
        best_agg_acc = init_acc[1][0]
        best_agg_idx = 0
        best_sel_acc = init_acc[1][1]
        best_sel_idx = 0
        best_whr_acc = init_acc[1][2]
        best_whr_idx = 0
        print 'Init dev acc_qm: %s\n  breakdown on (agg, sel, where): %s'%\
                init_acc
        if TRAIN_AGG:
            torch.save(model.agg_classifier.state_dict(), agg_m)
        if TRAIN_SEL:
            torch.save(model.sel_classifier.state_dict(), sel_m)
        if TRAIN_COND:
            torch.save(model.whr_classifier.state_dict(), whr_m)
        for i in range(100):
            print 'Epoch %d @ %s'%(i+1, datetime.datetime.now())
            print ' Loss = %s'%epoch_train(
                    model, optimizer, batch_size, 
                    sql_data, table_data, TRAIN_ENTRY)
            print ' Train acc_qm: %s\n   breakdown result: %s'%epoch_acc(
                    model, batch_size, sql_data, table_data, TRAIN_ENTRY)
            #val_acc = epoch_token_acc(model, batch_size, val_sql_data, val_table_data, TRAIN_ENTRY)
            val_acc = epoch_acc(model,
                    batch_size, val_sql_data, val_table_data, TRAIN_ENTRY)
            print ' Dev acc_qm: %s\n   breakdown result: %s'%val_acc
            if TRAIN_AGG:
                if val_acc[1][0] > best_agg_acc:
                    best_agg_acc = val_acc[1][0]
                    best_agg_idx = i+1
                    torch.save(model.agg_classifier.state_dict(),
                        'saved_model/epoch%d.agg_model'%(i+1))
                    torch.save(model.agg_classifier.state_dict(), agg_m)
            if TRAIN_SEL:
                if val_acc[1][1] > best_sel_acc:
                    best_sel_acc = val_acc[1][1]
                    best_sel_idx = i+1
                    torch.save(model.sel_classifier.state_dict(),
                        'saved_model/epoch%d.sel_model'%(i+1))
                    torch.save(model.sel_classifier.state_dict(), sel_m)
            if TRAIN_COND:
                if val_acc[1][2] > best_whr_acc:
                    best_whr_acc = val_acc[1][2]
                    best_whr_idx = i+1
                    torch.save(model.whr_classifier.state_dict(),
                        'saved_model/epoch%d.whr_model'%(i+1))
                    torch.save(model.whr_classifier.state_dict(), whr_m)
            print ' Best val acc = %s, on epoch %s individually'%(
                    (best_agg_acc, best_sel_acc, best_whr_acc),
                    (best_agg_idx, best_sel_idx, best_whr_idx))
