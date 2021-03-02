import torch
import time
from utils.plot_util import loss_acc_plot
from utils.model_util import load_model,save_model
import config as args
from transformers.optimization import AdamW
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


def fit(model, training_iter, eval_iter,test_iter,num_epoch, pbar, num_train_steps, verbose=1):
    # ------------------判断CUDA模式----------------------
    device = torch.device(args.device if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # ---------------------优化器-------------------------

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.005)

    optimizer = AdamW(model.parameters(),lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.005)
    # ---------------------模型初始化----------------------
    model.to(device)

# ------------------------训练------------------------------
    dict_f1 =[]
    best_f1 = 0
    start = time.time()
    global_step = 0
    for e in range(num_epoch):
        model.train()
        y_predicts,y_labels=[],[]
        for step, batch in enumerate(training_iter):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, output_mask = batch
            # print("input_id", input_ids)
            # print("input_mask", input_mask)
            # print("segment_id", segment_ids)
            #print(len(input_ids),len(segment_ids),len(input_mask))
            bert_encode = model(input_ids, segment_ids, input_mask).cpu()
            train_loss = model.loss_fn(bert_encode=bert_encode, tags=label_ids, output_mask=output_mask)

            if args.gradient_accumulation_steps > 1:
                train_loss = train_loss / args.gradient_accumulation_steps

            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            predicts = model.predict(bert_encode, output_mask)
            y_predicts.append(predicts)
            #print(len(predicts))
            label_ids = label_ids.view(1, -1)
            #print(label_ids)
            label_ids = label_ids[label_ids != -1]
            y_labels.append(label_ids)
            label_ids = label_ids.cpu()
            train_acc, f1, train_precision, train_recall = model.result(predicts, label_ids)
            pbar.show_process(train_acc, train_loss.item(), f1, time.time() - start, step)
            #print(len(label_ids))
        print("----------------------------train----------------------------")
        train_predicted = torch.cat(y_predicts, dim=0).cpu()
        train_labeled = torch.cat(y_labels, dim=0).cpu()
        train_acc, f1,train_precision,train_recall = model.result(train_predicted,train_labeled)
        print("train_acc",train_acc)
        print("train_f1",f1)
        print("train_precision",train_precision)
        print("train_recall",train_recall)
            #model.class_report(predicts,label_ids)


# -----------------------验证----------------------------
        model.eval()
        count = 0
        y_predicts, y_labels = [], []
        eval_loss, eval_acc, eval_f1 = 0, 0, 0
        with torch.no_grad():
            for step, batch in enumerate(eval_iter):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, output_mask = batch
                bert_encode = model(input_ids, segment_ids, input_mask).cpu()
                eval_los = model.loss_fn(bert_encode=bert_encode, tags=label_ids, output_mask=output_mask)
                eval_loss = eval_los + eval_loss
                count += 1
                predicts = model.predict(bert_encode, output_mask)
                y_predicts.append(predicts)

                label_ids = label_ids.view(1, -1)
                label_ids = label_ids[label_ids != -1]
                y_labels.append(label_ids)


            eval_predicted = torch.cat(y_predicts, dim=0).cpu()
            eval_labeled = torch.cat(y_labels, dim=0).cpu()

            eval_acc, eval_f1,eval_precision,eval_recall = model.result(eval_predicted, eval_labeled)
            #model.class_report(eval_predicted, eval_labeled)
            print('--------------------------------------eval-------------------------------')
            # 保存最好的模型
            if eval_f1 > best_f1:
                best_f1 = eval_f1
                save_model(model, args.output_path)
                print("best_epoch",e+1)
            print(
                '\n\nEpoch %d - train_loss: %4f - eval_loss: %4f - train_acc:%4f - eval_acc:%4f - eval_f1:%4f - eval_precision:%4f - eval_recall:%4f\n'
                % (e + 1,
                   train_loss.item(),
                   eval_loss.item()/count,
                   train_acc,
                   eval_acc,
                   eval_f1,
                   eval_precision,
                   eval_recall))
    '''
    #--------------------------------------测试----------------------------------
    import os
    model.load_state_dict(torch.load(args.output_path+os.sep+'pytorch_model.bin'))
    y_predicts, y_labels = [], []
    with torch.no_grad():
        for step, batch in enumerate(test_iter):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, output_mask = batch
            bert_encode = model(input_ids, segment_ids, input_mask).cpu()
            predicts = model.predict(bert_encode, output_mask)
            y_predicts.append(predicts)

            label_ids = label_ids.view(1, -1)
            label_ids = label_ids[label_ids != -1]
            y_labels.append(label_ids)

        test_predicted = torch.cat(y_predicts, dim=0).cpu()
        test_labeled = torch.cat(y_labels, dim=0).cpu()

        test_acc, test_f1,test_precision,test_recall = model.result(test_predicted, test_labeled)
        #model.class_report(test_predicted, test_labeled)
        print("test_acc=",test_acc)
        print("test_f1=",test_f1)
        print("test_precision=",test_precision)
        print("test_recall=",test_recall)
        model.class_report(test_predicted,test_labeled)
    '''




