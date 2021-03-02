import torch.nn as nn
import numpy as np
from model.crf import CRF
from sklearn.metrics import f1_score, classification_report,accuracy_score,precision_score,recall_score
import config  as args
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel


class Bert_CRF(BertPreTrainedModel):
    def __init__(self,
                 config,
                 num_tag,need_birnn=True,hidden_size=768,hidden_dropout_prob=0.5,rnn_dim=768):
        super(Bert_CRF, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.apply(self.init_bert_weights)
        out_dim = hidden_size
        self.need_birnn = need_birnn
        print("need_birnn=",need_birnn)

        # 如果为False，则不要BiLSTM层
        if need_birnn:
            self.bilstm = nn.LSTM(bidirectional=True, num_layers=2, input_size=768, hidden_size=768 // 2, batch_first=True)
        self.classifier = nn.Linear(out_dim, num_tag)
        self.crf = CRF(num_tag)

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                label_id=None,
                output_all_encoded_layers=False):
        outputs,_ = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        #sequence_output = sequence_output.numpy()
        #print(sequence_output)
        if self.need_birnn:
            sequence_output, _ = self.bilstm(sequence_output)
        sequence_output = self.dropout(sequence_output)
        output = self.classifier(sequence_output)
        return output

    def loss_fn(self, bert_encode, output_mask, tags):
        loss = self.crf.negative_log_loss(bert_encode, output_mask, tags)
        return loss

    def predict(self, bert_encode, output_mask):
        predicts = self.crf.get_batch_best_path(bert_encode, output_mask)
        #print(predicts)
        predicts = predicts.view(1, -1).squeeze()
        predicts = predicts[predicts != -1]
        return predicts

    def result(self, y_pred, y_true):
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        f_score = f1_score(y_true, y_pred, average="weighted")
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")
        return acc, f_score, precision, recall

    def class_report(self, y_pred, y_true):
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        classify_report = classification_report(y_true, y_pred)
        print('\n\nclassify_report:\n', classify_report)





