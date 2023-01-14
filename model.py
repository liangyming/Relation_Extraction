from transformers import BertModel
import torch
import torch.nn as nn
import config


class BertForRE(nn.Module):
    def __init__(self, args, tokenizer):
        super(BertForRE, self).__init__()
        self.bert = BertModel.from_pretrained(args.model_size)
        self.bert.resize_token_embeddings(len(tokenizer))
        # 冻结部分参数
        unfrozen_layers = [
            "classifier",
            "pooler",
            "encoder.layer.11",
            "classification_layer",
            "blanks_linear",
            "lm_linear",
            "cls"
        ]
        for name, param in self.bert.named_parameters():
            if not any([layer in name for layer in unfrozen_layers]):
                print("[FROZE]: %s" % name)
                param.requires_grad = False
            else:
                print("[FREE]: %s" % name)
                param.requires_grad = True

        self.fc = nn.Linear(768 * 2, args.num_classes)
        # 线型层 参数初始化
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.)

    def forward(self, input, e1_e2_start, token_type_ids, attention_mask):
        out = self.bert(input, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        output1 = torch.zeros((input.shape[0], 768 * 2)).to(config.device)
        for i in range(input.shape[0]):
            a = out[i, :, :]
            e1 = e1_e2_start[i, 0]
            e2 = e1_e2_start[i, 1]
            b = torch.tanh(torch.cat((a[e1], a[e2]), 0))
            output1[i] = b

        output = torch.log_softmax(self.fc(output1), dim=-1)
        return output
