import torch
import torch.nn as nn
import subprocess
from sklearn.metrics import f1_score
import config
from model import BertForRE
from mydata import *


def train(args):
    train_loader, test_loader, train_len, test_len = load_dataloaders(args)
    tokenizer = load_pickle("BERT_tokenizer.pkl")

    Rmodel = BertForRE(args, tokenizer).to(config.device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam([{"params": Rmodel.parameters(), "lr": args.lr}])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 6, 8, 12, 15, 18, 20, 22, 24, 26, 30],
                                               gamma=0.8)
    logger.info("Starting training process...")
    pad_id = tokenizer.pad_token_id
    best_pre = 0
    for epoch in range(0, args.num_epochs):
        Rmodel.train()
        # start_time = time.time()
        total_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            x, e1_e2_start, labels, _, _, _ = data
            attention_mask = (x != pad_id).float().to(config.device)
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long().to(config.device)

            # 编码层最后一层的向量([32, 42, 768])#batch_size,seq_len,embedding_len
            outputs = Rmodel(input=x, e1_e2_start=e1_e2_start, token_type_ids=token_type_ids,
                             attention_mask=attention_mask)
            # output:[32,19]

            loss = criterion(outputs, labels.squeeze(1))
            # loss = loss / args.gradient_acc_steps
            total_loss += loss
            # if (i % args.gradient_acc_steps) == 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print('total_loss:',total_loss)
        scheduler.step()

        results = infer(Rmodel, test_loader)
        logger.info("epoch: {}---f1: {}".format(epoch, results['f1']))
        if results['f1'] > best_pre:
            # print('F1:',results['f1'])
            best_pre = results['f1']
            torch.save({
                'state_dict': Rmodel.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()},
                os.path.join("./result/", "ReExt_f1_checkpoint.pth.tar"))
            best_model = Rmodel
        logger.info("Finished Training!")
    return best_model, test_loader


def infer(Rmodel, test_loader, save=False):
    tokenizer = load_pickle("BERT_tokenizer.pkl")
    pad_id = tokenizer.pad_token_id

    logger.info("Evaluating test samples...")
    acc = 0
    out_labels = []
    true_labels = []
    Rmodel.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            x, e1_e2_start, labels, _, _, _ = data
            attention_mask = (x != pad_id).float().to(config.device)
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long().to(config.device)

            outputs = Rmodel(input=x, e1_e2_start=e1_e2_start, token_type_ids=token_type_ids,
                             attention_mask=attention_mask)
            # [32,19]
            accuracy, (o, l) = evaluate_(outputs, labels, ignore_idx=-1)
            out_labels.extend([str(i) for i in o])
            true_labels.extend([str(i) for i in l])
            acc += accuracy

    accuracy = acc / (i + 1)
    results = {
        "predict_label": out_labels,
        "accuracy": accuracy,
        "f1": f1_score(true_labels, out_labels, average='macro')
    }
    if save:
        logger.info("save result!")
        rm = load_pickle('relations.pkl')
        prediction_file = open(config.prediction_path, 'w')
        truth_file = open(config.truth_path, 'w')

        for i in range(len(out_labels)):
            prediction_file.write("{}\t{}".format(i, rm.idx2rel[int(out_labels[i])]))
            truth_file.write("{}\t{}".format(i, rm.idx2rel[int(true_labels[i])]))

        prediction_file.close()
        truth_file.close()

        # perl语言文件的源程序
        process = subprocess.Popen([
            "perl",
            config.perl_path,
            config.prediction_path,
            config.truth_path
        ], stdout=subprocess.PIPE)
        for line in str(process.communicate()[0].decode("utf-8")).split("\\n"):
            print(line)
    return results


def evaluate_(output, labels, ignore_idx):
    idxs = (labels != ignore_idx).squeeze()
    o_labels = torch.softmax(output, dim=1).max(1)[1]
    l = labels.squeeze()[idxs]
    o = o_labels[idxs]

    if len(idxs) > 1:
        acc = (l == o).sum().item() / len(idxs)
    else:
        acc = (l == o).sum().item()
    l = l.cpu().numpy().tolist() if l.is_cuda else l.numpy().tolist()
    o = o.cpu().numpy().tolist() if o.is_cuda else o.numpy().tolist()

    return acc, (o, l)