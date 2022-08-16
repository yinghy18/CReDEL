import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from label_util import get_entity_type_from_json


def label_sty2sgr(labels, sty2sgr):
    sgrs = []
    for l in labels:
        sgrs.append(sty2sgr[l.split('|')[0]])
    return sgrs


def model_eval(args, model, dataloader, eval=True, predict=True, tokenizer=None, eval_path=None, predict_path=None, output_suffix=""):
    eval_dict = {}
    all_predict = []
    all_label = []
    
    predict_lines = []

    sty2id, sty2sgr = get_entity_type_from_json()
    id2sty = {id:sty for sty, id in sty2id.items()}

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            sentence = torch.LongTensor(batch[0]).to(args.device)
            entity_mark = torch.LongTensor(batch[1]).to(args.device)
            labels = batch[2]

            logits = model.predict(sentence, entity_mark)
            predict_idx = torch.max(logits, dim=1)[1].cpu().detach().numpy().tolist()

            all_predict.extend([id2sty[p] for p in predict_idx])

            if eval:
                if len(labels.shape) == 1:
                    label = labels.numpy().tolist()
                else:
                    label = []
                    for i in range(labels.shape[0]):
                        label.append("|".join([id2sty[idx] for idx, l in enumerate(labels[i]) if l == 1]))

                all_label.extend(label)

            if predict:
                for i in range(sentence.shape[0]):
                    sentence_string = tokenizer.decode(sentence[i][1:-1], skip_special_tokens=True)
                    entity = []
                    for idx, ch in enumerate(sentence[i]):
                        if entity_mark[i][idx] == 1:
                            entity.append(ch)
                    entity_string = tokenizer.decode(entity, skip_special_tokens=True)
                    if len(labels.shape) == 1:
                        true_label = id2sty[labels[i].item()]
                    else:
                        true_label = "|".join([id2sty[idx] for idx, l in enumerate(labels[i]) if l == 1])
                    predict_label = id2sty[predict_idx[i]]
                    predict_lines.append('\t'.join([sentence_string, entity_string, predict_label, true_label]))

    
    if predict:
        if predict_path is None:
            for line in predict_lines:
                print(line)
        else:
            with open(predict_path, "w", encoding="utf-8") as f:
                for line in predict_lines:
                    f.write(line.strip() + "\n")
    
    if eval:
        predict_sgr = label_sty2sgr(all_predict, sty2sgr)
        label_sgr = label_sty2sgr(all_label, sty2sgr)
        sty_correct = 0
        sgr_correct = 0
        for i in range(len(predict_sgr)):
            if all_predict[i] in all_label[i]:
                sty_correct += 1
            if predict_sgr[i] == label_sgr[i]:
                sgr_correct += 1
        eval_dict['sty_acc'] = sty_correct / len(predict_sgr)
        eval_dict['sgr_acc'] = sgr_correct / len(predict_sgr)
        if eval_path is not None:
            with open(eval_path, "a+", encoding="utf-8") as f:
                if output_suffix:
                    f.write(output_suffix + "\n")
                for key, value in eval_dict.items():
                    f.write(f"{key}:{value}\n")
                f.write("----\n")
        
    return eval_dict
