from model import EntityTypeClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from evaluation import model_eval
from data_util import Entity_Dataset, my_collate_fn
from tqdm import tqdm, trange
import torch
from torch import nn
import time
import os
import numpy as np
import argparse
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from load_cleanterms import CLEANTERMS
from label_util import get_entity_type_from_json
import json

import ipdb


def run(args):

    sty2id, sty2sgr = get_entity_type_from_json()
    id2sty = {id:sty for sty, id in sty2id.items()}
    cleanterms = CLEANTERMS(args.clean_term_path)
    predict_control_dict = {}#{'no_short_upper':True}

    predict_dataset = Entity_Dataset(args.predict_text_file, args.predict_match_file, cleanterms, args.model_name_or_path, mask_ratio=0.0, control_dict=predict_control_dict, debug=args.debug, predict=True)
    predict_dataloader = DataLoader(predict_dataset, batch_size=args.train_batch_size, shuffle=False, collate_fn=my_collate_fn, num_workers=args.num_workers)
    model = torch.load(os.path.join(args.save_model_folder, 'model', args.selected_pth)).to(args.device)

    shift_id = []
    with open(args.predict_match_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if args.debug:
        lines = lines[0:1000000]
    for idx, line in tqdm(enumerate(lines)):
        entities = eval(lines[idx].strip().lower())#json.loads(lines[idx])
        flag = False
        for entity in entities:
            if predict_dataset.filter_entity(entity):
                flag = True
        if not flag:
            shift_id.append(idx)

    output_basename = os.path.basename(args.predict_text_file)
    if output_basename.endswith('.txt'):
        output_basename = output_basename[:-4]
    if args.debug:
        output_basename = output_basename + "_debug2"
    output_basename = output_basename + "_match.txt"

    output_match_file = os.path.join(args.save_model_folder, output_basename)
    try:
        os.system(f'rm -rf {output_match_file}')
    except BaseException:
        pass

    start_idx = 0
    model.eval()
    now_json = []
    last_sentence_id = 0
    origin_id = 0
    """
    while origin_id in shift_id:
        with open(output_match_file, 'a+', encoding='utf-8') as f:
            f.write('[]\n')
        origin_id += 1
    """
    #ipdb.set_trace()
    with torch.no_grad():
        for batch in tqdm(predict_dataloader):
            end_idx = start_idx + batch[0].shape[0]
            sentence = torch.LongTensor(batch[0]).to(args.device)
            entity_mark = torch.LongTensor(batch[1]).to(args.device)
            labels = batch[2]

            label_sty_list = []
            for i in range(labels.shape[0]):
                stys = torch.arange(labels.shape[1]).to(args.device)[labels[i] == 1]
                label_sty_list.append([id2sty[sty.item()] for sty in stys])
            #label_grp_list = [set([sty2sgr[sty] for sty in label_sty]) for label_sty in label_sty_list]

            logits = model.predict(sentence, entity_mark)
            predict_idx = torch.max(logits, dim=1)[1].cpu().detach().numpy().tolist()
            predict_sty = [id2sty[sty] for sty in predict_idx]
            predict_grp = [sty2sgr[sty] for sty in predict_sty]

            for i in range(labels.shape[0]):
                now_sentence_id = predict_dataset.map[start_idx + i]
                now_entity = predict_dataset.entities[start_idx + i]
                now_entity['type'] = predict_sty[i]
                now_entity['group'] = predict_grp[i]

                if now_sentence_id == last_sentence_id:
                    #if predict_grp[i] in label_grp_list[i]:
                    now_json.append(now_entity)
                else:
                    while origin_id in shift_id:
                        with open(output_match_file, 'a+', encoding='utf-8') as f:
                            f.write('[]\n')
                        origin_id += 1
                    with open(output_match_file, 'a+', encoding='utf-8') as f:
                        output_line = json.dumps(now_json).strip()
                        f.write(output_line + "\n")
                    now_json = []
                    last_sentence_id = now_sentence_id
                    origin_id += 1
                    #if predict_grp[i] in label_grp_list[i]:
                    now_json.append(now_entity)

            start_idx = end_idx
            
        while origin_id in shift_id:
            with open(output_match_file, 'a+', encoding='utf-8') as f:
                f.write('[]\n')
            origin_id += 1
        with open(output_match_file, 'a+', encoding='utf-8') as f:
            output_line = json.dumps(now_json).strip()
            f.write(output_line + "\n")
        now_json = []
        last_sentence_id = now_sentence_id
        origin_id += 1
        #if predict_grp[i] in label_grp_list[i]:
        now_json.append(now_entity)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_text_file', type=str, default='./mention_type/NCBI-sents')
    parser.add_argument('--predict_match_file', type=str, default='./mention_type/NCBI-tags')
    parser.add_argument('--clean_term_path', type=str, default="./mention_type/pre_dict.txt")

    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--model_name_or_path', type=str, default='./pretraining_models/pubmedbert_abs/')
    parser.add_argument(
        "--save_model_folder",
        default="./output/lr5e-05_mask0.15min-word-count-2",
        type=str,
    )
    parser.add_argument(
        "--selected_pth",
        default="model_17000.pth",
        type=str,
    )
    parser.add_argument(
        "--window_size",
        default=32,
        type=int,
        help="Window size",
    )
    parser.add_argument(
        "--train_batch_size", default=512, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument("--device", type=str, default='cuda:1', help="device")
    parser.add_argument("--num_workers", default=1, type=int,
                        help="Num workers for data loader, only 0 can be used for Windows")

    args = parser.parse_args()

    run(args)

#eg:python predict.py --predict_text_file ./mention_type/NCBIparse-sents --predict_match_file ./mention_type/NCBIparse-tags
if __name__ == "__main__":
    main()
