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

def run(args):
    writer = SummaryWriter(comment='TypeClassificationWithMask')

    cleanterms = CLEANTERMS(args.clean_term_path)
    train_control_dict = {'max_sgr_n':1, 'no_short_upper':True, 'only_one':True, 'min_entity_word_count':1}
    #train_control_dict = {'max_sty_n':1, 'no_short_upper':True, 'only_one':True}
    single_eval_control_dict = {'max_sty_n':1, 'no_short_upper':True}
    multi_eval_control_dict = {'min_sty_n':2, 'no_short_upper':True}

    train_dataset = Entity_Dataset(args.train_text_file, args.train_match_file, cleanterms, args.model_name_or_path, mask_ratio=args.mask_ratio, control_dict=train_control_dict, debug=args.debug)
    single_eval_dataset = Entity_Dataset(args.eval_text_file, args.eval_match_file, cleanterms, args.model_name_or_path, mask_ratio=0.0, control_dict=single_eval_control_dict)
    multi_eval_dataset = Entity_Dataset(args.eval_text_file, args.eval_match_file, cleanterms, args.model_name_or_path, mask_ratio=0.0, control_dict=multi_eval_control_dict)

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=my_collate_fn, num_workers=args.num_workers)
    test_dataloader = DataLoader(single_eval_dataset, batch_size=args.train_batch_size, shuffle=False, collate_fn=my_collate_fn, num_workers=args.num_workers)
    test_multi_dataloader = DataLoader(multi_eval_dataset, batch_size=args.train_batch_size, shuffle=False, collate_fn=my_collate_fn, num_workers=args.num_workers)

    if args.output_dir == "":
        args.output_dir = f"output/lr{args.learning_rate}_mask{args.mask_ratio}"
        if args.debug:
            args.output_dir = args.output_dir + "_debugAlpha"
    #args.output_dir = args.output_dir + "max-sty-n-1"
    args.output_dir = args.output_dir + "min-word-count-1"

    try:
        os.system(f"rm -rf {args.output_dir}")
    except BaseException:
        pass
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, "model")):
        os.makedirs(os.path.join(args.output_dir, "model"))
    if not os.path.exists(os.path.join(args.output_dir, "eval")):
        os.makedirs(os.path.join(args.output_dir, "eval"))

    # model & optimizer & scheduler
    train_steps = int(args.train_epoch * len(train_dataloader))
    model = EntityTypeClassification(args.model_name_or_path, len(train_dataset.umls_labels)).to(args.device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * train_steps), num_training_steps=train_steps
        )
    
    global_step = 0
    for train_epoch_idx in range(args.train_epoch):
        model.train()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", ascii=True)
        epoch_loss = 0.
        for batch_idx, batch in enumerate(epoch_iterator):
            sentence = torch.LongTensor(batch[0]).to(args.device)
            entity_mark = torch.LongTensor(batch[1]).to(args.device)
            labels = torch.LongTensor(batch[2]).to(args.device)
            loss, _ = model(sentence, entity_mark, labels.argmax(-1))

            batch_loss = loss.item()
            epoch_loss += batch_loss
            
            writer.add_scalar('batch_loss', loss, global_step=global_step)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            epoch_iterator.set_description("Epoch_loss: %s, Batch_loss: %0.4f" %
                                           (epoch_loss / (batch_idx + 1), batch_loss))

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

            global_step += 1
            if global_step % args.save_step == 0 and global_step > 0:
                save_path = os.path.join(
                    args.output_dir, 'model', f'model_{global_step}.pth')
                torch.save(model, save_path)

            if global_step % args.eval_step == 0 and global_step > 0:
                test_eval_dict = model_eval(args, model, test_dataloader, eval=True, predict=True, tokenizer=train_dataset.tokenizer,
                                            eval_path=os.path.join(args.output_dir, 'single_eval_log.txt'),
                                            predict_path=os.path.join(args.output_dir, 'eval', f'test-{global_step}.csv'),
                                            output_suffix=f"Global step:{global_step}")
                multi_eval_dict = model_eval(args, model, test_multi_dataloader, eval=True, predict=True, tokenizer=train_dataset.tokenizer, eval_path=os.path.join(args.output_dir, 'multi_eval_log.txt'),
                           predict_path=os.path.join(args.output_dir, 'eval', f'multi-test-{global_step}.csv'))
                for key, value in test_eval_dict.items():
                    writer.add_scalar(key, value, global_step=global_step)
                for key, value in multi_eval_dict.items():
                    writer.add_scalar("mulit_" + key, value, global_step=global_step)
                print(test_eval_dict)
                print(multi_eval_dict)
    save_path = os.path.join(args.output_dir, 'model', 'last.pth')
    torch.save(model, save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_text_file', type=str, default='./data/100w_merged_terms_raw_train.txt')
    parser.add_argument('--train_match_file', type=str, default='./data/100w_merged_terms_tagged_train_cleanterms5.txt')
    parser.add_argument('--eval_text_file', type=str, default='./data/100w_merged_terms_raw_dev.txt')
    parser.add_argument('--eval_match_file', type=str, default='./data/100w_merged_terms_tagged_dev_cleanterms5.txt')
    parser.add_argument('--clean_term_path', type=str, default="./dict/cleanterms5.txt")

    parser.add_argument('--debug', action="store_true")

    parser.add_argument(
        "--model_name_or_path",
        default="./pretraining_models/pubmedbert_abs/",
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: ",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--save_step",
        default=10000,
        type=int,
        help="Save step",
    )
    parser.add_argument(
        "--eval_step",
        default=1000,
        type=int,
        help="Save step",
    )
    parser.add_argument(
        "--window_size",
        default=32,
        type=int,
        help="Window size",
    )
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument(
        "--train_batch_size", default=64, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=1e-4,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01,
                        type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8,
                        type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0,
                        type=float, help="Max gradient norm.")
    parser.add_argument("--device", type=str, default='cuda:1', help="device")
    parser.add_argument("--num_workers", default=1, type=int,
                        help="Num workers for data loader, only 0 can be used for Windows")
    parser.add_argument("--train_epoch", default=2, type=int)
    parser.add_argument("--mask_ratio", default=0.15, type=float)

    args = parser.parse_args()

    run(args)

if __name__ == "__main__":
    main()
