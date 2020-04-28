# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

from __future__ import absolute_import
import os
import sys
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
import torch.nn as nn
from itertools import cycle
from tqdm import tqdm, trange
from nltk.tokenize import WordPunctTokenizer
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from model import Model
logger = logging.getLogger(__name__)

class Example(object):
    """A single training/test example for the Event2Mind or ATOMIC dataset."""
    def __init__(self,
                 idx,
                 event,
                 category,
                 target,
                 ):
        self.idx = idx
        self.event = event
        self.category = category
        self.target = target

def read_examples(filename):
    """Read examples from Event2Mind or ATOMIC dataset.""" 
    examples=[]
    for idx,temp in enumerate(pickle.load(open(filename,'rb'))):
            examples.append(
                Example(
                        idx = idx,
                        event=temp[0],
                        category = temp[1],
                        target = temp[2],
                        ) 
            )
    return examples

class InputFeature(object):
    """A single features for an example."""
    def __init__(self,
                 example_id,
                 event_ids,
                 target_ids,
                ):
        self.example_id = example_id
        self.event_ids = event_ids
        self.target_ids = target_ids
        
def convert_examples_to_features(examples, tokenizer, args,stage=None):
    """convert examples to tensor feature"""
    features = []
    for idx, example in tqdm(enumerate(examples),total=len(examples)) if stage=="training" else  enumerate(examples):
        #event
        event_tokens = tokenizer.tokenize(example.category+" "+example.event)[:args.max_event_length-1]
        event_ids = tokenizer.convert_tokens_to_ids(event_tokens)
        padding_length = args.max_event_length - len(event_ids)
        event_ids+=[0]*padding_length        
        
        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")[:args.max_target_length - 2]
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]
    
        target_tokens = ['<s>']+target_tokens+["</s>"]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        padding_length = args.max_target_length - len(target_ids)
        target_ids+=[0]*padding_length
        

        if idx < 5:
            if stage=='training':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))
                logger.info("event_tokens: {}".format(event_tokens))
                logger.info("event_ids: {}".format(' '.join(map(str, event_ids))))             
                logger.info("target_tokens: {}".format(target_tokens))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                               
        features.append(
            InputFeature(
                 example.idx,
                 event_ids,
                 target_ids
            )
        )
    return features


def set_seed(args):
    """set random seed for reproduction"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
def train(args,config,tokenizer,model):     
    """train model"""
    #Load and prepare data
    train_examples = read_examples(os.path.join(args.data_dir, 'ppl-trn.pkl'))
    train_features = convert_examples_to_features(train_examples, tokenizer, args,stage='training')
    all_event_ids = torch.tensor([f.event_ids for f in train_features], dtype=torch.long)
    all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_event_ids,all_target_ids)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    train_dataloader=cycle(train_dataloader)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.train_steps)

    #Running training
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size= %d", args.train_batch_size)
    logger.info("  Batch size (including gradient_accumulation_steps)= %d", args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Num steps = %d", args.train_steps*args.gradient_accumulation_steps)

    dev_dataset={}
    model.train()
    global_step,tr_re_loss,tr_vae_loss,tr_perplexity,tr_clf_loss,nb_tr_examples, nb_tr_steps,best_loss,best_bleu,eval_flag = 0,0,0,0,0,0,0,1e4,0,True    
    bar = tqdm(range(args.train_steps*args.gradient_accumulation_steps),total=args.train_steps*args.gradient_accumulation_steps)
    for step in bar:
        batch = next(train_dataloader)
        batch = tuple(t.to(args.device) for t in batch)
        event_ids,target_ids = batch
        (re_loss,vae_loss,perplexity),_,_,_ = model(event_ids=event_ids,target_ids=target_ids)
        
        # mean() to average on multi-gpu.
        if args.n_gpu > 1:
            re_loss = re_loss.mean() 
            vae_loss = vae_loss.mean()
            perplexity=perplexity.mean()
            
        if args.fp16 and args.loss_scale != 1.0:
            re_loss = re_loss * args.loss_scale
            vae_loss = vae_loss * args.loss_scale
            perplexity = perplexity * args.loss_scale

        if args.gradient_accumulation_steps > 1:
            re_loss = re_loss / args.gradient_accumulation_steps
            vae_loss = vae_loss / args.gradient_accumulation_steps
            perplexity = perplexity / args.gradient_accumulation_steps

        #print loss information
        tr_re_loss += re_loss.item()
        tr_vae_loss += vae_loss.item()
        tr_perplexity += perplexity.item()
        train_re_loss=round(tr_re_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
        train_vae_loss=round(tr_vae_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
        train_perplexity=round(tr_perplexity*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
        bar.set_description("re_loss {}, vae_loss {}, perprlexity {}".format(train_re_loss,train_vae_loss,train_perplexity))
        nb_tr_examples += event_ids.size(0)
        nb_tr_steps += 1
        
        #backward 
        loss=re_loss+vae_loss
        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()
        
        #update parameter
        if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            eval_flag = True
        
        #Running evaluation
        if ((global_step + 1) %args.eval_steps == 0) and eval_flag:
            tr_re_loss,tr_vae_loss,tr_perplexity,tr_clf_loss,nb_tr_examples, nb_tr_steps,eval_flag = 0,0,0,0,0,0,False
            result,_=test(args,config,tokenizer,model,os.path.join(args.data_dir, 'ppl-dev.pkl'))
            result['global_step']= global_step+1
            result['train_loss']= round(train_re_loss,5)
            #print result
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
            logger.info("  "+"*"*20)    
            if result['eval_loss']<best_loss:
                logger.info("  Best ppl:%s",round(np.exp(result['eval_loss']),5))
                logger.info("  "+"*"*20)
                best_loss=result['eval_loss']
                # Save a trained model
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                
test_dataset={}                
def test(args,config,tokenizer,model,filename):            
    #Load and prepare data
    if filename in test_dataset:
        eval_examples,eval_dataloader=test_dataset[filename]
    else:
        eval_examples = read_examples(filename)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='dev')
        all_event_ids = torch.tensor([f.event_ids for f in eval_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)   
        eval_data = TensorDataset(all_event_ids,all_target_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        test_dataset[filename]=eval_examples,eval_dataloader

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)            
    eval_loss,tokens_num, eval_accuracy,nb_eval_steps, nb_eval_examples,eval_perplexity,eval_cls_loss,index = 0,0,0,0,0,0,0,0
    prior={}
    pdf={}
    encoding_all=None            
    model.eval()

    #Running evaluation
    for batch in eval_dataloader:
        batch = tuple(t.to(args.device) for t in batch)
        event_ids,target_ids = batch     
        with torch.no_grad():
            (re_loss,vae_loss,perplexity),tmp_eval_loss,num,encoding = model(event_ids=event_ids,target_ids=target_ids) 
        encoding=encoding.cpu().numpy()
        for i in range(encoding.shape[0]):
            if (eval_examples[index].event,eval_examples[index].category) not in prior:
                prior[(eval_examples[index].event,eval_examples[index].category)]=encoding[i].copy()
            else:
                prior[(eval_examples[index].event,eval_examples[index].category)]+=encoding[i]
            pdf[(eval_examples[index].event,eval_examples[index].category,eval_examples[index].target)]=np.argmax(encoding[i])
            if encoding_all is None:
                encoding_all=encoding[i].copy()
            else:
                encoding_all+=encoding[i]
            index+=1
        eval_loss += tmp_eval_loss.sum().item()
        tokens_num += num.sum().item()
        nb_eval_examples += event_ids.size(0)
        nb_eval_steps += 1
    encoding_all=encoding_all/np.sum(encoding_all)
    eval_perplexity=np.sum(-encoding_all*np.log(encoding_all+1e-7))
    eval_perplexity=np.exp(eval_perplexity)
    eval_loss = eval_loss / tokens_num
    result = {'eval_loss': round(eval_loss,5),
              'eval_ppl': round(np.exp(eval_loss),5),
              'codebook_ppl':round(eval_perplexity,5)}
    model.train()
    return result,(prior,pdf)

            

        
def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " )
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    ## Other parameters
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model" )    
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_event_length", default=64, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_bleu", action='store_true',
                        help="Whether to run eval bleu on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval bleu on the test set.")
    parser.add_argument("--do_label", action='store_true',
                        help="Whether to output posterior and prior distribution and codebook")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size for training per gpu.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size for evaluation per gpu.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="Eval over eval_steps")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="Total train steps")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--z_size", default=64, type=int,
                        help="Hidden size of latent variable.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="Beam size for beam search")    
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")    
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--preprocess_type', type=str, default='', help="How to process the input")
    args = parser.parse_args()
    logger.info(args)
    
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    
    
    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    
    # Set seed
    set_seed(args)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    config = GPT2Config.from_pretrained(args.model_name_or_path)
    decoder = GPT2LMHeadModel.from_pretrained(args.model_name_or_path,config=config)
    encoder = GPT2LMHeadModel.from_pretrained(args.model_name_or_path,config=config) 
    model = Model(encoder,decoder,config,args)
        
    if args.load_model_path is not None:
        logger.info("Load model from %s",args.load_model_path)
        model.load_state_dict(torch.load(args.load_model_path))
    logger.info("Training/evaluation parameters %s", args)     
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        train(args,config,tokenizer,model)
    if args.do_label:
        logger.info("Saving codebook to %s",os.path.join(args.output_dir, "codebook.bin"))  
        model_to_save = model.module.codebook._embedding if hasattr(model, 'module') else model.codebook._embedding
        torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, "codebook.bin"))
        for flag in ['dev','trn']:
            result,(prior,pdf)=test(args,config,tokenizer,model,os.path.join(args.data_dir, 'ppl-{}.pkl'.format(flag)))
            #print result
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
            logger.info("  "+"*"*20)   
            logger.info("Saving prior distribution to %s",os.path.join(args.output_dir, "prior-{}.pkl".format(flag)))
            pickle.dump(prior,open(os.path.join(args.output_dir, "prior-{}.pkl".format(flag)),'wb'))
            logger.info("Saving posterior distribution to %s",os.path.join(args.output_dir, "posterior-{}.pkl".format(flag)))
            pickle.dump(pdf,open(os.path.join(args.output_dir, "posterior-{}.pkl".format(flag)),'wb'))
            

        
                
if __name__ == "__main__":
    main()

