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
from nltk.tokenize import WordPunctTokenizer
from nltk import bleu
from nltk.translate.bleu_score import SmoothingFunction
logger = logging.getLogger(__name__)
weights = [1/2] * 2
def score(hyp, refs):
    return bleu(refs, hyp, weights=weights, smoothing_function=SmoothingFunction().method1)

def tokenize(sentence):
    sentence=WordPunctTokenizer().tokenize(sentence.lower())
    return sentence

def dist1(refs):
    item=[]
    for x in refs:
        item+=x
    return len(set(item))

def dist2(refs):
    item=[]
    n_gram=[]
    for x in refs:
        item+=x
        for i in range(len(x)-1):
            n_gram.append((x[i],x[i+1]))
    return len(set(n_gram))

class Example(object):
    """A single training/test example for the Event2Mind or ATOMIC dataset."""
    def __init__(self,
                 idx,
                 event,
                 category,
                 target,
                 posterior,
                 prior,
                 ):
        self.idx = idx
        self.event = event
        self.category = category
        self.target = target
        self.posterior=posterior
        self.prior=prior

def read_examples(filename, posterior_dic=None,prior_dic=None,dim=None):
    """Read examples from Event2Mind or ATOMIC dataset.""" 
    examples=[]
    for idx,temp in enumerate(pickle.load(open(filename,'rb'))):
        if dim is None or temp[1]==dim:
            examples.append(
                Example(
                        idx = idx,
                        event=temp[0],
                        category = temp[1],
                        target = temp[2],
                        posterior=posterior_dic[(temp[0],temp[1],temp[2])] if posterior_dic is not None else -1,
                        prior=prior_dic[(temp[0],temp[1])] if prior_dic is not None else [-1]
                        ) 
            )
    return examples

class InputFeature(object):
    """A single features for an example."""
    def __init__(self,
                 example_id,
                 event_ids,
                 context_ids,
                 target_ids,
                 posterior,
                 prior,
                ):
        self.example_id = example_id
        self.event_ids = event_ids
        self.context_ids=context_ids
        self.target_ids = target_ids
        self.posterior=posterior
        self.prior=prior
        
def convert_examples_to_features(examples, tokenizer, args,stage=None):
    """convert examples to tensor feature"""
    evidences=json.load(open(os.path.join('../data','evidence.json')))
    try:
        evidences_tokens=pickle.load(open(os.path.join('../data','evidence_tokens_{}.pkl').format(args.max_evidence_length),'rb'))
    except:
        evidences_tokens={}
        for key in tqdm(evidences,total=len(evidences)):
            for evidence in evidences[key]:
                evidences_tokens[evidence]=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(evidence)[:args.max_evidence_length])
                padding_length = args.max_evidence_length - len(evidences_tokens[evidence])
                evidences_tokens[evidence]+=[0] * padding_length
                evidences_tokens["None"]=[0]*args.max_evidence_length             
        pickle.dump(evidences_tokens,open(os.path.join('../data','evidence_tokens_{}.pkl').format(args.max_evidence_length),'wb'))             
    features = []
    for idx, example in tqdm(enumerate(examples),total=len(examples)) if stage=="training" else  enumerate(examples):
        #event
        event_tokens = tokenizer.tokenize(example.category+" "+example.event)[:args.max_event_length-1]
        event_ids = tokenizer.convert_tokens_to_ids(event_tokens)
        padding_length = args.max_event_length - len(event_ids)
        event_ids+=[0]*padding_length        
        posterior=int(example.posterior)
        prior=example.prior
        
        context=[]
        for x in evidences[example.event]:
            if x not in context and len(context)<args.num_evidence:
                context.append(x)
        context=context+["None"]*(args.num_evidence-len(context)+1)
        context_ids=[evidences_tokens[c] for c in context]

        
        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")[:args.max_target_length - 2]
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]
        target_tokens = ['##']+target_tokens+["</s>"]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        padding_length = args.max_target_length - len(target_ids)
        target_ids+=[0]*padding_length                     
       
        assert len(event_ids) == args.max_event_length
        assert len(target_ids) == args.max_target_length

        if idx < 5:
            if stage=='training' or stage=="test":
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))
                logger.info("vent_tokens: {}".format(event_tokens))
                logger.info("event_ids: {}".format(' '.join(map(str, event_ids))))  
                for i in range(5):
                    logger.info("context_text {}: {}".format(i,context[i]))
                    logger.info("context_ids {}: {}".format(i,context_ids[i]))
                logger.info("target_tokens: {}".format(target_tokens))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("posterior: {}".format(posterior))
                logger.info("prior: {}".format(' '.join(map(str, [round(x,2) for x in prior]))))
                               
        features.append(
            InputFeature(
                 example.idx,
                 event_ids,
                 context_ids,
                 target_ids,
                 posterior,
                 prior
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
    posterior_dic=pickle.load(open(os.path.join(args.posterior_dir, 'posterior-trn.pkl'),'rb'))
    train_examples = read_examples(os.path.join(args.data_dir, 'ppl-trn.pkl'),posterior_dic=posterior_dic)
    train_features = convert_examples_to_features(train_examples, tokenizer, args,stage='training')
    all_event_ids = torch.tensor([f.event_ids for f in train_features], dtype=torch.long)
    all_context_ids = torch.tensor([f.context_ids for f in train_features], dtype=torch.long)
    all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
    all_posterior = torch.tensor([f.posterior for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_event_ids,all_context_ids,all_target_ids,all_posterior)
    
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
    global_step,tr_re_loss,tr_context_loss,tr_reward,tr_clf_loss,nb_tr_examples, nb_tr_steps,best_bleu,best_bleu,eval_flag = 0,0,0,0,0,0,0,0,0,True    
    bar = tqdm(range(args.train_steps*args.gradient_accumulation_steps),total=args.train_steps*args.gradient_accumulation_steps)
    for step in bar:
        batch = next(train_dataloader)
        batch = tuple(t.to(args.device) for t in batch)
        event_ids,context_ids,target_ids,posterior = batch
        (re_loss,context_loss,reward),_,_ = model(event_ids=event_ids,context_ids=context_ids,target_ids=target_ids,posterior=posterior)

        # mean() to average on multi-gpu.
        if args.n_gpu > 1:
            re_loss = re_loss.mean() 
            context_loss = context_loss.mean()
            reward=reward.mean()
            
        if args.fp16 and args.loss_scale != 1.0:
            re_loss = re_loss * args.loss_scale
            context_loss = context_loss * args.loss_scale
            reward = reward * args.loss_scale

        if args.gradient_accumulation_steps > 1:
            re_loss = re_loss / args.gradient_accumulation_steps
            context_loss = context_loss / args.gradient_accumulation_steps
            reward = reward / args.gradient_accumulation_steps

        #print loss information
        tr_re_loss += re_loss.item()
        tr_context_loss += context_loss.item()
        tr_reward += reward.item()
        train_re_loss=round(tr_re_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
        train_context_loss=round(tr_context_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
        train_reward=round(tr_reward*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
        bar.set_description("re_loss {}, context_loss {}, reward {}".format(train_re_loss,train_context_loss,train_reward))
        nb_tr_examples += event_ids.size(0)
        nb_tr_steps += 1
        
        #backward 
        loss=re_loss+context_loss
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
            tr_re_loss,tr_context_loss,tr_reward,tr_clf_loss,nb_tr_examples, nb_tr_steps,eval_flag = 0,0,0,0,0,0,False
            result=evaluate(args,config,tokenizer,model,os.path.join(args.data_dir, 'ppl-dev.pkl'))
            category=set([e.category for e in train_examples])
            overall_bleu=0
            overall_dist=[]
            for c in category:
                bleu,dist=test(args,config,tokenizer,model,os.path.join(args.data_dir, 'gen-dev.pkl'),c,10,1000)
                result[c+' (bleu,dist1,dist2)']=[bleu,dist1(dist),dist2(dist)]
                result[c+' (bleu,dist1,dist2)']=' '.join([str(x) for x in result[c+' (bleu,dist1,dist2)']])
                overall_bleu+=bleu
                overall_dist+=dist
            overall_bleu=round(overall_bleu/len(category),2)
            result['Overall (bleu-2,dist1,dist2)']=[overall_bleu,dist1(overall_dist),dist2(overall_dist)]
            result['Overall (bleu-2,dist1,dist2)']=' '.join([str(x) for x in result['Overall (bleu-2,dist1,dist2)']])
            result['global_step']= global_step+1
            result['train_loss']= round(train_re_loss,5)
            logger.info("***** Result *****")
            #print result
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
            logger.info("  "+"*"*20)    

            if overall_bleu>best_bleu:
                logger.info("  Best bleu:%s",overall_bleu)
                logger.info("  "+"*"*20)
                best_bleu=overall_bleu
                # Save a trained model
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                
evaluate_dataset={}                
def evaluate(args,config,tokenizer,model,filename):            
    #Load and prepare data
    if filename in evaluate_dataset:
        eval_examples,eval_dataloader=evaluate_dataset[filename]
    else:
        flag='dev' if 'dev' in filename else 'tst'
        posterior_dic=pickle.load(open(os.path.join(args.posterior_dir, 'posterior-{}.pkl').format(flag),'rb'))
        eval_examples = read_examples(filename,posterior_dic=posterior_dic)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='dev')
        all_event_ids = torch.tensor([f.event_ids for f in eval_features], dtype=torch.long)
        all_context_ids = torch.tensor([f.context_ids for f in eval_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)   
        all_posterior = torch.tensor([f.posterior for f in eval_features], dtype=torch.long)   
        eval_data = TensorDataset(all_event_ids,all_context_ids,all_target_ids,all_posterior)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        evaluate_dataset[filename]=eval_examples,eval_dataloader

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)            
    eval_loss,tokens_num, eval_accuracy,nb_eval_steps, nb_eval_examples,eval_perplexity,eval_cls_loss,index = 0,0,0,0,0,0,0,0

    #Running evaluation
    for batch in eval_dataloader:
        batch = tuple(t.to(args.device) for t in batch)
        event_ids,context_ids,target_ids,posterior = batch
        with torch.no_grad():
            _,tmp_eval_loss,num = model(event_ids=event_ids,context_ids=context_ids,target_ids=target_ids,posterior=posterior)  
        eval_loss += tmp_eval_loss.sum().item()
        tokens_num += num.sum().item()
        nb_eval_examples += event_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss / tokens_num
    result = {'eval_loss': round(eval_loss,5),
              'eval_ppl': round(np.exp(eval_loss),5)}
    model.train()
    return result

test_dataset={}    
def test(args,config,tokenizer,model,filename,dim,topk,num_sample=None):
    if (filename,dim,num_sample) in test_dataset:
        eval_examples,eval_dataloader=test_dataset[(filename,dim,num_sample)]
    else:
        flag='dev' if 'dev' in filename else 'tst'
        prior_dic=pickle.load(open(os.path.join(args.prior_dir, 'prior-{}.pkl').format(flag),'rb'))
        eval_examples = read_examples(filename,prior_dic=prior_dic,dim=dim)
        if num_sample is not None:
            eval_examples = random.sample(eval_examples,num_sample)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
        all_event_ids = torch.tensor([f.event_ids for f in eval_features], dtype=torch.long)
        all_context_ids = torch.tensor([f.context_ids for f in eval_features], dtype=torch.long)
        all_prior = torch.tensor([f.prior for f in eval_features], dtype=torch.float)   
        eval_data = TensorDataset(all_event_ids,all_context_ids,all_prior)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        test_dataset[(filename,dim,num_sample)]=eval_examples,eval_dataloader

    logger.info("***** Running test *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)    
    logger.info("  Dimension = %s", dim)

    model.eval()
    pairs=[]
    index=0
    for batch in eval_dataloader:
        batch = tuple(t.to(args.device) for t in batch)
        event_ids,context_ids,prior = batch            
        with torch.no_grad():
            preds = model(event_ids=event_ids,context_ids=context_ids,prior=prior,topk=topk)
        for pred in preds:
            p=[]
            p1=[]
            pred=pred.cpu().numpy()
            for t in pred:
                t=list(t)
                if 0 in t:
                    t=t[:t.index(0)]
                text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                p.append(tokenize(text))
            ref=list(set(eval_examples[index].target))
            cat=eval_examples[index].category
            index+=1
            if "none" in ref and 1 / len(ref) > 1/3 and "atomic" in args.data_dir: 
                continue                                    
            ref=[tokenize(r) for r in ref]
            pairs.append((ref,p,cat))
    model.train()

    #calculate bleu score,dist-1,dist-2
    dist,bleu=[],[]
    example_dist,example_bl,result={},{},{}
    for pair in pairs:
        for gen in pair[1]:
            bleu.append(score(gen, pair[0]))  
        dist+=pair[1]
    return round(np.mean(bleu)*100,2),dist




        
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
    parser.add_argument("--codebook_path", default=None, type=str,
                        help="The path of pretrained vq-vae model.")    
    parser.add_argument("--prior_dir", default=None, type=str,
                        help="The directory of prior distribution.")
    parser.add_argument("--posterior_dir", default=None, type=str,
                        help="The directory of prior distribution.")    
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model" )    
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_evidence_length", default=64, type=int,
                        help="The maximum total evidence sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")    
    parser.add_argument("--num_evidence", default=45, type=int,
                        help="numbers of evidence per event")       
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
    parser.add_argument("--do_topk", action='store_true',
                        help="Obtain topK latent variables")
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
    encoder_layer = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)    
    codebook=nn.Embedding(args.z_size,config.n_embd)
    codebook.load_state_dict(torch.load(args.codebook_path))
    model = Model(encoder,decoder,codebook,config,args,sos_id=tokenizer.convert_tokens_to_ids(["##"])[0],eos_id=tokenizer.convert_tokens_to_ids(["</s>"])[0])
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
    if args.do_topk:
        if 'event2mind' in args.load_model_path:
            categories = ["<oReact>","<xIntent>","<xReact>"]
        else:
            categories = ["<oEffect>","<oReact>","<oWant>","<xAttr>","<xEffect>","<xIntent>","<xNeed>","<xReact>","<xWant>"]
        topk_dic={}
        for cat in categories:
            best_bleu=0
            topk_dic[cat]=5
            for topk in range(5,16):
                bleu,_=test(args,config,tokenizer,model,os.path.join(args.data_dir, 'gen-dev.pkl'),cat,topk,100)
                logger.info("  {} (topk bleu-2): {} {}".format(cat,topk,bleu))
                if bleu>=best_bleu:
                    best_bleu=bleu
                    topk_dic[cat]=topk
            logger.info("  "+"*"*20)  
            logger.info("  Best (topk bleu-2): {} {}".format(topk_dic[cat],best_bleu))
            logger.info("  "+"*"*20)  
        logger.info("Saving topk latent variables to %s",os.path.join(args.output_dir, "topk.pkl"))
        pickle.dump(topk_dic,open(os.path.join(args.output_dir, "topk.pkl"),'wb'))
        
    if args.do_test:
        logger.info("Load topk latent variables from %s",os.path.join(args.output_dir, "topk.pkl"))  
        topk_dic=pickle.load(open(os.path.join(args.output_dir, "topk.pkl"),'rb'))
        result={}
        overall_bleu=0
        overall_dist=[]
        for c in topk_dic:
            bleu,dist=test(args,config,tokenizer,model,os.path.join(args.data_dir, 'gen-tst.pkl'),c,topk_dic[c],None)
            result[c+' (bleu-2,dist1,dist2)']=[bleu,dist1(dist),dist2(dist)]
            result[c+' (bleu-2,dist1,dist2)']=' '.join([str(x) for x in result[c+' (bleu-2,dist1,dist2)']])
            overall_bleu+=bleu
            overall_dist+=dist
        overall_bleu=round(overall_bleu/len(topk_dic),2)
        result['Overall (bleu-2,dist1,dist2)']=[overall_bleu,dist1(overall_dist),dist2(overall_dist)]
        result['Overall (bleu-2,dist1,dist2)']=' '.join([str(x) for x in result['Overall (bleu-2,dist1,dist2)']])
        logger.info("***** Result *****")
        #print result
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        logger.info("  "+"*"*20)       
        
                
if __name__ == "__main__":
    main()

