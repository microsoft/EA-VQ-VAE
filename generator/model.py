# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss
from beam import Beam
import torch.nn.functional as F
import random
class Model(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. 2-layer transformer
        * `decoder`- decoder of seq2seq model. e.g. GPT2
        * `codebook`- codebook of VQ-VAE
        * `config`- configuration of encoder model. 
        * `args`- arguments.
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """
    def __init__(self, encoder,decoder,codebook,config,args,sos_id,eos_id):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder=decoder
        self.codebook = codebook
        self.config=config
        self.args=args
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)     
        self.lsm = nn.LogSoftmax(dim=-1)
        
        self.beam_size=args.beam_size
        self.max_length=args.max_target_length
        self.sos_id=sos_id
        self.eos_id=eos_id
        self.lm_head.weight=self.decoder.wte.weight       
        
    def forward(self, event_ids,context_ids,target_ids=None,posterior=None,prior=None,topk=None):   
        """
            Forward the VQ-VAE model.
            Parameters:
            * `event_ids`- event ids of examples
            * `target_ids`- target ids of examples
            * `context_ids`- context ids of examples
            * `posterior`- posterior distribution function p(z|x,y). 
            * `prior`- prior distribution function p(z|x). 
            * `topK`- using topK latent variables to select evidences. 
        """        
        if target_ids is None:
            return self.gen(event_ids,context_ids,prior,topk)
        
        #obtain latent variable
        z=self.codebook(posterior).detach()
   
        #obtain hidden states of context
        bs,cx,le=context_ids.shape
        context_ids_all=context_ids
        context_ids=context_ids.view(-1,le)
        c=self.encoder(context_ids)[0][:,-1].view(bs,cx,-1)
        
        # select nearest context 
        distances = ((c-z[:,None,:])**2).sum(-1)      
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices,cx)
        c=(c*encodings[:,:,None].float()).sum(1)
        context_loss = torch.mean((c - z)**2,-1)
        context_ids=(context_ids.view(bs,cx,le)*encodings[:,:,None].long()).sum(1)
        
        #randomly select context
        item=[]
        for i in range(bs):
            k=-1
            while k==-1 or k==encoding_indices[i]:
                k=random.randint(0,cx-1)
            item.append(context_ids_all[i:i+1,k])
        context_ids_random=torch.cat(item,0) 
        
        #calculate probability of target given empty context
        inputs_ids=torch.cat((context_ids_random,event_ids,target_ids),-1)
        hidden_states=self.decoder(inputs_ids)[0][:,-target_ids.size(1):]
        lm_logits = self.lsm(self.lm_head(hidden_states))
        labels_logits=lm_logits[:,:-1].gather(-1,target_ids[:,1:].unsqueeze(-1))[:,:,0]
        active_loss=target_ids.ne(0)[:,1:]
        sm_prob=labels_logits*active_loss.float()
        prob_1=torch.exp(sm_prob.sum(1))    

         #calculate probability of target given selected context
        inputs_ids=torch.cat((context_ids,event_ids,target_ids),-1)
        hidden_states=self.decoder(inputs_ids)[0][:,-target_ids.size(1):]        
        lm_logits = self.lsm(self.lm_head(hidden_states))            
        labels_logits=lm_logits[:,:-1].gather(-1,target_ids[:,1:].unsqueeze(-1))[:,:,0]
        active_loss=target_ids.ne(0)[:,1:]
        sm_prob=labels_logits*active_loss.float()     
        prob_2=torch.exp(sm_prob.sum(1))  


        # calculate target loss
        active_loss = target_ids[..., 1:].ne(0).view(-1) == 1
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],shift_labels.view(-1)[active_loss])
        #calculate context loss

        reward=(prob_2>prob_1)
        reward=reward.float()*2-1

        context_loss=torch.mean(reward*context_loss)
        
        outputs = (loss,context_loss,reward.mean()),loss*active_loss.sum(),active_loss.sum()
        return outputs
    
    def gen(self,event_ids,context_ids,prior,topk):
        preds,scores=[],[]
        zero=torch.cuda.LongTensor(1).fill_(0) 
        prob,topk_id=prior.topk(topk,-1)    
        context_ids_all=context_ids
        for i in range(event_ids.shape[0]):  
            beam = Beam(self.beam_size,self.sos_id,self.eos_id,prob[i])
            z=self.codebook.weight[topk_id[i]]
            
            ############################################################################    
            #calculate hidden state of evidences
            context_ids=context_ids_all[i:i+1].repeat(topk,1,1)
            bs,cx,le=context_ids.shape
            context_ids=context_ids.view(-1,le)
            c=self.encoder(context_ids)[0][:,-1].view(bs,cx,-1)
            
            #obtain evidence selected by latent variable
            distances = ((c-z[:,None,:])**2).sum(-1)      
            encoding_indices = torch.argmin(distances, dim=1)
            encodings = F.one_hot(encoding_indices,cx)
            context_ids=(context_ids.view(bs,cx,le)*encodings[:,:,None].long()).sum(1)
            ############################################################################    
            #concatenate evidence and event to obtain hidden states
            inputs_ids=torch.cat((context_ids,event_ids[i:i+1].repeat(topk,1)),-1)
            transformer_outputs=self.decoder(inputs_ids)
            past_x=[x.repeat(1,self.beam_size,1,1,1) for x in transformer_outputs[1]]
            z=z.repeat(self.beam_size,1)
            ############################################################################  
            
            #beam search
            input_ids=None
            for _ in range(self.max_length-1):
                if beam.done():
                    break
                if input_ids is None:
                    input_ids=beam.getCurrentState()
                else:
                    input_ids=torch.cat((input_ids,beam.getCurrentState()),-1)

                target_ids=input_ids.unsqueeze(1).repeat(1,topk,1).view(-1,input_ids.shape[-1])
                transformer_outputs=self.decoder(target_ids,past=past_x)
                hidden_states = transformer_outputs[0]   

                out = self.lsm(self.lm_head(hidden_states[:,-1,:])).data
                out = out.view(-1,topk,out.shape[-1])
                beam.advance(out)
                input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
            hyp= beam.getHyp(beam.getFinal())
            pred=beam.buildTargetTokens(hyp)[:10]
            pred=[torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
            preds.append(torch.cat(pred,0).unsqueeze(0))
        preds=torch.cat(preds,0)                
        return preds       
        
         
