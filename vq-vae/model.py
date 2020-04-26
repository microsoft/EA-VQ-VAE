# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss
from codebook import CodeBook
class Model(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. 2-layer transformer
        * `decoder`- decoder of seq2seq model. e.g. GPT2
        * `config`- configuration of encoder model. 
        * `args`- arguments.
    """
    def __init__(self, encoder,decoder,config,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder=decoder
        self.config=config
        self.args=args
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.codebook = CodeBook(args.z_size, config.n_embd,0.25)  
        self.codebook._embedding.weight.data.normal_(mean=0,std=0.1)
        self.lsm = nn.LogSoftmax(dim=-1)
        
        self.tie_weights()
        
    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight
                  
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.decoder.transformer.wte)        
    
    def embeddings(self, input_ids,z):
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long, device=input_ids.device)
        input_embeds=self.decoder.transformer.wte(input_ids)
        position_embeds=self.decoder.transformer.wpe(position_ids)       
        input_embeds=input_embeds+position_embeds.unsqueeze(0)+z.unsqueeze(1)     
        return input_embeds
        
    def forward(self, event_ids,target_ids):   
        """
            Forward the VQ-VAE model.
            Parameters:
            * `event_ids`- event ids of examples
            * `target_ids`- target ids of examples
        """  
        input_ids=torch.cat((event_ids,target_ids),-1)
        #obtain hidden of event+target by encoder
        hidden_xy=self.encoder.transformer(input_ids)[0][:,-1]

        #obtain latent variable z by coodebook
        vae_loss, z, perplexity, encoding=self.codebook(hidden_xy)

        #obtain hiddens of target 
        embeds=self.embeddings(input_ids,z=z)
        transformer_outputs=self.decoder.transformer(inputs_embeds=embeds)
        hidden_states = transformer_outputs[0][:,-target_ids.size(1):]

        #calculate loss
        lm_logits = self.lm_head(hidden_states)
        # Shift so that tokens < n predict n
        active_loss = target_ids[..., 1:-1].ne(0).view(-1) == 1
        shift_logits = lm_logits[..., :-2, :].contiguous()
        shift_labels = target_ids[..., 1:-1].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                        shift_labels.view(-1)[active_loss])

        outputs = (loss,vae_loss,perplexity),loss*active_loss.sum(),active_loss.sum(),encoding
        return outputs            
         
