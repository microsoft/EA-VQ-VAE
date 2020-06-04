# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss

class CodeBook(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(CodeBook, self).__init__()  
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings     
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)      
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # Calculate distances
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).cuda()
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight)
        
        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        q_latent_loss = torch.mean((quantized - inputs.detach())**2)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized, perplexity, encodings
    

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
        self.lm_head.weight=self.decoder.wte.weight     
    
    def forward(self, event_ids,target_ids):   
        """
            Forward the VQ-VAE model.
            Parameters:
            * `event_ids`- event ids of examples
            * `target_ids`- target ids of examples
        """  
        input_ids=torch.cat((event_ids,target_ids),-1)
        #obtain hidden of event+target by encoder
        hidden_xy=self.encoder(input_ids,special=True)[0][:,-1]

        #obtain latent variable z by coodebook
        vae_loss, z, perplexity, encoding=self.codebook(hidden_xy)

        #obtain hiddens of target 
        transformer_outputs=self.decoder(input_ids,z=z)
        hidden_states = transformer_outputs[0][:,-target_ids.size(1):]

        #calculate loss
        lm_logits = self.lm_head(hidden_states+z[:,None,:])
        # Shift so that tokens < n predict n
        active_loss = target_ids[..., 1:].ne(0).view(-1) == 1
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                        shift_labels.view(-1)[active_loss])

        outputs = (loss,vae_loss,perplexity),loss*active_loss.sum(),active_loss.sum(),encoding
        return outputs            
         
