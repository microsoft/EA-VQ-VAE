# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
class Model(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
    """
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
        
    def forward(self, event_ids,prior=None):   
        """
            Forward the prior distribution estimator model.
            Parameters:
            * `event_ids`- event ids of examples
            * `prioir`- prioir distribution function p(z|x,y). 
        """        
        logits=self.encoder(event_ids,attention_mask=event_ids.ne(1))[0]
        probs =F.softmax(logits,-1)
        if prior is not None:
            prior=prior.float()
            prior=prior/(prior.sum(-1)+1e-7)[:,None]
            loss=prior*torch.log(prior+1e-7)-prior*torch.log(probs+1e-7)
            loss=loss.sum(-1).mean()
            return loss
        else:
            return probs
         
