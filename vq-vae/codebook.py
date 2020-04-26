import torch
import torch.nn as nn
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


