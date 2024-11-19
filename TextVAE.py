import torch
from torch import nn, tensor
from transformers import GPT2Config, GPT2LMHeadModel


class TextVAE(nn.Module):
    gpt2_config = GPT2Config.from_pretrained("distilgpt2")
    gpt2_config.attn_pdrop = 0
    gpt2_config.embd_pdrop = 0
    gpt2_config.resid_pdrop = 0
    gpt = GPT2LMHeadModel.from_pretrained('distilgpt2', config=gpt2_config)
    for param in gpt.parameters():
        param.requires_grad = False
    h = gpt.transformer.h[-1]

    @staticmethod
    def gpt_forward(input_ids, mask, bias):
        hidden_states = TextVAE.gpt(input_ids, attention_mask = mask, 
                                    output_hidden_states = True).hidden_states[-2]
        residual = hidden_states
        hidden_states = TextVAE.h.ln_1(hidden_states)
        attn_output = TextVAE.h.attn(hidden_states)[0]
        hidden_states = attn_output + residual
        residual = hidden_states
        hidden_states = TextVAE.h.ln_2(hidden_states)
        hidden_states = TextVAE.h.mlp.c_fc(hidden_states)
        hidden_states = hidden_states + bias.unsqueeze(1)
        hidden_states = TextVAE.h.mlp.act(hidden_states)
        hidden_states = TextVAE.h.mlp.c_proj(hidden_states)
        hidden_states = TextVAE.h.mlp.dropout(hidden_states)
        hidden_states = residual + hidden_states
        logits = TextVAE.gpt.lm_head(TextVAE.gpt.transformer.ln_f(hidden_states))
        return logits
    
    class mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.ffn = nn.Sequential(
            nn.Linear(768*4, 768),
            nn.GELU(),
            nn.Linear(768, 768*4)
            )
            self.norm = nn.LayerNorm(768*4)
        def forward(self, x):
            x = self.ffn(x) + x
            return self.norm(x)
    

    def __init__(self):
        super().__init__()
        self.gpt = GPT2LMHeadModel.from_pretrained('distilgpt2', config = TextVAE.gpt2_config)
        self.h = self.gpt.transformer.h[-1]
        self.amplifier = nn.Linear(768*4, 768 * 8)
        self.decoder = nn.Sequential()
        for _ in range(6):
          self.decoder.append(TextVAE.mlp())
    

    def get_masked_average(self, vectors, mask):
        if mask is None: 
            return vectors.mean(dim=1)
        sum = (vectors * mask.unsqueeze(-1)).sum(dim=1)
        cnt = mask.sum(dim = 1, keepdim = True)
        return sum / cnt
    

    def get_hidden_states(self, input_ids, mask):
        hidden_states = self.gpt(input_ids, attention_mask = mask, 
                                 output_hidden_states = True).hidden_states[-2]
        residual = hidden_states
        hidden_states = self.h.ln_1(hidden_states)
        attn_output = self.h.attn(hidden_states)[0]
        hidden_states = attn_output + residual
        residual = hidden_states
        hidden_states = self.h.ln_2(hidden_states)
        hidden_states = self.h.mlp.c_fc(hidden_states)
        return hidden_states


    def get_bias(self, input_ids, mask, do_sample = True):
        # gpt
        hidden_states = self.get_hidden_states(input_ids, mask)
        hidden_states = self.amplifier(hidden_states)
        hidden_states = self.get_masked_average(hidden_states, mask)
        # sampling
        mu, logvar = hidden_states[:,:768*4], hidden_states[:,768*4:]
        kl_loss = 0.5 * (mu.pow(2) + torch.exp(logvar) - logvar - 1).sum()
        if do_sample:
            mu = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        # decoder
        bias = self.decoder(mu)
        return bias, kl_loss/len(input_ids)
    

    def forward(self, gpt_ids, gpt_mask = None, do_sample = True):
        # prepare
        device = next(self.gpt.parameters()).device
        gpt_ids, gpt_mask = gpt_ids.to(device) if gpt_ids is not None else None, gpt_mask.to(device) if gpt_mask is not None else None
        # get bias
        bias, kl_loss = self.get_bias(gpt_ids, gpt_mask, do_sample)
        logits = self.gpt_forward(gpt_ids[:,:-1], gpt_mask[:,:-1] if gpt_mask is not None else None, bias)
        # loss
        loss = nn.functional.cross_entropy(logits.view(-1, 50257), gpt_ids[:,1:].contiguous().view(-1), 
                                           reduction = 'none')
        loss = torch.mean(loss * gpt_mask[:,1:].contiguous().view(-1))
        print((loss).item(), kl_loss.item())
        return (loss + 0.1 * kl_loss, bias, logits)




if __name__ == '__main__':
    gpt = GPT2LMHeadModel.from_pretrained('distilgpt2')
    input_ids = tensor([[1,2,3]])
    mask = tensor([[1,1,0]])
    logits1 = gpt(input_ids, attention_mask = mask)[0] * mask.unsqueeze(-1)
    logits2 = TextVAE.gpt_forward(input_ids, mask, torch.zeros(1, 768*4)) * mask.unsqueeze(-1)
    print((logits1 - logits2).abs().sum().item())