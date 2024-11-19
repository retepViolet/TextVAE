import torch
from torch import nn, tensor
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel, DistilBertModel, DistilBertConfig

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


class TextVAE(nn.Module):
    # bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
    # for param in bert.parameters():
    #     param.requires_grad = False

    
    def __init__(self):
        super().__init__()
        gpt2_config = GPT2Config.from_pretrained("distilgpt2")
        gpt2_config.attn_pdrop = 0
        gpt2_config.embd_pdrop = 0
        gpt2_config.resid_pdrop = 0
        self.gpt = GPT2LMHeadModel.from_pretrained('distilgpt2', config=gpt2_config)
        # self.gpt.train()
        for param in self.gpt.parameters():
          param.requires_grad = False
        self.h = [self.gpt.transformer.h[-2], self.gpt.transformer.h[-1]]
        # for i in self.h:
        #   for param in i.parameters():
        #     i.requires_grad = True
        # bert_config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
        # bert_config.attn_pdrop = 0
        # bert_config.embd_pdrop = 0
        # bert_config.resid_pdrop = 0
        # self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased', config=bert_config)
        self.bert = GPT2LMHeadModel.from_pretrained('distilgpt2', config=gpt2_config)
        self.amplifier = nn.Linear(768*4, 768 * 8)
        self.decoder = nn.Sequential()
        for _ in range(6):
          self.decoder.append(mlp())
    

    def get_masked_average(self, vectors, mask):
        if mask is None: 
            return vectors.mean(dim=1)
        sum = (vectors * mask.unsqueeze(-1)).sum(dim=1)
        cnt = mask.sum(dim = 1, keepdim = True)
        return sum / cnt
    

    def get_gpt_bias(self, input_ids, mask, do_sample = True):
      bias = None
      hidden_states = self.bert(input_ids, attention_mask = mask, output_hidden_states = True).hidden_states[-2]
      for layer in [-1]:
          residual = hidden_states
          hidden_states = self.bert.transformer.h[layer].ln_1(hidden_states)
          attn_outputs = self.bert.transformer.h[layer].attn(hidden_states)
          attn_output = attn_outputs[0]
          hidden_states = attn_output + residual
          residual = hidden_states
          hidden_states = self.bert.transformer.h[layer].ln_2(hidden_states)
          feed_forward_hidden_states = self.h[layer].mlp.c_fc(hidden_states)
          # feed_forward_hidden_states = self.h[layer].mlp.act(feed_forward_hidden_states)
          # feed_forward_hidden_states = self.h[layer].mlp.c_proj(feed_forward_hidden_states)
          # feed_forward_hidden_states = self.h[layer].mlp.dropout(feed_forward_hidden_states)
          # hidden_states = residual + feed_forward_hidden_states
      hidden_states = self.amplifier(feed_forward_hidden_states)
      hidden_states = self.get_masked_average(hidden_states, mask)
      mu, logvar = hidden_states[:,:768*4], hidden_states[:,768*4:]
      kl_loss = 0.5 * (mu.pow(2) + torch.exp(logvar) - logvar - 1).sum()
      # if do_sample:
      #     mu = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        # bias = mu.view(-1, 2, 768*4) # (self.decoder(mu)).view(-1, 2, 768*4)
      bias = mu #self.decoder(mu)
      return bias, 0*kl_loss/len(input_ids)


    def get_bias(self, input_ids, mask, do_sample):
        hidden_states = self.bert(input_ids, attention_mask = mask, output_hidden_states = True).hidden_states[-1]
        hidden_states = self.amplifier(hidden_states)
        hidden_states = self.get_masked_average(hidden_states, mask)
        mu, logvar = hidden_states[:,:768*4], hidden_states[:,768*4:]
        kl_loss = 0.5 * (mu.pow(2) + torch.exp(logvar) - logvar - 1).sum()
        # if do_sample:
        #     mu = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        # bias = mu.view(-1, 2, 768*4) # (self.decoder(mu)).view(-1, 2, 768*4)
        bias = self.decoder(mu)
        return bias, kl_loss/len(input_ids)


    def mlp(self, layer, hidden_states, bias):
        hidden_states = self.h[layer].mlp.c_fc(hidden_states)
        hidden_states = hidden_states + bias.unsqueeze(1)
        hidden_states = self.h[layer].mlp.act(hidden_states)
        hidden_states = self.h[layer].mlp.c_proj(hidden_states)
        hidden_states = self.h[layer].mlp.dropout(hidden_states)
        return hidden_states


    def gpt_forward(self, input_ids, mask, bias):
      # input_embeds = self.gpt.transformer.wte(input_ids)
      # seq_length = input_ids.size(1)
      # position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
      # position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
      # position_embeds = self.gpt.transformer.wpe(position_ids)
      # hidden_states = input_embeds + position_embeds
      # for layer in self.gpt.transformer.h:
      #     residual = hidden_states
      #     hidden_states = layer.ln_1(hidden_states)
      #     attn_outputs = layer.attn(hidden_states)
      #     attn_output = attn_outputs[0]
      #     hidden_states = attn_output + residual
      #     residual = hidden_states
      #     hidden_states = layer.ln_2(hidden_states)
      #     hidden_states = layer.mlp.c_fc(hidden_states)
      #     hidden_states = hidden_states + bias.unsqueeze(1)
      #     hidden_states = layer.mlp.act(hidden_states)
      #     hidden_states = layer.mlp.c_proj(hidden_states)
      #     hidden_states = layer.mlp.dropout(hidden_states)
      #     hidden_states = residual + hidden_states
      # logits = self.gpt.lm_head(self.gpt.transformer.ln_f(hidden_states))
      # return logits
        hidden_states = self.gpt(input_ids, attention_mask = mask, output_hidden_states = True).hidden_states[-2]
        for layer in [1]:
            residual = hidden_states
            hidden_states = self.h[layer].ln_1(hidden_states)
            attn_outputs = self.h[layer].attn(hidden_states)
            attn_output = attn_outputs[0]
            hidden_states = attn_output + residual
            residual = hidden_states
            hidden_states = self.h[layer].ln_2(hidden_states)
            feed_forward_hidden_states = self.mlp(layer, hidden_states, bias)
            hidden_states = residual + feed_forward_hidden_states
        logits = self.gpt.lm_head(self.gpt.transformer.ln_f(hidden_states))
        return logits
    

    def forward(self, bert_ids, bert_mask = None, gpt_ids = None, gpt_mask = None, do_sample = True):
        device = next(self.bert.parameters()).device
        bert_ids, bert_mask = bert_ids.to(device), bert_mask.to(device) if bert_mask is not None else None
        gpt_ids, gpt_mask = gpt_ids.to(device) if gpt_ids is not None else None, gpt_mask.to(device) if gpt_mask is not None else None
        # bias, kl_loss = self.get_bias(gpt_ids, gpt_mask, do_sample)
        bias, kl_loss = self.get_gpt_bias(gpt_ids, gpt_mask)
        if gpt_ids is not None and gpt_mask is not None:
            logits = self.gpt_forward(gpt_ids[:,:-1], gpt_mask[:,:-1] if gpt_mask is not None else None, bias)
            loss = nn.functional.cross_entropy(logits.view(-1, 50257), gpt_ids[:,1:].contiguous().view(-1), 
                                               reduction = 'none')
            loss = torch.mean(loss * gpt_mask[:,1:].contiguous().view(-1))
            print((loss).item(), kl_loss.item())
            return (loss + 0.1 * kl_loss, bias, logits)
        return bias




if __name__ == '__main__':
    gpt = GPT2LMHeadModel.from_pretrained('distilgpt2')
    gpt.eval()
    model = TextVAE()
    input_ids = tensor([[1,2,3]])
    mask = tensor([[1,1,0]])
    logits1 = gpt(input_ids, attention_mask = mask)[0] * mask.unsqueeze(-1)
    logits2 = model.gpt_forward(input_ids, mask, torch.zeros(1, 2, 768*4)) * mask.unsqueeze(-1)
    print((logits1 - logits2).abs().sum().item())