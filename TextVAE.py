import torch
from torch import nn, tensor
from transformers import GPT2LMHeadModel, DistilBertModel


class TextVAE(nn.Module):
    gpt = GPT2LMHeadModel.from_pretrained('distilgpt2')
    gpt.train()
    for param in gpt.parameters():
        param.requires_grad = False
    blocks = [gpt.transformer.h[-2], gpt.transformer.h[-1]]

    # bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
    # for param in bert.parameters():
    #     param.requires_grad = False


    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        self.amplifier = nn.Linear(768, 768 * 2)
        self.decoder = nn.Sequential(
            nn.Linear(768, 768*2),
            nn.GELU(),
            nn.Linear(768*2, 768*4),
            nn.GELU(),
            nn.Linear(768*4, 768*8),
        )
    

    def get_masked_average(self, vectors, mask):
        if mask is None: 
            return vectors.mean(dim=1)
        sum = (vectors * mask.unsqueeze(-1)).sum(dim=1)
        cnt = mask.sum(dim = 1, keepdim = True)
        return sum / cnt
    

    def get_bias(self, input_ids, mask, do_sample):
        hidden_states = self.bert(input_ids, mask)[0]
        hidden_states = self.amplifier(hidden_states)
        hidden_states = self.get_masked_average(hidden_states, mask)
        mu, logvar = hidden_states[:,:768], hidden_states[:,768:]
        kl_loss = 0.5 * (mu.pow(2) + torch.exp(logvar) - logvar - 1).sum()
        if do_sample:
            mu = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        # print(mu.mean().item(), logvar.mean().item())
        # print(mu.std().item(), logvar.std().item())
        bias = self.decoder(mu).view(-1, 2, 768*4)
        return bias, kl_loss


    def mlp(self, layer, hidden_states, bias):
        hidden_states = TextVAE.blocks[layer].mlp.c_fc(hidden_states)
        hidden_states = hidden_states + bias
        hidden_states = TextVAE.blocks[layer].mlp.act(hidden_states)
        hidden_states = TextVAE.blocks[layer].mlp.c_proj(hidden_states)
        hidden_states = TextVAE.blocks[layer].mlp.dropout(hidden_states)
        return hidden_states


    def gpt_forward(self, input_ids, mask, bias):
        hidden_states = TextVAE.gpt(input_ids, attention_mask = mask, output_hidden_states = True).hidden_states[-3]
        for layer in [0, 1]:
            residual = hidden_states
            hidden_states = TextVAE.blocks[layer].ln_1(hidden_states)
            attn_outputs = TextVAE.blocks[layer].attn(
                hidden_states,
                attention_mask = mask,
            )
            attn_output = attn_outputs[0]
            hidden_states = attn_output + residual
            residual = hidden_states
            hidden_states = TextVAE.blocks[layer].ln_2(hidden_states)
            feed_forward_hidden_states = self.mlp(layer, hidden_states, bias[:,layer])
            hidden_states = residual + feed_forward_hidden_states
        logits = TextVAE.gpt.lm_head(TextVAE.gpt.transformer.ln_f(hidden_states))
        return logits
    

    def forward(self, bert_ids, bert_mask = None, gpt_ids = None, gpt_mask = None, do_sample = True):
        bias, kl_loss = self.get_bias(bert_ids, bert_mask, do_sample)
        if gpt_ids is not None and gpt_mask is not None:
            logits = self.gpt_forward(gpt_ids[:,:-1], gpt_mask[:,:-1], bias)
            loss = nn.functional.cross_entropy(logits.view(-1, 50257), gpt_ids[:,1:].view(-1), 
                                               reduction = 'none')
            loss = torch.sum(loss * gpt_mask[:,1:].view(-1))
            return loss, kl_loss
        return bias




if __name__ == '__main__':
    gpt = GPT2LMHeadModel.from_pretrained('distilgpt2')
    gpt.eval()
    model = TextVAE()
    input_ids = tensor([[1,2,3]])
    logits1 = gpt(input_ids)[0]
    logits2 = model(input_ids, input_ids)
    print((logits1 - logits2).abs().mean().item())