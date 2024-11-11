import torch
from torch import nn, tensor, randn, randn_like
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


class BiasGPT(nn.Module):
    gpt = GPT2LMHeadModel.from_pretrained('distilgpt2')
    for param in gpt.parameters():
        param.requires_grad = False
    @staticmethod
    def train_bias(input_ids, mask = None, epochs = 50, lr = 0.25, log = False):
        # 准备
        if mask is not None: loss_mask = mask[:,1:].view(-1)
        labels = input_ids[:,1:].view(-1)
        # 定义模型
        biasGPT = BiasGPT(len(input_ids))
        biasGPT.train()
        with torch.no_grad():
            hidden_states = BiasGPT.gpt(input_ids[:,:-1], output_hidden_states = True).hidden_states[-3]
            hidden_states.detach_()
        # 优化器
        optimizer = torch.optim.AdamW(biasGPT.parameters(), lr = lr, weight_decay = 0.5)
        scheduler = CosineAnnealingLR(optimizer, T_max = epochs, eta_min = 0)
        # 训练
        # BiasGPT.gpt.train()
        for i in range(epochs):
            logits = biasGPT(hidden_states)
            loss = nn.functional.cross_entropy(logits.view(-1, 50257), labels, reduction='none')
            if mask is not None: loss = loss * loss_mask
            loss = torch.mean(loss)
            kl_loss = torch.mean(biasGPT.bias_mu.pow(2)) # + torch.exp(biasGPT.bias_logvar) - biasGPT.bias_logvar - 1)
            if log: print(f"Epoch: {i+1}; Loss: {round(loss.item(), 3), round(kl_loss.item(), 3)}; lr: {scheduler.get_last_lr()[0]}")
            (loss + kl_loss * 0.0).backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        return biasGPT


    def __init__(self, batch_size = 1):
        super().__init__()
        self.batch_size = batch_size
        self.blocks = [BiasGPT.gpt.transformer.h[-2], BiasGPT.gpt.transformer.h[-1]]
        self.bias_mu = torch.nn.Parameter(torch.zeros(2, batch_size, 1, 3072))
        # self.bias_logvar = torch.nn.Parameter(torch.randn(2, batch_size, 1, 3072))
    

    def set_bias_to(self, gpt, batch_id, do_sample = False):
        with torch.no_grad():
            gpt.transformer.h[-2].mlp.c_fc.bias += self.get_bias(0, do_sample)[batch_id, 0]
            gpt.transformer.h[-1].mlp.c_fc.bias += self.get_bias(1, do_sample)[batch_id, 0]


    def get_bias(self, layer, do_sample = False):
        bias = self.bias_mu[layer]
        if do_sample:
            # std = torch.exp(0.5 * self.bias_logvar[layer]) 
            eps = randn_like(self.bias_mu[layer])
            # bias = torch.nn.functional.dropout(bias, 0.1)
        # ratio = BiasGPT.gpt.transformer.h[layer-2].mlp.c_fc.bias.std().pow(0.5)
        return bias # * self.ratio


    def mlp(self, layer, hidden_states):
        hidden_states = self.blocks[layer].mlp.c_fc(hidden_states)
        hidden_states = hidden_states + self.get_bias(layer, do_sample = True)
        hidden_states = self.blocks[layer].mlp.act(hidden_states)
        hidden_states = self.blocks[layer].mlp.c_proj(hidden_states)
        hidden_states = self.blocks[layer].mlp.dropout(hidden_states)
        return hidden_states


    def decoder(self, layer, hidden_states, mask):
        residual = hidden_states
        hidden_states = self.blocks[layer].ln_1(hidden_states)
        attn_outputs = self.blocks[layer].attn(
            hidden_states,
            attention_mask = mask,
        )
        attn_output = attn_outputs[0]
        hidden_states = attn_output + residual
        residual = hidden_states
        hidden_states = self.blocks[layer].ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(layer, hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        return hidden_states
    

    def forward(self, hidden_states, mask = None):
        hidden_states = hidden_states + torch.randn_like(hidden_states) * 1
        hidden_states = self.decoder(0, hidden_states, mask)
        hidden_states = self.decoder(1, hidden_states, mask)
        logits = BiasGPT.gpt.lm_head(BiasGPT.gpt.transformer.ln_f(hidden_states))
        return logits




if __name__ == '__main__':
    gpt = GPT2LMHeadModel.from_pretrained('distilgpt2')
    model = BiasGPT(2)
    input_ids = tensor([[1,2,3],[1,1,111]])
    output = gpt(input_ids, output_hidden_states = True)
    logits1 = output.logits
    hidden_states = output.hidden_states[-3]
    logits2 = model(hidden_states)
    print((logits1 - logits2).mean().item())
