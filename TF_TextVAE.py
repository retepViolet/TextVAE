import tensorflow as tf
from transformers import TFGPT2Model, GPT2Config


class TextVAE(tf.keras.Model):
    gpt2_config = GPT2Config.from_pretrained("distilgpt2")
    gpt2_config.attn_pdrop = 0
    gpt2_config.embd_pdrop = 0
    gpt2_config.resid_pdrop = 0
    gpt = TFGPT2Model.from_pretrained("distilgpt2", config = gpt2_config)
    gpt.trainable = False
    h = gpt.transformer.h[-1]


    @staticmethod
    def gpt_forward(input_ids, mask, bias):
        hidden_states = TextVAE.gpt(input_ids, attention_mask = mask,
                                    output_hidden_states = True).hidden_states[-2]
        residual = hidden_states
        hidden_states = TextVAE.h.ln_1(hidden_states)
        attn_output = TextVAE.h.attn(hidden_states, None, None, None, None, False, False, False)[0]
        hidden_states = attn_output + residual
        residual = hidden_states
        hidden_states = TextVAE.h.ln_2(hidden_states)
        hidden_states = TextVAE.h.mlp.c_fc(hidden_states)
        hidden_states += tf.expand_dims(bias, axis=1)
        hidden_states = TextVAE.h.mlp.act(hidden_states)
        hidden_states = TextVAE.h.mlp.c_proj(hidden_states)
        hidden_states = TextVAE.h.mlp.dropout(hidden_states)
        hidden_states = TextVAE.gpt.transformer.ln_f(hidden_states + residual)
        logits = tf.matmul(hidden_states, TextVAE.gpt.transformer.wte.weights, transpose_b=True)
        return logits


    class MLP(tf.keras.layers.Layer):
        def __init__(self):
            super(TextVAE.MLP, self).__init__()
            self.ffn = tf.keras.Sequential([
                tf.keras.layers.Dense(768, activation="gelu"),
                tf.keras.layers.Dense(768 * 4)
            ])
            self.norm = tf.keras.layers.LayerNormalization()

        def call(self, x):
            return self.norm(self.ffn(x) + x)


    def __init__(self):
        super(TextVAE, self).__init__()
        self.gpt = TFGPT2Model.from_pretrained("distilgpt2", config = TextVAE.gpt2_config)
        self.h = self.gpt.transformer.h[-1]
        self.amplifier = tf.keras.layers.Dense(768 * 8)
        self.decoder = tf.keras.Sequential([TextVAE.MLP() for _ in range(6)])
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")


    def get_masked_average(self, vectors, mask):
        if mask is None:
            return tf.reduce_mean(vectors, axis=1)
        sum_vectors = tf.reduce_sum(vectors * tf.expand_dims(mask, axis=-1), axis=1)
        count = tf.reduce_sum(mask, axis=1, keepdims=True)
        return sum_vectors / count


    def get_hidden_states(self, input_ids, mask):
        hidden_states = self.gpt(input_ids, attention_mask = mask,
                                 output_hidden_states = True).hidden_states[-2]
        residual = hidden_states
        hidden_states = self.h.ln_1(hidden_states)
        attn_output = self.h.attn(hidden_states, None, None, None, None, False, False, True)[0]
        hidden_states = attn_output + residual
        hidden_states = self.h.ln_2(hidden_states)
        hidden_states = self.h.mlp.c_fc(hidden_states)
        return hidden_states


    def get_bias(self, input_ids, mask, do_sample):
        # hidden_states = self.get_hidden_states(input_ids, mask)
        # hidden_states = self.get_masked_average(hidden_states, mask)
        # print(hidden_states)
        # return hidden_states, 0
        hidden_states = self.get_hidden_states(input_ids, mask)
        hidden_states = self.amplifier(hidden_states)
        hidden_states = self.get_masked_average(hidden_states, mask)
        mu, logvar = tf.split(hidden_states, num_or_size_splits=2, axis=-1)
        kl_loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(logvar) - logvar - 1)
        if do_sample:
            mu += tf.exp(0.5 * logvar) * tf.random.normal(shape=tf.shape(mu))
        bias = self.decoder(mu)
        return bias, kl_loss / tf.cast(tf.shape(input_ids)[0], tf.float32)


    def call(self, inputs, training = True):
        input_ids, mask = inputs['input_ids'], tf.cast(inputs['attention_mask'], tf.float32)
        bias, kl_loss = self.get_bias(input_ids, mask, False)
        logits = TextVAE.gpt_forward(input_ids[:, :-1], mask[:, :-1] if mask is not None else None, bias)
        loss = self.loss_fn(input_ids[:, 1:], logits)
        if mask is not None: loss = tf.reduce_sum(loss * mask[:, 1:]) / tf.reduce_sum(mask[:, 1:])
        else: loss = tf.reduce_mean(loss)
        if tf.math.reduce_any(tf.math.is_nan(loss + kl_loss)): 
          loss = tf.zeros_like(loss)
          kl_loss = tf.zeros_like(kl_loss)
        return loss #, 0.01 * kl_loss



if __name__ == '__main__':
    # gpt = TFGPT2LMHeadModel.from_pretrained('distilgpt2')
    # input_ids = tf.constant([[1, 2, 3]], dtype=tf.int32)
    # mask = tf.constant([[1, 1, 0]], dtype=tf.float32)
    # logits1 = gpt(input_ids, attention_mask = mask).logits * tf.expand_dims(mask, axis=-1)
    # bias = tf.zeros((1, 768 * 4), dtype=tf.float32)
    # logits2 = TextVAE.gpt_forward(input_ids, mask, bias) * tf.expand_dims(mask, axis=-1)
    # print(tf.reduce_sum(tf.abs(logits1 - logits2)))

    input_ids = tf.constant([[1, 2, 3]], dtype=tf.int32)
    mask = tf.constant([[1, 1, 0]], dtype=tf.float32)
    model = TextVAE()
    model({'input_ids':input_ids, 'attention_mask':mask})