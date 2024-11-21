import tensorflow as tf
from tensorflow.keras.optimizers import AdamW # type: ignore
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from TF_TextVAE import TextVAE
import argparse, math


# 环境设置
# TPU
# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.TPUStrategy(resolver)
# GPU
strategy = tf.distribute.MirroredStrategy()


# 命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--size', type = int, default = 1000)
parser.add_argument('--dataset', type = str, default = '20220301.simple')
parser.add_argument('--ratio', type=float, default = 1)
# 模型结构
parser.add_argument('--dim', type=int, default = 768*16)
parser.add_argument('--max_length', type=int, default = 128)
# 训练参数
parser.add_argument('--epoch', type=int, default = 1)
parser.add_argument('--lr', type=float, default = 5e-4)
parser.add_argument('--warmup', type=int, default = 1000)
parser.add_argument('--weight_decay', type=float, default = 0)
parser.add_argument('--dropout', type=float, default = 0)
parser.add_argument('--batch_size', type=int, default = 128)
parser.add_argument('--save_path', type=str, default = './result/TextVAE.weights.h5')
parser.add_argument('--validation_freq', type=int, default = 1)
# 测试参数
parser.add_argument('--test_num', type=int, default = 5)
args = parser.parse_args()


# 处理数据
wiki = load_dataset('wikipedia', args.dataset,
          trust_remote_code = True, split = 'train')
wiki_text = wiki.select(range(args.size))

tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
def mapping(x):
  text = ['\n' + text.replace('\n',' ') for text in x['text']]
  return tokenizer(text, return_tensors="tf", padding = 'max_length',
           max_length = 128, truncation = True)
dataset = wiki_text.map(mapping, batched = True, remove_columns = ['id','url','title','text'],
             load_from_cache_file = True) #, cache_file_name = './cache/'+args.dataset)
# dataset = load_dataset("arrow", data_files = './cache/'+args.dataset, split = 'train')
print("-" * 50)
print(dataset)


# 统计信息
val_size = dataset.num_rows // 10
train_size = dataset.num_rows - val_size
steps_per_epoch = math.ceil(train_size / args.batch_size * args.ratio)
tot_step = steps_per_epoch * args.epoch
print(f'train_size: {train_size}, val_size: {val_size}')
print(f'warmup: {args.warmup}, tot_step: {tot_step}, step_per_epoch: {steps_per_epoch}')
print("-" * 50)


# 数据打包
def generator():
  for example in dataset:
    yield (example, tf.convert_to_tensor([1], dtype = tf.int32))
tf_dataset = tf.data.Dataset.from_generator(
  generator, output_signature=({
    'input_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32),
    'attention_mask': tf.TensorSpec(shape=(None,), dtype=tf.float32),
  }, tf.TensorSpec(shape=(None,), dtype=tf.int32))
)
trainData = (tf_dataset
        .take(train_size)
        .shuffle(100000)
        .repeat()
        .batch(args.batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE))
evalData = (tf_dataset
        .skip(train_size)
        .batch(args.batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE))


# 模型训练
lr = tf.keras.optimizers.schedules.CosineDecay(  # lr scheduler
  name = 'CosineDecay',
  initial_learning_rate = 0.0,
  warmup_target = args.lr,
  warmup_steps = args.warmup,
  decay_steps = tot_step - args.warmup,
  alpha = 0.0,
)
checkpoint = tf.keras.callbacks.ModelCheckpoint(  # save
  filepath = args.save_path,
  monitor = 'val_loss',
  save_best_only = True,
  save_weights_only = True,
  mode = 'min',
  verbose = 1
)
with strategy.scope():  # tpu strategy
  model = TextVAE()
  # model.load_weights('result/TextVAE..weights.h5')
  model.compile(
    optimizer = AdamW(lr, weight_decay = args.weight_decay, clipnorm = 1.0),
    loss = lambda y_true, y_pred: y_pred
  )
model.fit(  # train
  trainData, 
  validation_data = evalData, 
  epochs = args.epoch,
  steps_per_epoch = steps_per_epoch,
  validation_freq = args.validation_freq,
  callbacks = [checkpoint]
)


# 进行测试
# def test(data_item):
#   logits = model(data_item[0], training = False)[2]
#   predicted_ids = tf.argmax(logits, axis=-1)
#   pred_text = tokenizer.batch_decode(predicted_ids)
#   targ_text = tokenizer.batch_decode(data_item[0]['input_ids'], skip_special_tokens=True)
#   for i in range(args.test_num):
#     print("pred: ", pred_text[i].replace('\n','\\n'))
#     print("targ: ", targ_text[i].replace('\n','\\n'))
#     print("-" * 50)
# print("\nTest on Training Data: ")
# test(next(trainData.take(1).as_numpy_iterator()))
# print("\nTest on Evaluation Data: ")
# test(next(evalData.take(1).as_numpy_iterator()))