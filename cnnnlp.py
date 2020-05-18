import os
import tensorflow as tf
import tensorflow.contrib.keras as kr
import numpy as np
from collections import Counter

from sklearn import metrics

base_dir = 'data\\cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.jion()
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径

class TCNNConfig(object):
    """CNN配置参数"""
    embedding_dim = 64  # 字的特征是64
    seq_length = 600  # 序列长度    句子长度600
    num_classes = 10  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 500  # 字数
    hidden_dim = 128  # 全连接层神经元
    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率0.001
    batch_size = 64  # 每批训练大小
    num_epochs = 10000  # 总迭代轮次
    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

class TextCNN(object):

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据

        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')#句子长度(句子数,600)
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')#标签类别(1,10)
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')#设置的dropout

        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 字向量映射
        with tf.device('/cpu:0'):  # 5000行64列代表一个字
            embedding = tf.get_variable('embedding',
                                        [self.config.vocab_size, self.config.embedding_dim])  # (5000,64)5000个字
            embeding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
            # 选取一个张量里面索引对应的元素 shape=(句子数, 600, 64)
        with tf.name_scope('cnn'):
            conv = tf.layers.conv1d(embeding_inputs, self.num_filters, self.kernel_size, name= "conv")
            gmp = tf.reduce_max(conv , reduction_indices=[1], name = "gmp")

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活gmp输入的数据，hidden_dim输出的维度大小
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')#shape=(64, 128),64为batch_size
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)#shape=(64, 128)
            fc = tf.nn.relu(fc)#shape=(64, 128)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')  # shape=(?, 10)
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别 shape=(?,)按列取

        with tf.name_scope("optimize"):
            #损失函数
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits= self.logits, labels= self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

            #优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(self.input_y , self.y_pred_cls)  # shape=(?,)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # shape=()

def main():
    print('this message is from main function')

def open_file(filename, mode='r'):
    return open(filename, mode, encoding='utf-8', errors='ignore')

def read_file(train_dir):
    contents, labels = [], []
    with open_file(train_dir, 'r') as f:
        for line in f:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(content))
                    labels.append(list(label))
        return contents, labels

def build_vocab():
    train_dir = os.path.join(base_dir, 'cnews.train.txt')
    data_train,_ = read_file(train_dir)  # 从训练集中读取数据
    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data);
    count_pairs = counter.most_common(5000)
    words, _ = list(zip(*count_pairs))
    words = ['<PAD>'] +list(words)
    open_file(vocab_dir, 'w').write('\n'.join(words) + '\n')

def read_category():
    """读取分类目录，固定"""
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    categories = [x for x in categories]#编码
    cat_to_id = dict(zip(categories, range(len(categories))))#{'体育': 0, '财经': 1, '房产': 2, '家居': 3, '教育': 4, '科技': 5, '时尚': 6, '时政': 7, '游戏': 8, '娱乐': 9}
    return categories, cat_to_id

def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def process_file(filename, word_to_id, cat_to_id, seq_length):
    contents, labels = read_file(val_dir)
    data_id, label_id = [], []

    for i in range(len(contents)):  # i 是一句话 word_to_id{'马':387} cat_to_id{'体育': 0, '财经': 1}
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

        # 使用keras提供的pad_sequences来将文本pad为固定长度 序列处理 长度为600 不足补0    data_id50000 个句子，每个句子对应的字id
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, seq_length)  # 返回的是二维的tenson
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示
    # label_id: 类别向量
    # num_classes:总共类别数
    # 将类别向量(从0到nb_classes的整数向量)映射为二值类别矩阵, 用于应用到以categorical_crossentropy为目标函数的模型中.
    return x_pad, y_pad


def batch_iter(x_train, y_train, batch_size):
    data_len = len(x_train)
    num_batch = int((data_len) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x_train[indices]
    y_shuffle = y_train[indices]

     #yield迭代器节约内存
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict

def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len

def trainModel():
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)  # 能够保存训练过程以及参数分布图并在tensorboard显示
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    # merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。如果没有特殊要求，一般用这一句就可一显示训练时的各种信息了。
    writer = tf.summary.FileWriter(tensorboard_dir)
    # 指定一个文件用来保存图。

    # 配置 Saver
    saver = tf.train.Saver()  # 保存的训练好的模型。
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, TCNNConfig.seq_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, TCNNConfig.seq_length)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    for epoch in range(TCNNConfig.num_epochs):
        batch_train = batch_iter(x_train, y_train, TCNNConfig.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, TCNNConfig.dropout_keep_prob)
            if total_batch % TCNNConfig.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

                if total_batch % TCNNConfig.print_per_batch == 0:
                    # 每多少轮次输出在训练集和验证集上的性能
                    feed_dict[model.keep_prob] = 1.0
                    loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                    loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                    if acc_val > best_acc_val:
                        # 保存最好结果
                        best_acc_val = acc_val
                        last_improved = total_batch
                        saver.save(sess=session, save_path=save_path)
                        improved_str = '*'
                    else:
                        improved_str = ''

                session.run(model.optim, feed_dict=feed_dict)  # 运行优化
                total_batch += 1

                if total_batch - last_improved > require_improvement:
                    # 验证集正确率长期不提升，提前结束训练
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break  # 跳出循环
            if flag:  # 同上
                break


def testDate():
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, TCNNConfig.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型
    loss_test, acc_test = evaluate(session, x_test, y_test)

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

        # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

if __name__ == '__main__':
    
    cnnconfig = TCNNConfig()
    main()
    vocab_dir = os.path.join(base_dir, 'vocab1.txt')
    if not os.path.exists(vocab_dir):
        build_vocab();
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)

    cnnconfig.vocab_size = len(words)
    model = TextCNN(cnnconfig)

    trainModel()
    testDate()




