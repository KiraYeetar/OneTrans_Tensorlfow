import tensorflow as tf
from tensorflow.keras import layers


# FFN 里面想加啥就加啥吧，这里简单的固定两层
class FFNLayer(layers.Layer):
    def __init__(self, unit_1=256, unit_2=128, **kwargs):
        super(FFNLayer, self).__init__()
        self.dense_1 = layers.Dense(unit_1, activation='relu')
        self.dense_2 = layers.Dense(unit_2, activation='relu')

    def call(self, x, training=False):
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x


class CausalMaskAttention(layers.Layer):
    def __init__(self, d_model=128, num_heads=4, if_mask=True, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = self.d_model // num_heads
        self.wq = layers.Dense(self.d_model)
        self.wk = layers.Dense(self.d_model)
        self.wv = layers.Dense(self.d_model)
        self.dense = layers.Dense(self.d_model)
        self.if_mask = if_mask

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    @staticmethod
    def create_causal_mask(x, y):
        mask = tf.linalg.band_part(tf.ones((x, y)), num_lower=-1, num_upper=0)
        causal_mask = (1.0 - mask) * -1e9
        return causal_mask

    def call(self, x):
        batch_size = tf.shape(x[0])[0]
        seq_len_k = tf.shape(x[0])[1]
        seq_len_q = tf.shape(x[1])[1]

        k = self.wk(x[0])
        q = self.wq(x[1])
        v = self.wv(x[2])
        k = self.split_heads(k, batch_size)
        q = self.split_heads(q, batch_size)
        v = self.split_heads(v, batch_size)

        # (batch_size, num_heads, seq_len, seq_len)
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(self.depth, tf.float32)
        # (batch_size, num_heads, seq_len, seq_len)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # mask
        if self.if_mask:
            causal_mask = self.create_causal_mask(seq_len_q, seq_len_k)  # (seq_len, seq_len)
            causal_mask = tf.expand_dims(tf.expand_dims(causal_mask, axis=0), axis=0)
            scaled_attention_logits += causal_mask

        # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        # (batch_size, num_heads, seq_len, seq_len)
        output = tf.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len, depth)
        # (batch_size, seq_len, num_heads, depth)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        # (batch_size, seq_len, d_model)
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        # (batch_size, seq_len, d_model)
        output = self.dense(output)

        return output


class OneTransBlock(layers.Layer):
    def __init__(self, d_model=128, num_heads=4, ffn_units=(256, 128), pyramid_stack_size=None, **kwargs):
        super().__init__()
        self.ffn = None
        self.cma = None
        self.rms_1 = None
        self.rms_0 = None
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_units = ffn_units
        self.pyramid_stack_size = pyramid_stack_size

    def build(self, input_shape):
        self.rms_0 = tf.keras.layers.LayerNormalization()
        self.rms_1 = tf.keras.layers.LayerNormalization()
        self.cma = CausalMaskAttention(d_model=self.d_model, num_heads=self.num_heads)
        self.ffn = FFNLayer(unit_1=self.ffn_units[0], unit_2=self.ffn_units[1])

    def call(self, x):
        """
        :param x: [batch_size, seq_len, d_model]
        :return:
        """
        x = self.rms_0(x)

        k_x, q_x, v_x = x, x, x
        if self.pyramid_stack_size is not None:
            q_x = tf.slice(x, [0, 0, 0], self.pyramid_stack_size)
        origin_x = q_x

        x = self.cma([k_x, q_x, v_x])
        x = origin_x + x
        origin_x = x

        x = self.rms_1(x)
        x = self.ffn(x)
        x = origin_x + x

        return x


class BaseOneTransBlock(layers.Layer):
    def __init__(self, d_model=128, num_heads=4, ffn_units=(256, 128), n=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_units = ffn_units
        self.n = n
        self.ns_otb_list = []
        self.s_otb_list = []

    def build(self, input_shape):
        for i in range(self.n):
            self.ns_otb_list.append(OneTransBlock(d_model=self.d_model, num_heads=self.num_heads, ffn_units=self.ffn_units))
            self.s_otb_list.append(OneTransBlock(d_model=self.d_model, num_heads=self.num_heads, ffn_units=self.ffn_units))

    def call(self, x):
        """
        :param x: 这里设计的有点丑陋，x[0]是序列特征编码结果，x[1]是非序列特征编码结果
        :return:
        """
        s_emb = x[0]
        ns_emb = x[1]
        ns_res = []
        s_res = []

        for otb in self.ns_otb_list:
            ns_res.append(otb(ns_emb))
        for otb in self.s_otb_list:
            s_res.append(otb(s_emb))

        # [n, batch_size, seq_len, dim]
        ns_res = tf.convert_to_tensor(ns_res)
        # [batch_size, seq_len, dim]
        ns_res = tf.reduce_mean(ns_res, axis=0)
        s_res = tf.convert_to_tensor(s_res)
        s_res = tf.reduce_mean(s_res, axis=0)

        return tf.concat([ns_res, s_res], axis=1)


class StackOneTransBlock(layers.Layer):
    def __init__(self, d_model=128, num_heads=4, ffn_units=(256, 128), n=4, pyramid_stack_size=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_units = ffn_units
        self.n = n
        self.pyramid_stack_size = pyramid_stack_size
        self.otb_list = []

    def build(self, input_shape):
        for i in range(self.n):
            self.otb_list.append(OneTransBlock(d_model=self.d_model, num_heads=self.num_heads, ffn_units=self.ffn_units, pyramid_stack_size=self.pyramid_stack_size))

    def call(self, x):
        res = []
        for otb in self.otb_list:
            res.append(otb(x))
        res = tf.convert_to_tensor(res)
        res = tf.reduce_mean(res, axis=0)
        return res


BATCH_SIZE = 4
SEQ_LEN = 3  # 序列特征长度
FEAT_DIM = 8  # 原始特征维度
D_MODEL = 16  # 模型输入输出维度，也是 token 的长度

# 假设有序列特征 [batch_size, seq_len, feat_dim]
seq_feature = tf.random.normal((BATCH_SIZE, SEQ_LEN, FEAT_DIM), dtype=tf.float32)
# tokenizer 编码后得到 [batch_size, seq_len, D_MODEL]
# 序列特征的 tokenizer 简单说就是把序列里的每个元素都过一个网络结构，然后映射到 D_MODEL 维度
s_feat = layers.Dense(D_MODEL)(seq_feature)
print("设有序列特征[batch_size, seq_len, feat_dim]: ", seq_feature.shape)
print("编码后[batch_size, seq_len, D_MODEL]: ", s_feat.shape)

# 假设有非序列特征拼接后 [batch_size, 随机维度]
n_seq_feature = tf.random.normal((BATCH_SIZE, 128), dtype=tf.float32)
# tokenizer 编码后得到 [batch_size, N, D_MODEL]
# 非序列特征的 tokenizer 就是把原始特征改造为序列结构，序列的长度 N 自己定
N = 4
ns_feat = layers.Dense(N * D_MODEL)(n_seq_feature)
ns_feat = tf.reshape(ns_feat, [BATCH_SIZE, N, D_MODEL])
print("设有非序列特征[batch_size, 随机维度]: ", n_seq_feature.shape)
print("编码后[batch_size, N, D_MODEL]: ", ns_feat.shape)
print()

# 定义block结构
NUM_HEAD = 4
MULTI_NUM = 8
FFN_UNITS = (64, D_MODEL)
# 最底层block，内有两个多层OneTransBlock，分别过序列特征和非序列特征，最后拼接得到一个大的序列
# [batch_size, seq_len, D_MODEL] + [batch_size, N, D_MODEL] = [batch_size, seq_len + N, D_MODEL]
base_block = BaseOneTransBlock(d_model=D_MODEL, num_heads=NUM_HEAD, ffn_units=FFN_UNITS, n=MULTI_NUM)
# 这里注意 - 序列特征在前 非序列特征在后，不然后续的压缩对象就错了（包括序列特征里的拼接顺序，也要按时间先后）
base_embedding = base_block([s_feat, ns_feat])
print("序列编码特征+非序列编码特征 → 过底层 OneBlock 结构后[batch_size, SEQ_LEN + N, D_MODEL]: ", base_embedding.shape)
# 然后是不断蒸馏、压缩这段序列向量，理论上是有 seq_len 个序列，就压缩 seq_len 次
# 形象的解释就是，把之前第 N 次行为，压缩到 N-1 次，再压缩到 N-2 次 .... 直到只剩下非序列特征
# 嘛，这种解释和工程复杂度就仁者见仁了
base_seq_len = base_embedding.shape[1]

# 第一层压缩，把序列长度从 base_seq_len 压缩到 base_seq_len - 1
stack_block_1 = StackOneTransBlock(d_model=D_MODEL, num_heads=NUM_HEAD, ffn_units=FFN_UNITS, n=MULTI_NUM, pyramid_stack_size=[-1, base_seq_len - 1, -1])
stack_embedding = stack_block_1(base_embedding)
print("过第一层压缩结构后[batch_size, SEQ_LEN + N - 1, D_MODEL]: ", stack_embedding.shape)
# 第二层压缩，把序列长度从 base_seq_len 压缩到 base_seq_len - 2
stack_block_2 = StackOneTransBlock(d_model=D_MODEL, num_heads=NUM_HEAD, ffn_units=FFN_UNITS, n=MULTI_NUM, pyramid_stack_size=[-1, base_seq_len - 2, -1])
stack_embedding = stack_block_2(stack_embedding)
print("过第二层压缩结构后[batch_size, SEQ_LEN + N - 2, D_MODEL]: ", stack_embedding.shape)
# 第三层压缩，把序列长度从 base_seq_len 压缩到 base_seq_len - 3
stack_block_3 = StackOneTransBlock(d_model=D_MODEL, num_heads=NUM_HEAD, ffn_units=FFN_UNITS, n=MULTI_NUM, pyramid_stack_size=[-1, base_seq_len - 3, -1])
stack_embedding = stack_block_3(stack_embedding)
print("过第三层压缩结构后[batch_size, SEQ_LEN + N - 3, D_MODEL]: ", stack_embedding.shape)
print()

# 因为这里的 seq_len 只有3，所以我们压缩3次就完成了，相当于把长度为 3 的行为序列特征都压缩进了最后的向量
# 而非序列特征都参与了所有压缩，充分交叉
print("这也就是最终的压缩结果[batch_size, N, D_MODEL]: ", stack_embedding.shape)
# 最后把这段向量 pooling 或者 concat 后再输入下游任务
final_embedding = tf.reduce_mean(stack_embedding, axis=-1)
print("输入下游任务前的 Pooling 结果: ", final_embedding.shape)
