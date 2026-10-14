import tensorflow as tf
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Token 布局约定（与原代码保持一致）：
#   [NS_1, ..., NS_ns_len, S_最近, S_次近, ..., S_最久]
# 即「非序列 token 在前，序列 token 在后，且时间近的在左」。
# 这等价于论文中 [S_最久, ..., S_最近, NS...] 布局的整体逆序，
# 所以论文中的下三角 causal mask，在这里对应「位置 i 只能看到 j >= i」的上三角 mask，
# pyramid stacking 保留前缀（NS + 最近行为），丢弃尾部（最久远行为）。
# ---------------------------------------------------------------------------


class RMSNorm(layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale", shape=(int(input_shape[-1]),), initializer="ones", trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        ms = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
        return x * tf.math.rsqrt(ms + self.epsilon) * self.scale


class MixedDense(layers.Layer):
    """混合参数化 Dense：前 ns_len 个 NS-token 各自独立参数，其余所有 S-token 共享一套参数。"""

    def __init__(self, ns_len, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.ns_len = ns_len
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        d_in = int(input_shape[-1])
        self.ns_kernel = self.add_weight(
            name="ns_kernel", shape=(self.ns_len, d_in, self.units), initializer="glorot_uniform"
        )
        self.ns_bias = self.add_weight(name="ns_bias", shape=(self.ns_len, self.units), initializer="zeros")
        self.s_kernel = self.add_weight(name="s_kernel", shape=(d_in, self.units), initializer="glorot_uniform")
        self.s_bias = self.add_weight(name="s_bias", shape=(self.units,), initializer="zeros")
        super().build(input_shape)

    def call(self, x):
        # x: [batch, L, d_in]，要求 L >= ns_len
        ns_x = x[:, :self.ns_len, :]
        s_x = x[:, self.ns_len:, :]
        ns_out = tf.einsum("bnd,nde->bne", ns_x, self.ns_kernel) + self.ns_bias
        s_out = tf.einsum("bld,de->ble", s_x, self.s_kernel) + self.s_bias
        return self.activation(tf.concat([ns_out, s_out], axis=1))


class MixedFFN(layers.Layer):
    """混合参数化 FFN：升维 + 激活 + 线性降回 d_model（输出层不加激活，便于残差）。"""

    def __init__(self, ns_len, d_model, hidden_units, activation="swish", **kwargs):
        super().__init__(**kwargs)
        self.up = MixedDense(ns_len, hidden_units, activation=activation)
        self.down = MixedDense(ns_len, d_model)

    def call(self, x):
        return self.down(self.up(x))


class CausalMaskAttention(layers.Layer):
    def __init__(self, ns_len, d_model=128, num_heads=4, if_mask=True, **kwargs):
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError(f"d_model({d_model}) 必须能被 num_heads({num_heads}) 整除")
        self.ns_len = ns_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.if_mask = if_mask
        self.q_proj = MixedDense(ns_len, d_model)
        self.k_proj = MixedDense(ns_len, d_model)
        self.v_proj = MixedDense(ns_len, d_model)
        self.out_proj = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    @staticmethod
    def create_causal_mask(len_q, len_k):
        # query 是 key 的前缀（pyramid 时 len_q <= len_k），位置 i 只能看到 j >= i
        # 返回 bool 矩阵，True 表示可见
        allowed = tf.linalg.band_part(tf.ones((len_q, len_k)), num_lower=0, num_upper=-1)
        return tf.cast(allowed, tf.bool)

    def call(self, q_x, kv_x):
        """
        :param q_x:  [batch, len_q, d_model]，为 kv_x 的前缀
        :param kv_x: [batch, len_k, d_model]
        """
        batch_size = tf.shape(kv_x)[0]
        len_q = tf.shape(q_x)[1]
        len_k = tf.shape(kv_x)[1]

        q = self.split_heads(self.q_proj(q_x), batch_size)
        k = self.split_heads(self.k_proj(kv_x), batch_size)
        v = self.split_heads(self.v_proj(kv_x), batch_size)

        # (batch, heads, len_q, len_k)
        logits = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.depth, q.dtype))

        if self.if_mask:
            allowed = self.create_causal_mask(len_q, len_k)[tf.newaxis, tf.newaxis, :, :]
            logits = tf.where(allowed, logits, tf.cast(-1e9, logits.dtype))

        weights = tf.nn.softmax(logits, axis=-1)
        output = tf.matmul(weights, v)                       # (batch, heads, len_q, depth)
        output = tf.transpose(output, perm=[0, 2, 1, 3])     # (batch, len_q, heads, depth)
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        return self.out_proj(output)


class OneTransBlock(layers.Layer):
    def __init__(self, ns_len, d_model, num_heads=4, ffn_units=(256, 128), pyramid_stack_len=None, **kwargs):
        super().__init__(**kwargs)
        if ffn_units[-1] != d_model:
            raise ValueError(f"FFN 输出维度 ffn_units[-1]({ffn_units[-1]}) 必须等于 d_model({d_model}) 才能做残差")
        if pyramid_stack_len is not None and pyramid_stack_len < ns_len:
            raise ValueError(f"pyramid_stack_len({pyramid_stack_len}) 不能小于 ns_len({ns_len})，NS-token 必须全部保留")
        self.ns_len = ns_len
        self.pyramid_stack_len = pyramid_stack_len
        self.norm_attn = RMSNorm()
        self.norm_ffn = RMSNorm()
        self.attn = CausalMaskAttention(ns_len=ns_len, d_model=d_model, num_heads=num_heads)
        self.ffn = MixedFFN(ns_len=ns_len, d_model=d_model, hidden_units=ffn_units[0])

    def call(self, x):
        """
        :param x: [batch, seq_len, d_model]
        :return:  [batch, pyramid_stack_len or seq_len, d_model]
        """
        h = self.norm_attn(x)
        if self.pyramid_stack_len is not None:
            x_q = x[:, :self.pyramid_stack_len, :]
            h_q = h[:, :self.pyramid_stack_len, :]
        else:
            x_q, h_q = x, h

        # Pre-Norm 残差：残差分支用未归一化的原始输入
        x = x_q + self.attn(h_q, h)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class MultiOneTransBlock(layers.Layer):
    """串行堆叠 n 层 OneTransBlock；pyramid 压缩只在第一层执行，后续层在压缩后的长度上继续计算。"""

    def __init__(self, ns_len=4, d_model=128, num_heads=4, ffn_units=(256, 128), n=4, pyramid_stack_len=None, **kwargs):
        super().__init__(**kwargs)
        self.otb_list = [
            OneTransBlock(
                ns_len=ns_len,
                d_model=d_model,
                num_heads=num_heads,
                ffn_units=ffn_units,
                pyramid_stack_len=pyramid_stack_len if i == 0 else None,
            )
            for i in range(n)
        ]

    def call(self, x):
        for otb in self.otb_list:
            x = otb(x)
        return x


if __name__ == "__main__":
    BATCH_SIZE = 4
    SEQ_LEN = 3   # 序列特征长度
    FEAT_DIM = 8  # 原始特征维度
    D_MODEL = 16  # 模型输入输出维度，也是 token 的长度

    # 序列特征 [batch_size, seq_len, feat_dim]，时间近的在左
    seq_feature = tf.random.normal((BATCH_SIZE, SEQ_LEN, FEAT_DIM), dtype=tf.float32)
    s_feat = layers.Dense(D_MODEL)(seq_feature)
    print("设有序列特征[batch_size, seq_len, feat_dim]: ", seq_feature.shape)
    print("编码后[batch_size, seq_len, D_MODEL]: ", s_feat.shape)

    # 非序列特征 [batch_size, 随机维度] -> [batch_size, ns_len, D_MODEL]
    n_seq_feature = tf.random.normal((BATCH_SIZE, 128), dtype=tf.float32)
    ns_len = 2
    ns_feat = layers.Dense(ns_len * D_MODEL)(n_seq_feature)
    ns_feat = tf.reshape(ns_feat, [-1, ns_len, D_MODEL])
    print("设有非序列特征[batch_size, 随机维度]: ", n_seq_feature.shape)
    print("编码后[batch_size, ns_len, D_MODEL]: ", ns_feat.shape)
    print()

    NUM_HEAD = 4
    MULTI_NUM = 2  # 每个 stage 串行堆叠的层数
    FFN_UNITS = (64, D_MODEL)

    # 非序列特征在前，序列特征在后（序列内时间近的在左）
    tokens = tf.concat([ns_feat, s_feat], axis=1)
    base_block = MultiOneTransBlock(ns_len=ns_len, d_model=D_MODEL, num_heads=NUM_HEAD, ffn_units=FFN_UNITS, n=MULTI_NUM)
    base_embedding = base_block(tokens)
    print("过底层 OneTrans 结构后[batch_size, SEQ_LEN + N, D_MODEL]: ", base_embedding.shape)

    base_seq_len = base_embedding.shape[1]
    stack_embedding = base_embedding
    # 逐层 pyramid 压缩，每次丢弃最久远的一个行为 token，直到只剩 NS-token
    for i in range(1, SEQ_LEN + 1):
        stack_block = MultiOneTransBlock(
            ns_len=ns_len, d_model=D_MODEL, num_heads=NUM_HEAD, ffn_units=FFN_UNITS,
            n=MULTI_NUM, pyramid_stack_len=base_seq_len - i,
        )
        stack_embedding = stack_block(stack_embedding)
        print(f"过第{i}层压缩结构后[batch_size, SEQ_LEN + N - {i}, D_MODEL]: ", stack_embedding.shape)
    print()

    stack_embedding = RMSNorm()(stack_embedding)
    print("最终的压缩结果[batch_size, N, D_MODEL]: ", stack_embedding.shape)
    final_embedding = tf.reduce_mean(stack_embedding, axis=1)
    print("输入下游任务前的 Pooling 结果: ", final_embedding.shape)
