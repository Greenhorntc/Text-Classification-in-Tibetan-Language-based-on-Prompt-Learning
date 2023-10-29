from transformers import XLMRobertaTokenizer, TFXLMRobertaModel
import tensorflow as tf

METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'),
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
]


def Prompt(cinomodel,dropout):
    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
    outputs = cinomodel.roberta(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']
    outputs = tf.keras.layers.Dropout(dropout)(outputs)
    cla_outputs = tf.keras.layers.Dense(2, activation=None)(outputs)
    model = tf.keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=[cla_outputs])
    return model



def TextCNN(cinomodel,dropout):
    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
    outputs = cinomodel.roberta(input_ids=input_ids, attention_mask=attention_mask,training=False)['last_hidden_state']
    reshape=tf.keras.layers.Reshape((500, 768,1))
    reshapetf=reshape(outputs)
    cnns = []
    filter_sizes = [3, 4, 5]
    for size in filter_sizes:
        conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(size, 768),
                             strides=1, padding='valid', activation='relu')(reshapetf)
        pool = tf.keras.layers.MaxPool2D(pool_size=(500 - size + 1, 1), padding='valid')(conv)
        cnns.append(pool)
    outt = tf.keras.layers.concatenate(cnns)
    cnnout=tf.keras.layers.Flatten()(outt)
    fc1=tf.keras.layers.Dense(56,activation="relu")(cnnout)
    fc1out=tf.keras.layers.Dropout(dropout)(fc1)
    finalout=tf.keras.layers.Dense(12, activation="softmax")(fc1out)
    model = tf.keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=[finalout])
    return model

def RNN(cinomodel,dropout):
    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
    outputs = cinomodel.roberta(input_ids=input_ids, attention_mask=attention_mask, training=False)['last_hidden_state']
    rnnout=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,dropout=dropout))(outputs)
    fc1 = tf.keras.layers.Dense(56, activation="relu")(rnnout)
    fc1out = tf.keras.layers.Dropout(dropout)(fc1)
    finalout = tf.keras.layers.Dense(12, activation="softmax")(fc1out)
    model = tf.keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=[finalout])
    return model

def TextRCNN(cinomodel,dropout):

    #1两个rnn网络得到的激活和原来的embedding结合
    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
    outputs = cinomodel.roberta(input_ids=input_ids, attention_mask=attention_mask, training=False)['last_hidden_state']
    lx, rx = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=dropout, return_sequences=True), merge_mode=None)(outputs)
    rnnall = tf.keras.layers.concatenate([lx, outputs, rx])
    # fc = tf.keras.layers.Dense(rnnall.shape[-1], activation="tanh")(rnnall)
    conv = tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu')(rnnall)
    print(conv.shape)
    # x=L.Conv1D(filters=1,kernel_size=3,activation="relu")(x)
    pool = tf.keras.layers.GlobalMaxPool1D()(conv)
    print(pool.shape)
    poolout = tf.keras.layers.Dropout(dropout)(pool)

    output = tf.keras.layers.Dense(12, activation="softmax")(poolout)
    model = tf.keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=[output])
    return model

def DPCNN(cinomodel,dropout):
    #暂定128
    num_filters=256
    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
    embed = cinomodel.roberta(input_ids=input_ids, attention_mask=attention_mask, training=False)['last_hidden_state']
    #cino 输出(none,none,768) 句子长度500 （none ,500,768）
    embedding_inputs = tf.expand_dims(embed, axis=-1)  # [None,500,768,1]
    print(embedding_inputs.shape)
    # # region_embedding  # [batch,seq-3+1,1,250]
    region_embedding = tf.keras.layers.Conv2D(num_filters,[3, 768],activation="relu")(embedding_inputs)
    print(region_embedding.shape)
    with tf.compat.v1.variable_scope("conv3_0", reuse=True):
    # with tf.variable_scope("conv3_0", reuse=Ture):
        conv3 = tf.keras.layers.Conv2D(num_filters,3,padding="same", activation=tf.nn.relu)(region_embedding)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
    with tf.compat.v1.variable_scope("conv3_1", reuse=True):
    # with tf.variable_scope("conv3_1", reuse=reuse):
        conv3 = tf.keras.layers.Conv2D(num_filters, 3, padding="same", activation=tf.nn.relu)(region_embedding)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
    print(conv3.shape)
    # resdul
    conv3 = conv3 + region_embedding
    print("+region")
    print(conv3.shape)
    with tf.compat.v1.variable_scope("pool_1", reuse=True):
    # with tf.variable_scope("pool_1", reuse=reuse):
        pool = tf.pad(conv3, paddings=[[0, 0], [0, 1], [0, 0], [0, 0]])
        pool = tf.nn.max_pool(pool, [1, 3, 1, 1], strides=[1, 2, 1, 1], padding='VALID')
    with tf.compat.v1.variable_scope("conv3_2", reuse=True):
    # with tf.variable_scope("conv3_2", reuse=reuse):
        conv3 = tf.keras.layers.Conv2D(num_filters, 3,padding="same", activation=tf.nn.relu)(pool)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
    with tf.compat.v1.variable_scope("conv3_3", reuse=True):
    # with tf.variable_scope("conv3_3", reuse=reuse):
        conv3 = tf.keras.layers.Conv2D(num_filters, 3,padding="same", activation=tf.nn.relu)(conv3)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
    #
    # # resdul
    conv3 = conv3 + pool
    pool_size = int((500 - 3 + 1) / 2)
    conv3 = tf.keras.layers.MaxPool1D(pool_size, 1)(tf.squeeze(conv3, [2]))
    conv3 = tf.squeeze(conv3, [1])  # [batch,250]
    print(conv3.shape)
    conv3 = tf.keras.layers.Dropout(dropout)(conv3)
    fc = tf.keras.layers.Dense(12, activation="softmax")(conv3)
    # output = tf.keras.layers.Dense(12, activation="softmax")(outputs)
    model = tf.keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=[fc])
    return model

