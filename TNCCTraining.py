from transformers import XLMRobertaTokenizer, TFXLMRobertaModel,AutoTokenizer
import tensorflow as tf
from matplotlib import pyplot as plt
#使用随训练批次更改的学习率,模型使用该数据三轮就能获得很高的表现
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from datasets import Dataset
from GetModel import Prompt
from ResultPlot import plotLoss,plotACC

from TNCCEvaluate import get_orlabel,getTarget

"""
模型名	MODEL_NAME
CINO-large-v2	hfl/cino-large-v2
CINO-base-v2	hfl/cino-base-v2
CINO-small-v2	hfl/cino-small-v2
CINO-large	hfl/cino-large
...sangjeedondrub/tibetan-roberta-base
"""

#参数
config1={"batch_size":8,"epochs":50,"checkpoint":"hfl/cino-small-v2","trainpath":"dataset/TNCC/final3prompt0/train","testpath":"dataset/TNCC/final3prompt0/test","valpath":"dataset/TNCC/final3prompt0/val","savemodelname":"savedmodel/TNCC/final3prompt0-small"}
config2={"batch_size":8,"epochs":50,"checkpoint":"hfl/cino-small-v2","trainpath":"dataset/TNCC/final3prompt1/train","testpath":"dataset/TNCC/final3prompt1/test","valpath":"dataset/TNCC/final3prompt1/val","savemodelname":"savedmodel/TNCC/final3prompt1-small"}
config3={"batch_size":8,"epochs":50,"checkpoint":"hfl/cino-small-v2","trainpath":"dataset/TNCC/final3prompt2/train","testpath":"dataset/TNCC/final3prompt2/test","valpath":"dataset/TNCC/final3prompt2/val","savemodelname":"savedmodel/TNCC/final3prompt2-small"}

config4={"batch_size":8,"epochs":50,"checkpoint":"sangjeedondrub/tibetan-roberta-base","trainpath":"dataset/TNCC/200prompt0/train","testpath":"dataset/TNCC/200prompt0/test","valpath":"dataset/TNCC/200prompt0/val","savemodelname":"savedmodel/TNCC/200prompt0-tibert"}
config5={"batch_size":8,"epochs":50,"checkpoint":"sangjeedondrub/tibetan-roberta-base","trainpath":"dataset/TNCC/200prompt1/train","testpath":"dataset/TNCC/200prompt1/test","valpath":"dataset/TNCC/200prompt1/val","savemodelname":"savedmodel/TNCC/200prompt1-tibert"}
config6={"batch_size":8,"epochs":50,"checkpoint":"sangjeedondrub/tibetan-roberta-base","trainpath":"dataset/TNCC/200prompt2/train","testpath":"dataset/TNCC/200prompt2/test","valpath":"dataset/TNCC/200prompt2/val","savemodelname":"savedmodel/TNCC/200prompt2-tibert"}


def getTFDataset(config,tokenzier):
    trainds = Dataset.load_from_disk(config["trainpath"])
    valds=Dataset.load_from_disk(config["valpath"])
    testds = Dataset.load_from_disk(config["testpath"])
    # trainds=trainds.select(range(1000))
    # valds=valds.select(range(1000))
    # testds=testds.select(range(1000))
    print(trainds)
    print(valds)
    print(testds)
    def tokenize_fun(example):
        return tokenzier(example["textfeatures"], truncation=True, padding='max_length', max_length=500)

    train_tokenized_ds = trainds.map(tokenize_fun, batched=True)
    val_toeknized_ds = valds.map(tokenize_fun,batched=True)
    test_tokenized_ds = testds.map(tokenize_fun, batched=True)
    tf_train_dataset = train_tokenized_ds.to_tf_dataset(
        columns=["attention_mask", "input_ids"],
        label_cols=["labels"],
        shuffle=True,
        # collate_fn=data_collator,
        batch_size=config["batch_size"],
    )

    tf_validation_dataset = val_toeknized_ds.to_tf_dataset(
        columns=["attention_mask", "input_ids"],
        label_cols=["labels"],
        shuffle=True,
        # collate_fn=data_collator,
        batch_size=config["batch_size"],
    )

    tf_test_dataset = test_tokenized_ds.to_tf_dataset(
        columns=["attention_mask", "input_ids"],
        label_cols=["labels"],
        shuffle=False,
        # collate_fn=data_collator,
        batch_size=12,
    )

    return tf_train_dataset, tf_validation_dataset,tf_test_dataset


def train(config):
    #0 使用科带讯飞的cino模型试试
    # tokenzier = XLMRobertaTokenizer.from_pretrained(config["checkpoint"])
    tokenzier = AutoTokenizer.from_pretrained(config["checkpoint"])
    # 1 加载数据集合
    tftrainds,tfvalds,tftestds=getTFDataset(config=config,tokenzier=tokenzier)

    #2调用模型
    print("model - training ")
    cinomodel = TFXLMRobertaModel.from_pretrained(config["checkpoint"],from_pt=True)
    # cinomodel = TFXLMRobertaModel.from_pretrained(config["checkpoint"])
    clsmodel = Prompt(cinomodel,dropout=0.1)
    clsmodel.summary()
    # 1：complie
    lossfun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # 改变optimizer
    num_train_step = len(tftrainds) * config["epochs"]
    lr_scheduleer = PolynomialDecay(initial_learning_rate=3e-5, end_learning_rate=0.0, decay_steps=num_train_step)
    optimizer = tf.keras.optimizers.Adam(lr_scheduleer)

    """
    使用tensorborad
    """

    keras_callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir="./logs"),
        # tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', min_delta=0.01,restore_best_weights=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", verbose=2, patience=5, mode="max",restore_best_weights=True)
    ]

    clsmodel.compile(
        optimizer=optimizer,
        loss=lossfun,
        metrics=["accuracy"]
        # metrics=METRICS
    )

    history = clsmodel.fit(
        x=tftrainds,
        epochs=config["epochs"],
        verbose=2,
        callbacks=keras_callbacks,
        validation_data=tfvalds,
    )

    basemodelfilepath = config["savemodelname"]
    clsmodel.save(basemodelfilepath)
    print("base Saved")

    """keras 老三套 compile fit predict"""
    #看一下测试集合结果
    predics = clsmodel.predict(tftestds)
    testds = Dataset.load_from_disk(config["testpath"])
    rellabels=get_orlabel(testds["labels"])
    getTarget(predics, rellabels)
    return history

def savehistory(history,filename):
    epoch=datatostr(history.epoch)
    loss=datatostr( history.history["loss"])
    val_loss= datatostr(history.history["val_loss"])
    acc=datatostr( history.history["accuracy"])
    valacc=datatostr( history.history["val_accuracy"])
    with open(file=filename,encoding="UTF-8",mode='w') as f:
        f.write(epoch+" ")
        f.write("\n")
        f.write(loss+" ")
        f.write("\n")
        f.write(val_loss+" ")
        f.write("\n")
        f.write(acc+" ")
        f.write("\n")
        f.write(valacc + " ")
        f.write("\n")
        f.close()

def datatostr(data):
    datstr=""
    for i in data:
        datstr=datstr+str(i)
    return datstr

print("training")
history1=train(config=config1)
savehistory(history1,filename="logs/logs1.txt")

history2=train(config=config2)
savehistory(history2,filename="logs/logs2.txt")
history3=train(config=config3)
savehistory(history3,filename="logs/logs3.txt")

# modelhistorylist = [history]
modelhistorylist = [history1,history2,history3]
# namelist = ["Using-Prompt-base1"]
namelist = ["Using-prompt1-base1","Using-Prompt2-small2","Using-Prompt3-small3"]

plotLoss(modelhistorylist, namelist=namelist)
plt.show()
plotACC(modelhistorylist, namelist=namelist)
plt.show()
