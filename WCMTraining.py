from transformers import XLMRobertaTokenizer, TFXLMRobertaModel
import tensorflow as tf
from matplotlib import pyplot as plt
#使用随训练批次更改的学习率,模型使用该数据三轮就能获得很高的表现
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from datasets import load_dataset,Dataset
from GetModel import Prompt
from ResultPlot import plotLoss,plotACC
from ComparativeExperiments import savehistory
"""
模型名	MODEL_NAME
CINO-large-v2	hfl/cino-large-v2
CINO-base-v2	hfl/cino-base-v2
CINO-small-v2	hfl/cino-small-v2
CINO-large	hfl/cino-large

"""
def getTrainconfig(floder,name):
    train = floder + "/wcmtrain"
    test = floder + "/wcmtest"
    savehisname='logs/WCM/'+name+".txt"
    savemodel = "savedmodel/WCM/Prompt/" +name
    config = {"batch_size": 8, "epochs": 50, "checkpoint": "hfl/cino-small-v2", "trainpath": train, "testpath": test,
               "savemodelname": savemodel,"savehis":savehisname}
    return config


def getTFDataset(config,tokenzier):
    trainds = Dataset.load_from_disk(config["trainpath"])
    testds = Dataset.load_from_disk(config["testpath"])
    print(trainds)
    print(testds)
    def tokenize_fun(example):
        return tokenzier(example["textfeatures"], truncation=True, padding=True, max_length=500)

    train_tokenized_ds = trainds.map(tokenize_fun, batched=True)
    test_tokenized_ds = testds.map(tokenize_fun, batched=True)
    tf_train_dataset = train_tokenized_ds.to_tf_dataset(
        columns=["attention_mask", "input_ids"],
        label_cols=["labels"],
        shuffle=True,
        # collate_fn=data_collator,
        batch_size=config["batch_size"],
    )

    tf_validation_dataset = test_tokenized_ds.to_tf_dataset(
        columns=["attention_mask", "input_ids"],
        label_cols=["labels"],
        shuffle=False,
        # collate_fn=data_collator,
        batch_size=config["batch_size"],
    )
    return tf_train_dataset, tf_validation_dataset


def train(config):
    #0 使用科带讯飞的cino模型试试
    tokenzier = XLMRobertaTokenizer.from_pretrained(config["checkpoint"])
    # 1 加载数据集合
    tftrainds,tftestds=getTFDataset(config=config,tokenzier=tokenzier)

    #2调用模型
    print("model - training ")
    cinomodel = TFXLMRobertaModel.from_pretrained(config["checkpoint"])
    clsmodel = Prompt(cinomodel,dropout=0.3)
    clsmodel.layers[2].trainable = False
    clsmodel.summary()
    # 1：complie
    lossfun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # 改变optimizer
    num_train_step = len(tftrainds) * config["epochs"]
    lr_scheduleer = PolynomialDecay(initial_learning_rate=1e-3, end_learning_rate=0.0, decay_steps=num_train_step)
    optimizer = tf.keras.optimizers.Adam(lr_scheduleer)

    """
    使用tensorborad
    """

    keras_callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir="./logs"),
        # tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', min_delta=0.01,restore_best_weights=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", verbose=2, patience=5, mode="max",
                                         restore_best_weights=True)
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
        validation_data=tftestds,
    )

    basemodelfilepath = config["savemodelname"]
    clsmodel.save(basemodelfilepath)
    print("base Saved")
    return history

def modelrun(floder,modelname):
    trainconfig=getTrainconfig(floder,name=modelname)
    his=train(trainconfig)
    savehistory(his,filename=trainconfig["savehis"])
    return his


data0="dataset/WCM/finaleditionprompt0"
history1=modelrun(data0,modelname="Nofinutedprompt0")

data1="dataset/WCM/finaleditionprompt1"
history2=modelrun(data1,modelname="Nofinutedprompt1")

data2="dataset/WCM/finaleditionprompt2"
history3=modelrun(data2,modelname="Nofinutedprompt2")

# # modelhistorylist = [history1]
# modelhistorylist = [history1,history2,history3]
# # # namelist = ["Using-Prompt-base"]
# namelist = ["Using-Prompt1-small","Using-Prompt2-small","Using-Prompt3-small"]
#
# plotLoss(modelhistorylist, namelist=namelist)
# plt.show()
# plotACC(modelhistorylist, namelist=namelist)
# plt.show()

# plotPrecsion(modelhistorylist, namelist=namelist)
# plt.show()
# plotRecall(modelhistorylist, namelist=namelist)
# plt.show()

"""keras 老三套 compile fit predict"""

# predics=clsmodel.predict(tf_validation_dataset)["logits"]
# class_preds=np.argmax(predics,axis=1)
# print(predics.shape,class_preds.shape)
# print("over")