#用于对比实验，包括textcnn textrnn textrcnn
from datasets import Dataset
import tensorflow as tf
from TNCCEvaluate import getReport
from transformers import TFXLMRobertaModel,AutoTokenizer,TFAutoModelForSequenceClassification
import numpy as np
#训练参数


#loaddataset
def getTFDataset(config,dftokenizer):
    trainds = Dataset.load_from_disk(config["trainpath"])
    valds=Dataset.load_from_disk(config["valpath"])
    testds = Dataset.load_from_disk(config["testpath"])
    print(trainds)
    print(valds)
    print(testds)
    #调测小数据集(可选)
    # trainds=trainds.select(range(100))
    # valds=valds.select(range(1000))
    # testds=testds.select(range(1000))

    def tokenize_fun(example):
        example["labels"]=np.array(example["labels"])
        return dftokenizer(example["textfeatures"], truncation=True, padding='max_length', max_length=500)
    train_tokenized_ds = trainds.map(tokenize_fun, batched=True)
    val_toeknized_ds = valds.map(tokenize_fun)
    test_tokenized_ds = testds.map(tokenize_fun)
    #转tfdataset
    tf_train_dataset = train_tokenized_ds.to_tf_dataset(
        columns=["attention_mask", "input_ids"],
        label_cols=["labels"],
        shuffle=True,
        batch_size=config['batch_size'],
    )

    tf_validation_dataset = val_toeknized_ds.to_tf_dataset(
        columns=["attention_mask", "input_ids"],
        label_cols=["labels"],
        shuffle=True,
        batch_size=config['batch_size'],
    )
    #使用原来标签就不需要batchize=12
    tf_test_dataset = test_tokenized_ds.to_tf_dataset(
        columns=["attention_mask", "input_ids"],
        label_cols=["labels"],
        shuffle=False,
        batch_size=config['batch_size'],
    )
    return tf_train_dataset,tf_validation_dataset,tf_test_dataset,testds

#看一下数据
# for text_batch,label_batch in tf_validation_dataset.take(1):
#     print(text_batch)
#     print(label_batch)

def ctrain(cmodel,name,config,fintued=False):
    usetokenizer=AutoTokenizer.from_pretrained(config["checkpoint"])
    tftrainds, tfvalds, tftestds,testds = getTFDataset(config=config, dftokenizer=usetokenizer)
    model=cmodel
    if fintued==False:
        model.layers[2].trainable = False
    else:
        model.layers[2].trainable=True
    # for layer in model.layers:
    #     print(layer)
    model.summary()
    # 1：complie
    lossfun = tf.keras.losses.sparse_categorical_crossentropy
    # 改变optimizer
    optimizer = tf.keras.optimizers.Adam(1e-3)
    """
    使用tensorborad
    """
    keras_callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir="./logs"),
        # tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', min_delta=0.01,restore_best_weights=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", verbose=2, patience=5, mode="max",restore_best_weights=True)
        ]

    model.compile(
            optimizer=optimizer,
            loss=lossfun,
            metrics=["accuracy"]
            # metrics=METRICS
        )
    history = model.fit(
            x=tftrainds,
            epochs=config['epochs'],
            verbose=2,
            callbacks=keras_callbacks,
            validation_data=tfvalds
        )

    savemodelfilepath = config["savemodelname"]+name
    model.save(savemodelfilepath)
    print(name+"Saved")

    """keras 老三套 compile fit predict"""
    #看一下测试集合结果
    predics = model.predict(tftestds)
    prelabel = tf.math.argmax(predics, 1).numpy()
    # print(prelabel)
    testrespath=config["savehis"]+name+"result.txt"
    getReport(prelabel, testds["labels"],save=testrespath)
    hispath=config["savehis"]+name+".txt"
    savehistory(history,filename=hispath)
    print(name+"trained over")

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



# from TNCCEvaluate import getTFDataset,getReport
#
# config={
#     "batch_size":8,
#     "epochs":50,
#     "checkpoint":"hfl/cino-small-v2",
#     "trainpath":"dataset/TNCC/TNCCpassage/Noprompt/final/train",
#     "valpath":"dataset/TNCC/TNCCpassage/Noprompt/final/val",
#     "testpath":"dataset/TNCC/TNCCpassage/Noprompt/final/test",
#     "savemodelname":"savedmodel/TNCC/TNCCpassage/",
#     "savehis":"logs/TNCCpassage/noprompt"
#
# }
# usetokenizer=AutoTokenizer.from_pretrained(config["checkpoint"])
# test_dataset_reloaded, tf_test_dataset = getTFDataset(config=config, tokenzier=usetokenizer)
# modelfilepath = config["savemodelname"]+"DPCNN"
# basemodel = tf.keras.models.load_model(modelfilepath)
# print("predict")
# # predics = basemodel.predict(tf_test_dataset.take(3))
# predics = basemodel.predict(tf_test_dataset)
# prelabel = tf.math.argmax(predics, 1).numpy()
# # print(prelabel)
# getReport(prelabel,test_dataset_reloaded["labels"])


