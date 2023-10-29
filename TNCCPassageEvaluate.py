import tensorflow as tf
from transformers import XLMRobertaTokenizer, TFXLMRobertaModel,AutoTokenizer
from datasets import Dataset
from TNCCEvaluate import get_orlabel,getTarget,ensamble,getPredictions
#参数
config1={"checkpoint":"hfl/cino-small-v2","testpath":"dataset/TNCC/TNCCpassage/Prompt/edition1prompt0/test","savemodelname":"savedmodel/TNCC/TNCCpassage/editionprompt0-small"}
config2={"checkpoint":"hfl/cino-small-v2","testpath":"dataset/TNCC/TNCCpassage/Prompt/edition1prompt1/test","savemodelname":"savedmodel/TNCC/TNCCpassage/editionprompt1-small"}
config3={"checkpoint":"hfl/cino-small-v2","testpath":"dataset/TNCC/TNCCpassage/Prompt/edition1prompt2/test","savemodelname":"savedmodel/TNCC/TNCCpassage/editionprompt2-small"}

config4={"checkpoint":"sangjeedondrub/tibetan-roberta-base","testpath":"dataset/TNCC/TNCCpassage/Prompt/edition1prompt0/test","savemodelname":"savedmodel/TNCC/TNCCpassage/editionprompt0-tibert"}
config5={"checkpoint":"sangjeedondrub/tibetan-roberta-base","testpath":"dataset/TNCC/TNCCpassage/Prompt/edition1prompt1/test","savemodelname":"savedmodel/TNCC/TNCCpassage/editionprompt1-tibert"}
config6={"checkpoint":"sangjeedondrub/tibetan-roberta-base","testpath":"dataset/TNCC/TNCCpassage/Prompt/edition1prompt2/test","savemodelname":"savedmodel/TNCC/TNCCpassage/editionprompt2-tibert"}

def getTFDataset(config,tokenzier):
    testds = Dataset.load_from_disk(config["testpath"])
    print(testds)
    def tokenize_fun(example):
        all = [(x + y) for x, y in zip(example["textfeatures"], example["textprompt"])]
        example["allfeature"] = all
        return tokenzier(example["allfeature"], truncation=True, padding='max_length', max_length=500)

    test_tokenized_ds = testds.map(tokenize_fun, batched=True, remove_columns = ["textfeatures","textprompt"])
    tf_test_dataset = test_tokenized_ds.to_tf_dataset(
        columns=["attention_mask", "input_ids"],
        label_cols=["labels"],
        shuffle=False,
        # collate_fn=data_collator,
        batch_size=12,
    )

    return testds,tf_test_dataset

def getPredictions(config):
    tokenzier = AutoTokenizer.from_pretrained(config["checkpoint"])
    test_dataset_reloaded, tf_test_dataset = getTFDataset(config=config, tokenzier=tokenzier)
    basemodelfilepath = config["savemodelname"]
    basemodel = tf.keras.models.load_model(basemodelfilepath)
    print("predict")
    # predics = basemodel.predict(tf_test_dataset.take(3))
    predics=basemodel.predict(tf_test_dataset)

    # 实际标签
    # rellabels = get_orlabel(test_dataset_reloaded["labels"][:36])
    rellabels=get_orlabel(test_dataset_reloaded["labels"])
    return predics,rellabels

predics0,rellabels0=getPredictions(config4)
getTarget(predics0,rellabels0)
predics1,rellabels1=getPredictions(config5)
getTarget(predics1,rellabels1)
predics2,rellabels2=getPredictions(config6)
getTarget(predics2,rellabels2)
#得到集成学习的结果后，进行预测评价
resultlist=[predics0,predics1,predics2]
print("使用集成学习")
ensambleresult1=ensamble(resultlist,model="avg")
ensambleresult2=ensamble(resultlist,model="max-min")
#评估
print("avg")
getTarget(ensambleresult1,rellabels1)

print("max")
getTarget(ensambleresult2,rellabels1)
print("over")