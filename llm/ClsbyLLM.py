"""
直接使用lLM对sub数据集进行抽取实验

"""
import openai
import os
import pandas as pd
import time
from transformers import AutoTokenizer, AutoModel
labeldic={'Politics': 0, 'Education': 1, 'Economics': 2, 'Environment': 3, 'Instruments': 4, 'Religion': 5, 'Medicine': 6, 'Tourism': 7, 'Arts': 8, 'Customs': 9, 'Literature': 10, 'Language': 11}

dffile='subset2.xlsx'
df=pd.read_excel(dffile)
print(df.head(2))

#
def generate_text(prompt,text):
    key="your key "
    openai.api_key = os.getenv('OPENAI_KEY', key)
    openai.api_base = "openai or other llm api"
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=1.0,
    )
    #也可以使用gpt4
    # response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=messages,
    #     temperature=1.0,
    # )
    result = ''
    for choice in response.choices:
        result += choice.message.content

    print("summary_result:\n", result)
    return result


result=[]
prompt="Specify the category of the following Tibetan news,assigning them the values of: ['Politics','Education', 'Economics','Environment','Instruments','Religion', 'Medicine','Tourism','Arts','Customs', 'Literature','Language'], Generate the category and do not answer other things.The content is :"

# for text in df['text']:
#     generate_text(prompt,text)
#     result.append(result)
#
# df['llm out']=result
#
# df.to_excel('result.xlsx')


modelcheckpoints='F:\ChatGLM2-6B-main\models\chatglm2-6b'
tokenizer = AutoTokenizer.from_pretrained(modelcheckpoints, trust_remote_code=True)
model = AutoModel.from_pretrained(modelcheckpoints, trust_remote_code=True, device='cuda')
model = model.eval()
# response, history = model.chat(tokenizer, "你好", history=[])
# print(response)

cnprompt="将以下藏文新闻进行分类 其类别为['Politics','Education', 'Economics','Environment','Instruments','Religion', 'Medicine','Tourism','Arts','Customs', 'Literature','Language'], 生成其对应类别不要回答其他问题.新闻内容为:"
for text in df['text']:
    content=cnprompt+text
    print('oooooooooooooooo')
    print(content)
    response, history = model.chat(tokenizer, content, history=[])
    print(response)
    print('oooooooooooooooo')
    result.append(response)
    time.sleep(2)

print('over')
for r in result:
    print(r)