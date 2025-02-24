# load parameters from yaml file.
import yaml
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

import asyncio
from googletrans import Translator
import pandas as pd

async def translate_text(text: str):
    translator = Translator()
    
    translated = await translator.translate(text, src='da', dest='en')
    
    return str(translated.text).strip()

text = "Hej, hvordan har du det?"
print(f"Original: {text}")

translated = asyncio.run(translate_text(text))
print(f"Translated: {translated}")

#df_pro = pd.read_csv(parameters['pro_media_dir'])
#df_pro['Full text'] = df_pro['Full text'].apply(lambda x: asyncio.run(translate_text(x)))
#df_pro.to_csv(parameters['pro_media_translated_dir'], index=False)
#print('pro_media.csv is Translated')

df_reg = pd.read_csv(parameters['reg_media_dir'])
df_reg['Content'] = df_reg['Content'].apply(lambda x: asyncio.run(translate_text(x)))
df_reg.to_csv(parameters['reg_media_translated_dir'], index=False)
print('reg_media.csv is Translated')
