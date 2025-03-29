# pip install azure-ai-inference azure-identity
from azure.ai.inference import ChatCompletionsClient
from azure.identity import DefaultAzureCredential

# for serverless API or managed compute
key = "GABChkvdxlKM5K1PWE9TLo8nuXlj6rEKmXFdZ08fMT29gHvJLxwaJQQJ99BCACfhMk5XJ3w3AAAAACOGIUen"
endpoint = "https://admg-m830c1h3-swedencentral.cognitiveservices.azure.com/openai/deployments/gpt-4o"

# Use Azure Active Directory (AAD) authentication
credential = DefaultAzureCredential()

endpointsclient = ChatCompletionsClient(
    endpoint=endpoint,
    credential=credential,
)

import os 
from azure.ai.inference import ChatCompletionsClient 
from azure.ai.inference.models import SystemMessage, UserMessage 

model_name = "gpt-4o" 
client = ChatCompletionsClient(     
    endpoint=endpoint,     
    credential=credential, ) 

response = client.complete(     
    messages=[  SystemMessage(content="You are a helpful assistant."),         
                UserMessage(content="I am going to Paris, what should I see?")     ],     
                max_tokens=4096,     
                temperature=1.0,     
                top_p=1.0,     
                model=model_name ) 

print(response.choices[0].message.content)
