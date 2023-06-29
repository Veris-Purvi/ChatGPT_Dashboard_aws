import boto3
import pandas as pd
import openai
import json
import numpy as np
from io import StringIO
from botocore.exceptions import ClientError
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('.env')

# Set the OpenAI API key from the environment variable
openai.api_key = os.environ['API_KEY']

s3 = boto3.client('s3')
def get_embedding(text, model):
    embedding = openai.Embedding.create(
    input=text, model=model
    )["data"][0]["embedding"]
    return embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def lambda_handler(event, context):
    print("New Event Recieved")
    print(event)
    
    Body=json.loads(event['body'])
    question = Body['question']
    folder_name = Body['folder_name']
    
    print(f"Folder Name - {folder_name}")
    print(f"Question - {question}")

    # Get the CSV file from S3
    try:
        print("Loading S3 Bucket")
        obj = s3.get_object(Bucket='csvfiles100', Key=f'{folder_name}/embedding.csv')
        print("S3 Bucket Loaded Successfully")
        print("Decoding Body")
        body = obj['Body'].read().decode('utf-8')
        csv_file = StringIO(body)
        print("Body Decoded Successfully")
    except ClientError as e:
        print(e)
        raise Exception("Error getting S3 object: {}".format(e))

    # Preprocess the data
    emb = pd.read_csv(csv_file)
    print("Dataframe Initialised")
    emb['embedding'] = emb['embedding'].apply(lambda x: '[]' if x is None or not isinstance(x, str) else x)
    emb['embedding'] = emb['embedding'].apply(eval)
    emb['embedding'] = emb['embedding'].apply(lambda x: [float(i) for i in x])
    question_vector = get_embedding(question, model='text-embedding-ada-002')
    emb["similarities"] = emb['embedding'].apply(lambda x: cosine_similarity(x, question_vector) if len(x) > 0 else 0)
    emb = emb.sort_values("similarities", ascending=False).head(4)
    # Generate prompt for OpenAI
    nontext = []
    for i, row in emb.iterrows():
        if isinstance(row['Combined'], str):
            nontext.append(row['Combined'])
    text = "\n".join(nontext)
    nontext = text
    prompt = f"""Answer the following question using only the context below.

    Context:
    {nontext}

    Q: {question}
    A:"""

    # Call OpenAI API to get the answer
    try:
        print("Calling OpenAI API")
        ans = openai.Completion.create(
            prompt=prompt,
            temperature=1,
            max_tokens=200,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            model="text-davinci-003",
        )["choices"][0]["text"].strip(" \n")
        print(f"Ans -: {ans}")
        res = {"msg": ans, "status": "success"}
        print(res)
        return {
            'statusCode': 200,
            'body': json.dumps(res)
            }
    except Exception as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'message': str(e)})
        }

