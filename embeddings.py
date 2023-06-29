import json
import boto3
import pandas as pd
import openai
import os

from io import StringIO
from botocore.exceptions import ClientError

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('.env')

# Set the OpenAI API key from the environment variable
openai.api_key = os.environ['API_KEY']
s3 = boto3.client('s3')

bucket_name = 'csvfiles100'
file_name = 'embedding.csv'

# create a set to store the processed files
processed_files = set()

def get_embedding(text, model):
    embedding = openai.Embedding.create(input=text, model=model)["data"][0]["embedding"]
    return embedding


def lambda_handler(event, context):
    try:
        # Get bucket name and key (file name) from the event
        bucket_name = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']
        
        # Check if the file has already been processed
        if key in processed_files:
            return {
                'statusCode': 200,
                'body': json.dumps('File already processed')
            }
        
        # Add file to the processed set
        processed_files.add(key)
        
        folder_name = key.split('/')[0]

        # Check if embeddings file exists, if not create a new file
        try:
            s3.head_object(Bucket=bucket_name, Key=folder_name + '/' + file_name)
            file_exists = True
        except:
            file_exists = False

        if not file_exists:
            # Create new embeddings file with 'Combined' column
            df = pd.DataFrame(columns=['Combined'])
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            s3.put_object(Bucket=bucket_name, Key=folder_name + '/' + file_name, Body=csv_buffer.getvalue())

        response = s3.get_object(Bucket=bucket_name, Key=key)
        file_content = response['Body'].read().decode('utf-8')

        # Parse CSV file into a Pandas DataFrame
        df = pd.read_csv(StringIO(file_content))
        ref = df.copy()

        # Add OpenAI embeddings to DataFrame
        embedding_list = []
        for row in ref['Combined']:
            try:
                embedding = get_embedding(row, model='text-embedding-ada-002')
                embedding_list.append(embedding)
            except:
                embedding_list.append(None)

        ref['embedding'] = embedding_list

        # Append the new embeddings to the existing embeddings file
        response = s3.get_object(Bucket=bucket_name, Key=folder_name + '/' + file_name)
        existing_content = response['Body'].read().decode('utf-8')
        existing_df = pd.read_csv(StringIO(existing_content))
        existing_df = pd.concat([existing_df, ref[['Combined', 'embedding']]], ignore_index=True)

        # Write the updated embeddings file back to S3
        csv_buffer = StringIO()
        existing_df.to_csv(csv_buffer, index=False)
        s3.put_object(Bucket=bucket_name, Key=folder_name + '/' + file_name, Body=csv_buffer.getvalue())

    except ClientError as e:
        print('Error:', e)
        raise e

    return {
        'statusCode': 200,
        'body': json.dumps('Embeddings added successfully!')
    }
