import numpy as np 
import pandas as pd
import os
import time
import json

import faiss
from sentence_transformers import SentenceTransformer
import torch


# grab model and index details from lambda environment variables
model_name_or_path = str(os.environ.get('model_name_or_path')) #"../artifacts/sentence_transformer_trained_model/"
index_path = str(os.environ.get('index_path')) #"../artifacts/index/"
data_path = str(os.environ.get('data_path')) #"../input/news_summary_more.csv"

def lambda_handler(event, context):
    
    try:
        # Parse the JSON input from the API Gateway
        request_body = json.loads(event['body']) 
        
        # Retrieve the user query and top K from the JSON input
        user_query = request_body.get('query', '')
        k = request_body.get('top_k', '')
        
        # Load the model, index and data
        torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer(model_name_or_path, device=torch_device)
        index = faiss.read_index(index_path)
        data = pd.read_csv(data_path)

        print("Sentence Transformer Model and Index are loaded.")

        response_data = search_your_query(data=data,
                                          index=index,
                                          model=model, 
                                          query=user_query,
                                          k=k)
        
        # Create a response body
        response_body = {
            'message': 'Success',
            'data': response_data
        }
        
        # Return the response to the API Gateway
        return {
            'statusCode': 200,
            'body': json.dumps(response_body)
        }
    
    except Exception as e:
        # Handle any errors and return an error response
        return {
            'statusCode': 500,
            'body': json.dumps({'message': 'Error', 'error': str(e)})
        }
    
def search_your_query(data, index, model, query, k):
    
    t=time.time()
    query_vector = model.encode([query]).astype(np.float32)
    faiss.normalize_L2(query_vector)
    
    similarities, similarities_ids = index.search(query_vector, k)
    print('totaltime: {}\n'.format(time.time()-t))
    
    similarities = np.clip(similarities, 0, 1)
    
    output = []
    for i in range(len(similarities_ids[0])):
        item = {
            'id': similarities_ids[0][i],
            'text': data.loc[similarities_ids[0][i], 'text'],
            'similarity_score': similarities[0][i]
        }
        output.append(item)
    
    return output
