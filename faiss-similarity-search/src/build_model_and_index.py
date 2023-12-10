import numpy as np 
import pandas as pd
import os
import time

import faiss
from sentence_transformers import SentenceTransformer
import torch

def create_index(data, text_column, model):
    embedding = model.encode(data[text_column].to_list())
    
    # dimension
    dimension = embedding.shape[1]
    
    # create the vector/embeddings and their IDs                                                                                                                                                                                                                                                embedding vectors and ids:
    db_vectors = embedding.copy().astype(np.float32)
    db_ids = data.index.values.astype(np.int64)

    faiss.normalize_L2(db_vectors)
    index = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIDMap(index)
    index.add_with_ids(db_vectors, db_ids)
    
    return index

if __name__=="__main__":
    df_news = pd.read_csv("../input/news-summary/news_summary_more.csv") 

    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer("all-MiniLM-L6-v2", device=torch_device)

    modelPath = "../artifacts/sentence_transformer_trained_model"
    model.save(path=modelPath)

    news_index = create_index(data=df_news,
                              text_column='text',
                              model=model)
    
    faiss.write_index(news_index, 'news_dataset_index')

    
