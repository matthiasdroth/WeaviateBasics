import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

class EmbeddingGenerator:
    def __init__(self, checkpoint):
        # initialize the tokenizer and model for HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint)

    def get_embedding(self, text):
        # get text embeddings from HuggingFace model
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            model_output = self.model(**inputs)[0]
        # taking the first element of model_output as embeddings
        return model_output[0].numpy()

    def get_embeddings(self, text_list):
        # turn a list of texts into a list of embeddings
        embeddings = []
        for text in text_list:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def get_mean_embedding(self, text):
        return np.mean(self.get_embedding(text), axis=0)

    def get_mean_embeddings(self, text_list):
        # turn a list of texts into a list of embeddings
        embeddings = []
        for text in text_list:
            embedding = self.get_mean_embedding(text)
            embeddings.append(embedding)
        return embeddings

class WeaviateOps:
    def __init__(self, client_url, client_api_key, schema):
        self.client=weaviate.Client(
            url=client_url,
            auth_client_secret=weaviate.AuthApiKey(api_key=client_api_key)
        )
        self.schema=schema

    def create_one(self, object_id, class_name, text, embedding):
        self.client.data_object.create({
            "id": f"{object_id}",
            "class": f"{class_name}",
            "properties": {"text": text, "vector": embedding}
        })
        # assuming "text" is the relevant text and "embedding" that text's corresponding embedding
        object_data = {
            "id": object_id,
            "class": class_name,
            "properties": {
                "text": text,
                "vector": embedding.tolist()
            }
        }
        # store the object in Weaviate
        client.data_object.create(object_data)
