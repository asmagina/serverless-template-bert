# This file runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model
from transformers import BertModel, BertTokenizer 

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    m = BertModel.from_pretrained('bert-base-uncased')
    t = BertTokenizer.from_pretrained('bert-base-uncased')

if __name__ == "__main__":
    download_model()