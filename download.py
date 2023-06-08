# This file runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model
from transformers import pipeline, BertForQuestionAnswering, BertTokenizer

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    m = BertForQuestionAnswering.from_pretrained('bert-base-uncased',          output_hidden_states = True,)
    t = BertTokenizer.from_pretrained('bert-base-uncased')
    pipeline('qfeature-extraction', model=m, tokenizer=t)

if __name__ == "__main__":
    download_model()