from transformers import pipeline, BertModel, BertTokenizer
import torch

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global tokenizer
    
    device = 0 if torch.cuda.is_available() else -1
    model = BertModel.from_pretrained('bert-base-uncased',          output_hidden_states = True, device=device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer

    # Parse out your arguments

    tokenized_dict = tokenizer.encode_plus(
        model_inputs.get("text"),
        add_special_tokens=True,
        max_length=5
    )

    tokenized_text = torch.tensor(tokenized_dict["input_ids"])
    with torch.no_grad():
        embeddings = model(torch.tensor(tokenized_text.unsqueeze(0)))

    # Return the results as a dictionary
    return {'answer': embeddings[0][:, 0, :].squeeze().numpy().tolist()}
