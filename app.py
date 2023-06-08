from transformers import pipeline, BertForQuestionAnswering, BertTokenizer
import torch
import time

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global tokenizer
    
    device = 0 if torch.cuda.is_available() else -1
    model =  BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    pipeline('question-answering', model=model, tokenizer=tokenizer)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer

    # Parse out your arguments
    question = model_inputs.get('question', None)
    text = model_inputs.get('text', None)

    start = time.time()

    input_ids = tokenizer.encode(question, text)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)


    sep_idx = input_ids.index(tokenizer.sep_token_id)
    #number of tokens in segment A (question) - this will be one more than the sep_idx as the index in Python starts from 0
    num_seg_a = sep_idx+1
    #number of tokens in segment B (text)
    num_seg_b = len(input_ids) - num_seg_a
    #creating the segment ids
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    #making sure that every input token has a segment id
    assert len(segment_ids) == len(input_ids)

    output = model(torch.tensor([input_ids]),  token_type_ids=torch.tensor([segment_ids]))
    answer = None
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    if answer_end >= answer_start:
        answer = " ".join(tokens[answer_start:answer_end+1])
    else:
        answer = "I am unable to find the answer to this question. Can you please ask another question?"
    stop = time.time()
    # Return the results as a dictionary
    return {'answer': answer, 'time': stop - start}
