from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import time
import re
import psutil

#if u load the model in a separate cell u can run this once at the start and the model will stay loaded for the rest of the script, so u wont need to reload the model if u
#change the prompt and rerun in the next cell
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

print("loading model")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
model.to(device)
print("finished loading")

def generate_output(inputStr):
    inputs = tokenizer(inputStr, return_tensors='pt').to(device)
    output = model.generate(inputs, max_new_tokens=10, num_return_sequences=1, no_repeat_ngram_size=2, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    new_output = output[0, inputs.shape[1]:]
    outputStr = tokenizer(new_output, skip_special_tokens=True)
    return outputStr

def score_review(sentence):
    prompt = f"""<|user|>
    Classify the following review as 'J', 'W' or 'V', where:
    'J' is positive or encouraging or constructive or specific or has a positive tone or consists of factual statements and a positive tone.
    'J' is positive.
    'J' is encouraging.
    'J' is constructive.
    'J' is specific.
    'J' has a positive tone.
    'J' consists of factual statements and a positive tone.
    'V' contains personal attacks.
    'V' contains excessive negativity without constructive feedback.
    'W' is not encouraging.
    'W' is not discouraging.

    Review = "{sentence}"

    Please classify this review as either 'J', 'W' or 'V'. Only output 'J', 'W' or 'V' with no additional explanation.
    Your classification:
    <|end of text|>
    <|assistant|>
    """

    score = generate_output(prompt)
    res = 0
    if 'v' in score.lower():
        res = '2'
    if 'j' in score.lower():
        res = '0'
    if 'w' in score.lower():
        res = '1'
    return res

agreementScore = 0

file_path = '../../../Data/CombinedReviews/comprehensive_combined_annotations.csv'

reviews = pd.read_csv(file_path)

print(reviews)

for index, review in reviews.iterrows():
    if review["majority"] == score_review(review["review"]):
        agreementScore += 1

print(f"Agreement Score with Mistral 7B: {agreementScore}/99")