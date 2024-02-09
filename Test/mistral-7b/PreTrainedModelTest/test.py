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
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=torch.float16, trust_remote_code=True)
model.to(device)
print("finished loading")

def generate_output(messages):
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)
    model.to(device)
    generated_ids = model.generate(model_inputs, max_new_tokens=10, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    
    return decoded[0]

def score_review(sentence):
    prompt = [
        {"role": "assistant", "content": f"""
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
        """}
    ]


    score = generate_output(prompt)
    print(score)
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
number = 0
print(reviews)

for index, review in reviews.iterrows():
    if review["majority"] == score_review(review["review"]):
        agreementScore += 1
        number += 1
        print(f"""Review {number}""")

print(f"Agreement Score with Mistral 7B: {agreementScore}/99")