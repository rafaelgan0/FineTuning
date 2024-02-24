from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import prompts
import torch
import torch.nn as nn
from parallelformers import parallelize
import time

print("Cuda Availability: ", torch.cuda.is_available())

class ModelLoader:
    def __init__(self, model_id, load_8bit=False, load_16bit=False):
        self.model_id = model_id
        self.load_8bit = load_8bit
        self.load_16bit = load_16bit
        self.model = None
        self.tokenizer = None

    def load_model(self):
        if self.load_8bit:
            print(f"Loading {self.model_id} in 8bit...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, load_in_8bit=True, trust_remote_code=True)
            print("Finished Loading.")
        elif self.load_16bit:
            print(f"Loading {self.model_id} in 16bit...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True)
            parallelize(self.model, num_gpus=4, fp16=True, verbose='detail')
            print("Finished Loading.")
        else:
            print(f"Loading {self.model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True, device_map="auto")
            print("Finished Loading.")

    def generate_output(self, max_new_tokens, inputStr):
        if self.model is None:
            print("Model not loaded.")
            return
        
        encoded_input = self.tokenizer.encode(inputStr, return_tensors='pt')
        output = self.model.generate(encoded_input, max_new_tokens=max_new_tokens, num_return_sequences=1, no_repeat_ngram_size=2, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        new_output = output[0, encoded_input.shape[1]:]
        outputStr = self.tokenizer.decode(new_output, skip_special_tokens=True)
        return outputStr

class Config:
    def __init__(self, prompt_func, model_name, load_8bit=False, load_16bit=False):
        self.model_name = model_name
        self.load_8bit = load_8bit
        self.load_16bit = load_16bit
        self.prompt_func = prompt_func

def score_review(model_loader, review, config):
    prompt = config.prompt_func(review)
    score = model_loader.generate_output(2, prompt)
    if 'v' in score.lower():
        score = '2' 
    elif 'j' in score.lower():
        score = '0'
    elif 'w' in score.lower():
        score = '1'
    return score

def evaluate_prompt(config, model_loader, input_file, output_file):
    match_counter = 0

    df = pd.read_csv(input_file)
    df = df.head(10)
    results = pd.DataFrame(columns=['review_id', 'word_count', 'score', 'score_time', 'majority'])

    for index, row in df.iterrows():
        review = row['review']
        majority = row['majority']
        review_id = row['id']
        review = review.replace('"', "'")

        if pd.notnull(review) and review != '':
            start_time = time.time()
            score = score_review(model_loader, review, config)
            end_time = time.time()
            print(score)
            word_count = len(review.split())
            score_time = end_time - start_time

            try:
                if int(score) == int(majority):
                    match_counter += 1
            except Exception:
                pass
            results.loc[len(results)] = [review_id, word_count, score, score_time, majority]

            review_padded = str(row['id']).ljust(15, ' ')
            print(review_id, score, majority)
    match_percentage = (match_counter / len(df)) * 100
    print(f"Match Count: {match_counter}")
    print(f"Match Percentage: {match_percentage}%")
    results.to_csv(f'{output_file}', index=False)
    torch.cuda.empty_cache()

num_gpus = torch.cuda.device_count()
print(f"Number of GPUs: {num_gpus}")

config = Config(prompts.mistral_p1, "stabilityai/stablelm-zephyr-3b")
model_loader = ModelLoader(config.model_name, load_8bit=config.load_8bit, load_16bit=config.load_16bit)
model_loader.load_model()
print(model_loader.model.memory_allocated())
print(model_loader.model.memory_reserved())
print(model_loader.model.memory_chached())

evaluate_prompt(config, model_loader, "/home/rgan2/FineTuning/FineTuning/Data/CombinedReviews/comprehensive_combined_annotations.csv", "test_output.csv")
