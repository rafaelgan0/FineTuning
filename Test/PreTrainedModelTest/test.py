from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import prompts
import torch
import time

########################################################################
# ModelLoader Class
# Load in 8bit: model_loader = ModelLoader("gpt2", load_8bit=True)
# Load in 16bit: model_loader = ModelLoader("gpt2", load_16bit=True)
# Load in 32bit: model_loader = ModelLoader("gpt2")
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
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float16, trust_remote_code=True).to(0)
            print("Finished Loading.")
        else:
            print(f"Loading {self.model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True).to(0)
            print("Finished Loading.")

    def generate_output(self, max_new_tokens, inputStr):
        if self.model == None:
            print("Model not loaded.")
            return
        
        encoded_input = self.tokenizer.encode(inputStr, return_tensors='pt').to(0)
        output = self.model.generate(encoded_input, max_new_tokens=max_new_tokens, num_return_sequences=1, no_repeat_ngram_size=2, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        new_output = output[0, encoded_input.shape[1]:]
        outputStr = self.tokenizer.decode(new_output, skip_special_tokens=True)
        return outputStr
    
def zephyr_p1(review):
    prompt = f"""<|user|>
Classify the following review as 'J', 'W', or 'V', where:
'J' is positive or encouraging or constructive or specific or has a positive tone or consists of factual statements and a positive tone.
'J' is positive.
'J' is encouraging.
'J' is constructive.
'J' is specific.
'J' has a positive tone.
'J' consists of factual statements and a positive tone.
'V' contains personal attacks
'V' contains excessive negativity without constructive feedback.
'W' is not encouraging.
'W' is not discouraging.
'W' consists of factual statements.

Review: "{review}"

Please classify this review as either 'J', 'W', or 'V'. Only output 'J', 'W', or 'V' with no additional explanation. Your classification:
<|endoftext|>
<assistant>
"""
    return prompt

def mistral_p1(review):
    prompt = f"""<s>[INST]
Classify the following review as 'J', 'W', or 'V', where:
'J' is positive or encouraging or constructive or specific or has a positive tone or consists of factual statements and a positive tone.
'V' contains personal attacks and a discouraging tone or excessive negativity without constructive feedback.
'W' meets the standards of professionalism and is not discouraging and is not encouraging or consists of factual statements.

Review: "{review}"

Please classify this review as either 'J', 'W', or 'V'. Only output 'J', 'W', or 'V' with no additional explanation. [/INST] Your classification: """
    return prompt
# Config Class
# Load in 8bit: config = Config(prompts.mistral_p1, "mistralai/Mistral-7B-Instruct-v0.1", load_8bit=True)
# Load in 16bit: config = Config(prompts.mistral_p1, "mistralai/Mistral-7B-Instruct-v0.1", load_16bit = True)
# Load in 32bit: config = Config(prompts.mistral_p1, "mistralai/Mistral-7B-Instruct-v0.1")
class Config:
    def __init__(self, prompt_func, model_name, load_8bit=False, load_16bit=False):
        self.model_name = model_name
        self.load_8bit = load_8bit
        self.load_16bit = load_16bit
        self.prompt_func = prompt_func
########################################################################
# functions
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

def evaluate_prompt(config):
    model_loader = ModelLoader(config.model_name, load_8bit=config.load_8bit, load_16bit=config.load_16bit)
    model_loader.load_model()
    match_counter = 0
    
    df = pd.read_csv('comprehensive_combined_annotations.csv')       
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
            
            word_count = len(review.split())
            score_time = end_time - start_time
            
            try:
                if int(score) == int(majority):
                    match_counter += 1
            except Exception:
                pass
            results.loc[len(results)] = [review_id, word_count, score, score_time, majority]
            
    match_percentage = (match_counter / len(df)) * 100
    print(f"Match Count: {match_counter}")
    print(f"Match Percentage: {match_percentage}%")
    results.to_csv('evaluate_model.csv', index=False)
    del model_loader.model
    torch.cuda.empty_cache()
########################################################################
    
config = Config(prompts.mistral_p1, "mistralai/Mistral-7B-Instruct-v0.1", load_16bit=True)
evaluate_prompt(config)