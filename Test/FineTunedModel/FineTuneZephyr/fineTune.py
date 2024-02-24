import pandas as pd
import torch
import sys
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from accelerate import Accelerator
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("CUDNN version:", torch.backends.cudnn.version())

# Load data from JSONLines file
data_path = "/home/rgan2/FineTuning/FineTuning/Data/IndividualReviews/firstTrainingSet.jsonl"

reviews = []
ratings = []

with jsonlines.open(data_path, 'r') as reader:
    for obj in reader:
        reviews.append(obj['review'])
        ratings.append(int(obj['rating']))

# Define a custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, reviews, ratings, tokenizer, max_length):
        self.reviews = reviews
        self.ratings = ratings
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        rating = self.ratings[idx]

        encoding = self.tokenizer(review, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(rating, dtype=torch.long)
        }
    
accelerator = Accelerator()
device = accelerator.device

max_memory = {i: '7000MB' for i in range(torch.cuda.device_count())}
max_memory[0] = '1000MB'
print(max_memory)
# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-zephyr-3b")
# model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-zephyr-3b", device_map="auto", max_memory=max_memory, num_labels=5, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-zephyr-3b", device_map="auto", trust_remote_code=True)
# model = torch.nn.DataParallel(model, device_ids=(0,) ).cuda()
# optimizer = torch.optim.Adam(model.parameters())
# Prepare dataset

train_dataset = CustomDataset(reviews, ratings, tokenizer, max_length=2048)
optimizer = torch.optim.Adam(model.parameters())

# model, optimizer, data = accelerator.prepare(model, optimizer, train_dataset)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=3,              
    per_device_train_batch_size=1,   
    warmup_steps=500,                
    weight_decay=0.01,               
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained('stablelm-zephyr-3b-50.1')
