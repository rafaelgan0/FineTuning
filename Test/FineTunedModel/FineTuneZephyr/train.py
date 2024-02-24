import pandas as pd
import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import adapters
import os
from peft import LoraConfig

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("CUDNN version:", torch.backends.cudnn.version())

# custom dataset 
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
    
df = pd.read_csv('/home/rgan2/FineTuning/FineTuning/Test/FineTunedModel/FineTuneZephyr/training_data.csv')

tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-zephyr-3b")
tokenizer.pad_token = tokenizer.eos_token


df['input'] = df['input'].astype(str)
df['output'] = df['output'].astype(str)
max_length = 2048 # or any other value depending on your model's maximum length

unique_labels = df['output'].unique()
label_to_id = {label: id for id, label in enumerate(unique_labels)}

inputs = tokenizer(df['input'].tolist(), padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
outputs = df['output'].apply(lambda x: label_to_id[x]).tolist()  # assuming label_to_id is a dictionary mapping labels to unique integers

train_dataset = CustomDataset(inputs, outputs)
    
print("Loading model...")
# 
model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-zephyr-3b", trust_remote_code=True, num_labels=len(df['output'].unique()))#.to(0)
print("Finished loading model.")

# adapter stuff, needed when loading model quantized?????? -> works if not 8bit but not enough memory lol
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['q_proj', 'k_proj', 'v_proj'],
)

model.add_adapter(peft_config)
training_args = TrainingArguments(   
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs (what is an epoch lol)
    per_device_train_batch_size=1,   # batch size per device during training 
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
)

# Define the trainer
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
)


print("train train train train train")
trainer.train()
print("finished training")
model.save_pretrained('stablelm-zephyr-3b-50.1')

#sys.stdout.close()