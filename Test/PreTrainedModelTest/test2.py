# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-zephyr-3b")
model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-zephyr-3b", device_map="auto")

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))

# import torch
# print("Cuda Availability: ", torch.cuda.is_available())
# torch.zeros(1).cuda()
