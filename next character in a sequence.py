import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'  # You can use other models like 'gpt2-medium' or 'gpt2-large' for more capacity
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def generate_next_character(text):
    # Encode input text and convert to PyTorch tensor
    input_ids = tokenizer.encode(text, return_tensors='pt')
   
    

    # Generate the next token (character)
    with torch.no_grad():
        output = model.generate(input_ids, max_length=input_ids.size(1) + 1)

    # Decode the generated token and return it as the next character
    next_character = tokenizer.decode(output[:, -1])

    return next_character

# Prompt the user for input
user_input = input("Enter text: ")

# Generate the next character
next_character = generate_next_character(user_input)

print("Next character:", next_character)
