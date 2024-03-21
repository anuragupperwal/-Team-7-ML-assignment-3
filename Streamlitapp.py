import streamlit as st
import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt # for making figures
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
from pprint import pprint


with open('shakespeare2.txt', 'r') as file:
    text = file.read()
tokens = text.split()

# Step 2: Create a vocabulary
vocab = sorted(set(tokens))
stoi = {word: i for i, word in enumerate(vocab)}
itos = {i: word for word, i in stoi.items()}
vocab_size = len(vocab)


class NextChar(nn.Module):
  def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, emb_dim)
    self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
    self.lin2 = nn.Linear(hidden_size, hidden_size)
    self.lin3 = nn.Linear(hidden_size, vocab_size)

  def forward(self, x):
    x = self.emb(x)
    x = x.view(x.shape[0], -1)
    x = torch.sin(self.lin1(x))
    x = torch.sin(self.lin2(x))
    x = self.lin3(x)
    return x
  
model10_150 = NextChar(10, len(stoi), 150, 64)
model10_150.load_state_dict(torch.load("model10_150.pth"))

model10_100 = NextChar(10, len(stoi), 100, 64)
model10_100.load_state_dict(torch.load("model10_100.pth"))

model7_150 = NextChar(7, len(stoi), 150, 64)
model7_150.load_state_dict(torch.load("model7_150.pth"))

model7_100 = NextChar(7, len(stoi), 100, 64)
model7_100.load_state_dict(torch.load("model7_100.pth"))

model5_150 = NextChar(5, len(stoi), 150, 64)
model5_150.load_state_dict(torch.load("model5_150.pth"))

model5_100 = NextChar(5, len(stoi), 100, 64)
model5_100.load_state_dict(torch.load("model5_100.pth"))



def generate_name(model, itos, stoi, block_size, context ,max_len=10):
    name = ''
    for i in range(max_len):
        x = torch.tensor(context).view(1, -1)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        ch = itos[ix]
        name =name+' '+ch
        context = context[1:] + [ix]
    return name


def createContext(input_text , block_size):
  input_text= input_text.split()
  input_text= input_text[-block_size:]
  context = [stoi[input_text[j]] for j in range(block_size)]
  return context


# Streamlit app
st.title("Next K Text Generation with MLP")
st.sidebar.title("Settings")

input_string = st.sidebar.text_input("Input String")
nextk = st.sidebar.number_input("Next K Tokens", min_value=1, max_value=500, value=150)
block_size = st.select_slider("Block Size", options=[5, 7, 10], value=7)
embedding_size = st.select_slider("Embedding Size", options=[50, 100, 150], value=100)


if st.sidebar.button("Generate Text"):
    
    if block_size == 5:
        context = createContext(input_string, 5)
        if embedding_size == 100:
            generated_text = generate_name(model5_100, itos, stoi, 5, context, max_len=nextk)
        else:
            generated_text = generate_name(model5_150, itos, stoi, 5, context, max_len=nextk)
    elif block_size == 7:
        context = createContext(input_string, 7)
        if embedding_size == 100:
            generated_text = generate_name(model7_100, itos, stoi, 7, context, max_len=nextk)
        else:
            generated_text = generate_name(model7_150, itos, stoi, 7, context, max_len=nextk)
    else:
        context = createContext(input_string, 10)
        if embedding_size == 100:
            generated_text = generate_name(model10_100, itos, stoi, 10, context, max_len=nextk)
        else:
            generated_text = generate_name(model10_150, itos, stoi, 10, context, max_len=nextk)
    st.write("Generated Text:")
    st.write(generated_text)
   


