import streamlit as st
import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt # for making figures
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
from pprint import pprint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = open('shakespeare2.txt', 'r').read()

unique_chars = list(set(''.join(data)))
unique_chars.sort()
to_string = {i:ch for i, ch in enumerate(unique_chars)}
to_int = {ch:i for i, ch in enumerate(unique_chars)}


class NextChar(nn.Module):
  def __init__(self, block_size, vocab_size, emb_dim, hidden_dims):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, emb_dim)
    self.lin1 = nn.Linear(block_size * emb_dim, hidden_dims[0])
    self.lin2 = nn.Linear(hidden_dims[0], hidden_dims[1])
    self.lin3 = nn.Linear(hidden_dims[1], vocab_size)

  def forward(self, x):
    x = self.emb(x)
    x = x.view(x.shape[0], -1)
    x = torch.sin(self.lin1(x))
    x = torch.sin(self.lin2(x))
    x = self.lin3(x)
    return x
  
# For context size 5 and embedding size 5
model_c5_e5 = NextChar(5, len(to_int), 5, [64, 64])
model_c5_e5.load_state_dict(torch.load("context_5_embedding_5.pth"))

# For context size 5 and embedding size 10
model_c5_e10 = NextChar(5, len(to_int), 10, [64, 64])
model_c5_e10.load_state_dict(torch.load("context_5_embedding_10.pth"))

# For context size 7 and embedding size 5
model_c7_e5 = NextChar(7, len(to_int), 5, [64, 64])
model_c7_e5.load_state_dict(torch.load("context_7_embedding_5.pth"))

# For context size 7 and embedding size 10
model_c7_e10 = NextChar(7, len(to_int), 10, [64, 64])
model_c7_e10.load_state_dict(torch.load("context_7_embedding_10.pth"))

random_seed = 3
g = torch.Generator()
g.manual_seed(random_seed)
torch.manual_seed(random_seed)
def generate_name(model,sentence, itos, stoi, block_size, max_len=10):
    original_sentence = sentence
    if len(sentence) < block_size:
        sentence = " " * (block_size - len(sentence)) + sentence
    using_for_predicction = sentence[-block_size:].lower()
    context = [stoi[word] for word in using_for_predicction]
    prediction = ""
    for i in range(max_len):
        x = torch.tensor(context).view(1, -1).to(device)
        print(type(model))
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        ch = itos[ix]
        prediction += ch
        context = context[1:] + [ix]

    return original_sentence + prediction

# Streamlit app
st.title("Next K Text Generation with MLP")
st.sidebar.title("Settings")


input_string = st.sidebar.text_input("Input String")
nextk = st.sidebar.number_input("Next K Tokens", min_value=1, max_value=500, value=150)
block_size = st.select_slider("Block Size", options=[5,7], value=5)
embedding_size = st.select_slider("Embedding Size", options=[5,10], value=5)


if st.sidebar.button("Generate Text"):

    if block_size == 5:
        context = input_string
        if embedding_size == 5:
            generated_text = generate_name(model_c5_e5,context, to_string, to_int, 5, max_len=nextk)
        else:
            generated_text = generate_name(model_c5_e10,context, to_string, to_int, 5, max_len=nextk)
    elif block_size == 7:
        context = input_string
        if embedding_size == 10:
            generated_text = generate_name(model_c7_e10, context, to_string, to_int, 7 ,max_len=nextk)
        else:
            generated_text = generate_name(model_c7_e5, context, to_string, to_int, 7, max_len=nextk)
    st.write("Generated Text:")
    st.write(generated_text)
   


