######################################################
# Use these package versions
#!pip install torchtext==0.6.0 torch==1.13.1
######################################################


import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] =':16:8' #This is a command to reduce non-deterministic behavior in CUDA
import warnings
warnings.simplefilter("ignore", UserWarning)
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import get_tokenizer
import sys
import argparse
from LanguageModel import LanguageModel
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')



def main():
  chkpt = "got_language_model"

  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logging.info('Using device: {}'.format(dev))

  logging.info("Loading tokenizer and vocab from vocab.pkl")  
  text_field = pickle.load(open("vocab.pkl", "rb"))
  vocab_size = len(text_field.vocab.itos)

  logging.info("Loading checkpoint {}".format(chkpt))
  lm = LanguageModel(vocab_size).to(dev)
  lm.load_state_dict(torch.load(chkpt, map_location=torch.device('cpu')))
  lm.eval()


  p = "the night is dark and full of terrors"

  # Torch is a bit frustrating at times and some things that ought to be deterministic are not.
  # This is an attempt to resolve that, but it doesn't work 100% of the time
  torch.use_deterministic_algorithms(True)
  seed = 42
  mlen = 150

  # torch.manual_seed(seed); np.random.seed(seed)
  # print("\n----------- Vanilla Sampling -----------")
  # print(sample(lm, text_field, prompt=p, max_len=mlen))

  # torch.manual_seed(seed); np.random.seed(seed)
  # print("\n------- Temp-Scaled Sampling 0.0001 -------")
  # print(sample(lm, text_field, prompt=p, temp=0.0001, max_len=mlen))

  # torch.manual_seed(seed); np.random.seed(seed)
  # print("\n------- Temp-Scaled Sampling 100 --------")
  # print(sample(lm, text_field, prompt=p, temp=100, max_len=mlen))

  # torch.manual_seed(seed); np.random.seed(seed)
  # print("\n----------- Top-k Sampling 1 -----------")
  # print(sample(lm, text_field, prompt=p, k=1, max_len=mlen))

  # torch.manual_seed(seed); np.random.seed(seed)
  # print("\n----------- Top-k Sampling 20 -----------")
  # print(sample(lm, text_field, prompt=p, k=20, max_len=mlen))

  # torch.manual_seed(seed); np.random.seed(seed)
  # print("\n----------- Top-p Sampling 0.001 -----------")
  # print(sample(lm, text_field, prompt=p, p=0.001, max_len=mlen))

  # torch.manual_seed(seed); np.random.seed(seed)
  # print("\n----------- Top-p Sampling 0.75 -----------")
  # print(sample(lm, text_field, prompt=p, p=0.75, max_len=mlen))

  # torch.manual_seed(seed); np.random.seed(seed)
  # print("\n----------- Top-p Sampling 1 -----------")
  # print(sample(lm, text_field, prompt=p, p=1, max_len=mlen))


  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Beam Search B=1 -----------")
  print(beamsearch(lm, text_field, prompt=p, beams=1, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Beam Search B=10 -----------")
  print(beamsearch(lm, text_field, prompt=p, beams=10, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Beam Search B=50 -----------")
  print(beamsearch(lm, text_field, prompt=p, beams=50, max_len=mlen))

  print()

############################################################################################
# TASK 2.1
############################################################################################

def expansions(model, h, c, cur_beams, vocab_size):
  # index values for all the words in the vocab
  all_words = torch.arange(0, vocab_size).unsqueeze(1)
  # Get the beams to be size num_beams x |V|
  expanded_beams = torch.repeat_interleave(cur_beams, vocab_size, 0)
  # Get the words to size |V| x L
  expanded_words = torch.repeat_interleave(all_words, len(cur_beams), 0)
  if cur_beams.dim() != 1:
    # Add the new words to the all the new beams
    beams_with_all_words = torch.cat((expanded_beams, expanded_words), dim=1)
  else:
    beams_with_all_words = expanded_words
  # Don't do batch first
  beams_with_all_words = beams_with_all_words.reshape(beams_with_all_words.shape[-1], -1)
  
  h = h.unsqueeze(1).expand(-1, beams_with_all_words.shape[1], -1)
  c = c.unsqueeze(1).expand(-1, beams_with_all_words.shape[1], -1)
  # z_out, z_h, z_c = model(beams_with_all_words, h, c)
  # Get the log_probs off all of the potential beams
  out = model(beams_with_all_words, h, c)[0]
  t = out[-1, torch.arange(0, vocab_size), torch.arange(0, vocab_size)]
  # new_log_probs = F.log_softmax(out[-1, ], dim=-1)
  new_log_probs = F.log_softmax(out[-1, torch.arange(0, vocab_size), torch.arange(0, vocab_size)], dim=-1)
  
  return beams_with_all_words, new_log_probs

def selection(beams_with_all_words, cur_log_probs, new_log_probs, num_beams):
  # cur_log_probs = torch.repeat_interleave(cur_log_probs, new_log_probs.shape[0], 0)
  # total_log_prob = cur_log_probs + new_log_probs
  new_log_probs += cur_log_probs
  vals, indices = torch.topk(new_log_probs, num_beams, dim=-1)
  # Select the topk best beams
  # t = indices[:num_beams, 0]
  t = indices[:num_beams]
  
  # Now get the best words from those beams
  to_ret = beams_with_all_words[torch.arange(num_beams).unsqueeze(1), t]
  z = new_log_probs[t]
  return to_ret, vals


def beamsearch(model, text_field, beams=5, prompt="", max_len=50):
  # Initial vals
  num_layers = 3
  h_c_out = 512
  h = torch.zeros(num_layers, h_c_out)
  c = torch.zeros(num_layers, h_c_out)

  # Run prompt through
  indices = text_field.process([text_field.tokenize(prompt.lower())])
  indices = indices.squeeze()
  out, h, c = model(indices[:-1], h, c)
  out, h, c = model(indices[-1:], h, c)
  vocab_size = len(text_field.vocab.itos)

  cur_beams = torch.empty(beams)
  cur_log_probs = F.log_softmax(out[-1], dim=-1)

  for _ in range(max_len):

    beams_with_all_words, new_log_probs = expansions(model, h, c, cur_beams, vocab_size)

    cur_beams, cur_log_probs = selection(beams_with_all_words, cur_log_probs, new_log_probs, beams)

  return prompt + " " + reverseNumeralize(cur_beams[0], text_field)

# def beamsearch(model, text_field, beams=5, prompt="", max_len=50):
#   # Initial vals
#   num_layers = 3
#   h_c_out = 512
#   h = torch.zeros(num_layers, h_c_out)
#   c = torch.zeros(num_layers, h_c_out)

#   # Run prompt through
#   indices = text_field.process([text_field.tokenize(prompt.lower())])
#   indices = indices.squeeze()
#   out, h, c = model(indices[:-1], h, c)
#   out, h, c = model(indices[-1:], h, c)
#   cur_log_prob = F.log_softmax(out, -1)
#   cur_log_probs = cur_log_prob.repeat(beams, 1)
#   vocab_size = len(text_field.vocab.itos)

#   all_beams = [] * beams
#   vocab_size = len(text_field.vocab.itos)

#   # Go over the whole length of the desired output
#   for k in range(max_len):
#     # For all the beams
#     for i in range(beams):
#       log_probs = torch.tensor([])
#       cur_log_prob = cur_log_probs[i]
#       # For each possible word
#       for j in range(vocab_size):
#         cur_string = []
#         if k != 0:
#           cur_string = all_beams[i]
#         cur_string.append(i)
#         out, h, c = model(torch.tensor(cur_string), h, c)
#         t = F.log_softmax(out, -1)
#         total_log_prob = cur_log_prob + F.log_softmax(out, -1)
#         log_probs = torch.cat((log_probs, total_log_prob))

#       vals, indices = torch.topk(log_probs, 1, dim=-1)
#       if k != 0:
#         all_beams[i].append(indices.item())
#       else:
#         all_beams.append([indices.item()])
#       cur_log_probs[i] += vals
    
  

############################################################################################
# TASK 1.1
############################################################################################

def vanilla(out):
  dist = F.softmax(out, dim=1)
  return torch.multinomial(dist, 1)[0]

def temp_scaled(out, temp):
  dist = F.softmax(out / temp, dim=1)
  return torch.multinomial(dist, 1)[0]

def top_k(out, k):
  dist = F.softmax(out, dim=1)
  vals, indices = torch.topk(dist, k)
  indices = indices.squeeze(-1)
  vals = vals.squeeze(-1)
  mask = torch.full((out.shape[1],), True)
  mask[indices] = False
  dist[0, mask] = 0.
  dist = dist / torch.sum(vals)
  return torch.multinomial(dist, 1)[0]

def top_p(out, p):
  dist = F.softmax(out, dim=1).squeeze()
  srted, srted_inds = torch.sort(dist)
  probs = torch.cumsum(srted, dim=-1)
  mask = probs < p
  dist[srted_inds[mask]] = 0.
  x = dist / torch.sum(dist)

  return torch.multinomial(x, 1)

def sample(model, text_field, prompt="", max_len=50, temp=1.0, k=0, p=1.):
  assert (k==0 or p==1), "Cannot combine top-k and top-p sampling"
  ret = []
  num_layers = 3
  h_c_out = 512
  h = torch.zeros(num_layers, h_c_out)
  c = torch.zeros(num_layers, h_c_out)

  indices = text_field.process([text_field.tokenize(prompt.lower())])
  indices = indices.squeeze()
  out, h, c = model(indices[:-1], h, c)
  out, h, c = model(indices[-1:], h, c)

  for _ in range(max_len):
    if temp != 1.0:
      index = temp_scaled(out, temp)
    elif k != 0:
      index = top_k(out, k)
    elif p != 1:
      index = top_p(out, p)
    else:
      index = vanilla(out)

    ret.append(index.item())

    out, h, c = model(index, h, c)

  return prompt + " " + reverseNumeralize(ret, text_field)

############################################################################################

def reverseNumeralize(numeralized_string, text_field):
  strings = [text_field.vocab.itos[i] for i in numeralized_string]
  return " ".join(strings)

if __name__ == "__main__":
  main()
