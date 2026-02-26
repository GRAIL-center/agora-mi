import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae.load_sae import load_sae_for_layer
from _common import read_config

model_id = 'google/gemma-2-2b-it' # Make sure it's the IT model if we want chat behavior, but the config uses 'google/gemma-2-2b'. Wait, the config uses 'google/gemma-2-2b'. Let me check what the model actually is.
# ACTUALLY, the paper uses 'google/gemma-2-2b' (base model). But the base model doesn't have an alignment tax in a chat sense. 
# Let me look up what model we used.
