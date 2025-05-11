import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
#from sentence_transformers import SentenceTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def mean_pooling(model_output, attention_mask):
  token_embeddings = model_output.last_hidden_state
  input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
  sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
  sum_mask = input_mask_expanded.sum(dim=1)
  return sum_embeddings / (sum_mask + 1e-8)

def encode(sentences, model_name, batch_size=32):
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModel.from_pretrained(model_name).to(device)
  model.eval()

  all_embeddings = []
  with torch.no_grad():
    for i in tqdm(range(0, len(sentences), batch_size), desc=f'Encoding {model_name}'):
      batch = sentences[i:i+batch_size]
      inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
      outputs = model(**inputs)
      embeddings = mean_pooling(outputs, inputs['attention_mask'])
      embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
      all_embeddings.append(embeddings.cpu())
  return torch.cat(all_embeddings, dim=0).numpy()

def to_xlmr_sentence_embeddings(df, model_name_def):
  if model_name_def == 'xlmr':
    model_name = 'xlm-roberta-base' 
  elif model_name_def == 'glot500':
    model_name = 'cis-lmu/glot500-base' 
  hsb_embeddings = encode(df['hsb'].tolist(), model_name, batch_size=32)
  de_embeddings = encode(df['de'].tolist(), model_name, batch_size=32)
  similarities = np.sum(hsb_embeddings * de_embeddings, axis=1)
  return hsb_embeddings, de_embeddings, similarities



# def to_labse_sentence_embeddings(data,output_path):
  model = SentenceTransformer('sentence-transformers/LaBSE')
  hsb_embeddings = model.encode(data['hsb'].tolist(), normalize_embeddings=True, show_progress_bar=True)
  de_embeddings = model.encode(data['de'].tolist(), normalize_embeddings=True, show_progress_bar=True)
  similarities = np.sum(hsb_embeddings * de_embeddings, axis=1)
  data['similarity'] = similarities
  data.to_csv(output_path, sep='\t', index=False)

