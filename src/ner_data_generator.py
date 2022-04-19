import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
import nltk
import argparse
from utils import *
nltk.download('punkt')
RND_SEED = 241

def get_offsets(text_sentences, resource_name, resource_type):
  """
  Matches a given resource_name (eg 'MEGALEX') of a given resource_type (eg RES) to a given list of sentences
  :param text_sentences: sentences to map the given resource_name to
  :param resource_name: resource_name to map
  :param resource_type: type of resource to map
  :return: list of words, word_indices, tags and sent_indices corresponding to the mapped sequence
  """
  word_indices = []
  tags = []
  words = []
  sent_indices = []
  
  for sent_idx, text in enumerate(text_sentences):
    text_tokens = text.split()
    text_tokens_stripped =  np.array([x.strip(string.punctuation).lower() for x in text_tokens])
    text_token_idx2tag = {}
    if resource_name == resource_name:

      resource_name_tokens = [x.lower() for x in resource_name.split()]
      resource_name_first_token = resource_name_tokens[0]

      match_indices = np.where(text_tokens_stripped == resource_name_first_token)[0]
      for match_idx in match_indices:
        found_match = True
        text_token_idx2tag[match_idx] = 'B-' + resource_type
        for i, resource_name_token in enumerate(resource_name_tokens[1:]):
          resource_name_token_in_range = ((i + match_idx + 1) < (len(text_tokens_stripped)))
          if (not resource_name_token_in_range) or (resource_name_token_in_range and (text_tokens_stripped[i + match_idx + 1] != resource_name_token)):
            found_match = False
          else:
             text_token_idx2tag[match_idx + i + 1] = 'I-' + resource_type
        if not found_match:
          for i, resource_name_token in enumerate(resource_name_tokens):
            text_token_idx2tag[match_idx + i] = 'O'

    for token_idx, token in enumerate(text_tokens):
      words.append(token)
      word_indices.append(token_idx)
      tags.append(text_token_idx2tag[token_idx] if token_idx in text_token_idx2tag else 'O')
      sent_indices.append(sent_idx)
  return words, word_indices, tags, sent_indices

def reconcile_tags(tags_arr1, tags_arr2):
  """
  Reconciles different set of tags in arrays corresponding to the same sequence of words.
  Eg: [B-RES, I-RES, O] and [O, O, O]
  Each word will get the more specific tag found in either of the arrays
  :param tags_arr1: array containing first list of tags
  :param tags_arr2: array containing second list of tags
  :return final_tags: array containing reconciled tags
  """
  final_tags = []
  for tag1, tag2 in zip(tags_arr1, tags_arr2):
    if tag1 == tag2:
      final_tags.append(tag1)
    elif tag1 != 'O':
      final_tags.append(tag1)
    else:
      final_tags.append(tag2)
  return final_tags

def BIO_scheme_transform(df):
  """
  Matches B-RES and I-RES tags according to the BIO-scheme for the mentions found under the 'name' and 'acronym' fields. 
  Matches on both the 'title' and 'abstract' fields, parsed to remove XML tags
  :param df: the given df. Must contain  the following fields: [id, title, abstract_parsed_xml, name, acronym]
  :return df: df containing sentences where mentions under 'name' and 'acronym' fields are being matched
  """
  pmids = df['id'].values
  titles = df['title'].values
  abstracts = df['abstract_parsed_xml'].values
  names = df['name'].values
  acronyms = df['acronym'].values
  
  all_words = []
  all_word_indices = []
  all_tags = []
  all_pmids = []
  all_sent_indices = []
  last_pmid  = -1
  last_final_tags = []
  last_words = []
  last_word_indices = []
  last_sent_indices = []
  for pmid, title, abstract, resource_name, acronym in zip(pmids, titles, abstracts, names, acronyms):
    title_abstract_sentences = nltk.sent_tokenize(title) + nltk.sent_tokenize(abstract)
    words, word_indices, tags, sentences_indices = get_offsets(title_abstract_sentences, resource_name, 'RES')
    _, _, acronyms_tags, acronyms_sentences_indices = get_offsets(title_abstract_sentences, acronym, 'RES')
    
    final_tags = reconcile_tags(tags, acronyms_tags)
    if pmid == last_pmid:
      final_tags = reconcile_tags(final_tags, last_final_tags)
    # seeing a new pmid -> append last information unless last_pmid is -1
    elif last_pmid != -1 :
      all_words.extend(last_words)
      all_word_indices.extend(last_word_indices)
      all_tags.extend(last_final_tags)
      all_pmids.extend([last_pmid] * len(last_words))
      all_sent_indices.extend(last_sent_indices)
      
    last_pmid = pmid
    last_final_tags = final_tags
    last_words = words
    last_word_indices = word_indices
    last_sent_indices = sentences_indices
    
  all_words.extend(last_words)
  all_word_indices.extend(last_word_indices)
  all_tags.extend(last_final_tags)
  all_pmids.extend([last_pmid] * len(last_words))
  all_sent_indices.extend(last_sent_indices)
  df = pd.DataFrame({'pmid' :all_pmids, 'sent_idx' : all_sent_indices, 'word' : all_words, 'word_idx' : all_word_indices, 'tag' : all_tags})
  return df

def process_df(df, filename):
  """
  Saves a df as a pickle file under a given filename
  :param filename: file under which to save the pickled df
  """
  df_grouped = df.groupby(['pmid', 'sent_idx']).agg(list).reset_index()
  df_grouped = df_grouped.rename(columns = {'word' : 'words', 'tag' : 'ner_tags'})
  df_grouped.to_pickle(filename)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_file', type=str, default = 'data/extracted_elements_2022-02-24_hji.csv')
  parser.add_argument('--output_dir', type=str, default = 'data/', help = 'Output directory. Where train/test/val files will be found')
  args, _ = parser.parse_known_args()

  print(f'args={args}')

  df = pd.read_csv(args.input_file)
  df = df[['id', 'title', 'abstract', 'name', 'acronym']]
  df['abstract_parsed_xml'] = df['abstract'].apply(get_parsed_xml) 
  df['title_parsed_xml'] = df['title'].apply(get_parsed_xml) 
  df = df.drop_duplicates()
  print('This is how the initial data looks like:')
  print(df.head())
  df = df[~df['name'].isna()]
  ner_df = BIO_scheme_transform(df)

  np.random.seed(RND_SEED)
  sent_ids = ner_df['pmid'].unique()
  sent_ids_train, sent_ids_val_test = train_test_split(sent_ids, test_size=0.3, random_state=RND_SEED)
  sent_ids_val, sent_ids_test = train_test_split(sent_ids_val_test, test_size=0.5, random_state=RND_SEED)

  train_df = ner_df[ner_df['pmid'].isin(sent_ids_train)]
  val_df = ner_df[ner_df['pmid'].isin(sent_ids_val)]
  test_df = ner_df[ner_df['pmid'].isin(sent_ids_test)]

  assert(len(set(sent_ids_train).intersection(set(sent_ids_val))) == 0)
  assert(len(set(sent_ids_train).intersection(set(sent_ids_test))) == 0)
  assert(len(set(sent_ids_val).intersection(set(sent_ids_test))) == 0)

  process_df(train_df, args.output_dir + 'ner_train.pkl')
  process_df(val_df, args.output_dir + 'ner_val.pkl')
  process_df(test_df, args.output_dir + 'ner_test.pkl')