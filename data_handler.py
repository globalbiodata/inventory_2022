import pandas as pd
from transformers import AutoTokenizer
from datasets import ClassLabel, Dataset
from torch.utils.data import DataLoader

class DataHandler:
  """
  Handles generating training, validation and testing dataloaders used for training and evaluation
  """
  def __init__(self, model_huggingface_version, train_file, val_file = None, test_file = None):
    """
    :param train_file: path to train file
    :param val_file: path to val file
    :param test_file: path to test file
    :param model_huggingface_version: Hugginface model version used to instantiate the tokenizer
    """
    if val_file:
      self.train_only = False
    else:
      self.train_only = True
    train_df, val_df, test_df = self.get_data_splits(train_file, val_file, test_file)
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = val_df
    self.tokenizer = AutoTokenizer.from_pretrained(model_huggingface_version)
  
  def get_parsed_xml(self, text):
    """
    Strips XML tags from a string

    :param text: string possibly containing XML tags
    :return: string parsed of XML tags
    """
    text_xml_parsed = ""
    substring = text
    while '<' in substring and '</' in substring and '>' in substring:
      start_tag_open = substring.find('<')
      start_tag_close = substring.find('>')

      end_tag_open = substring.find('</')
      end_tag_close = substring.find('>', end_tag_open)

      if start_tag_open != -1 and start_tag_close != -1 and end_tag_open != -1 and end_tag_close != 1:
        parsed_xml = substring[(start_tag_close + 1): end_tag_open]
        text_xml_parsed += substring[:start_tag_open] + parsed_xml + " "
      substring = substring[(end_tag_close + 1):]
    text_xml_parsed += substring
    return text_xml_parsed

  def parse_abstracts_xml(self):
    """
    Parses abstracts in train, val, test splits from XML tags
    Adds an extra field to the train, val, test splits: abstract_no_xml

    :param train_df: df containing training data
    :param val_df: df containing val data
    :param test_df: df containing test data
    """
    self.train_df['abstract_no_xml'] = self.train_df['abstract'].apply(lambda x: self.get_parsed_xml(x))
    if not self.train_only:
      self.val_df['abstract_no_xml'] = self.val_df['abstract'].apply(lambda x: self.get_parsed_xml(x))
      self.test_df['abstract_no_xml'] = self.test_df['abstract'].apply(lambda x: self.get_parsed_xml(x))

  def concatenate_title_abstracts(self):
    """
    Concatenates titles and abstracts in train, val, test splits
    Adds an extra field to the train, val, test splits: title_abstract

    :param train_df: df containing training data
    :param val_df: df containing val data
    :param test_df: df containing test data
    """
    self.train_df['title_abstract'] = self.train_df['title'] + '-' + self.train_df['abstract_no_xml']
    if not self.train_only:
      self.val_df['title_abstract'] = self.val_df['title'] + '-' + self.val_df['abstract_no_xml']
      self.test_df['title_abstract'] = self.test_df['title'] + '-' + self.test_df['abstract_no_xml']

  def get_data_splits(self, train_path, val_path, test_path):
    """
    Loads train, val and test splits from file

    :param train_path: path of training data
    :param val_path: path of val data
    :param test_path : path of test data
    :return: train, val, test dataframes
    """
    train_df = pd.read_csv(train_path)
    if not self.train_only:
      val_df = pd.read_csv(val_path)
      test_df = pd.read_csv(test_path)
      print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))
    else:
      val_df = None
      test_df = None
    return train_df, val_df, test_df

  def get_predictive_text_and_labels(self, train_df, val_df, test_df, predictive_field, score_field):
    """
    Returns arrays containing the train, val and test data that will be used for training, as well as corresponding labels

    :param train_df: df containing training data
    :param val_df: df containing val data
    :param test_df: df containing test data
    :param predictive_field: field that will be used for training; can be one of ['title', 'abstract', 'title_abstract']
    :param score_field: field that contains the scores, that will be used as labels

    :return train_text: array containing data used for training
    :return val_text: array containing data used for validation
    :return test_text: array containing data used for test 
    :return train_labels: labels corresponding to entries in train_text
    :return val_labels: labels corresponding to entries in val_text
    :return test_labels: labels corresponding to entries in test_text
    """
    train_text = train_df[predictive_field].tolist()
    train_labels = train_df[score_field].tolist()

    if not self.train_only:
      val_text = val_df[predictive_field].tolist()
      val_labels = val_df[score_field].tolist()

      test_text = test_df[predictive_field].tolist()
      test_labels = test_df[score_field].tolist()
      return train_text, val_text, test_text, train_labels, val_labels, test_labels
    else:
      return train_text, None, None, train_labels, None, None

  def tokenize_function(self, examples, tokenizer, max_len):
    """
    Tokenizes text entries up to a MAX_LEN length; Could involve truncation or padding of text entries

    :param examples: text entries to tokenize
    :param tokenizer: tokenizer used for tokenizing the data

    :return: tokenized data up to MAX_LEN
    """
    return tokenizer(examples["text"], padding = 'max_length', truncation = True, max_length = max_len)

  def get_tokenized_dataset(self, text_arr, tokenizer, class_label, max_len, labels = None):
    """
    Tokenizes a given array containing plain text entries using a given tokenizer

    :param text_arr: array containing plain_text data
    :param tokenizer: tokenizer used for tokenizing the data
    :param class_label: ClassLabel mapping the [0, 1] scores to ['not-bio-resouce', 'bio-resource'] classes
    :param labels: True if we have labels for the entries in text_arr; If None, the dataset won't contain a 'label' field

    :return: Dataset entry containing tokenized text data and corresponding labels
    """
    dataset_dict = {'text' : text_arr}
    if labels:
      dataset_dict['label'] = labels
    dataset = Dataset.from_dict(dataset_dict)
    tokenized_dataset = dataset.map(lambda x: self.tokenize_function(x, tokenizer, max_len), batched=True)
    if labels:
      tokenized_dataset = tokenized_dataset.cast_column("label", class_label)
      tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset.set_format("torch")
    return tokenized_dataset

  def get_dataloader(self, text_arr, labels, tokenizer, class_label, batch_size, max_len, shuffle = False, num_datapoints = -1):
    """
    Tokenizes a given text array and returns a DataLoader used for batching during training

    :param text_arr: array containing plain_text data
    :param labels: labels correponding to entries in text_arr
    :param tokenizer: tokenizer used for tokenizing the data
    :param class_label: ClassLabel mapping the [0, 1] scores to ['not-bio-resouce', 'bio-resource'] classes
    :param batch_size: batch_size that will be used to load the entries from the DataLoader
    :param shuffle: True if to shuffle the data when loading it from the DataLoader
    :param num_pts: if != -1 - consider only a limited number of points; helpful for debugging or sanity-checking

    :return: DataLoader containing tokenized data and corresponding labels
    """
    dataset = self.get_tokenized_dataset(text_arr, tokenizer, class_label, max_len, labels)
    if num_datapoints != -1:
      dataset = dataset.select(range(num_datapoints))
    return DataLoader(dataset, shuffle = shuffle, batch_size = batch_size)

  def generate_dataloaders(self, predictive_field, score_field, class_names, batch_size, max_len, sanity_check = False, num_datapoints = -1):
    """
    Returns train, val and test DataLoaders

    :param train_df: df containing training data
    :param val_df: df containing val data
    :param test_df: df containing test data
    :param predictive_field: field that will be used for training; can be one of ['title', 'abstract', 'title_abstract']
    :param score_field: field that contains the scores, that will be used as labels
    :param tokenizer: tokenizer used for tokenizing the data
    :param class_names: class names corresponding to the [0, 1] labels
    :param sanity_check: if True, train_dataloader will have num_datapoints entries
    :param num_datapoints: if != -1, number of entries in training_dataloader; meant for debugging, sanity-checking

    :return train_dataloader: DataLoader containing tokenized data and corresponding labels for training
    :return val_dataloader: DataLoader containing tokenized data and corresponding labels for validation
    :return test_dataloader: DataLoader containing tokenized data and corresponding labels for testing   
    """
    train_text, val_text, test_text, train_labels, val_labels, test_labels = self.get_predictive_text_and_labels(self.train_df, self.val_df, self.test_df, predictive_field, score_field)
    class_label = ClassLabel(num_classes = 2, names = class_names)
    if sanity_check:
      num_datapoints = num_datapoints
    else:
      num_datapoints = -1 
    self.train_dataloader = self.get_dataloader(train_text, train_labels, self.tokenizer, class_label, batch_size, max_len, True, num_datapoints)
    if not self.train_only:
      self.val_dataloader = self.get_dataloader(val_text, val_labels, self.tokenizer, class_label, batch_size, max_len)
      self.test_dataloader = self.get_dataloader(test_text, test_labels, self.tokenizer, class_label, batch_size, max_len)