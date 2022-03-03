from data_handler import DataHandler
import os
import argparse
from transformers import AutoModelForSequenceClassification, AdamW, get_scheduler
import torch
from tqdm.auto import tqdm
from datasets import load_metric
import plotly.express as px
import copy
import pandas as pd


# Mapping from generic model name to the Huggingface Version used to initialize the model
MODEL_TO_HUGGINGFACE_VERSION = {
  'bert' : 'bert_base_uncased',
  'biobert' : 'dmis-lab/biobert-v1.1',
  'scibert' : 'allenai/scibert_scivocab_uncased',
  'pubmedbert' : 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
  'pubmedbert_pmc' : 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
  'bluebert' : 'bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12',
  'bluebert_mimic3' : 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
  'sapbert' : 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
  'sapbert_mean_token' : 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token',
  'bioelectra' : 'kamalkraj/bioelectra-base-discriminator-pubmed',
  'bioelectra_pmc' : 'kamalkraj/bioelectra-base-discriminator-pubmed-pmc',
  'electramed' : 'giacomomiolo/electramed_base_scivocab_1M',
  'biomed_roberta' : 'allenai/biomed_roberta_base',
  'biomed_roberta_chemprot' : 'allenai/dsp_roberta_base_dapt_biomed_tapt_chemprot_4169',
  'biomed_roberta_rct_500' : 'allenai/dsp_roberta_base_dapt_biomed_tapt_rct_500'
}

class Trainer():
  """
  Handles training of the model
  """
  def __init__(self, model, optimizer, train_dataloader, val_dataloader, lr_scheduler, num_epochs, num_training_steps, device):
    """
    :param model: PyTorch model
    :param optimizer: optimizer used
    :param train_dataloader: DataLoader containing data used for training
    :param val_dataloader: DataLoader containing data used for validation
    :param lr_scheduler: learning rate scheduler; could be equal to None if no lr_scheduler is used
    :param num_epochs: number of epochs to train the model for
    :param num_training_steps: total number of training steps
    :param device: device used for training; equal to 'cuda' if GPU is available
    """  
    self.model = model
    self.optimizer = optimizer
    self.train_dataloader = train_dataloader
    self.val_dataloader = val_dataloader
    self.lr_scheduler = lr_scheduler
    self.num_epochs = num_epochs
    self.num_training_steps = num_training_steps
    self.device = device
    
  def get_metrics(self, dataloader):
    """
    Computes and returns metrics (P, R, F1 score) of a model on data present in a dataloader

    :param model: model used to compute the metrics
    :param dataloader: DataLoader containing tokenized text entries and corresponding labels

    :return: precision, recall, F1 score
    """
    precision = load_metric("precision")
    recall = load_metric("recall")
    f1 = load_metric("f1")
    total_loss = 0
    num_seen_datapoints = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = self.model(**batch)
        num_seen_datapoints += len(batch['input_ids'])
        logits = outputs.logits
        loss = outputs.loss
        predictions = torch.argmax(logits, dim=-1)
        precision.add_batch(predictions=predictions, references=batch["labels"])
        recall.add_batch(predictions=predictions, references=batch["labels"])
        f1.add_batch(predictions=predictions, references=batch["labels"])
        total_loss += loss.item()
    total_loss /= num_seen_datapoints
    return precision.compute()['precision'], recall.compute()['recall'], f1.compute()['f1'], total_loss

  def train_epoch(self, progress_bar):
    """
    Handles training of the model over one epoch

    :param model: PyTorch model
    :param optimizer: optimizer used
    :param lr_scheduler: learning rate scheduler; could be equal to None if no lr_scheduler is used
    :param train_dataloader: DataLoader containing data used for training
    :param device: device used for training; equal to 'cuda' if GPU is available
    :param progress_bar: tqdm instance for tracking progress
    """
    train_loss = 0
    num_train = 0 
    for batch in self.train_dataloader:
        batch = {k: v.to(self.device) for k, v in batch.items()}
        num_train += len(batch['input_ids'])
        outputs = self.model(**batch)
        loss = outputs.loss
        loss.backward()
        train_loss += loss.item()
        self.optimizer.step()
        if self.lr_scheduler:
          lr_scheduler.step()
        self.optimizer.zero_grad()
        progress_bar.update(1)
    return train_loss / num_train

  def train(self):
    """
    Handles training of the model over all epochs

    :param model: PyTorch model
    :param train_dataloader: DataLoader containing data used for training
    :param val_dataloader: DataLoader containing data used for validation
    :param optimizer: optimizer used
    :param lr_scheduler: learning rate scheduler; could be equal to None if no lr_scheduler is used
    :param num_epochs: number of epochs to train the model for
    :param num_training_steps:
    :param checkpt_name: name under which the checkpoint will be saved
    :param device: device used for training; equal to 'cuda' if GPU is available

    :return best_model: model checkpt that has the highest F1 score on the validation data
    :return best_epoch: epoch corresponding to best_model
    :return train_losses: list of training loss values over all epochs; helpful for plotting
    :return val_losses: list of validation loss values over all epochs; helpful for plotting

    """
    progress_bar = tqdm(range(num_training_steps))
    self.model.train()
    best_model = self.model
    train_losses = []
    val_losses = []
    best_val_f1 = 0
    best_train_f1 = 0
    best_val_p = 0
    best_train_p = 0
    best_val_r = 0
    best_train_r = 0
    best_epoch = 0
    best_train_loss = 0
    best_val_loss = 0

    for epoch in range(self.num_epochs):
        # training
        train_loss = self.train_epoch(progress_bar)
        # evaluation
        self.model.eval()
        train_p, train_r, train_f1, _ = self.get_metrics(self.train_dataloader)
        val_p, val_r, val_f1, val_loss = self.get_metrics(self.val_dataloader)

        if val_f1 > best_val_f1:
          best_val_f1 = val_f1
          best_train_f1 = train_f1
          best_val_p = val_p
          best_train_p = train_p
          best_val_r = val_r
          best_train_r = train_r
          best_val_loss = val_loss
          best_train_loss = train_loss
          best_model = copy.deepcopy(self.model)
          best_epoch = epoch

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print("Epoch", (epoch + 1),  ": Train Loss: %.5f Val Loss: %.5f" % (train_loss, val_loss))
        print("Train Precision: %.3f Train Recall: %.3f Train F1: %.3f Val Precision: %.3f Val Recall: %.3f Val F1: %.3f" % (train_p, train_r, train_f1, val_p, val_r, val_f1))
    print('Finished model training!')
    print('=' * 30)
    print("Best Train Precision: %.3f Best Train Recall: %.3f Best Train F1: %.3f Best Val Precision: %.3f Best Val Recall: %.3f Best Val F1: %.3f" % (best_train_p, best_train_r, best_train_f1, best_val_p, best_val_r, best_val_f1))
    self.best_model = best_model
    self.best_epoch = best_epoch
    self.best_f1_score = best_val_f1
    return best_model, best_epoch, train_losses, val_losses
  
  def save_best_model(self, checkpt_filename):
    """
    Saves a model checkpoint, epoch and F1 score to file

    :param model: model to save
    :param epoch: num_epoch corresponding to trained model
    :param f1_score: F1 score obtained by the model on validation data
    :param checkpt_filename: filename under which the model checkpoint will be saved
    """
    torch.save({
    'model_state_dict': self.best_model.state_dict(),
    'epoch' : self.best_epoch,
    'f1_val' : self.best_f1_score,
    }, checkpt_filename)

  def plot_losses(self, losses, labels, img_filename):
    """
    Plots training and val losses

    :param num_epochs: total number of epochs the model was trained on; corresponds to length of the losses array
    :param losses: array corresponding to [train_losses, val_losses]
    :param labels: labels used for plotting; usually ['Train Loss', 'Val Loss']

    :return: Generated plot
    """
    x = [i for i in range(self.num_epochs)]
    df = pd.DataFrame({'Epoch' : x})
    for loss_arr, label in zip(losses, labels):
      df[label] = loss_arr
    fig = px.line(df, x="Epoch", y=labels, title='Train/Val Losses')
    fig.show()
    fig.write_image(img_filename)
    return fig

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--train-file', type=str, default='data/train.csv', help = 'Location of training file')
	parser.add_argument('--val-file', type=str, default='data/val.csv', help = 'Location of validation file')
	parser.add_argument('--test-file', type=str, default='data/test.csv', help = 'Location of test file')
	parser.add_argument('--model-name', type=str, default='scibert', help = "Name of model to try. Can be one of: ['bert', 'biobert', 'scibert', 'pubmedbert', 'pubmedbert_pmc', 'bluebert', 'bluebert_mimic3', 'sapbert', 'sapbert_mean_token', 'bioelectra', 'bioelectra_pmc', 'electramed', 'biomed_roberta', 'biomed_roberta_chemprot', 'biomed_roberta_rct_500']")
	parser.add_argument('--predictive-field', type=str, default='title', help = "Field in the dataframes to use for prediction. Can be one of ['title', 'abstract', 'title-abstract']")
	parser.add_argument('--labels-field', type=str, default='curation_score', help = "Field in the dataframes corresponding to the scores (0, 1)")
	parser.add_argument('--descriptive-labels', type=str, default=['not-bio-resource', 'bio-resource'], help = "Descriptive labels corresponding to the [0, 1] numeric scores")
	parser.add_argument('--sanity-check', action = 'store_true', help = "True for sanity-check. Runs training on a smaller subset of the entire training data.")
	parser.add_argument('--num-training', type=int, default = -1, help = "Number of data points to run training on. If -1, training is ran an all the data. Useful for debugging.")
	parser.add_argument('--output-dir', type=str, default='output_dir/', help = 'Default directory to output checkpt and plot losses')
	parser.add_argument('--num-epochs', type=int, default=10, help = 'Number of Epochs')
	parser.add_argument('--batch-size', type=int, default=32, help = 'Batch Size')
	parser.add_argument('--max-len', type=int, default=256, help = 'Max Sequence Length')
	parser.add_argument('--learning-rate', type=float, default=2e-5, help = 'Learning Rate')
	parser.add_argument('--weight-decay', type=float, default=0.0, help = 'Weight Decay for Learning Rate')
	parser.add_argument('--lr-scheduler', action = 'store_true', help = 'True if using a Learning Rate Scheduler. More info here: https://huggingface.co/docs/transformers/main_classes/optimizer_schedules')

	args, _ = parser.parse_known_args()

	if not os.path.exists(args.output_dir):
		os.mkdir(args.output_dir)

	print(f'args={args}')

	# Train, val, test dataloaders generation
	print('Generating train, val, test dataloaders ...')
	print('=' * 30)
	model_huggingface_version = MODEL_TO_HUGGINGFACE_VERSION[args.model_name]
	data_handler = DataHandler(args.train_file, args.val_file, args.test_file, model_huggingface_version)
	data_handler.parse_abstracts_xml()
	data_handler.concatenate_title_abstracts()
	data_handler.generate_dataloaders(args.predictive_field, args.labels_field, args.descriptive_labels, args.batch_size, args.max_len, args.sanity_check, args.num_training)
	train_dataloader = data_handler.train_dataloader
	val_dataloader = data_handler.val_dataloader
	print('Finished generating dataloaders!')
	print('=' * 30)

	# Model Initialization
	print('Initializing', model_huggingface_version, 'model ...')
	print('=' * 30)
	model = AutoModelForSequenceClassification.from_pretrained(model_huggingface_version, num_labels=2)
	optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay = args.weight_decay)
	num_training_steps = args.num_epochs * len(train_dataloader)
	if args.lr_scheduler:
	  lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
	else:
	  lr_scheduler = None
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	model.to(device)


	# Model Training
	print('Starting model training...')
	print('=' * 30)
	trainer = Trainer(model, optimizer, train_dataloader, val_dataloader, lr_scheduler, args.num_epochs, num_training_steps, device)
	best_model, best_epoch, train_losses, val_losses = trainer.train()

	# Save best checkpoint
	checkpt_filename = args.output_dir + 'checkpt_' + args.model_name + '_' + str(best_epoch + 1) + '_epochs'
	trainer.save_best_model(checkpt_filename)
	print('Saved best checkpt to', checkpt_filename)

	# Plot losses
	img_filename = args.output_dir + args.model_name + '_' + str(best_epoch + 1) + '_epochs.png'
	trainer.plot_losses([train_losses, val_losses], ['Train Loss', 'Val Loss'], img_filename)
	print('=' * 30)