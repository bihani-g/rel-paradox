from datetime import datetime
from typing import Optional
import os
import math
import random
import matplotlib.pyplot as plt
import pickle
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from captum.attr import LayerIntegratedGradients
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
import datasets
import nltk
import string
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))



class DataModule(LightningDataModule):
    
    text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "SetFit/20_newsgroups": ["text"],
        "ag_news": ["text"],
        "trec": ["text"],
        "emotion": ["text"],
        "hate": ["text"],
        "irony": ["text"],
        "sentiment": ["text"],
        "imdb": ["text"]
     
    }

    task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "SetFit/20_newsgroups": 20,
        "ag_news": 4,
        "trec": 6,
        "emotion": 4,
        "hate": 2,
        "irony": 2,
        "sentiment": 3,
        "imdb": 2,
        
    }
    
    task_label_map = {
        "cola": "label",
        "sst2": "label",
        "SetFit/20_newsgroups": "label",
        "ag_news": "label",
        "trec": "coarse_label",
        "emotion": "label",
        "hate": "label",
        "irony": "label",
        "sentiment": "label",
        "imdb": "label",
    }
    
    

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        dataset_name: str = "glue",
        predict_split: str = "test",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        task_name: str = None,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.dataset_name = dataset_name
        self.predict_split = predict_split
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.task_name = task_name
        
        if self.dataset_name in ["glue", "tweet_eval"]:
            self.text_fields = self.text_field_map[task_name]
            self.num_labels = self.task_num_labels[task_name]
            self.task_label = self.task_label_map[task_name]
        else:
            self.text_fields = self.text_field_map[dataset_name]
            self.num_labels = self.task_num_labels[dataset_name]
            self.task_label = self.task_label_map[dataset_name]
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, 
                                                       use_fast=True)

        # try:
        #     self.tokenizer = AutoTokenizer.from_pretrained(
        #         self.model_name_or_path,
        #         use_fast=False,  # Attempt to use the fast tokenizer
        #         # add_special_tokens=True  # Avoid adding special tokens
        #         )
        # except ValueError:  # If there is an issue with the fast tokenizer
        #     self.tokenizer = AutoTokenizer.from_pretrained(
        #         self.model_name_or_path,
        #         use_fast=False,  # Fall back to the slow tokenizer
        #         add_special_tokens=False  # Avoid adding special tokens
        #     )

    def setup(self, stage: str):
        print("setting up")
        if self.dataset_name in ["glue", "tweet_eval"]:
            self.dataset = datasets.load_dataset(self.dataset_name, self.task_name)
        else:
            self.dataset = datasets.load_dataset(self.dataset_name)
    
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=[self.task_label],
            )
            
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        if self.dataset_name in ["glue", "tweet_eval"]:
            datasets.load_dataset(self.dataset_name, self.task_name)
        else:
            datasets.load_dataset(self.dataset_name)
        
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True, num_workers=0)


    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size, num_workers=0)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size, num_workers=0) for x in self.eval_splits]
        
    def predict_dataloader(self):
        if self.predict_split == "validation":
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size, num_workers=0)
            
        if self.predict_split == "test":
            if self.dataset_name in ["imdb"]:
                total_instances = len(self.dataset["test"])
                sample_indices = random.sample(range(total_instances), 1821)
                sampled_instances = [self.dataset["test"][i] for i in sample_indices]
                return DataLoader(sampled_instances, batch_size=self.eval_batch_size, num_workers=0)                
            else:
                return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size, num_workers=0)
            
        if self.predict_split == "train":
            return DataLoader(self.dataset["train"], batch_size=self.eval_batch_size, num_workers=0)


    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, pad_to_max_length=True, truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch[self.task_label]
               
        return features


    

    
class TCTransformers(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        dataset_name: str = "glue",
        task_name: str = None,
        learning_rate: float = 1e-5,
        gradient_clip_val: int = 1,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.softmax = torch.nn.Softmax(dim=1)
        self.validation_step_outputs = []
        
        if self.hparams.dataset_name == "glue":
            if self.hparams.task_name in ['sst2', 'cola']:
                self.metric = datasets.load_metric(
                    "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")) 
        elif self.hparams.dataset_name == "tweet_eval":
            if self.hparams.task_name == "irony":
                self.metric = evaluate.load("f1", pos_label = 1, average = "binary")
                
            if self.hparams.task_name == "sentiment":
                self.metric = evaluate.load("recall", average = "macro")
        else:
            self.metric = evaluate.load("f1", average = "macro")
            
        
        

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        
        outputs = self(**batch)
        labels = batch["labels"]
        loss, logits = outputs[:2]
        
        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
            probs = self.softmax(logits)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()
            
            
        return preds, probs, labels


    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        # optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        optimizer = AdamW(params=optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
    
    
    
def compile_results(results, dm, predict_split):
    predicted = []
    probabilities = []
    labels_all = []
    
    for i in range(len(results)):
        preds, probs, labels = results[i][:3]
        predicted.extend([x.item() for x in preds])
        probabilities.extend([x.numpy() for x in probs])
        labels_all.extend([x.item() for x in labels])

    predictions = []
    for i in range(len(predicted)):
        sample_dict = {}
        for key in dm.dataset[predict_split].features:
            if key not in ['input_ids', 'token_type_ids', 'attention_mask', 'labels']:
                sample_dict[key] = dm.dataset[predict_split][key][i]
            
        sample_dict["label"] = labels_all[i]
        sample_dict["predicted"] = predicted[i]
        sample_dict["probabilities"] = probabilities[i]
    
        
        predictions.append(sample_dict)
    
    return predictions



class integrated_gradients(object):
    def __init__(self,
                 results: [list],
                 topk: [list],
                 model_name_or_path: str,
                 text_field: str,
                 device,
                 model,
                 max_seq_length: int = 128,
                 ):
        self.results = results
        self.topk = topk
        self.max_seq_length = max_seq_length
        self.text_field = text_field
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, 
                                                       use_fast=True)
    

        
        self.device = device
        self.ref_token_id = self.tokenizer.pad_token_id  
        self.sep_token_id = self.tokenizer.sep_token_id  
        self.cls_token_id = self.tokenizer.cls_token_id
        self.model = model

    def construct_input_ref_pair(self, sentence):
        text_ids = self.tokenizer.encode(sentence, 
                                         add_special_tokens=False, 
                                         max_length=self.max_seq_length, 
#                                          pad_to_max_length=True, 
                                         truncation=True)     

        input_ids = [self.cls_token_id] + text_ids + [self.sep_token_id]
        ref_input_ids = [self.cls_token_id] + [self.ref_token_id] * len(text_ids) + [self.sep_token_id]

        return torch.tensor([input_ids], device=self.device), torch.tensor([ref_input_ids], device=self.device)

    
    def construct_input_ref_token_type_pair(self,input_ids):
        seq_len = input_ids.size(1)
        token_type_ids = torch.tensor([[1 for i in range(seq_len)]], device=self.device)
        return token_type_ids
    
    def construct_input_ref_pos_id_pair(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        return position_ids
        
    def construct_attention_mask(self,input_ids):
        return torch.ones_like(input_ids)
    
    def summarize_attributions(self, attributions):
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions
        
    def predict_bert(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        output = self.model.model(input_ids, token_type_ids=token_type_ids,
                 position_ids=position_ids, attention_mask=attention_mask, )    
        return output.logits
    
    def predict_roberta(self, inputs, attention_mask=None):
        output = self.model.model(inputs, attention_mask=attention_mask, )  
        return output.logits
    
    def predict_bart(self, inputs, attention_mask=None):
        output = self.model.model(inputs, attention_mask=attention_mask, )  
        return output.logits
    
    def predict_deberta(self, inputs, token_type_ids=None, attention_mask=None):
        output = self.model.model(inputs, token_type_ids=token_type_ids, attention_mask=attention_mask, )  
        return output.logits
    
    def predict_albert(self, inputs, token_type_ids=None, attention_mask=None):
        output = self.model.model(inputs, token_type_ids=token_type_ids, attention_mask=attention_mask, )  
        return output.logits
        

    def get_attributions(self):
        self.model.eval()
        self.model.zero_grad()
        self.model.to(self.device)
        
        with torch.no_grad():           
            for i in tqdm(range(len(self.results))):
                input_ids, ref_input_ids = self.construct_input_ref_pair(self.results[i][self.text_field])
                token_type_ids = self.construct_input_ref_token_type_pair(input_ids)
                position_ids = self.construct_input_ref_pos_id_pair(input_ids)
                attention_mask = self.construct_attention_mask(input_ids)

                indices = input_ids[0].detach().tolist()
                all_tokens = self.tokenizer.convert_ids_to_tokens(indices)
                if self.model_name_or_path == 'bert-base-uncased':
                    scores = self.predict_bert(input_ids,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               attention_mask=attention_mask)

                if self.model_name_or_path == 'roberta-base':
                    scores = self.predict_roberta(input_ids,
                                               attention_mask=attention_mask)
                    
                if self.model_name_or_path == 'facebook/bart-base':
                    scores = self.predict_bart(input_ids,
                                               attention_mask=attention_mask)
                    
                if self.model_name_or_path == 'microsoft/deberta-base':
                    scores = self.predict_deberta(input_ids,
                                               attention_mask=attention_mask)
                    
                if self.model_name_or_path == 'albert-base-v1':
                    scores = self.predict_albert(input_ids,
                                               attention_mask=attention_mask)

                predicted_label = self.results[i]['predicted']
                actual_label = self.results[i]['label']

                if self.model_name_or_path == 'bert-base-uncased':
                    embeddings = self.model.model.bert.embeddings
                if self.model_name_or_path == 'roberta-base':
                    embeddings = self.model.model.roberta.embeddings
                if self.model_name_or_path == 'facebook/bart-base':
                    embeddings = self.model.model.model.encoder.embed_tokens
                if self.model_name_or_path == 'microsoft/deberta-base':
                    embeddings = self.model.model.deberta.embeddings
                if self.model_name_or_path == 'albert-base-v1':
                    embeddings = self.model.model.albert.embeddings
                    
                if self.model_name_or_path == 'bert-base-uncased':
                    lig = LayerIntegratedGradients(self.predict_bert, embeddings)
                    attributions, delta = lig.attribute(inputs = input_ids,
                                                    baselines=ref_input_ids,
                                                    additional_forward_args=(token_type_ids, 
                                                                             position_ids, 
                                                                             attention_mask),
                                                    target = predicted_label,
                                                    return_convergence_delta=True)

                if self.model_name_or_path == 'roberta-base':
                    lig = LayerIntegratedGradients(self.predict_roberta, embeddings)
                    attributions, delta = lig.attribute(inputs = input_ids,
                                                    baselines=ref_input_ids,
                                                    additional_forward_args=(attention_mask),
                                                    target = predicted_label,
                                                    return_convergence_delta=True)
                    
                if self.model_name_or_path == 'facebook/bart-base':
                    lig = LayerIntegratedGradients(self.predict_bart, embeddings)
                    attributions, delta = lig.attribute(inputs = input_ids,
                                                    baselines=ref_input_ids,
                                                    additional_forward_args=(attention_mask),
                                                    target = predicted_label,
                                                    return_convergence_delta=True)
                    
                if self.model_name_or_path == 'microsoft/deberta-base':
                    lig = LayerIntegratedGradients(self.predict_deberta, embeddings)
                    attributions, delta = lig.attribute(inputs = input_ids,
                                                    baselines=ref_input_ids,
                                                    additional_forward_args=(token_type_ids, 
                                                                             attention_mask),
                                                    target = predicted_label,
                                                    return_convergence_delta=True)
                    
                if self.model_name_or_path == 'albert-base-v1':
                    lig = LayerIntegratedGradients(self.predict_albert, embeddings)
                    attributions, delta = lig.attribute(inputs = input_ids,
                                                    baselines=ref_input_ids,
                                                    additional_forward_args=(token_type_ids, 
                                                                             attention_mask),
                                                    target = predicted_label,
                                                    return_convergence_delta=True)


                attributions_sum = self.summarize_attributions(attributions).to("cpu").numpy()
                self.results[i]['attribution'] = attributions_sum
                self.results[i]['tokens'] = all_tokens
                
                del input_ids
                del ref_input_ids
                del token_type_ids 
                del position_ids

                topk_predicted = []
                topk_actual = []
                topk_common = []

                # How many tokens in topk for predicted class and actual class? How many tokens in topk for both? 
                for token in all_tokens:
                    if token in [x[0] for x in self.topk[actual_label]]  and token in [x[0] for x in self.topk[predicted_label]]:
                        topk_common.append(token)
                    else:
                        if token in [x[0] for x in self.topk[predicted_label]] and token not in [x[0] for x in self.topk[actual_label]]:
                            topk_predicted.append(token)
                        elif token in [x[0] for x in self.topk[actual_label]] and token not in [x[0] for x in self.topk[predicted_label]]:
                            topk_actual.append(token)


                self.results[i]['topk_predicted'] = topk_predicted
                self.results[i]['topk_actual'] = topk_actual
                self.results[i]['topk_common'] = topk_common

            
        return self.results
            
            
            
            
            


# def id2label(data_name, id_):
#     if data_name == 'glue':
#         if task_name == 'cola':
#             label_dict = {0: 'unacceptable', 1: 'acceptable'}
#         if task_name == 'sst2':
#             label_dict = {0: 'negative', 1: 'positive'}
            
#     if data_name == 'tweets':
#         if task_name == 'emotion':
#             label_dict = {0: 'anger', 1: 'joy', 2: 'optimism', 3: 'sadness'}
#         if task_name == 'hate':
#             label_dict = {0: 'non-hate', 1: 'hate'}
#         if task_name == 'irony':
#             label_dict = {0: 'non-irony', 1: 'irony'}
#         if task_name == 'sentiment':
#             label_dict = {0: 'negative', 1: 'neutral', 2: 'positive'}
            
#     if data_name == '20ng':
#         label_dict = {0: 'alt.atheism', 1: 'comp.graphics', 2: 'comp.os.ms-windows.misc', 3: 'comp.sys.ibm.pc.hardware', 
#                       4: 'comp.sys.mac.hardware', 5: 'comp.windows.x', 6: 'misc.forsale', 7: 'rec.autos', 
#                       8: 'rec.motorcycles', 9: 'rec.sport.baseball', 10: 'rec.sport.hockey', 11: 'sci.crypt',
#                      12: 'sci.electronics', 13: 'sci.med', 14: 'sci.space', 15: 'soc.religion.christian', 
#                       16: 'talk.politics.guns', 17: 'talk.politics.mideast', 18: 'talk.politics.misc', 
#                       19: 'talk.religion.misc'}
        
#     if data_name == 'agnews':
#         label_dict = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
        
#     if data_name == 'trec':
#         label_dict = {0: 'ABBR', 1: 'ENTY', 2: 'DESC', 3: 'DESC', 4: 'LOC', 
#                       5: 'NUM'}
        
#     if data_name == 'yelp':
#         label_dict = {0: '1 star', 1: '2 star', 2: '3 star', 3: '4 star', 4: '5 star'}
            
    
#     return label_dict[id_]


def label_word_counts(dm, split, tokenizer, label):
    tokenizer = tokenizer
    
    words = []
    for x in tqdm(dm.dataset[str(split)]):
        if x['labels'] == label:
            tokens = tokenizer.convert_ids_to_tokens(x['input_ids'])
            tokens  = [x for x in tokens if x not in ['[CLS]', '[SEP]', '[PAD]']]
            words.extend(tokens)                                 

    word_counts = pd.value_counts(np.array(words))
    
    return word_counts 


def get_count_dicts(dm, labels, split, tokenizer):
    
    total_word_count = 0
    word_dict = {}
    label_dict = {}
   
    for label in labels:
        print("getting word counts for label:", label)
        wc = label_word_counts(dm, split, tokenizer = tokenizer,
                               label = label)

        total_word_count += sum(wc)
    
        for word in tqdm(wc.index):
            if word not in word_dict.keys():
                word_dict[word] = {}
                word_dict[word]['total_count'] = 0

            word_dict[word][label] = wc[word]
            word_dict[word]['total_count'] += wc[word]

            if label not in label_dict.keys():
                label_dict[label] = 0

            label_dict[label] += wc[word]
            
            
    return total_word_count, word_dict, label_dict 


def get_lmi(word, label, word_dict, label_dict, total_word_count):
    count_wl = word_dict[word][label]
    total_word_count = total_word_count
    count_w = word_dict[word]['total_count']
    count_l = label_dict[label]
    lmi_ = (count_wl/total_word_count)*math.log2((count_wl/count_w)/(count_l/total_word_count))
    return lmi_, count_w


def get_topk_head(lmi_label, k):
    topk = round(len(lmi_label)*k)   
    return lmi_label[:topk+1]


def parse_args():
    parser = argparse.ArgumentParser(description='Run calibration experiment.')
    parser.add_argument('--data_name', type=str, required=True,
                        help='Dataset name (e.g., glue, 20ng, agnews)')
    parser.add_argument('--task_name', type=str, nargs='?', const=None,
                        help='Task name (for datasets like GLUE or TweetEval)')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model to finetune (e.g., bert, roberta)')
    return parser.parse_args()


def get_tokenizer(model_name):
    model_keys = {
        "bert": "bert-base-uncased",
        "roberta": "roberta-base",
        "bart": "facebook/bart-base",
        "deberta": "microsoft/deberta-base",
        "albert": "albert-base-v1",
    }
    return AutoTokenizer.from_pretrained(model_keys[model_name]), model_keys


def get_dataset_key(data_name):
    dataset_keys = {
        "glue": "glue",
        "20ng": "SetFit/20_newsgroups",
        "agnews": "ag_news",
        "trec": "trec",
        "tweets": "tweet_eval",
    }
    return dataset_keys[data_name]


def label_lmi(dm, tokenizer):
    labels = list(range(dm.num_labels))
    total_word_count, word_dict, label_dict = get_count_dicts(dm, labels, split='train', tokenizer=tokenizer)

    lmi_label, lmi_word = {}, {}
    for label in label_dict.keys():
        lmi = []
        for word in tqdm(word_dict.keys()):
            if label in word_dict[word]:
                lmi_, count_w = get_lmi(word, label, word_dict, label_dict, total_word_count)
                lmi_word.setdefault(label, {})[word] = lmi_
                lmi.append((word, count_w, lmi_))
        lmi.sort(key=lambda x: x[2], reverse=True)
        lmi_label[label] = lmi
    return {label: get_topk_head(lmi_label[label], k=0.05) for label in lmi_label}


def run_experiment(args):
    tokenizer, model_keys = get_tokenizer(args.model_name)
    dataset_key = get_dataset_key(args.data_name)
    predict_split = "validation" if args.data_name == "glue" else "test"

    dm = DataModule(model_keys[args.model_name], dataset_name=dataset_key,
                    task_name=args.task_name, predict_split=predict_split)
    dm.prepare_data()
    dm.setup("fit")

    topk = label_lmi(dm, tokenizer)
    # save the topk results
    save_lmi(args, topk)

    seed_everything(42, workers=True)
    model = TCTransformers(
        model_name_or_path=model_keys[args.model_name],
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        data_name=dm.dataset_name,
        task_name=dm.task_name,
        learning_rate=1e-5,
        gradient_clip_val=1,
        weight_decay=0,
    )

    trainer = Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=[0] if torch.cuda.is_available() else 0,
        enable_checkpointing=False,
        logger=False,
    )

    results = {}
    print("Evaluating pretrained model...")
    results['before'] = compile_results(trainer.predict(model, dm), dm, predict_split=predict_split)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attributed_results = {
        'before': integrated_gradients(
            results=results['before'],
            topk=topk,
            model_name_or_path=model_keys[args.model_name],
            text_field=dm.text_fields[0],
            device=device,
            model=model
        ).get_attributions()
    }

    print("Training model...")
    trainer.fit(model, dm)
    results['after'] = compile_results(trainer.predict(model, dm), dm, predict_split=predict_split)

    print("Evaluating finetuned model...")
    attributed_results['after'] = integrated_gradients(
        results=results['after'],
        topk=topk,
        model_name_or_path=model_keys[args.model_name],
        text_field=dm.text_fields[0],
        device=device,
        model=model
    ).get_attributions()

    save_results(args, attributed_results)


def save_lmi(args, topk):
    results_dir = f"results_{args.data_name}/{args.model_name}"
    os.makedirs(results_dir, exist_ok=True)

    prefix = args.task_name if args.data_name in ["glue", "tweets"] else args.data_name
    out_file = os.path.join(results_dir, f"lmi_{prefix}_{args.model_name}.pickle")

    with open(out_file, 'wb') as f:
        pickle.dump(topk, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"LMI results saved to {out_file}")

def save_results(args, attributed_results):
    results_dir = f"results_{args.data_name}/{args.model_name}"
    os.makedirs(results_dir, exist_ok=True)

    prefix = args.task_name if args.data_name in ["glue", "tweets"] else args.data_name

    for phase in ["before", "after"]:
        out_file = os.path.join(results_dir, f"{prefix}_{args.model_name}_{phase}.pickle")
        with open(out_file, 'wb') as f:
            pickle.dump(attributed_results[phase], f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Results saved to {results_dir}")



def identify_token_type(token, model, t):
    if model == 'bert':
        if token in string.punctuation:
            return 'punctuation'
        elif token in stop_words:
            return 'stopword'
        elif "#" in token:
            return 'subword'
        elif token in ['[CLS]', '[SEP]']:
            return 'eos'
        else:
            return 'word'
        
    if model in ['roberta', 'bart', 'deberta']:
        if t == 0:
            if token in string.punctuation:
                return 'punctuation'
            elif token in stop_words:
                return 'stopword'
            elif token in ['[CLS]', '[SEP]']:
                return 'eos'
            else:
                return 'word'
        else:
            if 'Ġ' in token:
                token = token.replace('Ġ', '') 
                if token in string.punctuation:
                    return 'punctuation'
                elif token in stop_words:
                    return 'stopword'
                elif token in ['[CLS]', '[SEP]']:
                    return 'eos'
                else:
                    return 'word'
            else:
                if token in string.punctuation:
                    return 'punctuation'
                else:
                    return 'subword'
            
    if model == 'albert':
        if t == 0:
            if token in string.punctuation:
                return 'punctuation'
            elif token in stop_words:
                return 'stopword'
            elif token in ['[CLS]', '[SEP]']:
                return 'eos'
            else:
                return 'word'
        else:
            if '▁' in token:
                token = token.replace('▁', '') 
                if token in string.punctuation:
                    return 'punctuation'
                elif token in stop_words:
                    return 'stopword'
                elif token in ['[CLS]', '[SEP]']:
                    return 'eos'
                else:
                    return 'word'
            else:
                return 'subword'    
