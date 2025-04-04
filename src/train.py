#import libraries
from utils import *
from tqdm import tqdm
import argparse
import pickle

parser = argparse.ArgumentParser(description='Experiment setup')
parser.add_argument('--data_name', type=str,
                    help='name of dataset for finetuning')
parser.add_argument('--task_name', type=str, nargs='?', const = None,
                    help='name of task for finetuning')
parser.add_argument('--model_name', type=str,
                    help='name of model to finetune')

args = parser.parse_args()

data_name = args.data_name
task_name = args.task_name
model_name = args.model_name

# repeat over 5 trials
# get model and tokenizer details
model_keys = {"bert": "bert-base-uncased",
              "roberta": "roberta-base",
              "bart": "facebook/bart-base",
              "deberta": "microsoft/deberta-base",
              "albert": "albert-base-v1",
             }


tokenizer = AutoTokenizer.from_pretrained(model_keys[model_name])

# get data
if data_name == "glue":
    predict_split = "validation"
else:
    predict_split = "test"
    

dataset_keys = {"glue": "glue",
               "20ng": "SetFit/20_newsgroups",
               "agnews": "ag_news",
               "trec": "trec",
               "tweets": "tweet_eval"}



dm = DataModule(model_keys[model_name], dataset_name = dataset_keys[data_name], task_name = task_name,
                predict_split = predict_split)
dm.prepare_data()
dm.setup("fit")


# get long tailed word distribution from training set

labels = list(range(dm.num_labels))
split = 'train'
        
total_word_count, word_dict, label_dict = get_count_dicts(dm, labels, split, tokenizer = tokenizer)

lmi_label = {}
lmi_word = {}

for label in label_dict.keys():
    lmi = []
    for word in tqdm(word_dict.keys()):
        if label in word_dict[word].keys():
            lmi_, count_w = get_lmi(word, label, word_dict, label_dict, total_word_count)
            if label not in lmi_word.keys():
                lmi_word[label] = {}
            
            lmi_word[label][word] = lmi_
            lmi.append((word, count_w, lmi_))
            
    lmi.sort(key=lambda a: a[2])
    lmi_label[label] = lmi
    
topk = {}
    
for label in lmi_label.keys():
    lmi_label[label].sort(key=lambda a:a[2], reverse = True)
    topk[label] = get_topk_head(lmi_label[label], k = 0.05)
    
    


# train model
seed_everything(42, workers=True)

model = TCTransformers(
    model_name_or_path=model_keys[model_name],
    num_labels=dm.num_labels,
    eval_splits=dm.eval_splits,
    data_name=dm.dataset_name,
    task_name=dm.task_name,
    learning_rate = 1e-5,
    gradient_clip_val = 1,
    weight_decay = 0
)

trainer = Trainer(
    max_epochs=3,
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  
    enable_checkpointing=False,
    logger = False
)


# results before fine-tuning (directly using pretrained model)
results = {}
results['before'] = trainer.predict(model, dm)
results['before'] = compile_results(results['before'], dm, predict_split = predict_split)

# get attributions 
attributed_results = {}

ig_ = integrated_gradients(results = results['before'],
                          topk = topk,
                          model_name_or_path = model_keys[model_name],
                          text_field = dm.text_fields[0],
                          device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                          model = model)


attributed_results['before'] = ig_.get_attributions()


#after training (using finetuned model)
trainer.fit(model, dm)
results['after'] = trainer.predict(model, dm)
results['after'] = compile_results(results['after'], dm, predict_split = predict_split)

# get attributions 

print("getting attributions...")
ig_ = integrated_gradients(results = results['after'],
                          topk = topk,
                          model_name_or_path = model_keys[model_name],
                          text_field = dm.text_fields[0],
                          device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                          model = model)


attributed_results['after'] = ig_.get_attributions()


if data_name in ["glue", "tweets"]:
    with open('results_'+task_name+'_'+model_name+'_'+'before.pickle', 'wb') as handle:
        pickle.dump(attributed_results['before'], handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('results_'+task_name+'_'+model_name+'_'+'after.pickle', 'wb') as handle:
        pickle.dump(attributed_results['after'], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
else:
    with open('results_'+data_name+'_'+model_name+'_'+'before.pickle', 'wb') as handle:
        pickle.dump(attributed_results['before'], handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('results_'+data_name+'_'+model_name+'_'+'after.pickle', 'wb') as handle:
        pickle.dump(attributed_results['after'], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
