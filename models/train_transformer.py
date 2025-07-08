import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
from utils import log_experiment

#load dataset via `datasets` library
raw = load_dataset('imdb')
metric = load_metric('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return metric.compute(predictions=preds, references=labels)

if __name__ == "__main__":
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    def tokenize_fn(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=256)
    dataset = raw.map(tokenize_fn, batched=True)
    dataset.set_format(type='torch', columns=['input_ids','attention_mask','label'])

    #hyperparameter grid
    for lr in [5e-5, 3e-5, 2e-5]:
        for bs in [16, 32]:
            args = TrainingArguments(
                output_dir=f'results/lr{lr}_bs{bs}',
                evaluation_strategy='epoch',
                learning_rate=lr,
                per_device_train_batch_size=bs,
                per_device_eval_batch_size=bs,
                num_train_epochs=3,
                weight_decay=0.01,
                logging_steps=500
            )
            model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=dataset['train'],
                eval_dataset=dataset['test'],
                compute_metrics=compute_metrics
            )
            #train & evaluate
            trainer.train()
            eval_res = trainer.evaluate()
            #log to DB
            params = {'lr': lr, 'batch_size': bs, 'epochs': 3}
            log_experiment('DistilBERT', params,
                           train_loss=None,
                           val_loss=eval_res['eval_loss'],
                           val_acc=eval_res['eval_accuracy'])