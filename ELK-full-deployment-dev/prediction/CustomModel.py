import evaluate
from datasets import load_dataset
import numpy as np
from transformers import(
    Trainer,
    TrainingArguments,
    DistilBertTokenizerFast,
    DistilBertForTokenClassification,
    DataCollatorForTokenClassification,
    pipeline
)

class Model():
    def __init__(self, model_path: str, tokenizer_path: str) -> None:
        self.pipe      = None
        self.model     = DistilBertForTokenClassification.from_pretrained(model_path, num_labels=9)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)

    def predict(self, data: str) -> list:
        if self.pipe is None:
            self.pipe = pipeline("token-classification", model = self.model, tokenizer = self.tokenizer)
        return( self.format_output( self.pipe(data) ) )
    
    def format_output(self, data: list) -> list:
        for entity in data:
            for key in entity.keys():
                entity[key] = str(entity[key])
        return(data)
    
    def tokenize_and_align_labels(self, examples, padding:str, truncation:bool, max_length, is_split_into_words:bool, label_column_name:str, text_column_name:str):
        
        tokenized_inputs = self.tokenizer(
            examples[text_column_name],
            padding				= padding,
            truncation			= truncation,
            max_length			= max_length,
            is_split_into_words = is_split_into_words,
        )

        label_to_id = self.convert_label_to_id()
        b_to_i_label = self.convert_b_to_i()

        labels = []

        for i, label in enumerate(examples[label_column_name]):

            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)

                elif word_idx != previous_word_idx:
                    label_ids.append( label_to_id[ label[word_idx] ] )

                else:
                    label_ids.append( b_to_i_label[ label_to_id[ label[word_idx] ] ] )

                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def compute_metrics(self, p) -> dict:
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            results = self.metric.compute(predictions=true_predictions, references=true_labels)

            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
    
    def convert_b_to_i(self) -> list:
        b_to_i_label = []
        for idx, label in enumerate(self.label_list):
            if label.startswith("B-") and label.replace("B-", "I-") in self.label_list:
                b_to_i_label.append(self.label_list.index(label.replace("B-", "I-")))
            else:
                b_to_i_label.append(idx)
        return b_to_i_label
    
    def convert_label_to_id(self) -> dict:
        return {i: i for i in range(len(self.label_list))}
    
    def train(self, dataset_path: str, output_dir: str, label_column_name:str = "ner_tags", text_column_name:str = "tokens") -> None:
        training_args = TrainingArguments(
            do_train					= True,
            do_eval						= False,
            output_dir					= output_dir,
            evaluation_strategy			= "epoch",
            save_strategy               = "epoch",
            resume_from_checkpoint      = output_dir,
            num_train_epochs			= 1,
            per_device_train_batch_size = 16,
            per_device_eval_batch_size	= 16,
            learning_rate				= 2e-5,
        )

        dataset         = load_dataset(dataset_path)
        train_dataset   = dataset["train"]
        eval_dataset    = dataset["validation"]
        features        = train_dataset.features
        self.label_list = features[label_column_name].feature.names

        self.metric = evaluate.load("seqeval")

        train_dataset = train_dataset.map\
        (
            self.tokenize_and_align_labels,
            batched = True,
            fn_kwargs = \
            {
                'padding'             : "max_length", 
                'truncation'          : True, 
                'max_length'          : None, 
                'is_split_into_words' : True, 
                'label_column_name'   : label_column_name,
                'text_column_name'    : text_column_name
            },
            # num_proc=data_args.preprocessing_num_workers,
            desc="Running tokenizer on train dataset",
        )

        eval_dataset = eval_dataset.map\
        (
            self.tokenize_and_align_labels,
            batched = True,
            fn_kwargs = \
            {
                'padding'             : "max_length", 
                'truncation'          : True, 
                'max_length'          : None, 
                'is_split_into_words' : True, 
                'label_column_name'   : label_column_name,
                'text_column_name'    : text_column_name
            },
            # num_proc=data_args.preprocessing_num_workers,
            desc="Running tokenizer on validation dataset",
        )

        data_collator = DataCollatorForTokenClassification(self.tokenizer, pad_to_multiple_of=8)

        trainer = Trainer(
                model           = self.model,
                args            = training_args,
                train_dataset   = train_dataset,
                eval_dataset    = eval_dataset,
                tokenizer       = self.tokenizer,
                data_collator   = data_collator,
                compute_metrics = self.compute_metrics,
         )

        trainer.train()
        trainer.save_model()

if __name__ == '__main__':
    model = Model("distilbert-base-cased", "distilbert-base-cased")
    model.train("conll2003", "test_model", "ner_tags", "tokens")