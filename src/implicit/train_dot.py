from rankers import ( 
                      RankerTrainingArguments, 
                      RankerModelArguments,
                      RankerDataArguments,
                      RankerTrainer, 
                      Dot,
                      TrainingDataset,
                      DotDataCollator,
                      )
from transformers import HfArgumentParser, get_constant_schedule_with_warmup
from torch.optim import AdamW
import os

def main():
    parser = HfArgumentParser((RankerModelArguments, RankerDataArguments, RankerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_data_file = data_args.training_dataset_file.split('/')[-1].replace('.jsonl', '')
    formatted_model_name = model_args.model_name_or_path.replace('/', '-')
    distilled = "distilled" if data_args.teacher_file is not None else "first"
    training_args.output_dir = training_args.output_dir + f'/dot-{formatted_model_name}-{training_args.loss_fn.name}-{training_data_file}-{training_args.group_size}-{distilled}'
    model = Dot.from_pretrained(model_args.model_name_or_path)

    dataset = TrainingDataset(data_args.training_dataset_file, corpus=data_args.ir_dataset, no_positive=data_args.no_positive, teacher_file=data_args.teacher_file, lazy_load_text=False)
    collate_fn = DotDataCollator(model.tokenizer)

    opt = AdamW(model.parameters(), lr=training_args.learning_rate)
    
    num_training_steps = len(dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs if training_args.max_steps < 1 else training_args.max_steps

    trainer = RankerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        optimizers=(opt, get_constant_schedule_with_warmup(opt, training_args.get_warmup_steps(num_training_steps))),
        loss_fn = training_args.loss_fn,
        )
    
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == '__main__':
    main()