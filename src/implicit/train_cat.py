from rankers import ( 
                      RankerTrainingArguments, 
                      RankerModelArguments,
                      RankerDataArguments,
                      RankerTrainer, 
                      Cat,
                      TrainingDataset,
                      CatDataCollator,
                      )
from transformers import HfArgumentParser, get_constant_schedule_with_warmup
from torch.optim import AdamW

def main():
    parser = HfArgumentParser((RankerModelArguments, RankerDataArguments, RankerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    formatted_model_name = model_args.model_name_or_path.replace('/', '-')
    distilled = "distilled" if data_args.teacher_file is not None else "first"
    training_args.output_dir = training_args.output_dir + f'/cat-{formatted_model_name}-{training_args.loss_fn.name}-{training_args.group_size}-{distilled}'

    model = Cat.from_pretrained(model_args.model_name_or_path)

    dataset = TrainingDataset(data_args.training_data, corpus=data_args.ir_dataset, use_positive=data_args.use_positive)
    collate_fn = CatDataCollator(model.tokenizer)

    opt = AdamW(model.parameters(), lr=training_args.learning_rate)

    trainer = RankerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        optimizers=(opt, get_constant_schedule_with_warmup(opt, training_args.warmup_steps)),
        loss_fn = training_args.loss_fn,
        )
    
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == '__main__':
    main()