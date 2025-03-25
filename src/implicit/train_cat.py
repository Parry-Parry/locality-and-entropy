from rankers import (
    RankerTrainingArguments,
    RankerModelArguments,
    RankerDataArguments,
    RankerTrainer,
    Cat,
    TrainingDataset,
    CatDataCollator,
)
from transformers import HfArgumentParser
import os


def main():
    parser = HfArgumentParser(
        (RankerModelArguments, RankerDataArguments, RankerTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_data_file = data_args.training_dataset_file.split("/")[-1].replace(
        ".jsonl", ""
    )
    training_args.output_dir = (
        training_args.output_dir
        + f"/cat-{training_args.loss_fn.name}-{training_data_file}-{training_args.group_size}"
    )
    if os.path.exists(os.path.join(training_args.output_dir, "config.json")):
        print(f"Model already exists at {training_args.output_dir}, exiting...")
        return
    model = Cat.from_pretrained(model_args.model_name_or_path, num_labels=2)

    dataset = TrainingDataset(
        data_args.training_dataset_file,
        group_size=training_args.group_size,
        corpus=data_args.ir_dataset,
        no_positive=data_args.no_positive,
        teacher_file=data_args.teacher_file,
        lazy_load_text=False,
    )
    collate_fn = CatDataCollator(model.tokenizer)

    trainer = RankerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        loss_fn=training_args.loss_fn,
    )

    # check for any checkpoints in output_dir
    if os.path.exists(training_args.output_dir):
        paths = os.listdir(training_args.output_dir)
        paths = [path for path in paths if "checkpoint" in path]
        if paths:
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()
