from rankers import (
    Dot,
    RankerTrainer,
    RankerArguments,
    DotDataCollator,
    seed_everything,
    CatDataCollator,
    Cat,
    TrainingDataset,
)
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
import wandb
from fire import Fire
import logging
import os

logger = logging.getLogger(__name__)


def train(
    model_name_or_path: str,
    loss_fn: str,
    output_dir: str,
    triple_file: str,
    teacher_file: str = None,
    wandb_project: str = None,
    epochs: int = 1,
    max_steps: int = -1,
    batch_size: int = 8,
    lr: float = 1e-5,
    gradient_accumulation_steps: int = 1,
    warmup_ratio: float = 0.0,
    seed: int = 42,
    group_size: int = 2,
    fp16: bool = False,
    dataloader_num_workers: int = 0,
    save_total_limit: int = 3,
    save_strategy: str = "steps",
    inbatch_loss: str = None,
    listwise: bool = False,
    cat: bool = False,
    resume_from_checkpoint: bool = False,
):
    formatted_output = f"{model_name_or_path}-{loss_fn}-{'cat' if cat else 'dot'}-{'listwise' if listwise else 'pairwise'}-{group_size}-{batch_size}-{gradient_accumulation_steps}-seed{seed}"
    output_dir = os.path.join(output_dir, formatted_output)
    os.makedirs(output_dir, exist_ok=True)
    seed_everything(seed)
    if wandb_project is not None:
        wandb.init(project=wandb_project)

    model_constructor = Cat if cat else Dot
    collate_fn_constructor = CatDataCollator if cat else DotDataCollator

    logging.info(f"Loading {model_name_or_path}...")

    model_args = {}
    if not cat:
        model_args["inbatch_loss"] = inbatch_loss

    model = model_constructor.from_pretrained(model_name_or_path, **model_args)
    collate_fn = collate_fn_constructor(
        AutoTokenizer.from_pretrained(model_name_or_path)
    )

    logging.info(f"Loading dataset from {triple_file}...")

    triples = pd.read_json(triple_file, lines=True, orient="records")
    logging.info(f"Instantiating dataset...")

    dataset = TrainingDataset(triples, teacher_file, group_size, listwise=listwise)

    callbacks = []

    args = RankerArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        num_train_epochs=epochs,
        max_steps=max_steps,
        seed=seed,
        report_to="wandb",
        group_size=group_size,
        fp16=fp16,
        dataloader_num_workers=dataloader_num_workers,
        save_total_limit=save_total_limit,
        save_strategy=save_strategy,
    )

    logging.info(f"Training {model_name_or_path} with {loss_fn} loss")
    if max_steps < 1:
        max_steps = (len(dataset) // batch_size) * epochs
    opt = AdamW(model.parameters(), lr=lr)
    sched = (
        get_linear_schedule_with_warmup(
            opt, int(args.warmup_ratio * max_steps), max_steps
        )
        if args.warmup_ratio > 0
        else None
    )

    trainer = RankerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collate_fn,
        loss_fn=loss_fn,
        callbacks=callbacks,
        optimizers=(opt, sched),
    )

    if resume_from_checkpoint:
        # find latest checkpoint
        checkpoints = os.listdir(output_dir)
        checkpoints = [
            int(x.split("-")[1]) for x in checkpoints if x.startswith("checkpoint")
        ]
        if len(checkpoints) > 0:
            latest_checkpoint = max(checkpoints)
            trainer.load_model(f"{output_dir}/checkpoint-{latest_checkpoint}")
            resume_from_checkpoint = f"{output_dir}/checkpoint-{latest_checkpoint}"
        else:
            logging.info("No checkpoint found, starting from scratch")
            resume_from_checkpoint = None

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(train)
