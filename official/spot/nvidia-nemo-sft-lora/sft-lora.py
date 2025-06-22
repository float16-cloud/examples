import os
import time
import pandas as pd
import fiddle as fdl
from datasets import Dataset
import lightning.pytorch as pl
from nemo import lightning as nl
from nemo.collections import llm
from lightning.pytorch.loggers import WandbLogger
from nemo.collections.llm.recipes.optim.adam import pytorch_adam_with_cosine_annealing

os.environ['WANDB_API_KEY'] = "WANDB_API_KEY"  # Replace with your actual WandB API key

def prepare_datasets(parquet_datat_path,tokenizer, batch_size):
    def formatting_prompts_func(example):
        # Custom formatting function for any template
        sentiment = None
        category = example.get('category', None)
        if str(category) == "0":
            sentiment = "positive"
        elif str(category) == "1":
            sentiment = "natural"
        elif str(category) == "2":
            sentiment = "negative"

        if sentiment is None:
            sentiment = "unknown"

        formatted_text = [
            f"Context: {example['texts']} Sentiment:",
            f" {sentiment}",
        ]
        # Custom formatting function for any template

        context_ids, answer_ids = list(map(tokenizer.text_to_ids, formatted_text))
        if len(context_ids) > 0 and context_ids[0] != tokenizer.bos_id and tokenizer.bos_id is not None:
            context_ids.insert(0, tokenizer.bos_id)
        if len(answer_ids) > 0 and answer_ids[-1] != tokenizer.eos_id and tokenizer.eos_id is not None:
            answer_ids.append(tokenizer.eos_id)

        return dict(
            labels=(context_ids + answer_ids)[1:],
            input_ids=(context_ids + answer_ids)[:-1],
            loss_mask=[0] * (len(context_ids) - 1) + [1] * len(answer_ids),
        )

    # Use pandas to read the parquet file
    train_df = pd.read_parquet(parquet_datat_path)
    columns = train_df.columns.tolist()
    train_data = train_df.to_dict(orient="list")
    train_dataset = Dataset.from_dict(train_data)
    datamodule = llm.HFDatasetDataModule(train_dataset, split="train", micro_batch_size=batch_size, pad_token_id=tokenizer.eos_id or 0)

    datamodule.map(
        formatting_prompts_func,
        batched=False,
        batch_size=1,
        remove_columns=columns
    )

    return datamodule


def checkpoint_callback(ckpt_folder, save_every_n_train_steps):
    ckpt = nl.ModelCheckpoint(
        save_last=True,
        every_n_train_steps=save_every_n_train_steps,
        monitor="reduced_train_loss",
        save_top_k=1,
        save_on_train_epoch_end=True,
        save_optim_on_train_end=True,
    )

    return nl.NeMoLogger(
        name="nemo2_sft",
        log_dir=ckpt_folder,
        use_datetime_version=False,  # must be false if using auto resume
        ckpt=ckpt,
        wandb=None,
    )


def make_strategy(model, adapter_only=False):
    return pl.strategies.SingleDeviceStrategy(device='cuda:0', checkpoint_io=model.make_checkpoint_io(adapter_only=adapter_only))

def main():

    BATCH_SIZE = 16
    model_path = "../../models/Qwen2.5-0.5B-Instruct"  # Path to the local model directory
    data_path = "../../datasets/wisesight_sentiment_train"  # Path to the local dataset directory
    checkpoint_path = "../../checkpoints"

    optimizer = fdl.build(pytorch_adam_with_cosine_annealing(max_lr=10e-5, warmup_steps=50))
    wandb_logger = WandbLogger(log_model="all", project="nemo-sft", name=f"{int(time.time())}-sft-wisesight-sentiment")

    model = llm.HFAutoModelForCausalLM(
        model_name=model_path, # Path to the local model directory
        trust_remote_code=False,
        load_in_4bit=False,
        device_map='cuda:0',
    )
    strategy = make_strategy(model, False)
    train_data = prepare_datasets(data_path,model.tokenizer, BATCH_SIZE)
    train_data.global_batch_size = BATCH_SIZE

    resume = nl.AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
    )
    
    llm.api.finetune(
        model=model,
        data=train_data,
        trainer=nl.Trainer(
            devices=1,
            num_nodes=1,
            # max_steps=100, # (actual_steps = max_steps * accumulate_grad_batches) or max_epochs
            max_epochs=1,
            accelerator='gpu',
            strategy=strategy,
            limit_val_batches=0.0,
            num_sanity_val_steps=0,
            log_every_n_steps=50,
            accumulate_grad_batches=1,
            use_distributed_sampler=False,
            enable_progress_bar=False, # Disable progress bar for cleaner output
            precision='bf16-mixed',
            logger=wandb_logger
        ),
        optim=optimizer,
        peft=llm.peft.LoRA(
            target_modules=['*_proj'],
            dim=8,
        ),
        log=checkpoint_callback(checkpoint_path, save_every_n_train_steps=500),
        resume=resume
    )

print("Starting NeMo SFT application...")
main()