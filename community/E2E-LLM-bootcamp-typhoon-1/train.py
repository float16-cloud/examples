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

os.environ['WANDB_API_KEY'] = ""  # Replace with your actual WandB API key

def prepare_datasets(parquet_datat_path, tokenizer, batch_size):
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
        elif str(category) == "3":
            sentiment = "question"

        if sentiment is None:
            sentiment = "unknown"

        # Create messages for chat template
        messages = [
            {"role": "user", "content": f"Analyze the sentiment of the following text:\nContext: {example['texts']}\nSentiment: "},
            {"role": "assistant", "content": f"The sentiment is {sentiment}"}
        ]
        
        # Apply chat template to get the formatted conversation
        formatted_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # Also get just the context (user message) to know where to start masking
        context_text = tokenizer.apply_chat_template(
            [messages[0]], 
            tokenize=False, 
            add_generation_prompt=True  # This adds the assistant prompt
        )
        
        # Tokenize both
        context_ids = tokenizer.text_to_ids(context_text)
        full_ids = tokenizer.text_to_ids(formatted_text)
        
        # Add BOS token if needed
        if len(context_ids) > 0 and context_ids[0] != tokenizer.bos_id and tokenizer.bos_id is not None:
            context_ids.insert(0, tokenizer.bos_id)
        if len(full_ids) > 0 and full_ids[0] != tokenizer.bos_id and tokenizer.bos_id is not None:
            full_ids.insert(0, tokenizer.bos_id)
            
        # Add EOS token if needed
        if len(full_ids) > 0 and full_ids[-1] != tokenizer.eos_id and tokenizer.eos_id is not None:
            full_ids.append(tokenizer.eos_id)

        
        # Create loss mask: 0 for context (user input), 1 for assistant response
        # Following the NeMo pattern from the documentation
        loss_mask = [0] * (len(context_ids) - 1) + [1] * (len(full_ids) - len(context_ids))
        #loss_mask = [0] * (context_len - 1) + [1] * (len(full_ids) - (context_len - 1))

        return dict(
            labels=full_ids[1:],
            input_ids=full_ids[:-1],
            loss_mask=loss_mask,
        )

    # Use pandas to read the parquet file
    train_df = pd.read_parquet(parquet_datat_path)
    train_df = train_df[train_df['category'].isin([0, 1, 2])]
    print(f"Removed 'q' class. Remaining categories: pos(0), neu(1), neg(2)")
    
    # Balance the dataset by sampling equal amounts from each class
    # Category mapping: {"pos": 0, "neu": 1, "neg": 2, "q": 3}
    class_counts = train_df['category'].value_counts()
    print(f"Original class distribution: {class_counts.to_dict()}")
    
    # Find the minimum class size for balanced sampling
    min_samples = class_counts.min()
    print(f"Sampling {min_samples} samples from each class")
    
    # Sample equal amounts from each class
    balanced_dfs = []
    for category in [0, 1, 2]:  # pos, neu, neg
        category_df = train_df[train_df['category'] == category]
        if len(category_df) > 0:
            # Sample with replacement if needed, without replacement if we have enough samples
            replace = len(category_df) < min_samples
            sampled_df = category_df.sample(n=min_samples, replace=replace, random_state=42)
            balanced_dfs.append(sampled_df)
    
    # Combine all balanced samples
    train_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Shuffle the balanced dataset
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    final_class_counts = train_df['category'].value_counts()
    print(f"Balanced class distribution: {final_class_counts.to_dict()}")
    print(f"Total samples after balancing: {len(train_df)}")
    
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
    model_path = "/share_weights/bootcamp/e2e-llm-typhoon-1/llama3.2-typhoon2-3b-instruct"  # Path to the local model directory
    data_path = "/share_weights/bootcamp/e2e-llm-typhoon-1/datasets"  # Path to the local dataset directory
    checkpoint_path = "../../checkpoints_typhoon2"

    optimizer = fdl.build(pytorch_adam_with_cosine_annealing(max_lr=10e-5, warmup_steps=10))
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
            max_epochs=3,
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
            target_modules=["linear_qkv", "linear_proj", "*_proj"],
            dim=8,
            alpha=16
        ),
        log=checkpoint_callback(checkpoint_path, save_every_n_train_steps=500),
        resume=resume
    )

print("Starting NeMo SFT application...")
main()