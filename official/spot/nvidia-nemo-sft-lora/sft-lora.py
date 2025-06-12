import fiddle as fdl
import lightning.pytorch as pl

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.recipes.optim.adam import pytorch_adam_with_cosine_annealing


def make_squad_hf_dataset(tokenizer, batch_size, fp8=False):
    def formatting_prompts_func(example):
        formatted_text = [
            f"Context: {example['context']} Question: {example['question']} Answer:",
            f" {example['answers']['text'][0].strip()}",
        ]
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

    datamodule = llm.HFDatasetDataModule(
        "rajpurkar/squad",
        split="train",
        micro_batch_size=batch_size,
        pad_token_id=tokenizer.eos_id or 0,
        # pad_seq_len_divisible=16 if fp8 else None,  # FP8 training requires seq length to be divisible by 16.
    )
    datamodule.map(
        formatting_prompts_func,
        batched=False,
        batch_size=2,
        remove_columns=["id", "title", "context", "question", 'answers'],
    )
    return datamodule


def make_strategy(strategy, model, devices, num_nodes, adapter_only=False, enable_cpu_offload=False):
    if strategy == 'auto':
        return pl.strategies.SingleDeviceStrategy(
            device='cuda:0',
            checkpoint_io=model.make_checkpoint_io(adapter_only=adapter_only),
        )
    elif strategy == 'ddp':
        return pl.strategies.DDPStrategy(
            checkpoint_io=model.make_checkpoint_io(adapter_only=adapter_only),
        )
    elif strategy == 'fsdp2':
        offload_policy = None
        if enable_cpu_offload:
            from nemo.lightning.pytorch.strategies.fsdp2_strategy import HAS_CPU_OFFLOAD_POLICY, CPUOffloadPolicy

            assert HAS_CPU_OFFLOAD_POLICY, "Could not import offload policy"
            offload_policy = CPUOffloadPolicy()

        return nl.FSDP2Strategy(
            data_parallel_size=devices * num_nodes,
            tensor_parallel_size=1,
            checkpoint_io=model.make_checkpoint_io(adapter_only=adapter_only),
            offload_policy=offload_policy,
        )
    else:
        raise NotImplementedError("Encountered unknown strategy")


def main():
    """Example script to run PEFT with a HF transformers-instantiated model on squad."""
    BATCH_SIZE = 24

    optimizer = fdl.build(pytorch_adam_with_cosine_annealing(max_lr=10e-5, warmup_steps=50))
    model_accelerator = None

    model = llm.HFAutoModelForCausalLM(
        model_name="Qwen/Qwen2.5-0.5B",
        model_accelerator=model_accelerator,
        trust_remote_code=False,
        load_in_4bit=False,
    )
    strategy = make_strategy("auto", model, 1, 1, False)
    train_data = make_squad_hf_dataset(model.tokenizer, BATCH_SIZE, False)
    train_data.global_batch_size = BATCH_SIZE

    llm.api.finetune(
        model=model,
        data=train_data,
        trainer=nl.Trainer(
            devices=1,
            num_nodes=1,
            max_steps=100,
            accelerator='gpu',
            strategy=strategy,
            log_every_n_steps=1,
            limit_val_batches=0.0,
            num_sanity_val_steps=0,
            accumulate_grad_batches=10,
            gradient_clip_val=1.0,
            use_distributed_sampler=False
        ),
        optim=optimizer,
        peft=llm.peft.LoRA(
            target_modules=['*_proj'],
            dim=8,
        )
    )

main()