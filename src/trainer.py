from trl import SFTTrainer, SFTConfig
import torch
from config import max_seq_length

def get_training_args():
    args = SFTConfig(
        #Batch Size
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        
        # LR
        warmup_steps = 5,
        learning_rate = 2e-4,
        num_train_epochs = 3,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",

        # Logging
        logging_strategy = "steps",
        logging_steps = 1,
        output_dir = "outputs",
        report_to = "wandb",
        seed = 3407,

        # Saving args
        save_strategy = "steps",
        save_steps = 100,
        save_total_limit = 5,
        load_best_model_at_end = False,
        save_only_model = False,

        # Dataset args
        max_seq_length = max_seq_length,
        dataset_text_field = "text",
        dataset_num_proc = 1,
        packing = False
    )
    return args

def initialize_trainer(model, tokenizer, dataset):
    args = get_training_args()
    
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset['train'],
        args = args
    )
    return trainer

def train(trainer):
    trainer_stats = trainer.train(resume_from_checkpoint=True)
    return trainer_stats