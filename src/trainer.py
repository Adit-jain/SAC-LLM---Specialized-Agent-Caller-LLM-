from transformers import TrainingArguments
from trl import SFTTrainer
import torch
from unsloth import unsloth_train
from config import max_seq_length

def get_training_args():
    args = TrainingArguments(
        per_device_train_batch_size = 8,  # Controls the batch size per device
        gradient_accumulation_steps = 2,  # Accumulates gradients to simulate a larger batch
        warmup_steps = 5,
        learning_rate = 2e-4,             # Sets the learning rate for optimization
        num_train_epochs = 3,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        optim = "adamw_8bit",
        weight_decay = 0.01,              # Regularization term for preventing overfitting
        lr_scheduler_type = "linear",     # Chooses a linear learning rate decay
        seed = 3407,                        
        output_dir = "outputs",             
        report_to = "wandb",              # Enables Weights & Biases (W&B) logging
        logging_steps = 1,                # Sets frequency of logging to W&B
        logging_strategy = "steps",       # Logs metrics at each specified step
        save_strategy = "no",               
        load_best_model_at_end = True,    # Loads the best model at the end
        save_only_model = False           # Saves entire model, not only weights
    )
    return args

def initialize_trainer(model, tokenizer, dataset):
    args = get_training_args()
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,        # Can make training 5x faster for short sequences.
        args = args
    )
    return trainer

def train(trainer):
    trainer_stats = unsloth_train(trainer)
    print(trainer_stats)
    return trainer_stats