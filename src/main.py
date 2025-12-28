from monitoring import setup_wandb, end_wandb
from model import get_model
from dataset import get_dataset, update_tokenizer_with_template, map_dataset_to_template
from trainer import initialize_trainer, train
from gpu_metrics import display_start_gpu_metrics, display_end_gpu_metrics

def main():
    setup_wandb(project_name="SAC-LLM", run_name="First")
    dataset = get_dataset()
    model, tokenizer = get_model()
    tokenizer = update_tokenizer_with_template(tokenizer)
    dataset = map_dataset_to_template(dataset, tokenizer)
    trainer = initialize_trainer(model, tokenizer, dataset)
    start_gpu_memory, max_memory = display_start_gpu_metrics()
    trainer_stats = train(trainer)
    display_end_gpu_metrics(start_gpu_memory, max_memory, trainer_stats)
    end_wandb()