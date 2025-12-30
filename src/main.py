# from monitoring import setup_wandb, end_wandb
from model import get_model
from dataset import get_dataset, update_tokenizer_with_template, map_dataset_to_template
from trainer import initialize_trainer, train

def main():
    # setup_wandb(project_name="SAC-LLM", run_name="First")
    dataset = get_dataset()
    model, tokenizer = get_model()
    tokenizer = update_tokenizer_with_template(tokenizer)
    dataset = map_dataset_to_template(dataset, tokenizer)
    trainer = initialize_trainer(model, tokenizer, dataset)
    train(trainer)
    # end_wandb()

if __name__ == "__main__":
    main()