from monitoring import setup_wandb
from model import get_model
from dataset import get_dataset, update_tokenizer_with_template, map_dataset_to_template

def main():
    dataset = get_dataset()
    model, tokenizer = get_model()
    tokenizer = update_tokenizer_with_template(tokenizer)
    dataset = map_dataset_to_template(dataset, tokenizer)