from unsloth import FastLanguageModel
from config import max_seq_length, dtype, load_in_4bit

def get_original_model(model_name="unsloth/Meta-Llama-3.1-8B-Instruct"):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,  
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    return model, tokenizer

def get_lora_model(model):
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,   # LoRA rank - suggested values: 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,   # Supports any, but = 0 is optimized
        bias="none",      # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # Ideal for long context tuning
        random_state=3407,
        use_rslora=False,   # Disable rank-sensitive LoRA for simpler tasks
        loftq_config=None   # No LoftQ, for standard fine-tuning
    )

    return model

def get_model(model_name="unsloth/Meta-Llama-3.1-8B-Instruct"):
    model, tokenizer = get_original_model(model_name)
    model = get_lora_model(model)
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = get_model()
    print("Model loaded successfully!")
    print(model)