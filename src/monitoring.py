import os
import wandb
from dotenv import load_dotenv
load_dotenv()

def setup_wandb(project_name: str, run_name: str):
    # Set up your API KEY
    try:
        api_key = os.getenv("WANDB_API_KEY")
        wandb.login(key=api_key)
        print("Successfully logged into WandB.")
    except KeyError:
        raise EnvironmentError("WANDB_API_KEY is not set in the environment variables.")
    except Exception as e:
        print(f"Error logging into WandB: {e}")
    
    # Optional: Log models
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    os.environ["WANDB_WATCH"] = "all"
    os.environ["WANDB_SILENT"] = "true"
    
    # Initialize the WandB run
    try:
        wandb.init(project=project_name, name=run_name)
        print(f"WandB run initialized: Project - {project_name}, Run - {run_name}")
    except Exception as e:
        print(f"Error initializing WandB run: {e}")