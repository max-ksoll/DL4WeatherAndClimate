import wandb

from src.sweep_config import getSweepID


def train():
    entity = "philippgrill"
    project = "CaseStudiesOfAIImplementationResults"
    with wandb.init(project=project, entity=entity) as run:
        epochs = wandb.config.get("dataset")
        wandb.log(
            {
                "mse": 1,
                "epochs": epochs,
            }
        )

if __name__ == "__main__":
    wandb.agent(getSweepID(), train)

