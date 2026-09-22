import wandb

from main_ppo import run_ppo


def train():
    with wandb.init():
        config = wandb.config

        # config_name = (
        #     f"lr={config.learning_rate}_"
        #     f"div={config.diversity_scale}_"
        #     f"ent={config.entropy_loss_coefficient}"
        # )

        # wandb.run.group = config_name

        run_ppo(
            env_id=config.env_id,
            env_is_discrete=config.env_is_discrete,
            total_timesteps=config.total_timesteps,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            entropy_loss_coefficient=config.entropy_loss_coefficient,
            diversity_scale=config.diversity_scale,
            seed=config.seed,
            use_wandb=True,
        )


if __name__ == "__main__":
    train()