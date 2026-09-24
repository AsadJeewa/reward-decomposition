import wandb

from main_ppo import run_ppo


def train():
    with wandb.init():
        config = wandb.config

        run_ppo(
            env_id=config.env_id,
            env_is_discrete=config.env_is_discrete,
            total_timesteps=config.total_timesteps,
            learning_rate=config.learning_rate,
            num_rollout_steps=config.num_rollout_steps,
            entropy_loss_coefficient=config.entropy_loss_coefficient,
            gamma=config.gamma,
            seed=config.seed,
            use_wandb=True,
        )


if __name__ == "__main__":
    train()