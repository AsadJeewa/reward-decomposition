
import wandb

from main_ppo import run_ppo


def train():
    with wandb.init():
        config = wandb.config

        batch_size = config.num_envs * config.num_rollout_steps

        # config_name = (
        #     f"lr={config.learning_rate}_"
        #     f"batch={batch_size}_"
        #     f"ent={config.entropy_loss_coefficient}_"
        #     f"obsnorm={config.normalize_observations}"
        # )
        # wandb.run.group = config_name

        run_ppo(
            env_id=config.env_id,
            env_is_discrete=config.env_is_discrete,
            total_timesteps=config.total_timesteps,
            num_rollout_steps=config.num_rollout_steps,
            learning_rate=config.learning_rate,
            entropy_loss_coefficient=config.entropy_loss_coefficient,
            normalize_observations=config.normalize_observations,
            seed=config.seed,
            use_wandb=True,
            # wandb_group=config_name, 
        )


if __name__ == "__main__":
    train()