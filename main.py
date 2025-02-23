import trainer


# DDQN training
# chk_name = r"saved_models\flappybird_ddqn.pth"
# args = trainer.TrainingArguments(load_checkpoint = chk_name, initial_epsilon=0.0001, observe=10000, explore=1, batch_size=128, lr_value=1e-6, replay_memory=10000)
# trainer = trainer.DQN(args)
# trainer.train()

# VPG training
# chk_name = (r"saved_models\flappybird_vpg_actor_0.pth", r"saved_models\flappybird_vpg_critic_0.pth")
# args = trainer.TrainingArguments(horizon = 8192, save_interval=50, load_checkpoint = chk_name)
# trainer = trainer.VPG(args)
# trainer.train()


# PPO training
# chk_name = (r"saved_models\flappybird_ppo_actor_0.pth", r"saved_models\flappybird_ppo_critic_0.pth")
# args = trainer.TrainingArguments(horizon = 8192, save_interval=50, lr_policy=1e-5, lr_value=1e-4, load_checkpoint = chk_name)
# trainer = trainer.PPO(args)
# trainer.train()