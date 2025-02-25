import trainer as trainer


# DDQN training
chk_name = r"saved_models\flappybird_ddqn.pth"
# next line is for testing only, no training
args = trainer.TrainingArguments(initial_epsilon=0., final_epsilon=0., observe=100000, load_checkpoint=chk_name)
trainer = trainer.DQN(args)
trainer.train()

# VPG training
# chk_name = (r"saved_models\flappybird_vpg_actor_0.pth", r"saved_models\flappybird_vpg_critic_0.pth")
# args = trainer.TrainingArguments(horizon = 8192, save_interval=50, load_checkpoint = chk_name)
# trainer = trainer.VPG(args)
# trainer.train()


# PPO training
# chk_name = (r"saved_models\flappybird_ppo_actor_100.pth", r"saved_models\flappybird_ppo_critic_100.pth")
# args = trainer.TrainingArguments(horizon = 8192, save_interval=50, lr_policy=1e-5, lr_value=1e-4, load_checkpoint=chk_name)
# trainer = trainer.PPO(args)
# trainer.train()