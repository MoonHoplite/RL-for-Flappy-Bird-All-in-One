import model
import trainer



# DDQN training
chk_name = r"saved_models\flappybird_ddqn.pth"
args = trainer.TrainingArguments(load_checkpoint = chk_name, observe=0, explore=1, batch_size=256)
trainer = trainer.DQN(args)
trainer.train()

# VPG training
# chk_name = (r"saved_models\flappybird_vpg_actor_100.pth", r"saved_models\flappybird_vpg_critic_100.pth")
# args = trainer.TrainingArguments(batch_size=1024, save_interval=100, load_checkpoint = chk_name)
# trainer = trainer.VPG(args)
# trainer.train()


# PPO training
# chk_name = (r"saved_models\flappybird_ppo_actor_800.pth", r"saved_models\flappybird_ppo_critic_800.pth")
# args = trainer.TrainingArguments(batch_size=1024, save_interval=100, load_checkpoint = chk_name)
# trainer = trainer.PPO(args)
# trainer.train()