import pygame
import sys
import random
from game import wrapped_flappy_bird

# 初始化游戏状态
game_state = wrapped_flappy_bird.GameState()
total_reward = 0

while True:
    # 默认不做任何操作
    action = 0

    # 捕获键盘事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            action = 1  # 按下空格键时跳跃

    # 更新游戏状态
    image_data, reward, terminal = game_state.frame_step(action)
    total_reward += reward

    # 如果游戏结束，重新初始化游戏状态
    if terminal:
        print(f"Game Over! Total reward: {total_reward}")
        total_reward = 0
        game_state = wrapped_flappy_bird.GameState()