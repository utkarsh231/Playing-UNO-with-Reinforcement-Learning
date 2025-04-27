import sys
import threading
import numpy as np
from agent import UnoAgent
from environment import UnoEnvironment
import wandb

# Hyperparameters
PLAYER_COUNT = 4
COLLECTOR_THREADS = 2
INITIAL_EPSILON = 1.0
EPSILON_DECAY = 0.999999
MIN_EPSILON = 0.01

def run(agent):
    env = UnoEnvironment(PLAYER_COUNT)
    epsilon = INITIAL_EPSILON
    counter = 0

    while True:
        done = False
        state = None
        rewards = []

        while not done:
            if state is None or np.random.rand() < epsilon or not agent.initialized:
                action = np.random.randint(env.action_count())
            else:
                action = agent.act(state)

            new_state, reward, done, _ = env.step(action)
            rewards.append(reward)

            if state is not None:
                agent.update_replay_memory((state, action, reward, new_state, done))
                if len(agent.replay_memory) >= agent.BATCH_SIZE:
                    agent.train(counter)
                    counter += 1

            state = new_state

            if agent.initialized:
                epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        # Log episode-level metrics
        cumulative_reward = np.sum(rewards)
        mean_reward = np.mean(rewards)
        episode_length = len(rewards)

        agent.logger.scalar('cumulative_reward', cumulative_reward)
        agent.logger.scalar('mean_reward', mean_reward)
        agent.logger.scalar('game_length', episode_length)
        agent.logger.scalar('epsilon_runtime', epsilon)
        agent.logger.flush()

        wandb.log({
            "cumulative_reward": cumulative_reward,
            "mean_reward": mean_reward,
            "game_length": episode_length,
            "epsilon_runtime": epsilon
        })

        env.reset()


if __name__ == '__main__':
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    dummy_env = UnoEnvironment(1)
    agent = UnoAgent(dummy_env.state_size(), dummy_env.action_count(), model_path)
    del dummy_env

    for i in range(COLLECTOR_THREADS):
        thread = threading.Thread(target=run, args=(agent,), daemon=True, name=f"Thread-{i+1}")
        thread.start()

    # Keep main thread alive
    while True:
        pass
