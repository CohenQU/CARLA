import matplotlib.pyplot as plt
import copy

from carEnv2 import CarEnv
from carEnv1 import CarEnv
from dqn import DQN

MEMORY_CAPACITY = 2000

def main():
    dqn = DQN()
    env = CarEnv()
    episodes = 400
    print("Collecting Experience....")
    reward_list = []
    plt.ion()
    fig, ax = plt.subplots()
    for i in range(episodes):
        state = env.reset()
        ep_reward = 0
        while True:
            # env.render()
            action = dqn.choose_action(state)
            next_state, reward , done= env.step(action)
            # x, x_dot, theta, theta_dot = next_state
            # reward = reward_func(env, x, x_dot, theta, theta_dot)

            dqn.store_transition(state, action, reward, next_state)
            ep_reward += reward

            if dqn.memory_counter >= MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
            if done:
                break
            state = next_state
        env.destroy()
        r = copy.copy(reward)
        reward_list.append(r)
        ax.set_xlim(0,300)
        #ax.cla()
        ax.plot(reward_list, 'g-', label='total_loss')
        plt.pause(0.001)


if __name__ == '__main__':
    main()
