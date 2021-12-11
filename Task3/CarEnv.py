import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
import random
import time
import numpy as np
import cv2
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import gym
import matplotlib.pyplot as plt
import copy


BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100

# env = gym.make("CartPole-v0")
# env = env.unwrapped
NUM_ACTIONS = 3
NUM_STATES = 9
#ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 50)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(50,30)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(30,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob

class DQN():
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.randn() <= EPISILO:# greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]
            #if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else: # random policy
            action = np.random.randint(0,NUM_ACTIONS)
            action = action
            #if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        return action


    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):

        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        #sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2])
        batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:])

        #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


SECONDS_PER_EPISODE = 40

class CarEnv:
    STEER_AMT = 1.0

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(200.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.bp_obstacle = self.blueprint_library.filter('cybertruck')[0]
        self.bp_agent = self.blueprint_library.filter('model3')[0]

        self.actor_list = []

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        self.spawn_point_obstacle = carla.Transform(carla.Location(x=150.000000, y=143.600000, z=0.300000), carla.Rotation(pitch=0.000000, yaw=0.000000, roll=0.000000))
        self.vehicle_obstacle = self.world.spawn_actor(self.bp_obstacle, self.spawn_point_obstacle)
        self.vehicle_obstacle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))
        self.actor_list.append(self.vehicle_obstacle)


        self.spawn_point_agent = carla.Transform(carla.Location(x=50.000000, y=143.400000, z=0.300000), carla.Rotation(pitch=0.000000, yaw=0.000000, roll=0.000000))
        self.vehicle_agent = self.world.spawn_actor(self.bp_agent, self.spawn_point_agent)
        self.vehicle_obstacle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))
        self.actor_list.append(self.vehicle_agent)

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle_agent)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))


        self.episode_start = time.time()
        self.vehicle_agent.apply_control(carla.VehicleControl(throttle=0.5, brake=0.0))

        agent_loc = self.vehicle_agent.get_location()
        agent_vel = self.vehicle_agent.get_velocity()
        agent_acc = self.vehicle_agent.get_acceleration()
        state = [
            agent_loc.x, agent_loc.y, agent_loc.z,
            agent_vel.x, agent_vel.y, agent_vel.z,
            agent_acc.x, agent_acc.y, agent_acc.z
        ]
        return state

    def destroy(self):
        for actor in self.actor_list:
            actor.destroy()

    def collision_data(self, event):
        self.collision_hist.append(event)

    ## a demo of applying a brake when the agent vehicle approaches the obstacle vehicle
    def apply_brake(self):
        throttle_agent = 0.5
        steer_agent = 0
        brake_agent = 0

        while True:
            # Retrieve a snapshot of the world at current frame.
            world_snapshot = self.world.get_snapshot()
            frame = world_snapshot.frame
            timestamp = world_snapshot.timestamp.elapsed_seconds # Get the time reference

            self.vehicle_agent.apply_control(carla.VehicleControl(throttle=throttle_agent, steer=steer_agent, brake=brake_agent))

            # Get thee location of the agent vehicle and the obstacle vehiclee
            agent_loc = self.vehicle_agent.get_location()
            agent_vel = self.vehicle_agent.get_velocity()
            # agent_ang_vel = vehicle_agent.get_angular_velocity()
            agent_acc = self.vehicle_agent.get_acceleration()
            obstacle_loc = self.vehicle_obstacle.get_location()

            # calculate distance
            # distance = agent_loc.distance(obstacle_loc)
            distance = abs(agent_loc.x - obstacle_loc.x)

            print('Frame:{%s}, Timestamp:{%s}, Agent Location:{%s}, Agent velocity:{%s}, Agent accelaration:{%s}' %(frame,timestamp,agent_loc,agent_vel,agent_acc))

            ## An example to apply a brake if the distance are too close
            if distance < 15:
                throttle_agent = 0
                brake_agent = 1

                if distance < 2:
                    print("Collision")
                    self.destroy()

                    break

    def step(self, action):

        ### TODO: 1. customize the action of steer and throttle
        if action == 0:
            self.vehicle_agent.apply_control(carla.VehicleControl(throttle=0.6, steer= 0))
        elif action == 1:
            self.vehicle_agent.apply_control(carla.VehicleControl(throttle=0.3, steer=-1*self.STEER_AMT))
        elif action == 2:
            self.vehicle_agent.apply_control(carla.VehicleControl(throttle=0.3, steer=1*self.STEER_AMT))

        v = self.vehicle_agent.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        ### TODO: 2. customize the reward function
        done = False
        reward = 1

        speed_index = (kmh - 50)
        loc1 = self.vehicle_agent.get_location()
        loc2 = self.vehicle_obstacle.get_location()
        away_index = int((loc1.x - loc2.x) / (abs(loc1.y - loc2.y) + 1))

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        else:
            reward = 0.4 * speed_index + 0.6 * away_index

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        ### TODO: 3. declare the state:{location, velocity, acceleration}
        agent_loc = self.vehicle_agent.get_location()
        agent_vel = self.vehicle_agent.get_velocity()
        agent_acc = self.vehicle_agent.get_acceleration()
        state = [
            agent_loc.x, agent_loc.y, agent_loc.z,
            agent_vel.x, agent_vel.y, agent_vel.z,
            agent_acc.x, agent_acc.y, agent_acc.z
        ]
        # return state, reward, done
        return state, reward, done

def main():
    env = CarEnv()
    dqn = DQN()
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

# if __name__ == '__main__':
#     env = CarEnv()
#     env.reset()
#     env.destroy()
#     env.destroy()
#     env.apply_brake()
