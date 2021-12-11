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
### TODO: import the DQN model
# import DQN_model

SECONDS_PER_EPISODE = 20
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

                if agent_vel.x == 0 and agent_vel.y == 0 and agent_vel.z == 0:
                    time.sleep(10)
                    print('destroying actors')
                    self.destroy()
                    print('done.')

    def step(self, action):
        
        ## TODO: 1. customize the action of steer and throttle
        if action == 0:
            self.vehicle_agent.apply_control(carla.VehicleControl(throttle=0.5, steer= 0))
        elif action == 1:
            self.vehicle_agent.apply_control(carla.VehicleControl(throttle=0.3, steer=-1*self.STEER_AMT))
        elif action == 2:
            self.vehicle_agent.apply_control(carla.VehicleControl(throttle=0.3, steer=1*self.STEER_AMT))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        ## TODO: 2. customize the reward function

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        
        agent_loc = self.vehicle_agent.get_location()
        agent_vel = self.vehicle_agent.get_velocity()
        agent_acc = self.vehicle_agent.get_acceleration()
        state = [
            agent_loc.x, agent_loc.y, agent_loc.z,
            agent_vel.x, agent_vel.y, agent_vel.z,
            agent_acc.x, agent_acc.y, agent_acc.z
        ]
        return state, reward, done


if __name__ == '__main__':
    env = CarEnv()
    env.reset()
    env.apply_brake()

    