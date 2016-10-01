import random
import pprint
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import defaultdict
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.Q = {}
        self.penalty = 0
        self.success = 0
        self.count = 0
        self.iteration = 0
        self.totalReward = 0
        self.last_10_penalty = 0
        self.last_10_success = 0
        self.last_10_count = 0
        self.last_10_Reward = 0
        self.Q = defaultdict(lambda: 0.0, self.Q)
        self.valid_actions = ['left','right','forward',None]

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.iteration += 1
        
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        state = []
        for i in inputs.values():
            state.append(i)
        state.append(self.next_waypoint)
        state.append(deadline)
        state = tuple(state)

        # TODO: Select action according to your policy
        maximum = -100
        for i in self.valid_actions:
            if self.Q[(state,i)] > maximum:
                maximum = self.Q[(state,i)]
                action = i

        epsilon = 1/(self.iteration)
        random_action = random.choice(self.valid_actions)
        action = np.random.choice(np.asarray([action,random_action]), p = [1-epsilon,epsilon])
        # Execute action and get reward
        reward = self.env.act(self, action)

        #Calculate Total success rate, total reward and penalty ratio
        self.totalReward += reward
        if reward > 11:
            self.success += 1
        if reward < 0:
            self.penalty += 1
        
        self.count += 1

        #Calculate Last 10 success rate, total reward and penalty ratio
        if self.iteration > 90:
            self.last_10_Reward += reward
            if reward > 11:
                self.last_10_success += 1
            if reward < 0:
                self.last_10_penalty += 1
        
            self.last_10_count += 1
        
        
        next_inputs = self.env.sense(self)
        
        next_state = []
        for i in next_inputs.values():
            next_state.append(i)
        next_state.append(self.next_waypoint)
        next_state.append(deadline)
        next_state = tuple(next_state)

        # TODO: Learn policy based on state, action, reward
        alpha = 0.8
        gamma = 0.4
        utility = -1000000
        for i in self.valid_actions:
            utility = max(utility, self.Q[(next_state,i)])
        self.Q[(state, action)] = (1 - alpha)*self.Q[(state, action)] + (alpha)*(reward + gamma*(utility))
        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

        #Print success rate,total reward and penalty ratio for total iteration and last 10 iterations
        if self.iteration == 100 and (reward > 11 or deadline == 0):
            print "\n\nTotal Success Rate:", self.success
            print "Total Reward:",self.totalReward
            print "Total Penalty Ratio:", float(self.penalty)/float(self.count)
            print "Last 10 success rate:", self.last_10_success
            print "Last 10 total Reward:", self.last_10_Reward
            print "Last 10 Penalty Ratio:", float(self.last_10_penalty)/float(self.last_10_count) 


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0005, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
