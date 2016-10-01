import random
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
        self.state = (None,None,None,None,None)
        self.Q = {}
        self.Q = defaultdict(lambda: 0.0, self.Q)
        self.valid_actions = [None,'left','right','forward']

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.totalReward = 0
        self.state = (None,None,None,None,None)

    def update(self, t,gamma = 0.1,epsilon = 0.3,alpha = 0.8):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        laststate = self.state
        state = []
        for i in inputs.values():
            state.append(i)
        state.append(self.next_waypoint)
        self.state = state
        self.state  = tuple(self.state)

        # TODO: Select action according to your policy
        maximum = -100
        for i in self.valid_actions:
            if self.Q[(laststate,i)] > maximum:
                maximum = self.Q[(laststate,i)]
                action = i

        random_action = random.choice(self.valid_actions)
        action = np.random.choice(np.asarray([action,random_action]), p = [1-epsilon,epsilon])
        # Execute action and get reward
        reward = self.env.act(self, action)
        # TODO: Learn policy based on state, action, reward
        utility = -1000000
        for i in self.valid_actions:
            utility = max(utility, self.Q[(self.state,i)])
        self.Q[(laststate, action)] = (1-alpha)*self.Q[(laststate, action)] + alpha*(reward + gamma*(utility))
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


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
