import gym
import random
import sys
import time
from matplotlib import pyplot


def main(number_of_episodes, timesteps_per_episode):
    env = gym.make('MountainCar-v0')

    algorithms = set_up_algorithm_objects(env.action_space, env.observation_space)

    for algorithm in algorithms:
        for i_episode in range(number_of_episodes):
            print('Starting episode {}'.format(i_episode))
            observation = env.reset()
            print('Observation: {}'.format(observation))

            positions = [observation[0]]
            velocities = [observation[1]]
            times = [-1]

            for t in range(timesteps_per_episode):
                print('t = {}'.format(t))
                env.render()
                action = algorithm.choose_next_action(observation)
                next_observation, reward, done, info = env.step(action)

                positions.append(next_observation[0])
                velocities.append(next_observation[1])
                times.append(t)

                algorithm.learn(observation, next_observation, action, reward)

                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    print('Done: {}'.format(done))
                    break

                observation = next_observation
                #time.sleep(0.5)
        #pyplot.plot(times, positions)
        pyplot.plot(times, velocities)
        pyplot.show()

        print('Max: {}'.format(max([state[1] for state in algorithm.states])))
    env.close()


def set_up_algorithm_objects(actions, states, number_of_states=50):
    algorithm_objects = []

    # Translate the actions into a list
    print(actions)
    actions = [action for action in range(actions.n)]
    print(actions)
    print(type(actions[0]))

    # Make the states discrete instead of continuous
    states_difference = states.high - states.low
    step_size = states_difference / (number_of_states - 1)
    discrete_states = []
    for state_number in range(number_of_states):
        state = states.low + (step_size * state_number)
        state = [float(element) for element in state]
        state = tuple(state)
        discrete_states.append(state)
    
    algorithm_objects.append(QLearning(discrete_states, actions))

    # TODO: Delete later.
    #algorithm_objects.append(NicoleLearningHowGymWorks())

    return algorithm_objects


class QLearning:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        self.learning_rate = 0.5
        self.discount = 0.5
        self.q_table = {}
        for state in states:
            for action in actions:
                self.q_table[state, action] = 0

    def choose_next_action(self, observation):
        # Choose the state-action pairs from the q-table
        state = self.observation_to_state(observation)
        viable_state_actions = [state_action for state_action in self.q_table.keys() if state_action[0] == state]

        # Get the Q-values for each state-action
        q_values = [self.q_table[state_action] for state_action in viable_state_actions]
        biggest_q_value = max(q_values)
        if q_values.count(biggest_q_value) < 1:
            # If multiple Q-values are tied, return a random action from the possible actions
            highest_state_actions = [viable_state_actions[index] for index in range(len(viable_state_actions)) if q_values[index] == biggest_q_value]
            return random.choice(highest_state_actions)

        # Otherwise, just choose the largest Q-value
        chosen_index = q_values.index(max(q_values))
        action = viable_state_actions[chosen_index][1]
        print('State: {}'.format(state))
        print('Action: {}'.format(action))
        return action

    def learn(self, last_state, next_state, action, reward):
        last_state = self.observation_to_state(last_state)
        next_state = self.observation_to_state(next_state)
        old_q_value = self.q_table[last_state, action]
        learned_value = reward + self.discount * max([self.q_table[next_state, action] for action in self.actions])
        updated_q_value = old_q_value + self.learning_rate * (learned_value - old_q_value)
        print('Updated q value: {}'.format(updated_q_value))
        print('  other q values: {}'.format([self.q_table[last_state, action] for action in self.actions]))
        self.q_table[last_state, action] = updated_q_value

    def observation_to_state(self, observation):
        closest_state = None
        closest_distances = None
        for state in self.states:
            distances = [abs(state[index] - observation[index]) for index in range(len(state))]
            if closest_state is None:
                # This is the first time through the loop. Set the closest state & distance
                closest_state = state
                closest_distances = distances

            distance_delta = [closest_distances[i] - distances[i] for i in range(len(state))]
            if min(distance_delta) >= 0:
                # This state is closer than the previous closest state
                closest_state = state
                closest_distances = distances

        return closest_state


class NicoleLearningHowGymWorks:
    # This is my (non-RL) toy example of how to get the car to the top of the hill.
    def __init__(self, states=None, actions=None):
        # Pretend that we start off moving left
        self.is_moving_left = True

    def choose_next_action(self, observation):
        if self.is_moving_left:
            action = 0  # Move to the left
        else:
            action = 2  # Move to the right

        return action

    def learn(self, last_state, next_state, action, reward):
        velocity = next_state[1]  # Negative is to the left, positive to the right
        if velocity < 0.0:
            self.is_moving_left = True
        else:
            self.is_moving_left = False


if __name__ == '__main__':
    # Parse the command line args
    if len(sys.argv) >= 2:
        number_of_episodes = int(sys.argv[1])
    else:
        number_of_episodes = 1

    if len(sys.argv) >= 3:
        timesteps_per_episode = int(sys.argv[2])
    else:
        timesteps_per_episode = 201

    main(number_of_episodes, timesteps_per_episode)
