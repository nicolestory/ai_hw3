import gym
import random
import sys
import time
from matplotlib import pyplot


def main(number_of_runs_per_algorithm, gui=False):
    env = gym.make('MountainCar-v0')

    algorithms = set_up_algorithm_objects(env.action_space, env.observation_space, show_the_gui=gui)

    episode_counts = {}

    for algorithm in algorithms:
        episode_counts[algorithm.algorithm_name] = []
        for run_num in range(number_of_runs_per_algorithm):
            print('Starting Run {} of {} algorithm'.format(run_num, algorithm.algorithm_name))
            episode_count = algorithm.run(env)
            episode_counts[algorithm.algorithm_name].append(episode_count)
            algorithm.reset()

    print(episode_counts)

    '''
    for algorithm in algorithms:
        reached_the_goal = False
        episode_count = 0
        while not reached_the_goal:
            print('Starting episode {}'.format(episode_count))
            observation = env.reset()
            #print('Observation: {}'.format(observation))

            positions = [observation[0]]
            velocities = [observation[1]]
            times = [-1]
            actions = []
            t = 0

            while not reached_the_goal:
                #env.render()
                action = algorithm.choose_next_action(observation)
                next_observation, reward, done, info = env.step(action)

                positions.append(next_observation[0])
                velocities.append(next_observation[1])
                times.append(t)
                actions.append(action)

                algorithm.learn(observation, next_observation, action, reward)

                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    #print('Final observation: {}'.format(next_observation))

                    #print('Actions: {}'.format(actions))
                    print('Max position: {}'.format(max(positions)))
                    print('Min position: {}'.format(min(positions)))
                    print('Max velocity: {}'.format(max(velocities)))
                    print('Min velocity: {}'.format(min(velocities)))

                    if 'TimeLimit.truncated' in info:
                        print('Time out\n')
                    else:
                        # If done and not TimeLimit.truncated, then we've reached the goal.
                        reached_the_goal = True
                        print('Reached the goal in {} episodes, and {} extra timesteps'.format(episode_count, t))
                    break

                observation = next_observation
                t += 1
                #time.sleep(0.5)
            #pyplot.plot(times, positions)
            #pyplot.plot(times, velocities)
            #pyplot.show()
            #pyplot.clf()

            episode_count += 1

    '''

    env.close()


def set_up_algorithm_objects(actions, states, number_of_states=8, show_the_gui=False):
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

    for position_number in range(number_of_states):
        for velocity_number in range(number_of_states):
            position = states.low[0] + step_size[0] * position_number
            velocity = states.low[1] + step_size[1] * velocity_number
            state = (position, velocity)
            discrete_states.append(state)

    algorithm_objects.append(QLearning(discrete_states, actions, show_the_gui))
    algorithm_objects.append(SARSA(discrete_states, actions, show_the_gui))
    algorithm_objects.append(ExpectedSARSA(discrete_states, actions, show_the_gui))

    # TODO: Delete later.
    #algorithm_objects.append(NicoleLearningHowGymWorks())

    return algorithm_objects


class QLearning:
    algorithm_name = 'QLearning'

    def __init__(self, states, actions, show_the_gui=False):
        self.states = states
        self.actions = actions
        self.show_gui = show_the_gui
        self.learning_rate = 0.0
        self.discount = 0.0
        self.epsilon = 0.0
        self.q_table = {}

        self.reset()

    def reset(self):
        self.learning_rate = 0.8
        self.discount = 0.95
        self.epsilon = 0.05
        self.q_table = {}
        for state in self.states:
            for action in self.actions:
                self.q_table[state, action] = 0

    def choose_next_action(self, observation):
        # Choose the state-action pairs from the q-table
        state = self.observation_to_state(observation)
        viable_state_actions = [state_action for state_action in self.q_table.keys() if state_action[0] == state]

        # Decide if we'll return a random action or the one with the best Q-value
        if random.random() < self.epsilon:
            random_state_action = random.choice(viable_state_actions)
            return random_state_action[1]

        # Get the Q-values for each state-action
        q_values = [self.q_table[state_action] for state_action in viable_state_actions]
        biggest_q_value = max(q_values)
        if q_values.count(biggest_q_value) > 1:
            # If multiple Q-values are tied, return a random action from the possible actions
            highest_state_actions = [viable_state_actions[index][1] for index in range(len(viable_state_actions)) if q_values[index] == biggest_q_value]
            return random.choice(highest_state_actions)

        # Otherwise, just choose the largest Q-value
        chosen_index = q_values.index(max(q_values))
        action = viable_state_actions[chosen_index][1]
        return action

    def learn(self, last_state, next_state, action, reward):
        last_state = self.observation_to_state(last_state)
        next_state = self.observation_to_state(next_state)
        old_q_value = self.q_table[last_state, action]
        learned_value = reward + self.discount * max([self.q_table[next_state, action] for action in self.actions])
        updated_q_value = old_q_value + self.learning_rate * (learned_value - old_q_value)
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

    def run(self, env):
        reached_the_goal = False
        episode_count = 0
        while not reached_the_goal:
            observation = env.reset()
            #print('Observation: {}'.format(observation))

            positions = [observation[0]]
            velocities = [observation[1]]
            times = [-1]
            actions = []
            t = 0

            while not reached_the_goal:
                if self.show_gui:
                    env.render()
                action = self.choose_next_action(observation)
                next_observation, reward, done, info = env.step(action)

                positions.append(next_observation[0])
                velocities.append(next_observation[1])
                times.append(t)
                actions.append(action)

                self.learn(observation, next_observation, action, reward)

                if done:
                    #print("Episode finished after {} timesteps".format(t+1))
                    #print('Final observation: {}'.format(next_observation))

                    #print('Actions: {}'.format(actions))
                    #print('Max position: {}'.format(max(positions)))
                    #print('Min position: {}'.format(min(positions)))
                    #print('Max velocity: {}'.format(max(velocities)))
                    #print('Min velocity: {}'.format(min(velocities)))

                    if 'TimeLimit.truncated' in info:
                        print('Episode {}: Time out'.format(episode_count))
                    else:
                        # If done and not TimeLimit.truncated, then we've reached the goal.
                        reached_the_goal = True
                        print('Reached the goal in {} episodes, and {} extra timesteps\n'.format(episode_count, t))
                    break

                observation = next_observation
                t += 1
                #time.sleep(0.5)
            #pyplot.plot(times, positions)
            #pyplot.plot(times, velocities)
            #pyplot.show()
            #pyplot.clf()

            episode_count += 1

        return episode_count


class SARSA(QLearning):
    algorithm_name = 'SARSA'

    def learn(self, last_state, action, reward, next_state, next_action):
        last_state = self.observation_to_state(last_state)
        next_state = self.observation_to_state(next_state)
        old_q_value = self.q_table[last_state, action]
        learned_value = reward + self.discount * self.q_table[next_state, next_action]
        updated_q_value = old_q_value + self.learning_rate * (learned_value - old_q_value)
        self.q_table[last_state, action] = updated_q_value

    def run(self, env):
        reached_the_goal = False
        episode_count = 0
        while not reached_the_goal:
            print('Starting episode {}'.format(episode_count))
            observation = env.reset()
            action = self.choose_next_action(observation)

            positions = [observation[0]]
            velocities = [observation[1]]
            times = [-1]
            actions = []
            t = 0

            while not reached_the_goal:
                if self.show_gui:
                    env.render()
                next_observation, reward, done, info = env.step(action)

                positions.append(next_observation[0])
                velocities.append(next_observation[1])
                times.append(t)
                actions.append(action)

                next_action = self.choose_next_action(observation)

                self.learn(observation, action, reward, next_observation, next_action)

                if done:
                    #print("Episode finished after {} timesteps".format(t+1))
                    #print('Final observation: {}'.format(next_observation))

                    #print('Actions: {}'.format(actions))
                    #print('Max position: {}'.format(max(positions)))
                    #print('Min position: {}'.format(min(positions)))
                    #print('Max velocity: {}'.format(max(velocities)))
                    #print('Min velocity: {}'.format(min(velocities)))

                    if 'TimeLimit.truncated' in info:
                        print('Episode {}: Time out'.format(episode_count))
                    else:
                        # If done and not TimeLimit.truncated, then we've reached the goal.
                        reached_the_goal = True
                        print('Reached the goal in {} episodes, and {} extra timesteps\n'.format(episode_count, t))
                    break

                observation = next_observation
                action = next_action
                t += 1
                #time.sleep(0.5)
            #pyplot.plot(times, positions)
            #pyplot.plot(times, velocities)
            #pyplot.show()
            #pyplot.clf()

            episode_count += 1

        return episode_count


class ExpectedSARSA(QLearning):
    algorithm_name = 'Expected SARSA'

    def learn(self, last_state, next_state, action, reward):
        last_state = self.observation_to_state(last_state)
        next_state = self.observation_to_state(next_state)
        old_q_value = self.q_table[last_state, action]
        learned_value = reward + self.discount * sum([self.pi(next_state, action) * self.q_table[next_state, action] for action in self.actions])
        updated_q_value = old_q_value + self.learning_rate * (learned_value - old_q_value)
        self.q_table[last_state, action] = updated_q_value

    def pi(self, state, action):
        # First, get the state-action pairs that match this action
        state_action_pairs = [state_action for state_action in self.q_table.keys() if state_action[0] == state]

        # Get the Q-values for each state-action
        q_values = [self.q_table[state_action] for state_action in state_action_pairs]
        biggest_q_value = max(q_values)
        #if q_values.count(biggest_q_value) > 1:

        # If the given action has the biggest q value, then it (and others with the same q-value)
        # will be chosen with a (1-epsilon) chance.
        if self.q_table[state, action] == biggest_q_value:
            return (1 - self.epsilon) / q_values.count(biggest_q_value)

        # Otherwise, there's an (epsilon) chance of selecting an action with a lower q-value.
        return self.epsilon / (len(q_values) - len(q_values.count(biggest_q_value)))


class NicoleLearningHowGymWorks(QLearning):
    algorithm_name = 'Nicole\'s Hard-Coded Not-Actual-Learning algorithm'

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
    if '--show-gui' in sys.argv:
        show_gui = True
        sys.argv.remove('--show-gui')
    else:
        show_gui = False

    if len(sys.argv) >= 2:
        number_of_runs = int(sys.argv[1])
    else:
        number_of_runs = 1

    # Run the algorithms and make plots
    main(number_of_runs, show_gui)
