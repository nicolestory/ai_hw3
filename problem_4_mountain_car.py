import gym
import sys


def main(number_of_episodes, timesteps_per_episode):
    env = gym.make('MountainCar-v0')

    print('obs space, high, low')
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)
    print('--------------------')

    algorithms = set_up_algorithm_objects(env.action_space, env.observation_space)

    for algorithm in algorithms:
        for i_episode in range(number_of_episodes):
            print('Starting episode {}'.format(i_episode))
            observation = env.reset()
            print('Observation: {}'.format(observation))
            for t in range(timesteps_per_episode):
                print('t = {}'.format(t))
                env.render()
                action = algorithm.choose_next_action(observation)
                next_observation, reward, done, info = env.step(action)
                algorithm.learn(observation, next_observation, action, reward)

                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    print('Done: {}'.format(done))
                    break

                observation = next_observation
    env.close()


def set_up_algorithm_objects(actions, states, number_of_states=20):
    algorithm_objects = []

    # TODO: Delete later.
    algorithm_objects.append(NicoleLearningHowGymWorks())

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

    return algorithm_objects


class QLearning:
    def __init__(self, states, actions):
        self.q_table = {}
        for state in states:
            for action in actions:
                self.q_table[state, action] = 0

    def choose_next_action(self, observation):
        return 1

    def learn(self, last_state, next_state, action, reward):
        pass


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
    if len(sys.argv) >= 2:
        number_of_episodes = int(sys.argv[1])
    else:
        number_of_episodes = 20

    if len(sys.argv) >= 3:
        timesteps_per_episode = int(sys.argv[2])
    else:
        timesteps_per_episode = 100

    main(number_of_episodes, timesteps_per_episode)
