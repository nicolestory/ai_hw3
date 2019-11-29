from matplotlib import pyplot
import sys


def alpha(t):
    if problem_num == 0:
        return 0.5
    elif problem_num == 1:
        return 1 / 8
    elif problem_num == 2:
        return 1
    elif problem_num == 3:
        return 1 / (t + 1)
    elif problem_num == 4:
        return -0.5
    elif problem_num == 5:
        return 1.5
    else:
        return 0


def main():
    signals = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0]

    times = []
    predictions = []
    estimate = 0.0
    print('\nPoints:')

    for time in range(len(signals)):
        times.append(time)
        predictions.append(estimate)
        estimate = estimate + alpha(time) * (signals[time] - estimate)
        print('({}, {})'.format(times[time], predictions[time]))

    error = 0
    for time in range(10, 16):
        error += abs(signals[time] - predictions[time])

    print('\nError in predictions 10-15:')
    print('Total error: {}, avg error: {}'.format(error, error / 6))

    pyplot.title('Signal prediction with step size {}'.format(step_size_string))
    pyplot.plot(times, predictions, marker='o')
    pyplot.plot(times, signals)
    pyplot.grid()
    #pyplot.show()
    pyplot.savefig('problem_3_update_rule_{}.png'.format(problem_num))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        problem_num = int(sys.argv[1])
    else:
        problem_num = 0

    if problem_num == 0:
        step_size_string = '1/2'
    elif problem_num == 1:
        step_size_string = '1/8'
    elif problem_num == 2:
        step_size_string = '1'
    elif problem_num == 3:
        step_size_string = '1/(t+1)'
    elif problem_num == 4:
        step_size_string = '-1/2'
    elif problem_num == 5:
        step_size_string = '1.5'
    else:
        step_size_string = '0.0'
    print('Using step size {}'.format(step_size_string))

    main()
