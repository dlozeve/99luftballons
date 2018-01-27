import numpy as np
import matplotlib.pyplot as plt


def getOnePath(algo, city, day, cityData, predMatrix, *args, **kwargs):
    dayGrid = predMatrix[day, :, :, :]

    start = (cityData.iloc[0, 1], cityData.iloc[0, 2])
    goal = (cityData.iloc[city, 1], cityData.iloc[city, 2])

    return start, goal, algo(start, goal, dayGrid, *args, **kwargs)


def toHourMin(step):
    step = step*2
    hour = step//60
    minute = step % 60
    return "%02d:%02d" % (hour, minute)


def formatPath(path, city, day, data='validation'):
    if data == 'validation':
        offset = 1
    elif data == 'test':
        offset = 6
    else:
        raise ValueError('data argument must be "validation" or "test"')

    return np.array([(city, day+offset, toHourMin(t), pos[0], pos[1])
                     for t, pos in enumerate(path)])


def checkPath(path, start, goal, trueDayGrid):
    """Checks with the real data to see if a given path crashes or not"""
    # /!\ The checker assumes that the path is correctly outputted,
    # aka one entry every two minutes. If there is a time gap in the
    # moves, the checker will not know it and its output will be
    # irrelevant. So check that your algo REALLY outputs one position
    # for every two minutes (however no further entries are necesary
    # when the goal has been reached)
    prev = path[0]
    step = 0
    if path[0] != start:
        print('Wrong path start')
        return False
    for pos in path:
        if step > 30*18:
                print('Incorrect: path too long')
                return False
        if step > 0:
            if abs(pos[0]-prev[0]+abs(pos[1]-prev[1])) > 1:
                print("Incorrect: invalid move")
                return False
        if pos[0] < 0 or pos[1] < 0 or pos[0] >= 548 or pos[1] >= 421:
            print("Incorrect: out of grid")
            return False
        hour = (step*2)//60
        if trueDayGrid[hour, pos[0], pos[1]] > 15:
            print('Incorrect: balloon crashed by wind')
            return False
        step += 1
        prev = pos
    if path[-1] != goal:
        print('Incorrect: path does not reach destination')
        return False
    print('Path valid')
    return True


def getSubmission(algo, cityData, predMatrix, nValidationDays, data='validation', trueGrid=None, *args, **kwargs):
    res = []
    if data == 'validation':
        successes = 0
    tries = 0
    for day in range(nValidationDays):
        for city in range(10):
            tries += 1
            start, goal, path = getOnePath(algo, city+1, day,
                                           cityData, predMatrix,
                                           *args, **kwargs)
            if data == 'validation':
                successes += checkPath(path, start, goal,
                                       trueGrid[day, :, :, :])
            res.append(formatPath(path, city, day, data))
    if data == 'validation':
        print('Successes:', successes)
        print('Failures', tries - successes)
    return np.concatenate(res)


def showPath1h(start, goal, path, trueWind, i, figsize=(20, 10),
               figax=None, disp=True):
    zipped = list(zip(*path))
    xPath = list(zipped[0])
    yPath = list(zipped[1])
    if figax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = figax

    ax.plot(xPath[:(30*(i+1))], yPath[:(30*(i+1))], color='green', linewidth=4)
    ax.imshow(trueWind[i, :, :].T)
    ax.plot(goal[0], goal[1], 'ro', markersize=10)
    ax.plot(start[0], start[1], 'bo', markersize=10)

    if disp:
        plt.draw()


def showFullPath(start, goal, path, trueWind):
    fig, axs = plt.subplots(nrows=5, ncols=4, figsize=(30, 24))
    step = 0

    for x in range(5):
        for y in range(4):
            if step < 18:
                if len(path) >= 30*(4*x+y):
                    showPath1h(start, goal, path, trueWind, 4*x+y, None,
                               (fig, axs[x][y]), False)
            step += 1

    plt.show()
