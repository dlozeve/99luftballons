# This file should contain the pathfinding algorithm
# It takes as input a matrix *dayGrid* which is the output of the
# prediction model for the given day
# it is for shape 18*548*421, ie hour*x*y
# The start and goal coordinates are (x,y) tuples
# Its output should be a path formatted as a list of tuples (x,y)


def stupidAlgo(start, goal, dayGrid):
    pos = start
    path = [pos]
    step = 0
    while pos != goal:
        # This part is quite useless here and just to illustrate
        # getting the current score from the matrix
        hour = (step*2)//30
        pos_score = dayGrid[hour, pos[0], pos[1]]
        if pos_score > 0.9:
            print("Almost certain crash")
            return path

        if pos[0] < goal[0]:
            pos = (pos[0] + 1, pos[1])
        elif pos[0] > goal[0]:
            pos = (pos[0]-1, pos[1])
        elif pos[1] < goal[1]:
            pos = (pos[0], pos[1]+1)
        elif pos[1] > goal[1]:
            pos = (pos[0], pos[1]-1)
        else:
            print("This should not happen")
        path.append(pos)
        step += 1
    return path
