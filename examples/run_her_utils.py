def check_goal_reached(env, state, info, goal):
    return info["x_position"] == goal[0]


def get_goal(state, info, goal):
    return [info["x_position"], 0]
