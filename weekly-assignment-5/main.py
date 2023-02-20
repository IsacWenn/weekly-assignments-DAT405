import numpy

def main():
    reward = numpy.asarray([[0, 0, 0], [0, 10, 0], [0, 0, 0]]).astype(float)
    states = numpy.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 0]]).astype(float)
    epsilon = 0.01
    gamma = 0.9
    print(reward)
    print(states)


# V_k[s] = max_a Σ_s' p(s′|s,a)*(r(a,s,s′) + γ*V_k−1[s′])

def actions(x_coord: int, y_coord: int, x_dim: int, y_dim: int):
    action_list = []
    if x_coord > x_dim or x_coord < 0 or y_coord > y_dim or y_coord < 0:
        return action_list
    if x_coord + 1 <= x_dim:
        action_list.append((x_coord + 1, y_coord))
    if y_coord + 1 <= y_dim:
        action_list.append((x_coord, y_coord + 1))
    if x_coord - 1 >= 0:
        action_list.append((x_coord - 1, y_coord))
    if y_coord - 1 >= 0:
        action_list.append((x_coord, y_coord - 1))
    return action_list


def reward_calc(reward, state, cur_coord: tuple, next_coord: tuple, p_action, p_no_action, gamma: float):
    cur_x, cur_y = cur_coord
    next_x, next_y = next_coord
    action_reward = p_action*(reward[next_x][next_y] + gamma * state[next_x][next_y])
    + p_no_action*(reward[cur_x][cur_y] + gamma * state[cur_x][cur_y])
    return action_reward


def value_iteration(states, next_state, reward, gamma: float, epsilon: float):
    p_action = 0.8
    p_no_action = 1-p_action
    V_k = numpy.copy(states)
    diff = 1
    while diff > epsilon:
        for s in states:
            pass





if __name__ == "__main__":
    main()
