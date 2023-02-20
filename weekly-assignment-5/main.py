import numpy


def main():
    reward = numpy.array([[0, 0, 0, 0, 10, 0, 0, 0, 0]])
    epsilon = 0.01
    gamma = 0.9


# V_k[s] = max_a Σ_s' p(s′|s,a)*(r(a,s,s′) + γ*V_k−1[s′])


def actions(x_coord: float, y_coord: float, x_dim: float, y_dim: float):
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


def value_iteration(array: list, reward: list, gamma: float, epsilon: float):
    V_k = numpy.zeros(len(array))
    diff = 1
    while diff > epsilon:
        for state in array:
            pass


if __name__ == "__main__":
    main()
