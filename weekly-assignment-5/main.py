import numpy


def main():
    reward = numpy.array([[0, 0, 0, 0, 10, 0, 0, 0, 0]])
    epsilon = 0.01
    gamma = 0.9


# V_k[s] = max_a Σ_s' p(s′|s,a)*(r(a,s,s′) + γ*V_k−1[s′])

def value_iteration(array: list, reward: list, gamma: float, epsilon: float):
    V_k = numpy.zeros(len(array))
    diff = 1
    while diff > epsilon:
        for state in array:
            pass




if __name__ == "__main__":
    main()
