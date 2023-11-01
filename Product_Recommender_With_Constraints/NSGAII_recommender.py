import random
import numpy as np
from matplotlib import pyplot as plt
from pymoo.util.display.output import Output
from MyMutation_cost import MyMutation
from MyProblem_recommender import MyProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
import pymoo.optimize as optimize
from duplicates import MyDuplicateElimination
from CR import MyCrossover
from initialization import MySampling
print("START")

class MyOutput(Output):

    def __init__(self):
        super().__init__()

    def update(self, algorithm):
        super().update(algorithm)
        print(algorithm.pop.get("X"))

algorithm_name = 'NSGA2'

if algorithm_name == 'NSGA2':

    algorithm = NSGA2(pop_size=100,
                      sampling=MySampling(),
                      crossover=MyCrossover(0.5),
                      mutation=MyMutation(), eliminate_duplicates=MyDuplicateElimination(), save_history=True
                      )
plt.figure(figsize=(7, 5))

problem_1 = MyProblem(376)
res = optimize.minimize(problem_1,
               algorithm,
               ('n_gen', 100),
               seed=1, output=MyOutput(),
               verbose=True)
ip = 1
for ij in res.history:
    c = '#'
    if ip % 10 == 0:
        for i in range(0, 6):
            c = c+random.choice(['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'])

        lists = []

        for k in ij.opt:
            lists.append(k.F.tolist())
        print(lists)

        plt.scatter(np.array(lists)[:, 0], np.array(lists)[:, 1], s=ip, facecolors=c, edgecolors=c, label=str(ip)+'th Iteration')

    ip = ip+1

plt.title("Objective Space")
plt.legend()
plt.show()


