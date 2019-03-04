import mlrose
import numpy as np
import matplotlib.pyplot as plt
import time
import math

problem_name = "ModMax"
file = open(problem_name + ".txt", 'w')

def filewrite_array(title, array):
    file.write(title + "\n")
    file.write(' '.join(str(e) for e in array) + "\n")
    file.write('[' + ', '.join(str(e) for e in array) + "]\n")
    file.write("\n")

def plot(RHC, SA, GA, MM, time_RHC, time_SA, time_GA, time_MM, array):
    plt.plot(RHC, color='red', alpha=0.8, label='RHC')
    plt.plot(SA, color='blue', alpha=0.8, label='SA')
    plt.plot(GA, color='green', alpha=0.8, label='GA')
    plt.plot(MM, color='yellow', alpha=0.8, label='Mimic')
    
    plt.title("Fitness over Iterations", fontsize=14)
    plt.ylabel('Fitness')
    plt.xlabel('Iterations')
    plt.xticks(np.arange(len(array)), array)
    plt.legend(loc='best')
    dwn = plt.gcf()
    plt.savefig(problem_name + "_Fitness")
    plt.show()

    plt.plot(time_RHC, color='red', alpha=0.8, label='RHC')
    plt.plot(time_SA, color='blue', alpha=0.8, label='SA')
    plt.plot(time_GA, color='green', alpha=0.8, label='GA')
    plt.plot(time_MM, color='yellow', alpha=0.8, label='Mimic')
    
    plt.title("Time over Iterations", fontsize=14)
    plt.ylabel('Time')
    plt.xlabel('Iterations')
    plt.xticks(np.arange(len(array)), array)
    plt.legend(loc='best')
    dwn = plt.gcf()
    plt.savefig(problem_name + "_Time")
    plt.show()

def fit(length, fitness):
    problem = mlrose.DiscreteOpt(length = length, fitness_fn = fitness, maximize = True, max_val = 2)

    iterations = [10,50,100,200,400,800,1600,3200]
    RHC, SA, GA, MM = ([],[],[],[])
    time_RHC, time_SA, time_GA, time_MM = ([],[],[],[])

    for iter in iterations:
        print ("max iterations = " + str(iter))
        start_time = time.time()
        best_fitness = 0
        for times in range(10):
          best_state, best_fitness = mlrose.random_hill_climb(problem, max_attempts = 10, max_iters = iter, restarts = 0, init_state = np.random.randint(2, size=(length,)))
          best_fitness = max(best_fitness, best_fitness)
          #print(best_state)
        RHC.append(best_fitness)
        print(best_fitness)
        time_RHC.append((time.time() - start_time)/10)
        
        start_time = time.time()
        best_fitness = 0
        for times in range(10):
          best_state, best_fitness = mlrose.simulated_annealing(problem, schedule = mlrose.GeomDecay(), max_attempts = 10, max_iters = iter, init_state = np.random.randint(2, size=(length,)))
          best_fitness = max(best_fitness, best_fitness)
          #print(best_state)
        SA.append(best_fitness)
        print(best_fitness)
        time_SA.append((time.time() - start_time)/10)

        start_time = time.time()
        best_fitness = 0
        best_state, best_fitness = mlrose.genetic_alg(problem, pop_size = 200, mutation_prob = 0.1, max_attempts = 10, max_iters = iter)
        #print(best_state)
        GA.append(best_fitness)
        print(best_fitness)
        time_GA.append((time.time() - start_time))

        start_time = time.time()
        best_fitness = 0
        best_state, best_fitness = mlrose.mimic(problem, pop_size = 200, keep_pct = 0.2, max_attempts = 10, max_iters = iter)
        #print(best_state)
        MM.append(best_fitness)
        print(best_fitness)
        time_MM.append((time.time() - start_time))
    
    plot(RHC, SA, GA, MM, time_RHC, time_SA, time_GA, time_MM, iterations)
    filewrite_array("iterations:", iterations)
    filewrite_array("Fitness(RHC):", RHC)
    filewrite_array("Fitness(SA):", SA)
    filewrite_array("Fitness(GA):", GA)
    filewrite_array("Fitness(MM):", MM)
    filewrite_array("Fitness(time_RHC):", time_RHC)
    filewrite_array("Fitness(time_SA):", time_SA)
    filewrite_array("Fitness(time_GA):", time_GA)
    filewrite_array("Fitness(time_MM):", time_MM)

state = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0])
def cust_fn(state): 
  arr = np.reshape(state, (-1, 8))
  ans = 1
  for row in arr:
    integer = row.dot(1 << np.arange(row.size)[::-1])
    ans *= (math.pow((integer % 13),2) % 7) + abs(math.sin(integer))
  return ans

fitness = mlrose.CustomFitness(cust_fn)
print(fitness.evaluate(state))
fit(48, fitness)
file.close()