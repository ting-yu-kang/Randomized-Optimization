import pandas as pd
import numpy as np
import mlrose
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import time
import datetime

# The file will record the training and testing result named with current time
filename = datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')
file = open(filename + '.txt', 'w')

# Read the dataset
url = "https://www.openml.org/data/get_csv/44/dataset_44_spambase.arff"
df = pd.read_csv(url)

def filewrite_array(title, array):
    file.write(title + "\n")
    file.write(' '.join(str(e) for e in array) + "\n")
    file.write('[' + ', '.join(str(e) for e in array) + "]\n")
    file.write("\n")

def sample_preprocess(frac_now):
    # Sampling
    print('data percentage: ' + str(frac_now))
    letterdata = df.sample(frac=frac_now)
    print('data shape: ' + str(letterdata.shape))

    # Preprocess
    X = letterdata.drop('class', axis=1).astype(float)
    y = letterdata['class']
    return X, y

def predict(gradient_descent, cv, iter, restart, X_train, X_test, y_train, y_test):
    y_train_accuracy = 0
    y_test_accuracy = 0
    for i in range(restart):
        # Standardlization
        scaler = MinMaxScaler()  
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)  
        X_test = scaler.transform(X_test)

        # Train Model
        if algorithm == 'gradient_descent':
            nn_model = mlrose.NeuralNetwork(hidden_nodes = [10,10], activation = 'relu',
                                        algorithm = algorithm, max_iters = iter,
                                        bias = True, is_classifier = True, learning_rate = 0.0001,
                                        early_stopping = True, max_attempts = 10)#, clip_max = 5)
        if algorithm == 'random_hill_climb':
            nn_model = mlrose.NeuralNetwork(hidden_nodes = [10,10], activation = 'relu',
                                        algorithm = algorithm, max_iters = iter,
                                        bias = True, is_classifier = True,
                                        early_stopping = True, max_attempts = 100)#, clip_max = 5)
        if algorithm == 'simulated_annealing':
            schedule = mlrose.GeomDecay(init_temp=100, decay=0.5, min_temp=0.1)
            nn_model = mlrose.NeuralNetwork(hidden_nodes = [10,10], activation = 'relu',
                                        algorithm = algorithm, max_iters = iter,
                                        bias = True, is_classifier = True,
                                        early_stopping = True, max_attempts = 100)#, schedule = schedule)#, clip_max = 5)
        if algorithm == 'genetic_alg':
            nn_model = mlrose.NeuralNetwork(hidden_nodes = [10,10], activation = 'relu',
                                        algorithm = algorithm, max_iters = iter,
                                        bias = True, is_classifier = True,
                                        pop_size = 200, mutation_prob = 0.1,
                                        early_stopping = True, max_attempts = 1000)#, clip_max = 5)
        nn_model_cv = nn_model

        nn_model.fit(X_train, y_train)

        # Cross Validation
        if cv == True:
            pipeline = Pipeline([('transformer', scaler), ('estimator', nn_model_cv)])
            cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

        # Scores
        y_train_pred = nn_model.predict(X_train)
        y_train_accuracy = max(y_train_accuracy, accuracy_score(y_train, y_train_pred))

        y_test_pred = nn_model.predict(X_test)
        y_test_accuracy = max(y_test_accuracy, accuracy_score(y_test, y_test_pred))
        
    print('Training accuracy: ', y_train_accuracy)
    print('Test accuracy: ', y_test_accuracy)

    if cv == True:
        print('Cross validation accuracy: ', cv_scores.mean())
        return y_train_accuracy, y_test_accuracy, cv_scores.mean()
        
    return y_train_accuracy, y_test_accuracy, y_test_accuracy

def plot(scores_train, scores_test, scores_cv10, array, title, algorithm, cv):
    # print(np.average(scores_train), np.average(scores_test), np.average(scores_cv10))
    # print(scores_train)
    # print(scores_test)
    if cv == True:
        print(scores_cv10)
    filewrite_array("iterations:", array)
    filewrite_array("train_scores: " + str(np.average(scores_train)), scores_train)
    filewrite_array("train_tests: " + str(np.average(scores_test)), scores_test)

    #plt.plot(scores_train, color='green', alpha=0.8, label='Train')
    plt.plot(scores_test, color='magenta', alpha=0.8, label='Test')
    if cv == True:
        plt.plot(scores_cv10, color='blue', alpha=0.8, label='CV-10')
    plt.title("Accuracy over " + title + "(" + algorithm + ")", fontsize=14)
    plt.ylabel('Accuracy')
    plt.xlabel(title)
    plt.xticks(np.arange(len(array)), array)
    plt.legend(loc='best')
    dwn = plt.gcf()
    plt.savefig("Accuracy over " + title + "(" + algorithm + ")_" + filename)
    plt.show()

def plot_time(times, array, title, algorithm):
    filewrite_array("times:", times)

    plt.plot(times, color='red', alpha=0.8, label='Time')
    plt.title("Time over " + title + "(" + algorithm + ")", fontsize=14)
    plt.ylabel('Time (s)')
    plt.xlabel(title)
    plt.xticks(np.arange(len(array)), array)
    plt.legend(loc='best')
    dwn = plt.gcf()
    plt.savefig("Time over " + title + "(" + algorithm + ")_" + filename)
    plt.show()
    
# Variables
SAMPLE_PERCENTAGE = 0.2
TEST_PERCENTAGE = 0.2
RESTART = 10
CROSS_VALIDATION = False

# Preprocess
(X, y) = sample_preprocess(SAMPLE_PERCENTAGE)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_PERCENTAGE)

# Choose which algorithms to run
algorithms = ['gradient_descent', 'random_hill_climb', 'simulated_annealing', 'genetic_alg']
# Choose which iterations to run
iters = [5, 10]
#iters = [100, 500, 1000, 5000, 10000, 15000, 20000]

for algorithm in algorithms:
    print("<<<Algorithm: " + algorithm + ">>")
    file.write("<<<Algorithm: " + algorithm + ">>\n")
    scores_train = []
    scores_test = []
    scores_cv10 = []
    times = []
    for iter in iters:
        print("<<Iteration: " + str(iter) + ">>")
        start_time = time.time()
        score_train, score_test, score_cv10 = predict(algorithm, CROSS_VALIDATION, iter, RESTART, X_train, X_test, y_train, y_test)
        scores_train.append(score_train)
        scores_test.append(score_test)
        scores_cv10.append(score_cv10)
        print("--- %.2f (s) ---" % (time.time() - start_time))
        times.append((time.time() - start_time)/RESTART)
    plot(scores_train, scores_test, scores_cv10, iters, 'Iterations', algorithm, CROSS_VALIDATION)
    plot_time(times, iters, 'Iterations', algorithm)

file.close()