import numpy as np
import random
import matplotlib.pyplot as plt
import math
import decimal
import sys

##############################################
# the method create and returns a list of points,
# the points are created randomly.
# each point heas its own class (0,1 or 2) and a value
##############################################
def create_list_of_points():
    listi = []
    for a in range(3):
        for i in range(100):
            x = np.random.normal(2*a, 1)
            tup = [x,a]
            listi.append(tup)
    return listi

##############################################
# the method recieves:
# x - a value,
# w and b - vectors
# and the nethod returns the softmax value
##############################################
def softmax(x,w,b):
    vec = [1.0,1.0,1.0]
    s1 = w[0]*x + b[0]
    s2 = w[1]*x + b[1]
    s3 = w[2]*x + b[2]
    vec = np.array([s1,s2,s3])
    exps = np.exp(vec - np.max(vec))
    result = exps/exps.sum()
    return result

##############################################
# the method recieves ls - a list of all the points
# and iterates over the points, and uses SGD in order
# to update w and b
##############################################
def iterate_over_points(ls):
    #create w and b and initialize the vecotrs
    w = [0,0,0]
    b = [0,0,0]

    #run over the points
    for i in range(100):
        for j in range(300):
            #get the point's value
            x = ls[j][0]
            #get the point's class
            class_index = ls[j][1]
            #call softmax
            softmax_result = softmax(x, w, b)
            #now update w and b
            for k in range(3):
                if(k == class_index):
                    b[k] =  b[k] - (softmax_result[k] - 1)*0.2
                    w[k] =  w[k] - (softmax_result[k]*x - x)*0.2
                else:
                    w[k] = w[k] - (softmax_result[k]*x)*0.2
                    b[k] =  b[k] - (softmax_result[k])*0.2

    return b,w

##############################################
# the method recieves:
# x - a value,
# w and b - vectors
# and the class of the point.
# and the nethod returns the softmax value
##############################################
def softmax2(x,w,b,class_index):

    sum = 0
    for i in range(3):
        sum = sum + np.exp(w[i]*x + b[i])
    return np.exp(w[class_index]*x + b[class_index])/sum


##############################################
# the main.
# creates the points, iterate over them, and then
# plot the graph.
##############################################
def calc_estimated_posterior_probability(x):
    return (1 / (math.sqrt((2 * math.pi))) * np.exp(-1 * math.pow((x - 2), 2) / 2) /
                         ((1 / (math.sqrt((2 * math.pi))) * np.exp(-1 * math.pow((x - 2), 2) / 2)) +
                          (1 / (math.sqrt((2 * math.pi))) * np.exp(-1 * math.pow((x - 4), 2) / 2)) +
                          (1 / (math.sqrt((2 * math.pi))) * np.exp(-1 * math.pow((x - 6), 2) / 2))))

##############################################
# the main.
# creates the points, iterate over them, and then
# plot the graph.
##############################################
def main():
    #create 300 points, each point has a class
    ls = create_list_of_points()
    random.shuffle(ls)
    #iterate over the points
    iteration = iterate_over_points(ls)
    #get the updated values of w and b
    b = iteration[0]
    w = iteration[1]
    ls2 = []
    ls3 = []
    class_index_for_graph = 0
    #plot the results on a graph
    for i in range(10):
        f = softmax2(i,w,b,class_index_for_graph)
        ls2.append(f)
        ls3.append(i)
    plt.plot(np.array(ls3), np.array(ls2), "r-", label='Softmax')
    ls4 = []
    for x in range(0, 10):
        ls4.append(calc_estimated_posterior_probability(x))

    plt.plot(np.array(ls3), np.array(ls4),"b-", label='Real')
    plt.show()


if __name__ == '__main__':
    main()