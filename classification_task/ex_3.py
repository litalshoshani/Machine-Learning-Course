import numpy as np

#sigmoid = lambda x: 1.0/(1.0 + np.exp(-x))

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

##############################################
# the method recieves:
# x - a value,
# w and b - vectors
# and the nethod returns the softmax value
##############################################
def softmax(x):
    exps = np.exp(x - x.max())
    result = exps/exps.sum()
    return result


##############################################
# the method recieves:
# x - a value,
# w and b - vectors
# and the class of the point.
# and the nethod returns the softmax value
##############################################
def softmax2(x,w,b,class_index,size):

    sum = 0
    for i in range(size):
        sum = sum + np.exp(w[i]*x + b[i])
    return np.exp(w[class_index]*x + b[class_index])/sum


##############################################
# return the y hat - the probability vector
##############################################
def forward(x,params,h_size):
    w1,b1,w2,b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    #x=np.shape(16,1)
    x.shape =(784,1)
    z1=np.dot(w1, x) + b1

    # temp = np.dot(x,w1.T).reshape(16,1)
    # z1 = temp + b1

    h = sigmoid(z1)

    temp = np.dot(w2,h)
   # for i in range(10):
    #    temp[i] += b2[i]
    z2 = temp + b2

    y_hat = softmax(z2)
    y_hat_vec = np.argmax(softmax(z2))

    ret = {'x': x, 'z1': z1, 'h': h, 'z2': z2, 'y_hat': y_hat,'y_hat_vec': y_hat_vec, 'w1': w1, 'w2': w2}
    for key in params:
        ret[key] = params[key]
    return ret


def compute_loss(y_hat_vec, y):
    class_index = int(y)
    return -(np.log(y_hat_vec[class_index]))

def backwards(x,y,params):
    x,z1,h,z2,y_hat,y_hat_vec,w1,w2 = [params[key] for key in ('x', 'z1', 'h', 'z2','y_hat', 'y_hat_vec','w1', 'w2')]
    # gradients:
    class_index = int(y)
    softmax_dl = y_hat

    softmax_dl[class_index] -= 1
    #softmax_dl = y_hat_vec - y_vec
    dz2 = softmax_dl
    dw2 = np.dot(dz2, h.T)
    db2 = softmax_dl

    temp =  sigmoid(z1) * (1 - sigmoid(z1))
    temp2 = np.dot(w2.T, softmax_dl)
    dz1 = temp2 * temp
    dw1 = np.dot(dz1, x.T)
    db1 = dz1

    #temp = np.dot(np.transpose(w2), db2)
    #sig1 = sigmoid(z1)
    #sig2 = 1-sigmoid(z1)
    #temp2 = sig1 * sig2
    #db1 = np.dot(temp, temp2)
    #db1 = []
    #for v1,v2 in zip(temp,temp2):
    #    db1.append(v1*v2)
    #db1 = np.array(db1)

    #db1 = temp * temp2

    #db1 = temp * (np.exp(-z1)/(1+np.exp(-z1))^2)
    #dw1 = np.dot(db1, x.T)
    #dw1 = db1 * x.T
        #np.outer((y_hat_vec - y_vec) * w2 , (np.exp(-z1)/(1+np.exp(-z1))^2) * x)

    return {'db1': db1, 'dw1': dw1, 'db2': db2, 'dw2': dw2}

def predict_dev(params, dev_x, dev_y):
    sum_loss = 0.0
    #count the times the learning was correct
    good = 0.0
    for x,y in zip(dev_x, dev_y):
        result = forward(x,params,100)
        loss = compute_loss(result['y_hat'], y)
        sum_loss += loss
        if result['y_hat'].argmax() == y:
            good += 1
    good_count = good / dev_x.shape[0]
    avg = sum_loss / dev_x.shape[0]
    return avg,good_count

def train(params, lr, train_x,train_y,dev_x,dev_y, h):

    w1,b1,w2,b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    #in range of epochs
    for i in range(100):
        sum_loss = 0.0
        # shuffle the trains in order to help the model to learn
        np.random.shuffle(zip(train_x, train_y))
        for x,y in zip(train_x, train_y):
            forward_result = forward(x,params, h)
            loss = compute_loss(forward_result['y_hat'],y)
            sum_loss += loss
            gradient = backwards(x,y,forward_result)
            #update the weights:
            db1, dw1, db2, dw2 = [gradient[key] for key in ('db1', 'dw1', 'db2', 'dw2')]
            w1 -= lr*dw1
            w2 -= lr*dw2
            b1 -= lr*db1
            b2 -= lr*db2

        dev_loss, good_count = predict_dev(params,dev_x,dev_y)
        print i, sum_loss / train_x.shape[0], dev_loss, "{}%".format(good_count * 100)

def main():
    # load the data from given files into matrixes
    train_x = np.loadtxt("train_x")
    train_y = np.loadtxt("train_y")
    test_x = np.loadtxt("test_x")

    for item in train_x:
        for i in range(784):
            item[i] = item[i]/255.0

    # now create two sets: train and dev.
    #first shuffle the x and y trains (using shuffle didnt work,
    # so I used a friend's advice on how to shuffle the trains.
    #np.random.seed(0)
    np.random.shuffle(zip(train_x,train_y))

    train_size = len(train_x)
    dev_size = train_size * 0.2

    # train contains 80% of the data
    new_train_x = train_x[:(int)(-dev_size),:]
    new_train_y = train_y[:(int)(-dev_size)]
    # while dev contains 20% of the data
    dev_x = train_x[(int)(-dev_size):,:]
    dev_y = train_y[(int)(-dev_size):]


    # initialize the random parameters and inputs
    # first, choose H
    h = 100
    w1 = np.random.uniform(-0.08, 0.08,[h,784])
    b1 = np.random.uniform(-0.08, 0.08,[h,1])
    w2 = np.random.uniform(-0.08, 0.08,[10, h])
    b2 = np.random.uniform(-0.08, 0.08,[10,1])
    params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    lr = 0.01
    train(params, lr, new_train_x, new_train_y, dev_x, dev_y, h)
    f = open("test.pred", "w")
    # et = {'x': x, 'z1': z1, 'h': h, 'z2': z2, 'y_hat': y_hat,'y_hat_vec': y_hat_vec, 'w1': w1, 'w2': w2}
    for x in test_x:
        result = forward(x,params,h)
        s = result['y_hat'].argmax(axis=0)
        f.write(str(s[0]) + '\n')
    f.close


if __name__ == '__main__':
    main()