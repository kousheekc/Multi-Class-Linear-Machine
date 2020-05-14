import numpy as np

# actual equation of hyperplanes: g(x)=w_t x + w0
# goal: determine w_ts, w0s
# simplify: define a_t = [w0, w_t] and yi = [1, xi] 
# least squares equation YA = B
# goal is to determine A, whose columns contain weight vectors [w0, w_t] for classification

def generate_B(classes, numOfClasses):
    
    B = np.zeros((len(classes), numOfClasses))

    # column c of be = [1,1,..1], rest 0
    for i,c in enumerate(classes):
        B[i][int(c)] = 1

    return B

def split_classes_and_features(raw_data, n=-1):
    # sort raw data based on classes
    sorted_data = raw_data[raw_data[:, n].argsort()]

    # last column of raw data contains classes
    classes = sorted_data[:,-1]
    # rest of the values are the features
    features = np.delete(sorted_data, -1, axis=1)
    # insert colums of [1,1...1] at the beginning of features
    features = np.insert(features, [0], np.ones((len(features), 1)), axis=1)
    return classes, features

def compute_A(Y, B):
    # least sqaures solution: A = (Y_t Y)^(-1)(Y_t B)
    YtB = np.dot(np.transpose(Y), B)
    YtY_inv = np.linalg.inv(np.dot(np.transpose(Y), Y))
    A = np.dot(YtY_inv, YtB)
    return A
        
                 

# use train dataset for finding weight vectors
raw_data = np.loadtxt("pendigits_train.txt", delimiter=",")
classes, Y = split_classes_and_features(raw_data)

B = generate_B(classes, 10)

A = compute_A(Y, B)

# save weight matrix A for classification
np.save('weights.npy', A)
