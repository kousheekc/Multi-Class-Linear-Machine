import numpy as np
import matplotlib.pyplot as plt

def predict(x, A):
    biggest = 0
    prediction = 0

    # check all weight vectors in A (columns of A) 
    for i in range(len(A[0])):
        ai = A[:,i]

        w0 = ai[0]
        w = ai[1:]

        # equation of the hyperplane
        g_of_x = np.dot(w, x) + w0

        # prediction is the biggest value of g(xi)
        if g_of_x > biggest:
            biggest = g_of_x
            prediction = i

    return prediction

def predict_all(X, A):
    predicted_classes = []

    for x in X:
        biggest = 0
        prediction = 0

        for i in range(len(A[0])):
            ai = A[:,i]

            w0 = ai[0]
            w = ai[1:]

            g_of_x = np.dot(w, x) + w0

            if g_of_x > biggest:
                biggest = g_of_x
                prediction = i

        predicted_classes.append(prediction)

    return np.array(predicted_classes)

def evaluate_performance(predicted, actual):
    correct = 0
    for i in range(len(predicted)):
        if (predicted[i] == actual[i]):
            correct += 1
    
    return correct/len(actual)*100

def evaluate_confusion_matrix(predicted, actual, num_of_classes):
    confusion_matrix = np.zeros((num_of_classes, num_of_classes))
    
    for i in range(len(predicted)):
        confusion_matrix[int(actual[i]), int(predicted[i])] += 1

    return confusion_matrix

# use test dataset to evaluate performance of A
raw_data = np.loadtxt("pendigits_test.txt", delimiter=",")
# load A
A = np.load('weights.npy')

classes = raw_data[:,-1]
features = np.delete(raw_data, -1, axis=1)

predicted_classes = predict_all(features, A)
performance = evaluate_performance(predicted_classes, classes)
confusion_matrix = evaluate_confusion_matrix(predicted_classes, classes, 10)

print("Performance of A: ", performance, "%")

plt.xlabel("predicted")
plt.ylabel("actual")
plt.imshow(confusion_matrix, cmap=plt.cm.Blues)

plt.show()
