# Multi-Class-Linear-Machine
Using hyperplanes in the feature-space to classify the data into classes

In a **multiclass linear classification machine** (linear machine), hyperplanes are used to split the **feature space** into regions based on the classes corresponding to the various feature vectors. In the training process, the hyperplanes are determined using a **train dataset** (feature vectors with known classes). Once the hyperplanes are computed, they can be used to predict the class for an unknown feature vector.

## Least Squares
The general form of a hyperplane is: ![g(\overrightarrow{x}_{i}) = \overrightarrow{w}^T \cdot\overrightarrow{x} _{i} +\overrightarrow{w}_{0} ](https://render.githubusercontent.com/render/math?math=g(%5Coverrightarrow%7Bx%7D_%7Bi%7D)%20%3D%20%5Coverrightarrow%7Bw%7D%5ET%20%5Ccdot%5Coverrightarrow%7Bx%7D%20_%7Bi%7D%20%2B%5Coverrightarrow%7Bw%7D_%7B0%7D%20) which can be improved and written as: ![g(\overrightarrow{y}_{i}) = \overrightarrow{a}^T \cdot\overrightarrow{y} _{i}](https://render.githubusercontent.com/render/math?math=g(%5Coverrightarrow%7By%7D_%7Bi%7D)%20%3D%20%5Coverrightarrow%7Ba%7D%5ET%20%5Ccdot%5Coverrightarrow%7By%7D%20_%7Bi%7D) by defining ![\overrightarrow{y}_{i} \coloneqq \[1, \overrightarrow{x}_{i}\]](https://render.githubusercontent.com/render/math?math=%5Coverrightarrow%7By%7D_%7Bi%7D%20%5Ccoloneqq%20%5B1%2C%20%5Coverrightarrow%7Bx%7D_%7Bi%7D%5D) and ![\overrightarrow{a}^{T} \coloneqq \[\overrightarrow{w}_{0}, \overrightarrow{w}\]](https://render.githubusercontent.com/render/math?math=%5Coverrightarrow%7Ba%7D%5E%7BT%7D%20%5Ccoloneqq%20%5B%5Coverrightarrow%7Bw%7D_%7B0%7D%2C%20%5Coverrightarrow%7Bw%7D%5D). Finally we can create a matrix ![\textit{Y} \coloneqq \[Y1, Y2, ... , Yc\]](https://render.githubusercontent.com/render/math?math=%5Ctextit%7BY%7D%20%5Ccoloneqq%20%5BY1%2C%20Y2%2C%20...%20%2C%20Yc%5D) where ![Y1, Y2, ... , Yc](https://render.githubusercontent.com/render/math?math=Y1%2C%20Y2%2C%20...%20%2C%20Yc) contain the feature vectors of the **C** classes and ![\textit{A} \coloneqq \[\overrightarrow{a}_{1}, \overrightarrow{a}_{2}, ... , \overrightarrow{a}_{c}\]](https://render.githubusercontent.com/render/math?math=%5Ctextit%7BA%7D%20%5Ccoloneqq%20%5B%5Coverrightarrow%7Ba%7D_%7B1%7D%2C%20%5Coverrightarrow%7Ba%7D_%7B2%7D%2C%20...%20%2C%20%5Coverrightarrow%7Ba%7D_%7Bc%7D%5D) where ![\overrightarrow{a}_{1}, \overrightarrow{a}_{2}, ... , \overrightarrow{a}_{c}](https://render.githubusercontent.com/render/math?math=%5Coverrightarrow%7Ba%7D_%7B1%7D%2C%20%5Coverrightarrow%7Ba%7D_%7B2%7D%2C%20...%20%2C%20%5Coverrightarrow%7Ba%7D_%7Bc%7D) contain the **C** weight vectors for the hyperplanes.

*A* can be found by applying **least squares** on the equation: ***YA = B*** where *B* is defined as:![\textit{B}^{T} \coloneqq \[{B}_{1},{B}_{2}, ... , {B}_{c}\]](https://render.githubusercontent.com/render/math?math=%5Ctextit%7BB%7D%5E%7BT%7D%20%5Ccoloneqq%20%5B%7BB%7D_%7B1%7D%2C%7BB%7D_%7B2%7D%2C%20...%20%2C%20%7BB%7D_%7Bc%7D%5D) where the column ***i*** of ***Bi*** is [1,1,...,1] and the rest of the columns are [0,0,...,0]. The least square equation is: ![\textit{A}=(\textit{Y}^{T}\textit{Y})^{-1}\cdot\textit{Y}^{T}\textit{B}](https://render.githubusercontent.com/render/math?math=%5Ctextit%7BA%7D%3D(%5Ctextit%7BY%7D%5E%7BT%7D%5Ctextit%7BY%7D)%5E%7B-1%7D%5Ccdot%5Ctextit%7BY%7D%5E%7BT%7D%5Ctextit%7BB%7D) 

## MNIST
The requirements are
* [Numpy](https://numpy.org/)
* [Matplotlib](https://matplotlib.org/)

The linear machine was tested on the **MNIST dataset for handwritten digits**.
* The model can be trained using the linear_machine_train.py script
  * The **split_classes_and_features** function splits the raw input data into the feature and class sets.
  * The **generate_B** function creates the B matrix from the right hand side of the least squares equation.
  * The weight vectors are computed using the **compute_A** function and the matrix is stored as a numpy array.
* Predictions on unknown feature vectors can be made using the **predict** function in the linear_machine_performance.py script.
  * Entire sets of unknown feature vectors can be predicted using **predict_all** function
* The performance of the model can be evaluated using test datasets (datasets with feature vectors and their corresponding class)
  * A percentage value of the performance can be found using the **evaluate_performance** function
  * A confusion matrix can be generated using the **evaluate_confusion_matrix** function
  
## Results
A performance of 82% was achieved when using the datasets attached in the repository

### 200 figures overlapped
<img src="media/200 figures overlapped.png">

### 5 figures overlapped
<img src="media/5 figures overlapped.png">

### Average of all figure
<img src="media/all figures average.png">

### Confusion matrix
<img src="media/confusion matrix test.png">


