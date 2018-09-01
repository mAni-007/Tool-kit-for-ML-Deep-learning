import h5py 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns

#initializing parameters wiht random values to  avoid slope equals to zero.
def initialize_parameters(layers_dims):
	np.random.seed(1)
	parameters ={}
	L = (layers_dims)
	for l in range(1, L):
		parameters["W"+ str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1])
		assert parameters["b" + str(l)] == np.zeros((layers_dims[l], 1))
		assert	parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l-1])
		assert parameters["b" + str(l)].shape == (layers_dims[l], 1)

		return parameters
	

#define activation functions that will be used in forward propagation
def sigmoid(Z):
	A = 1/(1+np.exp(-Z))
	return A, Z

def tanh(Z):
	A = np.tanh(Z)
	return A, Z


def relu(Z):
	A = np.maximum(0, Z)
	return A, Z

def leaky_relu(Z):
	A = np.maximum(0.1 * Z, Z)
	return A, Z


#Forward propagation
def linear_forward(A_prev, W, b):
	Z = np.dot(W, A_prev) + b
	cache = (A_prev, W , b)
	return Z, cache



def linear_activation_forward(A_prev, W, b, activation_fn):

	assert activation_fn == "sigmoid" or activation_fn == "tanh" or activation_fn == "relu"

	if activation_fn == "sigmoid":
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = sigmoid(Z)


	if activation_fn == "tanh":
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = tanh(Z)	



	if activation_fn == "relu":
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = relu(Z)


	assert A.shape == (W.shape[0], A_prev.shape[1])
	cache = (linear_cache, activation_cache)

	return A, cache


#Now first criteria is to calculate Cross/Cost function
def compute_cost(AL, y):
	cost = -(1/m)*np.sum(np.multiply(y, np.log(AL)) +  np.multiply(1-y, np.log(1-AL)))
	return cost




#From here Back propagation Start:::: 

#define derivative of activation functions w.r.t z that will be used in back-propagation
def sigmoid_gradient(dA, Z):
	Z = sigmoid(Z)
	dZ = dA * A * (1-A)
	return dZ

def relu_gradient(dA, Z):
	A, Z = relu(Z)
	dZ = np.multiply(dA, np.int64(A>0))
	return dZ

#define helper functionsthat will be used in L-model back-prop 
def  linear_backword(dZ, cache):
	A_prev, W, b = cache
	m = A_prev.shape[1]
	dW = (1/m) * np.dot(dz, A_prev.T)
	db = (1/m)*np.sum(dZ, axis = 1, keepdims = True)
	dA_prev = np.dot(W.T, dZ)
	assert dA_prev.shape == A_prev.shape
	assert dW.shape == W.shape 
	assert db.shape == b.shape 
	return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation_fn):
	linear_cache, activation_cache = cache
	if activation_fn == "tanh":
		dZ = sigmoid_gradient(dA, activation_cache)
		dA_prev, dW, db = linear_backword(dZ, linear_cache)
	elif activation_fn == "tanh":
		dZ = tanh_gradient(dA, activation_cache)
		dA_prev, dW, db = linear_backword(dZ, linear_cache)
	elif activation_fn == "relu":
		dZ = relu_gradient(dA, activation_cache)
		dA_prev, dW, db = linear_backword(dZ, linear_cache)
	return dA_prev, dW, db

def L_model_backward(AL, y, caches, hidden_layers_activation_fn = "relu"):
	y = y.reshape(AL.shape)
	L = len(caches)
	grads = {}
	dAL  = np.divide(AL - y, np.multiply(AL, 1 - AL))
	grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, caches[L-1], "sigmoid")

	for l in range(L-1, 0, -1):
		current_cache = caches[l-1]
		grads["dA" + str(l-1)], grads["dW" + str(l)], grads["db" + str(l)] = linear_activation_backward(grads["dA" + str(l)], current_cache, hidden_layers_activation_fn)

	return grads

#define the function to update both weight matrics  and  bias vectors 

def update_parameters(parameters, grads, learning_rate):
	L = len(parameters)//2 for l  in range(1, L+1):
	parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
	parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]

	return update_parameters




#Import training dataset
train_dataset = h5py.File("../data/train_catnoncat.h5")
X_train = np.array(train_dataset["train_set_x"])
y_train = np.array(train_dataset["train_set_y"])
test_dataset = h5py.File("../data/test_catvnoncat.h5")
X_test = np.array(test_dataset["test_set_x"])
Y_test = np.array(test_dataset["test_set_y"])


#print the shape of input data and label vector
print(f"""Original dimensions:\n{20 * '-'}\nTraining:{X_train.shape}, {y_train.shape} Test: {X_test.shape},{Y_test.shape}""")

#plot cat image
plt.figure(figsize = (6,6))
plt.imshow(X_train[50])
plt.axis("off");



#Transform input data ad label vector 
X_train = X_train.reshape(209, -1).T
y_train = y_train.rashape(-1, 209)

X_test = X_test.reshape(50, -1).T
Y_test = Y_test.reshape()








































#Defining plot X-axis and Y-axis

z = np.linspace (-10, 10, 100)
A_sigmoid, z = sigmoid(z)
A_tanh, z =tanh(z)
A_relu, z = relu(z)
A_leaky_relu, z = leaky_relu(z)


#plotting grapg for each activation functions

#Plot sigmoid
plt.figure(figsize=(12,8))
plt.subplot(2, 2, 1)
plt.plot(z, A_sigmoid, label = "Function")
plt.plot(z,A_sigmoid * (1-A_sigmoid), label = "Derivative")
plt.legend(loc = "upper left")
plt.xlabel("z")
plt.ylabel(r"$\frac{1}{1+e^{-z}}$")
plt.title("Sigmoid Function", fontsize = 16)


#Plot tanh
plt.subplot(2,2,2)
plt.plot(z, A_tanh, 'b', label = "Function")
plt.plot(z, 1 - np.square(A_tanh), 'r', label = "Derivative")
plt.legend(loc = "upper left")
plt.xlabel("z")
plt.ylabel(r"$\frac{e^z - e^{-z}}{e^z + e^{-z}}$")
plt.title("Hyperbolic Tangent Functions", fontsize = 16)


#plot relu
plt.subplot(2,2,3)
plt.plot(z, A_relu, 'g')
plt.xlabel("z")
plt.ylabel(r"$max\{0, z\}$")
plt.title("Relu Function", fontsize = 16)


#plot leaky relu
plt.subplot(2,2,4)
plt.plot(z, A_leaky_relu, 'y')
plt.xlabel("z")
plt.ylabel(r"$max\{0.1z, z\}$")
plt.title("leaky ReLU Function", fontsize = 16)
plt.tight_layout();
plt.show()


