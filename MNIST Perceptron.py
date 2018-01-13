import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def multilayer_perceptron(x,weights,biases):
    '''
    x: placeholder for the data input
    weigths: dictionary of weights
    biases: dictionary of bias values
    '''
    
    # First Hidden Layer with RELU Activation
    # (X * W + B)
    layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    # RELU(X * W + B) -> f(x) = max(0,x)
    layer_1 = tf.nn.relu(layer_1)
    
    #Second Hidden Layer
    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    
    # Last Output layer
    out_layer = tf.matmul(layer_2,weights['out']) + biases['out']
    
    return out_layer


mnist = input_data.read_data_sets('/tmp/data',one_hot=True)

sample = mnist.train.images[2].reshape(28,28)

learning_rate = 0.001 # how quickly we adjust the cost function
training_epochs = 15 # no of training cycles we go through
batch_size = 100 # size of the batches of the training data

# network parameters
n_classes = 10 # because digits in our data goe from 0 to 9
n_samples = mnist.train.num_examples # 55000
n_input = 784

# no of neurons we want in the 2 hidden layers of our neural network
n_hidden_1 = 256
n_hidden_2 = 256

weights = {
    'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
}

biases = {
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}

x = tf.placeholder('float',[None,n_input])
y = tf.placeholder('float',[None,n_classes])

pred = multilayer_perceptron(x,weights,biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Training the model
Xsamp,ysamp = mnist.train.next_batch(1)

sess = tf.InteractiveSession()

init = tf.initialize_all_variables() # initialize all variables defined previously

sess.run(init)

# 15 loops
for epoch in range(training_epochs):
    
    # Cost
    avg_cost = 0.0
    
    total_batch = int(n_samples/batch_size)
    
    for i in range(total_batch):
        
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        
        _,c = sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
        
        avg_cost += c/total_batch
        
    print('Epoch: {} cost: {:.4f}'.format(epoch+1,avg_cost))

print("Model has completed {} Epochs of Training".format(training_epochs))

# Model evaluation

correct_predictions = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))

print(correct_predictions[0])

correct_predictions = tf.cast(correct_predictions, "float")

print(correct_predictions[0])
print(correct_predictions[0])

accuracy = tf.reduce_mean(correct_predictions)

mnist.test.labels
mnist.test.images

print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

