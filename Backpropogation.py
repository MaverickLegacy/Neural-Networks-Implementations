# Kashyap, Sreesha
# Backpropogation algorithms implementation for CIFAR Image data Classification
# This implementation uses the Theano Framework for Deep Learning
import numpy as np
import matplotlib
import scipy.misc,sys,os
from os import listdir
from os.path import isfile, join
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from random import shuffle

def read_one_image_and_convert_to_vector(file_name):
    img = scipy.misc.imread(file_name).astype(np.float64) # read image and convert to float
    return img.reshape(-1,1) # reshape to column vector and return it

best_arch ={
 "no_of_hidden_nodes":400,
  "num_classes":10,
  "min_initial_weights":-0.00001,
  "max_initial_weights":0.00001,
  "alpha":0.001,
  "lbda": 0.1,
  "epoch":100,
 }
 
class multilayer_network(object):
 def __init__(self,settings,filedr,option):
 	self.__dict__.update(settings)
 	self.filedr = filedr
 	self.load_data(filedr+'/cifar_data_100_10/train')
 	self.load_data(filedr+'/cifar_data_100_10/test',False)
   	self.initialize_weights(self.no_of_hidden_nodes,self.num_classes,self.num_params)
 	if(option==1):
 	 self.task_1()
 	elif(option == 2):
 	 self.task_2()
 	elif(option == 3):
 	 self.task_3()
 	elif(option ==4):
 	 self.task_4()
 	else:
 	 self.task_5()  
 	
 def initialize_weights(self,no_of_hidden_nodes,num_classes, num_params):
 	'''
 	Args: 
 	no_of_hidden_nodes; type: int; no of nodes in hidden layer
 	num_class; type:int; number of classes in the output layer
 	num_params; type:int; number of input parameters  
 	'''
    self.w_1 = theano.shared(np.random.uniform(self.min_initial_weights,self.max_initial_weights,(no_of_hidden_nodes,num_params)))       
  	self.b_1 = theano.shared(np.random.uniform(self.min_initial_weights,self.max_initial_weights,(no_of_hidden_nodes,1)), broadcastable=(False,True))
  	self.w_2 = theano.shared(np.random.uniform(self.min_initial_weights,self.max_initial_weights,(num_classes,no_of_hidden_nodes)))
  	self.b_2 = theano.shared(np.random.uniform(self.min_initial_weights,self.max_initial_weights,(num_classes,1)),broadcastable=(False,True))
  		
 def load_data(self, fdr, is_training=True):
 	'''
 	Args:
 	fdr; type:int; the file directory containing the training data
 	is_training; type:boolean; if its training data
 	'''
 	onlyfiles = [f for f in listdir(fdr) if isfile(join(fdr, f))]
 	if(is_training):
 	 shuffle(onlyfiles)
 	 self.num_samples = len(onlyfiles)
 	 self.num_params = read_one_image_and_convert_to_vector(fdr +'/'+onlyfiles[0]).shape[0]
 	 self.training_data = np.ones(shape=(self.num_params,self.num_samples),dtype=np.float64)#
    	 self.target = 0 * np.ones(shape=(self.num_classes,self.num_samples),dtype=np.float64)#
    	 for i in range(0,len(onlyfiles)):
  	 	ind_cl =  int(onlyfiles[i][0])
  	 	vec1 = read_one_image_and_convert_to_vector(fdr +'/'+onlyfiles[i])
         	self.target[ind_cl,i] = 1
  	 	self.training_data[:,i:i+1] = vec1 / 255.0
         	
 	else:
 	    self.no_of_test_samples = len(onlyfiles)
 	    self.test_data = np.ones(shape=(self.num_params,self.no_of_test_samples),dtype=np.float64)
 	    self.test_target = 0 * np.ones(shape=(self.num_classes,self.no_of_test_samples),dtype=np.float64)#
 	    for i in range(0,len(onlyfiles)):
  	 	ind_cl =  int(onlyfiles[i][0])
  	 	vec1 = read_one_image_and_convert_to_vector(fdr +'/'+onlyfiles[i])
         	self.test_target[ind_cl,i] = 1
  	 	self.test_data[:,i:i+1] = vec1 / 255.0
 	    
 def create_trainer(self,is_regular=False,t_fn="relu"):  	#passing the input as a column vector
 	'''
 	Creating the Training Neural Network using Theano.
 	Args:
 	is_regular, type:int, If regularization is used
 	t_fn; type: boolean;  Transfer function ie. RELU
 	'''
   	input_p = T.dmatrix('input')
   	target = T.dmatrix('target')
   	if(t_fn == "sigmoid"):
   	 activation_1 = T.nnet.sigmoid(T.dot(self.w_1,input_p)+self.b_1) #calculating the output of layer 1
   	else:
   	 activation_1 = T.nnet.relu(T.dot(self.w_1,input_p)+self.b_1)
   	activation_2 = T.transpose(T.dot(self.w_2,activation_1) + self.b_2) # transposing the result to row vector for softmax function
   	output = T.nnet.softmax(activation_2)
   	if(is_regular):
   	  loss = T.mean(T.nnet.categorical_crossentropy(output,T.transpose(target))) + (self.lbda) * T.sum((self.w_1)**2) + (self.lbda) * T.sum((self.w_2)**2)
   	else:
   	  loss = T.mean(T.nnet.categorical_crossentropy(output,T.transpose(target)))
   	grad_w1, grad_w2, grad_b1, grad_b2 = T.grad(cost=loss, wrt=[self.w_1,self.w_2,self.b_1, self.b_2])
   	self.train =theano.function(inputs=[input_p, target],outputs=loss, updates=[[self.w_1, self.w_1 - self.alpha * grad_w1], [self.w_2,self.w_2 - self.alpha*grad_w2],[self.b_1, self.b_1 - self.alpha * grad_b1],[self.b_2, self.b_2 - self.alpha* grad_b2]])
 
 def create_predictor(self,t_fn="relu"):
 	'''
 	Creating the predictor Neural Network using Theano
 	Args:
 	is_regular, type:int, If regularization is used
 	t_fn; type: boolean;  Transfer function ie. RELU
	 '''
 	input_p = T.dmatrix('inputs')
   	if(t_fn=="relu"):
   	 activation_1 = T.nnet.relu(T.dot(self.w_1,input_p)+self.b_1)
   	else:
   	 activation_1 = T.nnet.sigmoid(T.dot(self.w_1,input_p)+self.b_1)
   	activation_2 = T.transpose(T.dot(self.w_2,activation_1) + self.b_2)
   	outputs = T.transpose(T.nnet.softmax(activation_2))
   	self.predictor = theano.function(inputs=[input_p],outputs= outputs)  	
 
 def calc_error_rate(self,g_outputs,g_targets):
 	 """
 	 Calculating error rate
 	 Args:
 	 g_outputs; type:int array; given predicted output
 	 g_targets; type:int array, target array
 	 """
      max_output_index = np.argmax(g_outputs, axis=0)
      max_targets_index = np.argmax(g_targets, axis=0)
      incorrect = np.sum(max_output_index != max_targets_index)
      return (float(incorrect) / self.num_samples) *100			
 
 def comp_confmatrix(self,g_outputs,g_targets):
 	 '''
 	 Computing the confusion matrix
 	 Args:
 	 g_outputs; type:int array; given predicted output
 	 g_targets; type:int array, target array
 	 '''
     conf_matrix = np.zeros((self.num_classes,self.num_classes),dtype=np.int)
     max_output_index = np.argmax(g_outputs, axis=0)
     max_targets_index = np.argmax(g_targets, axis=0)
     for i in range(g_outputs.shape[1]):
      conf_matrix[max_targets_index[i],max_output_index[i]] += 1
     return conf_matrix 
      
 def task_1(self):
 	'''
	Creating Neural Network with relu transfer function with the graph of loss function and error rate 
 	'''
     self.create_trainer()
     self.create_predictor()
     error_rate=[]
     loss_list=[]
     x_values = range(1,self.epoch+1) 
     for count in range(self.epoch):
      for i in range(self.num_samples):
       loss = self.train(self.training_data[:,i:i+1],self.target[:,i:i+1])
       #print loss
      predicted = self.predictor(self.training_data)
      error_rate.append(self.calc_error_rate(predicted,self.target))             
      loss_list.append(loss)
     predicted = self.predictor(self.test_data)
     print self.comp_confmatrix(predicted,self.test_target)
     plt.figure("loss function: task 1")
     plt.xlabel("epochs")
     plt.ylabel("Cross entropy loss")
     plt.plot(x_values, loss_list,'r-')
     plt.figure("Error rate: task 1")
     plt.xlabel("epochs")
     plt.ylabel("error rate")
     plt.plot(x_values,error_rate,'b-')
     plt.show()
     
 def task_2(self):
 	'''
	Creating Neural Network with sigmoid transfer function with the graph of loss function and error rate 
 	'''
     self.create_trainer(t_fn="sigmoid")
     self.create_predictor(t_fn="sigmoid")
     error_rate=[]
     loss_list=[]
     x_values = range(1,self.epoch+1) 
     for count in range(self.epoch):
      for i in range(self.num_samples):
       loss = self.train(self.training_data[:,i:i+1],self.target[:,i:i+1])
       #print loss
      predicted = self.predictor(self.training_data)
      error_rate.append(self.calc_error_rate(predicted,self.target))             
      loss_list.append(loss)
     predicted = self.predictor(self.test_data)
     print self.comp_confmatrix(predicted,self.test_target)
     plt.figure("loss function: task 2")
     plt.xlabel("epochs")
     plt.ylabel("Cross entropy loss")
     plt.plot(x_values, loss_list,'r-')
     plt.figure("Error rate: task 2")
     plt.xlabel("epochs")
     plt.ylabel("error rate")
     plt.plot(x_values,error_rate,'b-')
     plt.show()
 
 def task_3(self):
 	 '''
 	 Experiment with neural network having different num of neurons in hidden layer. 
 	 '''
     error_rate=[]
     x_values = range(25,126,25) 
     for nhn in x_values:
      self.initialize_weights(nhn,self.num_classes,self.num_params)
      self.create_trainer()
      self.create_predictor()
      for count in range(self.epoch):
       for i in range(self.num_samples):
        loss = self.train(self.training_data[:,i:i+1],self.target[:,i:i+1])
      predicted = self.predictor(self.training_data)
      error_rate.append(self.calc_error_rate(predicted,self.target))             
     plt.figure("Error rate: task 3")
     plt.xlabel("hidden layer nodes")
     plt.ylabel("error rate")
     plt.plot(x_values,error_rate,'b-')
     plt.show()
  
 def task_4(self):
     self.initialize_weights(500,self.num_classes,self.num_params)
     lbda_values = [0.1,0.2,0.3,0.4,0.5]
     x_values = range(1,self.epoch+1)
	 for l in lbda_values:
      error_rate=[]
      loss_list=[]
	  self.lbda = l    
      self.create_trainer(is_regular=True)
      self.create_predictor()
      for count in range(self.epoch):
       for i in range(self.num_samples):
        loss = self.train(self.training_data[:,i:i+1],self.target[:,i:i+1])
       predicted = self.predictor(self.training_data)
       error_rate.append(self.calc_error_rate(predicted,self.target))             
       loss_list.append(loss)     
	  predicted = self.predictor(self.test_data)
      print self.comp_confmatrix(predicted,self.test_target)
	  plt.figure("loss function: task 4")
      plt.xlabel("epochs")
      plt.ylabel("Cross entropy loss")
      plt.plot(x_values, loss_list,'r-')
      plt.figure("Error rate: task 4")
      plt.xlabel("epochs")
      plt.ylabel("error rate")
      plt.plot(x_values,error_rate,'b-')
      plt.show()
	  
 def task_5(self):
     self.load_data(self.filedr+'/cifar_data_1000_100/train')
     self.load_data(self.filedr+'/cifar_data_1000_100/test',False)   
     self.__dict__.update(best_arch)
     self.initialize_weights(self.no_of_hidden_nodes,self.num_classes,self.num_params)
     self.create_trainer(is_regular=True,t_fn="sigmoid")
     self.create_predictor(t_fn="sigmoid")
     error_rate=[]
     loss_list=[]
     x_values = range(1,self.epoch+1) 
     for count in range(self.epoch):
      for i in range(self.num_samples):
       loss = self.train(self.training_data[:,i:i+1],self.target[:,i:i+1])
       #print loss
      predicted = self.predictor(self.training_data)
      error_rate.append(self.calc_error_rate(predicted,self.target))             
      loss_list.append(loss)
     predicted = self.predictor(self.test_data)
     print self.comp_confmatrix(predicted,self.test_target)
     plt.figure("loss function: task 5")
     plt.xlabel("epochs")
     plt.ylabel("Cross entropy loss")
     plt.plot(x_values, loss_list,'r-')
     plt.figure("Error rate: task 5")
     plt.xlabel("epochs")
     plt.ylabel("error rate")
     plt.plot(x_values,error_rate,'b-')
     plt.show()
      
if __name__ == '__main__':
 # python Backpropogation.py -f data_dir -task task_num
 filedr = sys.argv[2]
 settings = {
  "no_of_hidden_nodes":100,
  "num_classes":10,
  "min_initial_weights":-0.00001,
  "max_initial_weights":0.00001,
  "alpha":0.001,
  "lbda": 0.1,
  "epoch":200
 }
  
 multi_network1= multilayer_network(settings,filedr,int(sys.argv[4]))
