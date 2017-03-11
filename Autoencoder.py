# Kashyap, Sreesha
# Autoencoder in Deep Learning implementation with the Theano framework
# This uses the MNIST dataset for training  
import numpy as np
import matplotlib
import scipy.misc,sys,os
from os import listdir
from os.path import isfile, join
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from random import shuffle
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy import linalg as la

def read_one_image_and_convert_to_vector(file_name):
    img = scipy.misc.imread(file_name).astype(np.float64) # read image and convert to float
    return img.reshape(-1,1) # reshape to column vector and return it

def PCA(data, num_comp =2):
 m,n = data.shape
 data -= data.mean(axis=0)
 R = np.cov(data, rowvar=False)
 evals, evecs = la.eigh(R)
 idx = np.argsort(evals)[::-1]
 evecs = evecs[:,idx]
 evals = evals[idx]
 return evecs[:,:num_comp]  

class multilayer_network(object):
 def __init__(self,settings,filedr,option):
 	self.__dict__.update(settings)
 	self.filedr = filedr
 	self.num_samples_ds1 ,self.ds1 = self.load_data(filedr+'/train')
 	self.num_samples_ds2 ,self.ds2 = self.load_data(filedr+'/set2_2k')
 	self.num_samples_ds3 ,self.ds3 =self.load_data(filedr+'/set3_100')
   	self.initialize_weights(self.no_of_hidden_nodes,self.num_params,self.num_params)
 	self.num_classes = self.num_params
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
 	'''
 	self.w_1 = theano.shared(np.random.uniform(self.min_initial_weights,self.max_initial_weights,(no_of_hidden_nodes,num_params)))       
  	self.b_1 = theano.shared(np.random.uniform(self.min_initial_weights,self.max_initial_weights,(no_of_hidden_nodes,1)), broadcastable=(False,True))
  	self.w_2 = theano.shared(np.random.uniform(self.min_initial_weights,self.max_initial_weights,(num_classes,no_of_hidden_nodes)))
  	self.b_2 = theano.shared(np.random.uniform(self.min_initial_weights,self.max_initial_weights,(num_classes,1)),broadcastable=(False,True))
  		
 def load_data(self, fdr):
 	 '''
 	 '''
 	onlyfiles = [f for f in listdir(fdr) if isfile(join(fdr, f))]
 	shuffle(onlyfiles)
 	num_samples = len(onlyfiles)
 	self.num_params = read_one_image_and_convert_to_vector(fdr +'/'+onlyfiles[0]).shape[0]
 	training_data = np.ones(shape=(self.num_params,num_samples),dtype=np.float64)#
    for i in range(0,len(onlyfiles)):
  	 	ind_cl =  int(onlyfiles[i][0])
  	 	vec1 = read_one_image_and_convert_to_vector(fdr +'/'+onlyfiles[i])
        training_data[:,i:i+1] = vec1 / 255.0
    return num_samples, training_data
         	
 def create_trainer(self,is_regular=False):  	#passing the input as a column vector
   	'''
   	'''
   	input_p = T.dmatrix('input')
   	target = T.dmatrix('target')
   	activation_1 = T.nnet.relu(T.dot(self.w_1,input_p)+self.b_1)
   	output = T.dot(self.w_2,activation_1) + self.b_2 # transposing the result to row vector for softmax function
   	if(is_regular):
   	  loss = T.mean((output - target)**2)+ (self.lbda) * T.sum((self.w_1)**2) + (self.lbda) * T.sum((self.w_2)**2)
   	else:
   	  loss = T.mean((output-target)**2)
   	grad_w1, grad_w2, grad_b1, grad_b2 = T.grad(cost=loss, wrt=[self.w_1,self
   	.w_2,self.b_1, self.b_2])
   	self.train =theano.function(inputs=[input_p, target],outputs=[loss,self.w_1,self.w_2], updates=[[self.w_1, self.w_1 - self.alpha * grad_w1], [self.w_2,self.w_2 - self.alpha*grad_w2],[self.b_1, self.b_1 - self.alpha * grad_b1],[self.b_2, self.b_2 - self.alpha* grad_b2]])
 
 def create_predictor(self,is_regular=False):
 	'''
 	'''
 	input_p = T.dmatrix('inputs')
   	target = T.dmatrix('target')
   	activation_1 = T.nnet.relu(T.dot(self.w_1,input_p)+self.b_1)
   	output = T.dot(self.w_2,activation_1) + self.b_2
   	if(is_regular):
   	  loss = T.mean((output - target)**2)+ (self.lbda) * T.sum((self.w_1)**2) + (self.lbda) * T.sum((self.w_2)**2)
   	else:
   	  loss = T.mean((output - target)**2)
   	self.predictor = theano.function(inputs=[input_p,target],outputs=[loss,output])  	
 
 def task_1(self):
 	 '''
 	 '''
    self.create_trainer(False)
    self.create_predictor(False)
    mse1=[]
    mse2=[]
    x_values = range(1,self.epoch+1) 
    for count in range(self.epoch):
      for i in range(self.num_samples_ds1):
       loss,w1,w2 = self.train(self.ds1[:,i:i+1],self.ds1[:,i:i+1])
       #print loss
      predicted_mse,out1 = self.predictor(self.ds1,self.ds1)
      mse1.append(predicted_mse)
      predicted_mse,out2 = self.predictor(self.ds2,self.ds2)
      mse2.append(predicted_mse)
    plt.plot(x_values,mse1,'b-',label='MSE First DS')
    plt.plot(x_values,mse2,'r-',label='MSE Second DS')
    plt.xlabel('No of epochs')
    plt.ylabel('Mean squared error')
    plt.legend()
    plt.show()
     
 def task_2(self):
    '''
    Trains the Autoencoder network with variying the number of nodes in the hidden layer. Plots the mean
    squared error vs number of hidden nodes.
    '''

    x_values = range(20,101,20)
    mse1=[]
    mse2=[]
    for num_hd in x_values:
 		self.initialize_weights(num_hd,self.num_params,self.num_params)
 		self.create_trainer()
     	self.create_predictor()
     	for count in range(self.epoch):
      	 for i in range(self.num_samples_ds1):
       		loss,w1,w2 = self.train(self.ds1[:,i:i+1],self.ds1[:,i:i+1])
        predicted_mse,out1 = self.predictor(self.ds1,self.ds1)
        mse1.append(predicted_mse)
      	predicted_mse,out2 = self.predictor(self.ds2,self.ds2)
      	mse2.append(predicted_mse)
    plt.plot(x_values,mse1,'b-',label='MSE First DS')
    plt.plot(x_values,mse2,'r-',label='MSE Second DS')
    plt.xlabel('No of hidden nodes')
    plt.ylabel('Mean squared error')
    plt.legend()
    plt.show()
     	
 def task_3(self):
    self.create_trainer(False)
    self.create_predictor(False)
    self.epoch=100
    #x_values = range(1,self.epoch+1) 
    for count in range(self.epoch):
      #print 'epoch:', count
      for i in range(self.num_samples_ds1):
       loss,w1,w2= self.train(self.ds1[:,i:i+1],self.ds1[:,i:i+1])
    fig = plt.figure(1,(20.,20.)) 
    grid = ImageGrid(fig,111,nrows_ncols=(10,10),axes_pad=0.1)
    for i in range(self.no_of_hidden_nodes):
      grid[i].imshow(np.reshape(w1[i],(28,28)), cmap='gray')
    plt.suptitle('task3')
    plt.show()
    self.task_4()
    self.task_5()

 def task_4(self):
    loss, out3 = self.predictor(self.ds3,self.ds3)
    fig1 = plt.figure(1,(20.,20.)) 
    grid1 = ImageGrid(fig1,111,nrows_ncols=(10,10),axes_pad=0.1)
    for i in range(self.no_of_hidden_nodes):
      grid1[i].imshow(np.reshape(self.ds3[:,i],(28,28)), cmap='gray')
    plt.suptitle('task4-original')
    fig2 = plt.figure(2,(20.,20.)) 
    grid2 = ImageGrid(fig2,111,nrows_ncols=(10,10),axes_pad=0.1)
    for i in range(self.no_of_hidden_nodes):
      grid2[i].imshow(np.reshape(out3[:,i],(28,28)), cmap='gray')
    plt.suptitle('task4-reconstructed')
    plt.show()
     
 def task_5(self):
    '''
	Compares the PCA compression with autoencoder compression with visualizing it.   	  
    '''
    evecs1 = PCA(self.ds2.T,num_comp=100)
    loss, output = self.predictor(self.ds2,self.ds2)
    evecs2 = PCA(output.T,num_comp=100)
    fig3 = plt.figure(1,(20.,20.)) 
    grid3 = ImageGrid(fig3,111,nrows_ncols=(10,10),axes_pad=0.1)
    for i in range(self.no_of_hidden_nodes):
      grid3[i].imshow(np.reshape(evecs1[:,i],(28,28)), cmap='gray')
    plt.suptitle('task5-DATA2')
    fig4 = plt.figure(2,(20.,20.)) 
    grid4 = ImageGrid(fig4,111,nrows_ncols=(10,10),axes_pad=0.1)
    for i in range(self.no_of_hidden_nodes):
      grid4[i].imshow(np.reshape(evecs2[:,i],(28,28)), cmap='gray')
    plt.suptitle('task5-Regenerated')
    plt.show()
     
if __name__ == '__main__':
 # python kashyap_assignment_05.py -f data_dir -task task_num
 pwd = os.getcwd()
 #filedr = pwd+"/"+sys.argv[2]
 filedr = sys.argv[2]
 settings = {
  "no_of_hidden_nodes":100,
  "min_initial_weights":-0.00001,
  "max_initial_weights":0.00001,
  "alpha":0.001,
  "lbda": 0.1,
  "epoch":50
 }
  
 multi_network1= multilayer_network(settings,filedr,int(sys.argv[4]))
