# Kashyap, Sreesha
# Implementation of Hebbian Learning Algorithm

import numpy as np
import Tkinter as Tk
import matplotlib
import scipy.misc,sys,os
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from random import shuffle

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import colorsys

def read_one_image_and_convert_to_vector(file_name):
    img = scipy.misc.imread(file_name).astype(np.float64) # read image and convert to float
    return img.reshape(-1,1) # reshape to column vector and return it
    
class ClDataSet:
    # This class encapsulates the data set
    # The data set includes input samples and targets
    def __init__(self, samples=[[0., 0., 1., 1.], [0., 1., 0., 1.]], targets=[[0., 1., 1., 0.]]):
        # Note: input samples are assumed to be in column order.
        # This means that each column of the samples matrix is representing
        # a sample point
        self.samples = samples
        if targets != None:
            self.targets = targets
        else:
            self.targets = None
            
class ClNNGui2d:
    """
    This class presents an experiment to demonstrate
    Hebbian learning.
    """

    def __init__(self, master,settings):
        self.__dict__.update(settings) # constructor with settings
        self.master = master
        self.xmin = 0
        self.xmax = self.epoch
        self.ymin = 0
        self.ymax = 100
        self.master.update()
        # network data structures
        self.error_rate =[]
        if( self.initialize!= True):
          self.weight_new = np.zeros(shape=(self.no_of_neurons,self.no_of_params+1),dtype=np.float64)
        else:
          self.weight_new = np.random.uniform(-1, 1,(self.no_of_neurons,self.no_of_params+1))
        
        self.training_data = self.data_set.samples
        self.target = self.data_set.targets
        self.error_rates = np.random.randint(low=0,high=100,size=self.epoch)
        #GUI settings
        self.master.rowconfigure(0, weight=2, uniform="group1")
        self.master.rowconfigure(1, weight=1, uniform="group1")
        self.master.columnconfigure(0, weight=2, uniform="group1")
        self.master.columnconfigure(1, weight=1, uniform="group1")
	self.canvas = Tk.Canvas(self.master)
        self.display_frame = Tk.Frame(self.master)
        self.display_frame.grid(row=0, column=0, columnspan=2, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.display_frame.rowconfigure(0, weight=1)
        self.display_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("Hebbian Learning")
        self.axes = self.figure.add_subplot(111)
        plt.title("Hebbian Learning")
        plt.scatter(0, 0)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.display_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # Create sliders frame
        self.sliders_frame = Tk.Frame(self.master)
        self.sliders_frame.grid(row=1, column=0)
        self.sliders_frame.rowconfigure(0, weight=10)
        self.sliders_frame.rowconfigure(1, weight=2)
        self.sliders_frame.columnconfigure(0, weight=1, uniform='xx')
        # Set up the sliders
        ivar = Tk.IntVar()
        self.learning_rate_slider_label = Tk.Label(self.sliders_frame, text="Learning Rate")
        self.learning_rate_slider_label.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.learning_rate_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=0.00, to_=1.00, resolution=0.0001, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=20,length=800,
                                             command=lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.set(self.alpha)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.gamma_value_slider_label = Tk.Label(self.sliders_frame, text="Gamma Value")
        self.gamma_value_slider_label.grid(row=3, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.gamma_value_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=0.00, to_=1.00, resolution=0.0001,bg="#DDDDDD",
                                             activebackground="#AF0000",
                                             highlightcolor="#00FFEF", width=20,length=800,
                                             command=lambda event: self.gamma_value_slider_callback())
        self.gamma_value_slider.set(self.gamma)
        self.gamma_value_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())

        self.gamma_value_slider.grid(row=4, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
	#########################################################################
        #  Set up the frame for button(s)
        #########################################################################

        self.buttons_frame = Tk.Frame(self.master)
        self.buttons_frame.grid(row=1,column=1)
        self.buttons_frame.rowconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1,uniform='xx')
        self.label_for_entry_box = Tk.Label(self.buttons_frame, text = "Learning Type", justify="center")
        self.label_for_entry_box.grid(row=0, column=0,sticky=Tk.N+Tk.E+Tk.S+Tk.W)

        self.learning_method_variable=Tk.StringVar()
        self.learning_method_dropdown=Tk.OptionMenu(self.buttons_frame,self.learning_method_variable, "Filtered learning","Delta rule","Unsupervised hebbian",command=lambda  event: self.learning_method_dropdown_callback())
        self.learning_method_variable.set("Delta learning")
        self.learning_method_dropdown.grid(row=1,column=0,sticky=Tk.N+Tk.E+Tk.S+Tk.W)
        self.adjust_weights_button = Tk.Button(self.buttons_frame,
                                               text="Adjust Weights (Learn)",
                                               bg="yellow", fg="red",
                                               command=lambda: self.adjust_weights_button_callback())
        self.adjust_weights_button.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        #self.initialize()
        self.refresh_display()

    #def initialize(self):
    #    self.nn_experiment.neural_network.randomize_weights()

    def refresh_display(self):
        x_axis =range(1,self.epoch+1)
        self.axes.cla()
        self.axes.plot(x_axis,self.error_rates)
        self.axes.xaxis.set_visible(True)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        plt.title(self.learning_method)
        self.canvas.draw()

    def learning_rate_slider_callback(self): # NEEDS TO BE CORRECTED
        self.alpha = self.learning_rate_slider.get()
                
        

    def adjust_weights_button_callback(self):
        temp_text = self.adjust_weights_button.config('text')[-1]
        self.adjust_weights_button.config(text='Please Wait')
        self.hebbian_learning() #returns error_rates
        self.refresh_display()
        self.adjust_weights_button.config(text=temp_text)
        self.adjust_weights_button.update_idletasks()
    
    def learning_method_dropdown_callback(self):
        self.learning_method=self.learning_method_variable.get()
        
        
    def gamma_value_slider_callback(self):
    	self.gamma = self.gamma_value_slider.get()
    
    def hebbian_learning(self):           ## Method implementing Hebbian learning 
    	if(self.learning_method == "Delta rule"):  # Delta Rule
    		for run in range(0,self.epoch):
  				for cnt in range(0, self.no_of_samples):
   	 				A = np.matrix.dot(self.weight_new, self.training_data) #predict output
 	 				err_matrix = target - A
 	 				self.weight_new += self.alpha * np.matrix.dot(err_matrix[:,cnt:cnt+1], np.transpose(self.training_data[:,cnt:cnt+1]))
        			self.error_rates[run] = self.calc_error_rate(self.target,A,self.no_of_samples)
     
    	elif(self.learning_method == "Filtered learning"): #  Filtered Rule
        	for run in range(0,self.epoch):
   	 			for cnt in range(0, self.no_of_samples):
  					self.weight_new = (1-self.gamma) * self.weight_new + self.alpha *self.target[:,cnt:cnt+1].dot( np.transpose(self.training_data[:,cnt:cnt+1]))
   					A = np.matrix.dot(self.weight_new, self.training_data) #predict output
     	 			self.error_rates[run] = self.calc_error_rate(self.target,A,self.no_of_samples)
     	 		 
     	else:
     		for run in range(0,self.epoch):  # unsupervised version of Hebbian Learning
   	 			for cnt in range(0, self.no_of_samples):	
 					A = np.matrix.dot(self.weight_new, self.training_data[:,cnt:cnt+1]) #predict output
 					self.weight_new += self.alpha * A.dot(np.transpose(self.training_data[:,cnt:cnt+1]))
   	 				A = np.matrix.dot(self.weight_new, self.training_data)
	 				self.error_rates[run] = self.calc_error_rate(self.target,A,self.no_of_samples)  
   	
    def calc_error_rate(self,given_target, given_output, no_of_samples):
   		max_out_indices = np.argmax(given_output,axis=0)
   		max_target_indices = np.argmax(given_target,axis=0)
   		return (no_of_samples-sum(max_out_indices==max_target_indices)) * 100 / float(no_of_samples) 	 
    	
if __name__=="__main__":
 #pwd = os.getcwd()
 filedr = sys.argv[1]
 onlyfiles = [f for f in listdir(filedr) if isfile(join(filedr, f))]
 shuffle(onlyfiles)
 num_samples = len(onlyfiles)
 num_params = read_one_image_and_convert_to_vector(filedr +'/'+onlyfiles[0]).shape[0]
 num_neurons = 10
 
 training_data = np.ones(shape=(num_params+1,num_samples),dtype=np.float64)
 target = -1 * np.ones(shape=(num_neurons,num_samples),dtype=np.float64)
 for i in range(0,len(onlyfiles)):
  ind_cl =  int(onlyfiles[i][0])
  vec1 = read_one_image_and_convert_to_vector(filedr +'/'+onlyfiles[i])
  target[ind_cl,i] = 1
  training_data[:num_params,i:i+1] = vec1 / 255.0
  
 nn_experiment_settings = {
        "min_initial_weights": -1.0,  # minimum initial weight
        "max_initial_weights": 1.0,  # maximum initial weight
        "no_of_params": num_params,  # number of inputs to the network
        "alpha": 0.001,  # learning rate
        "data_set": ClDataSet(training_data,target),
        'epoch': 200,
        "learning_method":"Delta rule",
        "no_of_samples": num_samples,
        "no_of_neurons": num_neurons,
        "gamma": 0.0,
        "initialize": True
    }
 main_frame = Tk.Tk()
 main_frame.title("Hebbian Learning")
 main_frame.geometry('640x480')
 ob_nn_gui_2d = ClNNGui2d(main_frame,nn_experiment_settings)
 main_frame.mainloop()        	
    	            
