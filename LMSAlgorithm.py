# Kashyap, Sreesha
# Least Mean Square Learning Algorithms implementation to predict Stock prices and Volume

import numpy as np
import Tkinter as Tk
import matplotlib,sys,math
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import colorsys

def read_csv_as_matrix(file_name):
    data = np.loadtxt(file_name, skiprows=1, delimiter=',', dtype=np.float64)
    return data
    
class ClNNGui2d:
   def __init__(self, master,settings):
        self.__dict__.update(settings)
        self.master = master
        self.xmin = 0
        self.ymin = 0
        self.ymax = 2
        self.xmax = 300
        self.master.update()
        # network data structures
        self.sample_size = math.ceil((self.sample_size_percentage / 100.0) * self.no_of_samples)
        self.training_data = self.training_samples[:,:self.sample_size]
        self.max_price ,self.max_vol = np.max(self.training_data,axis=1)
        self.training_data[0] = self.training_data[0] / self.max_price
  	    self.training_data[1] = self.training_data[1] / self.max_vol
  	    self.weights = np.random.uniform(self.min_initial_weights,self.max_initial_weights,(self.no_of_neurons,2*(self.no_of_delayed_elements + 1)+1))
  	    self.mse_p,self.mse_vol, self.max_abs_p,self.max_abs_vol = [],[],[],[]
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
        self.figure = plt.figure("LMS algorithm")
        self.axes = self.figure.add_subplot(111)
        plt.title("LMS algorithm")
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
                                             highlightcolor="#00FFFF", width=10,length=800,
                                             command=lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.set(self.alpha)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.iterations_slider_label = Tk.Label(self.sliders_frame, text="No. of iterations")
        self.iterations_slider_label.grid(row=3, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.iterations_value_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=1.0, to_=10, resolution=1,bg="#DDDDDD",
                                             activebackground="#AF0000",
                                             highlightcolor="#00FFEF", width=10,length=800,
                                             command=lambda event: self.iterations_value_slider_callback())
        self.iterations_value_slider.set(self.no_of_iterations)
        self.iterations_value_slider.bind("<ButtonRelease-1>", lambda event: self.iterations_value_slider_callback())
        self.iterations_value_slider.grid(row=4, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
	    self.delayed_elements_slider_label = Tk.Label(self.sliders_frame, text="No. of delayed elements")
        self.delayed_elements_slider_label.grid(row=5, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.delayed_elements_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=1.0, to_=10, resolution=1,bg="#DDDDDD",
                                             activebackground="#AF0000",
                                             highlightcolor="#00FFEF", width=10,length=800,
                                             command=lambda event: self.delayed_elements_slider_callback())
        self.delayed_elements_slider.set(self.no_of_delayed_elements)
        self.delayed_elements_slider.bind("<ButtonRelease-1>", lambda event: self.delayed_elements_slider_callback())
        self.delayed_elements_slider.grid(row=6, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.sample_size_slider_label = Tk.Label(self.sliders_frame, text="sample size percentage")
        self.sample_size_slider_label.grid(row=7, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.sample_size_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=1.0, to_=100, resolution=1,bg="#DDDDDD",
                                             activebackground="#AF0000",
                                             highlightcolor="#00FFEF", width=10,length=800,
                                             command=lambda event: self.sample_size_slider_callback())
        self.sample_size_slider.set(self.sample_size_percentage)
        self.sample_size_slider.bind("<ButtonRelease-1>", lambda event: sample_size_slider_callback())
        self.sample_size_slider.grid(row=8, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.batch_size_slider_label = Tk.Label(self.sliders_frame, text="batch size")
        self.batch_size_slider_label.grid(row=9, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.batch_size_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=1.0, to_=self.sample_size, resolution=1,bg="#DDDDDD",
                                             activebackground="#AF0000",
                                             highlightcolor="#00FFEF", width=10,length=800,
                                             command=lambda event: self.batch_size_slider_callback())
        self.batch_size_slider.set(self.batch_size)
        self.batch_size_slider.bind("<ButtonRelease-1>", lambda event: self.batch_size_slider_callback())
        self.batch_size_slider.grid(row=10, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        #########################################################################
        #  Set up the frame for button(s)
        #########################################################################

        self.buttons_frame = Tk.Frame(self.master)
        self.buttons_frame.grid(row=1,column=1)
        self.buttons_frame.rowconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1,uniform='xx')
        self.set_weights_to_zero_button = Tk.Button(self.buttons_frame,
                                               text="Set weights to Zero",
                                               bg="yellow", fg="red",
                                               command=lambda: self.set_weights_to_zero_button_callback())
        self.set_weights_to_zero_button.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.adjust_weights_button = Tk.Button(self.buttons_frame,
                                               text="Adjust Weights (Learn)",
                                               bg="yellow", fg="red",
                                               command=lambda: self.adjust_weights_button_callback())
        self.adjust_weights_button.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        #self.initialize()
        

    #def initialize(self):
    #    self.nn_experiment.neural_network.randomize_weights()

    def refresh_display(self):
        x_axis =range(1,len(self.mse_p)+1)
        self.axes.cla()
        #self.axes.clf()
        self.xmax = math.ceil(self.sample_size * self.no_of_iterations / self.batch_size)+50
        print len(x_axis),len(self.mse_p)
        plt.plot(x_axis,self.mse_p,'r--',label="Mean squared error- price")
      	plt.plot(x_axis, self.mse_vol, 'b--',label = "Mean squared error- volume")
      	plt.plot(x_axis,self.max_abs_p,'r-',label="Maximum absolute error- price")
      	plt.plot(x_axis,self.max_abs_vol,'b-',label="Maximum absolute error- volume")
        self.axes.xaxis.set_visible(True)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        plt.xlabel("no of batches")
        plt.ylabel("error")
        plt.title("LMS Algorithms")
        plt.legend()
        self.canvas.draw()
        plt.gcf().clear()

    def learning_rate_slider_callback(self): # NEEDS TO BE CORRECTED
        self.alpha = self.learning_rate_slider.get()
                
        

    def adjust_weights_button_callback(self):
        temp_text = self.adjust_weights_button.config('text')[-1]
        self.adjust_weights_button.config(text='Please Wait')
        self.leastMeanSquare() #returns error_rates
        self.refresh_display()
        self.adjust_weights_button.config(text=temp_text)
        self.adjust_weights_button.update_idletasks()
    
    def delayed_elements_slider_callback(self):
        self.no_of_delayed_elements = self.delayed_elements_slider.get()
        self.weights = np.random.uniform(self.min_initial_weights,self.max_initial_weights,(self.no_of_neurons,2*(self.no_of_delayed_elements + 1)+1))   
    def iterations_value_slider_callback(self):
        self.no_of_iterations = self.iterations_value_slider.get()
    
    def sample_size_slider_callback(self):
    	self.sample_size_percentage = self.sample_size_slider.get()
        self.sample_size = self.sample_size = math.ceil((self.sample_size_percentage / 100.0) * self.no_of_samples)
    
    def batch_size_slider_callback(self):
    	self.batch_size = self.batch_size_slider.get()
    
    def set_weights_to_zero_button_callback(self):
    	self.weights *= 0
    
    def calc_error_rates(self, first_index, last_index):
    '''

    '''
   		total_input = self.no_of_delayed_elements +1
   		mse = np.zeros(shape=(2,1),dtype = np.float64)
   		max_p,max_vol =0,0
   		for i in range(first_index, last_index+1):
   	  		delayed_elements = np.reshape(np.append(self.training_data[:,i-total_input:i].flatten(),[1]),(2*(total_input)+1,1))
   	  		#print delayed_elements
   	  		output = self.weights.dot(delayed_elements)
   	  		error = (self.training_data[:,i:i+1] - output) #* np.array([[self.max_price],[self.max_vol]])
   	  		mse += (error)**2
   	  		abs_err = np.abs(error )
   	  		if( abs_err[0,0] > max_p):
   	  			max_p = abs_err[0,0]
   	  		if( abs_err[1,0] > max_vol):
   	     		max_vol = abs_err[1,0]
   		self.mse_p.append(mse[0,0]*10/(last_index-first_index+1))   	
   		self.mse_vol.append(mse[1,0]*10/(last_index-first_index+1))
   		self.max_abs_p.append(max_p)
   		self.max_abs_vol.append(max_vol)
   	 
    def leastMeanSquare(self):
    '''
    Function implements the LMS algorithm. Trains the Neural Network to prediction of Volume and price of Stocks in market.
    '''
     for count in range(self.no_of_iterations):
      first_index = self.no_of_delayed_elements+1
      last_index = self.batch_size - 1
      total_input = self.no_of_delayed_elements +1 
      for i in range(first_index,self.training_data.shape[1]):
       delayed_elements = np.reshape(np.append(self.training_data[:,i-total_input:i].flatten(),[1]),(2*(total_input)+1,1))
       output = self.weights.dot(delayed_elements)
       err = self.training_data[:,i:i+1] - output
       self.weights += 2 * self.alpha * err.dot(np.transpose(delayed_elements))
       if( i == last_index ):
        self.calc_error_rates(first_index,last_index)
        first_index = last_index +1
        if ( self.training_data[:,first_index:].shape[1] < self.batch_size):
         last_index = first_index + self.training_data[:,first_index:].shape[1]-1
        else:
         last_index = first_index + self.batch_size -1	

if(__name__ == "__main__"):
 train_set = read_csv_as_matrix(sys.argv[1])
 nn_experiment_settings = {
        "min_initial_weights": -1.0,  # minimum initial weight
        "max_initial_weights": 1.0,  # maximum initial weight
        "number_of_params": 2,  # number of inputs to the network
        "alpha": 0.001,  # learning rate
        "training_samples": np.transpose(train_set),
        "no_of_samples": train_set.shape[0],
        "no_of_neurons": 2,
        "no_of_iterations":10,
        "no_of_delayed_elements": 7,
        "sample_size_percentage":100,
        "batch_size": 100
    }
 main_frame = Tk.Tk()
 main_frame.title("LMS Learning")
 main_frame.geometry('640x480')
 ob_nn_gui_2d = ClNNGui2d(main_frame, nn_experiment_settings)
 main_frame.mainloop()    	
