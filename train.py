# %% [markdown]
# <a href="https://colab.research.google.com/github/uanushkatkd/CS6910-Assignment-1/blob/main/Dl1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
'''!pip install wandb -qU
import wandb
!wandb login 
'''
# %%
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.datasets import mnist

from sklearn.model_selection import train_test_split
import math

from tqdm import tqdm



# %%
'''

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="CS6910_Assignment_1",
)
# Loading the fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#class names for fashion-MNIST
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# creating 2x5 grid 
img={}

for i in range(10):
    # to find first image in the training set with class label i
    idx = np.where(y_train == i)[0][0]
    # Plot the image
    img[class_names[i]]=(wandb.Image(x_train[idx], caption=class_names[i]))
    
wandb.log(img)    
# [optional] finish the wandb run, necessary in notebooks
wandb.finish()
'''
# %%

# one hot encoding
#from keras.utils import to_categorical
def onehot_encoding(a,n_class):
  temp = []
  for i in a:
    t1 = np.zeros(n_class)
    t1[i] = 1
    temp.append(t1)
  temp=np.array(temp)
  return temp

# Loading the fashion-MNIST dataset

def load_data(dataset):
  if dataset=='fashion_mnist':
    (x_tr, y_tr), (x_test, y_test) = fashion_mnist.load_data()
    #class names for fashion-MNIST
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

  elif dataset=='mnist':
    (x_tr, y_tr), (x_test, y_test) = mnist.load_data()
    #class names for MNIST
    class_names = [0,1,2,3,4,5,6,7,8,9]

    

  x_train,x_val,y_train, y_val = train_test_split(x_tr,y_tr ,random_state=104,test_size=0.10, shuffle=True)
  # creating 2x5 grid 
  fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))
  ax1=ax.flat
  for i in range(10):
    # to find first image in the training set with class label i
    idx = np.where(y_train == i)[0][0]
    # Plot the image
    ax1[i].imshow(x_train[idx], cmap='gray')
    ax1[i].set_xlabel(class_names[i])
    ax1[i].set_xticklabels([])
    ax1[i].set_yticklabels([])
    # Display the plot
  plt.show()
  # Normalize data
  x_train,x_val,x_test= x_train/255.0,x_val/255.0,x_test/255.0
  #one hot encoding
  #Reshaping train and test data
  x_train,x_val,x_test=x_train.reshape(len(x_train),28*28),x_val.reshape(len(x_val),28*28),x_test.reshape(len(x_test),28*28)
  # one hot encoding
  y_train=onehot_encoding(y_train,10)  #run once only
  y_val = onehot_encoding(y_val,10)
  y_test = onehot_encoding(y_test,10)
  print(y_val)
  #y_test = to_categorical(y_test, dtype ="uint8")
  # Labels after applying the function
  # Training set labels
  #print(x_train.shape)
  print(y_val.shape)
  
  return x_train,x_test,x_val,y_train,y_test,y_val,class_names


'''# %%
hidden_layer=[256,128,64]
no_of_class=10
layer_dim=[x_train.shape[1]]+hidden_layer+[no_of_class]
print(layer_dim)
def initialise_params(train,label,layers):

  params={}
  L=len(layers)
  w=[]
  b=[]
  for i in range(1,L):
    params['W'+str(i)]= np.random.randn(layers[i], layers[i-1]) * 0.05
    params['b' + str(i)] =  np.zeros((layers[i], 1))

  return params  

def forward_prop(train,label,layers,params):
  L=len(layers)
  a={}
  h={}
  h['h'+str(0)]=(train.T)
  
  for i in range(1,L-1):
    #preactivation calculation
    print(i)
    a['a'+str(i)]= params['W'+str(i)] @ h['h'+str(i-1)]+ params['b'+str(i)]
    
    #activation calculation
    h['h'+str(i)]=sigmoid(a['a'+str(i)])
    
  a['a'+str(L-1)]= params['W'+str(L-1)] @ (h['h'+str(L-2)]) +params['b'+str(L-1)]
  y_prob=[]
  for i in range(len(a['a'+str(L-1)][0])):
    y_prob.append(softmax(a['a'+str(L-1)][:,i]))
  y_prob=np.array(y_prob)
  h['h'+str(L-1)]=y_prob
  
  return a,h,y_prob

def sigmoid(x):
  return 1/(1+np.exp(-x))

def softmax(x):
   return (np.exp(x)/np.exp(x).sum())



p=initialise_params(x_train,y_train,layer_dim)
for key,val in p.items():
  print(key,'->',val.shape)
  
a1,h1,y_h=forward_prop(x_train[:10,:],y_train[:10,:],layer_dim,p)
print((y_h[0]))
for key,val in a1.items():
  print(key,'->',val.shape)
for key,val in h1.items():
  print(key,'->',val.shape)

#print(y_train)

# %%

#y_train=y_train[:5,:]
#x_train=x_train[:5,:]
#print(y_train)


'''
np.random.seed(1)
class NN:
  def __init__(self,layers,epochs,lr,activation_func,loss_func,optimizer,initialize,batch_size,dataset,m,beta,beta1,beta2,epsilon,weight_decay ):
    self.layers = layers
    self.epochs = epochs
    self.lr = lr
    self.activation_func=activation_func
    self.loss_func=loss_func
    self.optimizer=optimizer
    self.initialize=initialize
    self.weight_decay=weight_decay
    self.batch_size=batch_size
    self.dataset=dataset
    self.m=m
    self.beta=beta
    self.beta1=beta1
    self.beta2=beta2
    self.epsilon=epsilon
    self.params=self.initialise_params()
    self.L=len(self.layers)

    
  def initialise_params(self):
    params={}
    L=len(self.layers)
    
    for i in range(1,L):
      if self.initialize=='random':
        params['W'+str(i)]= np.random.randn(self.layers[i], self.layers[i-1]) * 0.1
      elif self.initialize=='xavier':
        params['W'+str(i)]= np.random.randn(self.layers[i], self.layers[i-1]) * np.sqrt(2/ (self.layers[i - 1] + self.layers[i]))
      elif self.initialize=='he_normal':
         params['W' + str(i)] = np.random.randn(self.layers[i], self.layers[i-1]) * np.sqrt(2/self.layers[i-1])
      elif self.initialize=='he_uniform':
         params['W' + str(i)] = np.random.uniform(low=-np.sqrt(6 / self.layers[i-1]), high=np.sqrt(6 /self.layers[i-1]), size=(self.layers[i], self.layers[i-1]))
  
      params['b' + str(i)] =  np.zeros((self.layers[i], 1))

    return params  
  
  def updates(self):
    updates={}
    L=len(self.layers)
   
    for i in range(1,self.L):
      updates['W'+str(i)]= np.zeros((self.layers[i], self.layers[i-1])) 
      updates['b' + str(i)] =  np.zeros((self.layers[i], 1))

    return updates  

  def sigmoid(self, x, derivative=False):
    if derivative:
      return (np.exp(-x))/((np.exp(-x)+1)**2)
    return 1/(1 + np.exp(-x))

  def identity(self, x, derivative=False):
    if derivative:
      return 1
    return x

  def tan_h(self, x, derivative=False):
    t=np.tanh(x)
    if derivative:
      return 1-t**2
    return t

  def relu(self, x, derivative=False):
    if derivative:
      return 1*(x>0)
    return np.maximum(0,x)


  def softmax(self, x, derivative=False):
    exps = np.exp(x - x.max())
    if derivative:
      return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    return exps/np.sum(exps, axis=0)
  
  def regularization_loss(self):
    # Calculate the L2 regularization loss
    L=len(self.layers)
    regularization_loss = 0.0
    for i in range(1,L):
      regularization_loss += np.sum(np.square(self.params['W'+str(i)]))
    regularization_loss *= self.weight_decay
    return regularization_loss

  def forward_prop(self,train):
    params = self.params
    L=len(self.layers)
   
    a={}
    h={}
    train=train.T
    h['h'+str(0)]=train.reshape(len(train),1)
    for i in range(1,L-1):
      #preactivation calculation
      #print(i)
      a['a'+str(i)]= params['W'+str(i)] @ h['h'+str(i-1)]+ params['b'+str(i)]
        #activation calculation
      if self.activation_func=='tanh':
        h['h'+str(i)]=self.tan_h(a['a'+str(i)])
      elif self.activation_func=='sigmoid':
        h['h'+str(i)]=self.sigmoid(a['a'+str(i)])
      elif self.activation_func=='relu':
        h['h'+str(i)]=self.relu(a['a'+str(i)])
      elif self.activation_func=='identity':
        h['h'+str(i)]=self.identity(a['a'+str(i)])


    a['a'+str(L-1)]= params['W'+str(L-1)] @ (h['h'+str(L-2)]) +params['b'+str(L-1)]
    y_prob=[]
    for i in range(len(a['a'+str(L-1)][0])):
      y_prob.append(self.softmax(a['a'+str(L-1)][:,i]))
    y_prob=np.array(y_prob)
    h['h'+str(L-1)]=y_prob
  
    return a,h,y_prob
  
  def backward_prop(self, y_train, y_hat,a,h):
    params = self.params
    delta_params = {}
    L=len(self.layers)
   
    y_train=y_train.reshape(len(y_train),1)
    # Compute output gradient
        # Gardient with respect to last layer
    if self.loss_func == 'cross_entropy':
      delta_params['a' + str(L-1)] = (y_hat - y_train)
    elif self.loss_func == 'squared_error':
      delta_params['a' + str(L-1)] = (y_hat - y_train)*y_hat*(1-y_hat)
    
    
    for i in range(L-1,0,-1):
      #gradients w rt parameters 
      delta_params['W' + str(i)]=(delta_params['a' + str(i)]@(h['h'+str(i-1)].T))+self.weight_decay*params['W'+str(i)]
      delta_params['b' + str(i)]=np.sum(delta_params['a' + str(i)],axis=1,keepdims=True)

      #gradients w rt layer below

      delta_params['h' + str(i-1)]=(params['W' + str(i)].T)@ delta_params['a' + str(i)]

      #gradients w rt layer below(preactivation)

      if i > 1:
        if self.activation_func=='tanh':
          delta_params['a' + str(i-1)] = delta_params['h' + str(i-1)] * self.tan_h(a['a' + str(i-1)], derivative=True)  
        elif self.activation_func=='sigmoid':
          delta_params['a' + str(i-1)] = delta_params['h' + str(i-1)] * self.sigmoid(a['a' + str(i-1)], derivative=True)  
        elif self.activation_func=='relu':
          delta_params['a' + str(i-1)] = delta_params['h' + str(i-1)] * self.relu(a['a' + str(i-1)], derivative=True)  
        elif self.activation_func=='identity':
          delta_params['a' + str(i-1)] = delta_params['h' + str(i-1)] * self.identity(a['a' + str(i-1)], derivative=True)  


    return delta_params
  
  
  def loss_fun(self,y,y_hat):
    if self.loss_func == 'cross_entropy':
      i=np.argmax(y)
      p=y_hat[i]
      loss=-np.log(p)+self.regularization_loss()
      return loss
    elif self.loss_func == 'squared_error':
      return np.sum((y-y_hat)**2)+self.regularization_loss()
 
 
  def modelPerformance(self, x_test, y_test):
    predictions = []
    y_true = []
    y_pred = []
    losses = []
    for x,y in tqdm(zip(x_test ,y_test), total=len(x_test)):
      a,h,y_p = self.forward_prop(x)
      predictedClass = np.argmax(y_p)
      y.reshape(len(y),1)
      actualClass = np.argmax(y)
      y_true.append(actualClass)
      y_pred.append(predictedClass)
      predictions.append(predictedClass == actualClass)
      losses.append(self.loss_fun(y.T,y_p.T))
    accuracy = (np.sum(predictions)*100)/len(predictions)
    loss = np.sum(losses)/len(losses)
    
    return accuracy, loss, y_true, y_pred

     
  def sgd(self,x_train,y_train,x_test,y_test,x_val,y_val):
    weights=self.params
    e=self.epochs
    for i in range(e):
      t=0
      dw_db=self.updates()
      print("epoch",i+1)
      for x,y in (zip(x_train,y_train)):
        a,h,y_p=self.forward_prop(x)
        
        delta_theta=self.backward_prop(y.T,y_p.T,a,h)
        
        for i in dw_db:
           dw_db[i]+=delta_theta[i]

        t=t+1
        if (t%self.batch_size==0):
          for key in weights:
            weights[key]=weights[key] - (self.lr)*dw_db[key]
          
          dw_db=self.updates()

      
      val_acc,val_loss,y_true1,y_perd1=self.modelPerformance(x_val,y_val)
      print("Val Accuracy = " + str(val_acc))
      print("Val Loss = " + str(val_loss))

      train_acc,train_loss,y_true2,y_perd2=self.modelPerformance(x_train,y_train)
      print("Train Accuracy = " + str(train_acc))
      print("Train Loss = " + str(train_loss))

      
      

      #wandb.log({"val_acc": val_acc, "train_acc": train_acc, "test_acc": test_acc, "val_loss": val_loss, "train_loss": train_loss,"test_loss":test_loss,"epoch": e+1})
	
    self.params=weights
    return weights

  def momentum(self,x_train,y_train,x_test,y_test,x_val,y_val):
    beta=0.9
    
    weights=self.params
    
    update= self.updates()
    e=self.epochs
    for i in range(e):
      t=0
      dw_db=self.updates()
      lookahead=self.updates()
      print("epoch",i+1)
      for x,y in (zip(x_train,y_train)):
        a,h,y_p=self.forward_prop(x)
        
        delta_theta=self.backward_prop(y.T,y_p.T,a,h)
        
        for i in dw_db:
           dw_db[i]+=delta_theta[i]

        t=t+1
        if (t%self.batch_size==0):
          for key in lookahead:
            lookahead[key]=beta*update[key]+self.lr*dw_db[key]
          
          for key in weights:
           weights[key]=weights[key] - lookahead[key]

          
          for key in update:
             update[key] =lookahead[key]
          
          dw_db=self.updates()
  

      
      val_acc,val_loss,y_true1,y_perd1=self.modelPerformance(x_val,y_val)
      print("Val Accuracy = " + str(val_acc))
      print("Val Loss = " + str(val_loss))

      train_acc,train_loss,y_true2,y_perd2=self.modelPerformance(x_train,y_train)
      print("Train Accuracy = " + str(train_acc))
      print("Train Loss = " + str(train_loss))

      
      

      #wandb.log({"val_acc": val_acc, "train_acc": train_acc, "test_acc": test_acc, "val_loss": val_loss, "train_loss": train_loss,"test_loss":test_loss,"epoch": e+1})
	
    self.params=weights
    return weights

    
   
     
  def nestrov(self,x_train,y_train,x_test,y_test,x_val,y_val):

    
    beta=0.9
    
    weights=self.params
    update= self.updates()
    
    e=self.epochs
    for i in range(e):
      dw_db= self.updates()

      lookahead= self.updates()

      #do partial updates
      for key in lookahead:
        lookahead[key]=beta*update[key]
    
      t=0
   
      print("epoch",i+1)
      for x,y in (zip(x_train,y_train)):
        
   
        a,h,y_p=self.forward_prop(x)
        
        delta_theta=self.backward_prop(y.T,y_p.T,a,h)
      
        for i in dw_db:
           dw_db[i]+=delta_theta[i]

        t=t+1
        if (t%self.batch_size==0):
          for key in lookahead:
            lookahead[key]=beta*update[key]+self.lr*dw_db[key]
          
          for key in weights:
           weights[key]=weights[key] - lookahead[key]

          
          for key in update:
             update[key] =lookahead[key]
          dw_db=self.updates()

       
      val_acc,val_loss,y_true1,y_perd1=self.modelPerformance(x_val,y_val)
      print("Val Accuracy = " + str(val_acc))
      print("Val Loss = " + str(val_loss))

      train_acc,train_loss,y_true2,y_perd2=self.modelPerformance(x_train,y_train)
      print("Train Accuracy = " + str(train_acc))
      print("Train Loss = " + str(train_loss))

      
      

      #wandb.log({"val_acc": val_acc, "train_acc": train_acc, "test_acc": test_acc, "val_loss": val_loss, "train_loss": train_loss,"test_loss":test_loss,"epoch": e+1})
	
    self.params=weights
    return weights
       

  def rmsprop(self,x_train,y_train,x_test,y_test,x_val,y_val):

    
    beta=0.9
    eps=1e-8
    
    weights=self.params
    update= self.updates()
    
    e=self.epochs
    for i in range(e):
      t=0
    
      dw_db= self.updates()

      
      print("epoch",i+1)
      for x,y in (zip(x_train,y_train)):
        
   
        a,h,y_p=self.forward_prop(x)
        
        #print("y_hat",y[0])
        delta_theta=self.backward_prop(y.T,y_p.T,a,h)
       
        for i in dw_db:
           dw_db[i]+=delta_theta[i]

        t=t+1
        if (t%self.batch_size==0):
          for key in update:
            update[key]=beta*update[key]+(1-beta)*(dw_db[key]**2)
          
          for key in weights:
           weights[key]=weights[key] - (self.lr/np.sqrt(update[key]+eps))*dw_db[key]
          
          
          dw_db=self.updates()

       
      val_acc,val_loss,y_true1,y_perd1=self.modelPerformance(x_val,y_val)
      print("Val Accuracy = " + str(val_acc))
      print("Val Loss = " + str(val_loss))

      train_acc,train_loss,y_true2,y_perd2=self.modelPerformance(x_train,y_train)
      print("Train Accuracy = " + str(train_acc))
      print("Train Loss = " + str(train_loss))

      
      

      #wandb.log({"val_acc": val_acc, "train_acc": train_acc, "test_acc": test_acc, "val_loss": val_loss, "train_loss": train_loss,"test_loss":test_loss,"epoch": e+1})
	
    self.params=weights
    return weights
    
  def adam(self,x_train,y_train,x_test,y_test,x_val,y_val):

    beta1=0.9
    beta2=0.999
    eps=1e-8
    
    weights=self.params
    mw_mb= self.updates()
    vw_vb=self.updates()
    
    mw_mb_hat= self.updates()
    vw_vb_hat=self.updates()
    
    e=self.epochs
    for i in range(e):
      t=0

      dw_db= self.updates()

      
      print("epoch",i+1)
      for x,y in (zip(x_train,y_train)):
        
   
        a,h,y_p=self.forward_prop(x)
        
        #print("y_hat",y[0])
        delta_theta=self.backward_prop(y.T,y_p.T,a,h)
       
        for j in dw_db:
           dw_db[j]+=delta_theta[j]

        t=t+1
        if (t%self.batch_size==0):
          for key in mw_mb:
            mw_mb[key]=beta1*mw_mb[key]+(1-beta1)*(dw_db[key])

          for key in vw_vb:
            vw_vb[key]=beta2*vw_vb[key]+(1-beta2)*(dw_db[key]**2)

          for key in weights:
            mw_mb_hat[key]=mw_mb[key]/(1-(beta1**(i+1)))
            vw_vb_hat[key]=vw_vb[key]/(1-(beta2**(i+1)))
            

          for key in weights:
           weights[key]=weights[key] - (self.lr/np.sqrt(vw_vb_hat[key]+eps))*mw_mb_hat[key]

          
          dw_db=self.updates()

       
      val_acc,val_loss,y_true1,y_perd1=self.modelPerformance(x_val,y_val)
      print("Val Accuracy = " + str(val_acc))
      print("Val Loss = " + str(val_loss))

      train_acc,train_loss,y_true2,y_perd2=self.modelPerformance(x_train,y_train)
      print("Train Accuracy = " + str(train_acc))
      print("Train Loss = " + str(train_loss))

      
     

      #wandb.log({"val_acc": val_acc, "train_acc": train_acc, "test_acc": test_acc, "val_loss": val_loss, "train_loss": train_loss,"test_loss":test_loss,"epoch": e+1})
	
    self.params=weights
    return weights
         
 

     
    
  def nadam(self,x_train,y_train,x_test,y_test,x_val,y_val): #update nadam

    beta1=0.9
    beta2=0.999
    eps=1e-8
    
    weights=self.params
    update= self.updates()
  
    mw_mb= self.updates()
    vw_vb=self.updates()
    mw_mb_hat= self.updates()
    vw_vb_hat=self.updates()
    
    beta=0.95
    e=self.epochs
    for i in range(e):
      dw_db= self.updates()
      lookahead=self.updates()

      t=0

      #do partial updates
      for key in lookahead:
        lookahead[key]=beta*update[key]

     #one point to add here same in nestrov .check all the algos once   


      print("epoch",i+1)
      for x,y in (zip(x_train,y_train)):
        
   
        a,h,y_p=self.forward_prop(x)
        
        #print("y_hat",y[0])
        delta_theta=self.backward_prop(y.T,y_p.T,a,h)
       
        for j in dw_db:
           dw_db[j]+=delta_theta[j]

        t=t+1
        if (t%self.batch_size==0):
          
          for key in lookahead:
            lookahead[key]=beta*update[key]+self.lr*dw_db[key]

          for key in mw_mb:
            mw_mb[key]=beta1*mw_mb[key]+(1-beta1)*(dw_db[key])

          for key in vw_vb:
            vw_vb[key]=beta2*vw_vb[key]+(1-beta2)*(dw_db[key]**2)

          for key in weights:
            mw_mb_hat[key]=mw_mb[key]/(1-(beta1**(i+1)))
            vw_vb_hat[key]=vw_vb[key]/(1-(beta2**(i+1)))
            

          for key in weights:
           weights[key]=weights[key] - (self.lr/np.sqrt(vw_vb_hat[key]+eps))*mw_mb_hat[key]
          
          
          for key in update:
             update[key] =lookahead[key]

          dw_db=self.updates()       
      
      val_acc,val_loss,y_true1,y_perd1=self.modelPerformance(x_val,y_val)
      print("Val Accuracy = " + str(val_acc))
      print("Val Loss = " + str(val_loss))

      train_acc,train_loss,y_true2,y_perd2=self.modelPerformance(x_train,y_train)
      print("Train Accuracy = " + str(train_acc))
      print("Train Loss = " + str(train_loss))

      

      #wandb.log({"val_acc": val_acc, "train_acc": train_acc, "test_acc": test_acc, "val_loss": val_loss, "train_loss": train_loss,"test_loss":test_loss,"epoch": e+1})
	
    self.params=weights
    return weights
 

  def fit(self,x_train,y_train,x_test,y_test,x_val,y_val):
    if self.optimizer == 'sgd':
      w=self.sgd(x_train,y_train,x_test,y_test,x_val,y_val)
    elif self.optimizer == 'mgd':
      w=self.momentum(x_train,y_train,x_test,y_test,x_val,y_val)
    elif self.optimizer == 'nestrov':
      w=self.nestrov(x_train,y_train,x_test,y_test,x_val,y_val)
    elif self.optimizer == 'rmsprop':
      w=self.rmsprop(x_train,y_train,x_test,y_test,x_val,y_val)
    elif self.optimizer == 'adam':
      w=self.adam(x_train,y_train,x_test,y_test,x_val,y_val)
    elif self.optimizer == 'nadam':
      w=self.nadam(x_train,y_train,x_test,y_test,x_val,y_val)

      




'''hidden_layer=[256,256,256,256,256]
no_of_class=10
layer_dim=[x_train.shape[1]]+hidden_layer+[no_of_class]
print(layer_dim)
lr=0.001
#nn = NN(layer_dim,10,0.001,activation_func='tanh',loss_func='cross_entropy',optimizer='nadam',initialize='xavier',weight_decay=0.005,batch_size=32)

#nn.fit(x_train,y_train,x_test,y_test,x_val,y_val)

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
nn = NN(layer_dim,10,0.001,'relu','cross_entropy','nadam','he_uniform',0,64)

nn.fit(x_train,y_train,x_test,y_test,x_val,y_val)

'''


# %%
'''test_acc,test_loss,y_true,y_pred=nn.modelPerformance(x_test,y_test)

print("Train Accuracy = " + str(test_acc))
print("Train Loss = " + str(test_loss))
class_names1 = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


cm = confusion_matrix(y_true,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=class_names1)


fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax)
plt.show()
'''

'''# %%
sweep_config={
    'method' : 'bayes' ,
    'metric' : { 'name' : 'val_acc' , 'goal' : 'maximize' } ,
    'parameters' : {
        'epochs' : { 'values' : [5,10,20] },
        'n_hidden_layers' : {'values' : [3,4,5]},
        'n_hidden_layer_size' : { 'values' : [16,32,64,128,256]},
        'batch_size' : { 'values' : [16,32,64,128]},
        'learning_rate' : { 'values' : [0.001, 0.0001,0.0002,0.0003]},
        'optimizer' : { 'values' : ["sgd", "mgd", "nestrov", "rmsprop", "adam", "nadam"] },
        'activations' : { 'values' : ["sigmoid", "tanh", "relu"] },
        'loss_function' : {'values' : ['cross_entropy']},
        'weight_ini' : {'values' : ['random','xavier','he_normal','he_uniform']},
        'weight_decay' : { 'values' : [0,0.0005]}
    }
}

def train():
  config_default={
      'weight_ini':'bayes',
      'n_hidden_layers':3,
      'n_hidden_layer_size':32,
      'optimizer':'sgd',
      'learning_rate':0.01,
      'epoch':10,
      'batch_size':32
  }
  wandb.init(config=config_default)

  c= wandb.config
  name = "op_"+str(c.optimizer)+"_ac_"+str(c.activations)+"_hl_"+str(c.n_hidden_layers)+"_hls_"+str(c.n_hidden_layer_size)+"_ep_"+str(c.epochs)+"_n_"+str(c.learning_rate)+"_bs_"+str(c.batch_size)+"_wi_"+str(c.weight_ini)
  wandb.init(name=name)
  n_points , n_input = np.shape(x_train)

  hn = [n_input]+[c.n_hidden_layer_size]*c.n_hidden_layers +[no_of_class] 
  hl = c.n_hidden_layers
  act = c.activations
  loss=c.loss_function
  opt = c.optimizer
  ep = c.epochs
  bs = c.batch_size
  lr = c.learning_rate
  wi = c.weight_ini
  wd=c.weight_decay

 
  nn = NN(hn,ep,lr,act,loss,opt,wi,wd,bs)

  nn.fit(x_train,y_train,x_test,y_test,x_val,y_val)

  return
sweep_id = wandb.sweep(sweep_config, project="CS6910_Assignment_1")
wandb.agent(sweep_id, function=train,count=20)
'''
# %%
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-wp','--wandb_project', type=str, default='CS6910_Assignment_1')
parser.add_argument('-we','--wandb_entity', type=str, default='cs22s015')
parser.add_argument('-d','--dataset', type=str, default='fashion_mnist')
parser.add_argument('-e','--epochs', type=int, default=10)
parser.add_argument('-b','--batch_size', type=int, default=64)
parser.add_argument('-l','--loss', type=str, default='cross_entropy')
parser.add_argument('-o','--optimizer', type=str, default='nadam')
parser.add_argument('-lr','--learning_rate', type=float, default=0.001)
parser.add_argument('-m','--momentum', type=float, default=0.9)
parser.add_argument('-beta','--beta', type=float, default=0.95)
parser.add_argument('-beta1','--beta1', type=float, default=0.9)
parser.add_argument('-beta2','--beta2', type=float, default=0.999)
parser.add_argument('-eps','--epsilon', type=float, default=1e-8)
parser.add_argument('-w_d','--weight_decay', type=float, default=0)
parser.add_argument('-w_i','--weight_init', type=str, default='he_uniform')
parser.add_argument('-sz','--hidden_size', type=int, default=256)
parser.add_argument('-nhl','--num_layers', type=int, default=5)
parser.add_argument('-a','--activation', type=str, default='relu')
args = parser.parse_args()

if __name__=='__main__':
  layers=[]
  layers.append(784)
  num_layers=args.num_layers
  hlayer_size=args.hidden_size
  for i in range(num_layers):
    layers.append(hlayer_size)
  layers.append(10)


  nn = NN(layers,args.epochs,args.learning_rate,args.activation,args.loss,args.optimizer,args.weight_init,args.batch_size,args.dataset,args.momentum,args.beta,args.beta1,args.beta2,args.epsilon,args.weight_decay)
  x_train,x_test,x_val,y_train,y_test,y_val,class_names=load_data(args.dataset)
  nn.fit(x_train,y_train,x_test,y_test,x_val,y_val)

  # Testing
  test_acc, test_loss, y_true, y_pred = nn.modelPerformance(x_test, y_test)
  print("Testing Accuracy = " + str(test_acc))
  print("Testing Loss = " + str(test_loss))
  #wandb.log({"test_acc": test_acc})
  #wandb.log({"Confusion_Matrix": wandb.sklearn.plot_confusion_matrix(y_true, y_pred, lab)})



