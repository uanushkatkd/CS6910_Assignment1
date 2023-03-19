# CS6910-Assignment-1
## Deep learning assignment 1
[Link to wandb report](https://api.wandb.ai/links/cs22s015/9q5djici)
In this assignment we need to perform classification task on fashion mnist dataset.
* ```Load_data()``` **function:**
 This takes string as an argument which specify the name of the dataset.It loads that dataset , do preprocessing steps which includes splitting of dataset into train and validation, normalization and <b>one hot encoding</b> for labels.It plot the first images in each class and  returns the list consisting of class names and preprocessed data.
* ```NN()``` **Class:**
 This takes all the arguments that are provided in code specification. It consists of saparate functions for <b>weight initialization</b> ,<b>forward propagation</b>, <b>back propagation</b>, <b>activation function</b> and <b>optimizers</b>. We can include any new optimizer or activation function by creating a function inside that class ```NN``` . 

### To run the code
* **Usage**
```
usage: train.py
       [-h]
       [-wp WANDB_PROJECT]
       [-we WANDB_ENTITY]
       [-d DATASET]
       [-e EPOCHS]
       [-b BATCH_SIZE]
       [-l LOSS]
       [-o OPTIMIZER]
       [-lr LEARNING_RATE]
       [-m MOMENTUM]
       [-beta BETA]
       [-beta1 BETA1]
       [-beta2 BETA2]
       [-eps EPSILON]
       [-w_d WEIGHT_DECAY]
       [-w_i WEIGHT_INIT]
       [-sz HIDDEN_SIZE]
       [-nhl NUM_LAYERS]
       [-a ACTIVATION]
  ```
* Run ``` python -m train.py ``` on cmd. It'll train the model for the best configuration obtained.
```
config={
        'epochs' :10 ,
        'n_hidden_layers' :5,
        'n_hidden_layer_size' :256,
        'batch_size' :64,
        'learning_rate' : 0.001,
        'optimizer' : "nadam",
        'activations' : "relu",
        'loss_function' : 'cross_entropy',
        'weight_ini' : 'he_uniform',
        'weight_decay' : 0
    }
```
