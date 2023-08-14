## AI generating music

I created an AI generator based on a script developed by [Andrej Karpathy](https://github.com/karpathy) on [this code](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing).

## How does it work?
- 2
 - You have these hyperparameters,
 
 | hyperparameter | value | explanation
 | --- | --- | --- |
 | batch_size | 16 | Batch size (number of examples) used when training the AI at each iteration.
 | block_size | 32 | The size of the block used to split data into sequences or batches. | 
 | max_iters | 1000000 | The maximum number of training iterations (steps) to be performed. | 
 | eval_interval | 10000 | How often AI performance evaluation is performed (in number of iterations). | 
 | learning_rate | 0.001 | Learning rate, which determines the size of adjustments made to model weights with each update. | 
 | eval_iters | 200 | Number of iterations between each performance assessment (e.g. loss calculation) of the AI. | 
 | n_embd | 64 | Number of units in embedding. | 
 | n_head | 4 | Number of heads in multi-head attention mechanisms. | 
 | n_layer | 4 | Total number of layers in the AI model. | 
 | dropout | 0.0 | Probability of deactivation of neurons during training, where 0 means no dropout. | 

 You can change these hyperparameters in the hyperparameters.json file.
