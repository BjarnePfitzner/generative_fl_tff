# generative_fl_tff
Generative Models trained with federated learning using TensorFlow Federated.

# Setup
Code was executed using Python 3.8

Please run ```pip install -r requirements.txt```

# Code Execution
We use hydra experiment configuration. 
The config files can be found in the ```./config``` directory. 
They are split thematically and define the experiment structure 

# Notes
- Hydra mutliruns (```python main.py -m ...```) don't work without defining more sophisticated launchers (such as ```hydra-submitit```). The issue is that ```@tf.function``` decorated functions are already instantiated for the 2nd and further runs which throws an exception.