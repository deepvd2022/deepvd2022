"""
Example showing how to create experiment with Python code.

https://nni.readthedocs.io/en/stable/Tutorial/HowToLaunchFromPython.html
https://github.com/microsoft/nni/blob/9a4d0d6750/examples/trials/mnist-tfv2/launch.py

"""

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path

from nni.experiment import Experiment


search_space = {
    "batch_size": {"_type":"choice", "_value": [16, 32, 64]},
    "hidden_dim":{"_type":"choice","_value":[32, 64, 128]},
    "learning_rate":{"_type":"choice","_value":[0.001, 0.0001, 0.00001]},
    "dropout":{"_type":"choice","_value":[0.1, 0.2, 0.3, 0.4]}
}

experiment = Experiment('local')
experiment.config.experiment_name = 'DeepVD'
experiment.config.trial_concurrency = 1
experiment.config.max_trial_number = 2
experiment.config.search_space = search_space
experiment.config.trial_command = 'python main.py'
experiment.config.trial_code_directory = Path(__file__).parent
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize' # If ‘maximize’, the tuner will try to maximize metrics. If ‘minimize’, the tuner will try to minimize metrics.
experiment.config.training_service.use_active_gpu = True

experiment.run(8015)