<p align="center">
    <br>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers_logo_name.png" width="400"/>
    <br>
<p>

<p align="center">
<br>
  <img src="https://assets.website-files.com/5ac6b7f2924c656f2b13a88c/6066c22135b8983b61ad7939_weights-and-biases-logo.svg" alt="" width="380">
    <br>
</p>

# NLP Project Template

This directory contains a template for NLP projects in Huggingface [Transformers](https://huggingface.co/) library.
The template also includes bash scripts to support execution of experiments in scale using [Wandb](https://wandb.ai/site) logging and sweeps.

## Template Basics

### Setup

The `venv.sh` script in the scripts directory contains shell command for:
* Installing [Anaconda](https://www.anaconda.com/)
* Setting up a virtual environment
* Set up [Wandb](https://wandb.ai/site)
* Set up a directory
* Define a shell alias for your project

### Configurations

For configuration we use the [Transformers](https://huggingface.co/) HfArgumentParser class which allows to parse command
line arguments into python dataclasses.
The configuration's dataclasses are defined in the `args` directory. For example in the `model_args.py` we define the
configurations of the model
```python
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default='roberta-base',
        metadata={"help": "Path to pretrained teacher model or model identifier from huggingface.co/models"}
    )
    trainer_type: str = field(
    ...
```

Then we can run the `main.py` script with the following arguments
```shell
python main.py --model_name_or_path roberta-base
```

Inside the script the command-line arguments will be parsed into their respective dataclass

```python
parser = HfArgumentParser((DataTrainingArguments, ModelArguments, ProjectTrainingArguments))
data_args, model_args, training_args = parser.parse_args_into_dataclasses()
print(model_args.model_name_or_path)
# roberta-base
```
Note that the `ProjectTrainingArguments` class inherits the Transformers `TrainingArguments` class.

## Customization

### Dataset

Any dataset in Huggingface can be loaded by passing `--dataset dataset_name`. To load a dataset that isn't on huggingface 
replace the 
```
raw_datasets = load_dataset(data_args.dataset)
``` 
call with a method of your own that loads your dataset into a Huggingface `DatasetDict` object. 
`data_utils.py` contains the `load_dataset_from_files` method that will suffice for most local datasets.

### Preprocessing

The `preprocess_datasets` method tokenizes the text. You can add any preprocessing procedures you wish.

### Training

The training is based on the Huggingface trainer. If you wish to use a different training framework, override the `train_model` method. 
Make sure to add an initialization of Wandb and report metrics in order to enable sweeps.

### Model, Metrics and Trainer

The model, the `comput_metrics` method and the trainer are obtained via a getter function in `train_utils.py` and have a dedicated config argument.
For example, to add your own trainer, add it to the `get_trainer` method:
```python
def get_trainer(trainer_type):
    if trainer_type == STANDARD:
        return Trainer
    elif trainer_type == CUSTOM:
        return CustomTrainer
    else:
        raise ValueError(f"Trainer type {trainer_type} is not supported. Available types are {ALL_TRAINER_TYPES}")
```

## Experimentation at Scale

Perhaps the template's most feature is the support of seamless execution of Wandb Sweeps which enable running many experiments in an efficient manner. 

### Sweep

Sweeps help us automate hyper-parameter tuning. To initialize a sweep, first configure the hyper-parameters you would like
to search over in the `sweep.yaml` file
```yaml
parameters:
  num_train_epochs:
    values: [3, 4, 5]
  learning_rate:
    values: [1e-5, 5e-5, 1e-4]
  per_device_train_batch_size:
    values: [2, 4, 8]
```

Then run

```shell
sh scripts/sweeps.sh

wandb: Creating sweep from: sweep.yaml
wandb: Created sweep with ID: 170jjitj
wandb: View sweep at: https://wandb.ai/navehp/project_template/sweeps/170jjitj
wandb: Run sweep agent with: wandb agent navehp/project_template/170jjitj
```

Make sure to copy the sweep ID to the `agents.sh` script.

Sweeps support grid search hyper-parameter optimization as well as 
[bayesian optimization](https://docs.wandb.ai/guides/sweeps/configuration#method), 
and early-stopping algorithms such as [Hyperband](https://docs.wandb.ai/guides/sweeps/configuration#early_terminate). 
Read more [here](https://docs.wandb.ai/guides/sweeps/configuration)

### Screen Session Control

A sweep is like a queue of experiments, but now we would like to run those experiments in different screen sessions.
The `screens.sh` script contains commands for controlling screen sessions in scale.

The script has "switches" to control different actions on your screen sessions: Initialization, termination, setting up a venv, configuring gpus, and canceling runs.
For example, this is the switch for initialization
```shell
CREATE=true
#CREATE=false
```

Setting it to `true` means the screen sessions will be created. 
You can easily toggle the switches in Pycharm by placing the caret on the upper line and pressing `Ctrl + /` twice which will comment and uncomment the lines.


### Agents

Now that we have a sweep and screen sessions to run experiments in, we need to set up an agent in each screen session.
An agent is a worker that takes tasks from the sweep's queue and executes it.

To set up agents, copy the sweep ID to the `agents.sh` script and run it.

### Multiple Servers

You can set up your code and venv in multiple servers to better utilize your resources. 
In order to run experiments on multiple servers all you need to do is set up a sweep on one server, and the run agents on all of them.

### Results

I will add a script that helps move all the results to a single server.
