# Kedro SLURM 

kedro-slurm is a library that integrates Kedro pipelines with SLURM to enable distributed execution of tasks on high-performance computing (HPC) clusters. 
This library provides seamless integration for defining, submitting, and monitoring jobs on SLURM, leveraging SLURM's job scheduling capabilities while adhering to Kedro's pipeline structure.

**INSTALLATION:** `pip install kedro-slurm`

## How do I use Kedro SLURM?

To define a SLURM-enabled node, use the `kedro_slurm.pipeline.node` function. 
This allows you to specify SLURM resource requirements and job configurations for each node in your pipeline.

``` python
from kedro_slurm.pipeline import node


def function(input_data):
    # Your node logic here
    return processed_data


node = node(
    func=function,
    inputs="input_data",
    outputs="processed_data",
    name="my_slurm_node",
    resources=slurm.Resources(cpus=4, memory=16, gpus=1),
    configuration=slurm.Configuration(time_limit="2:00:00", partition_name="gpu"),
)
```

Define your pipeline by combining SLURM nodes with standard Kedro nodes. 
Kedro nodes will run using the library's default resource settings.

``` python
from kedro.pipeline import Pipeline, node
from kedro_slurm.pipeline import node as slurm_node

pipeline = Pipeline([
    slurm_node(
        func=function_1,
        inputs="input_data",
        outputs="processed_data",
        name="slurm_node_1",
        resources=slurm.Resources(cpus=8, memory=32),
        configuration=slurm.Configuration(time_limit="4:00:00"),
    ),
    node(
        func=function_2,
        inputs="input_data",
        outputs="processed_data",
        name="node_1",
    ),
    # Add more nodes here
])
```

To run your pipeline on SLURM, use the custom SLURMRunner by executing the following shell command:


``` sh
kedro run --async --runner=kedro_slurm.runner.SLURMRunner  
```

## Monitoring SLURM Jobs

The library offers abstractions for submitting and monitoring jobs on SLURM. 
You can submit a SLURM job using the `kedro_slurm.slurm.Job `class with its submit method, and monitor the job using the `kedro_slurm.slurm.Future` class.


``` python
from kedro_slurm import slurm


resources = slurm.Resources(cpus=8, memory=32, gpus=2)
configuration = slurm.Configuration(time_limit="4:00:00", partition_name="gpu")

job = slurm.Job(
    resources=resources,
    configuration=configuration,
    name="example_job",
    command="python train_model.py",
    path="./logs/%j",
)

future = job.submit()
while not future.done:
    future.update()
    
    print(f"Job status: {future._state}")
    
    time.sleep(5)
```

A Future can transition through the following job states:
- RUNNING
- COMPLETED
- PENDING
- FAILED
- CANCELLED
- PREEMPTED
- SUSPENDED
- STOPPED

## Default SLURM Resource Configuration


```
_DEFAULT_RESOURCES = slurm.Resources(cpus=4, memory=10)
_DEFAULT_CONFIGURATION = slurm.Configuration(time_limit="1:00:00")
```
