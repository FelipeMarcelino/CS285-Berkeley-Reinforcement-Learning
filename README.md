# CS285
This repository contains notes about class CS285(Deep Reinforcement Learning) and homeworks with solutions. 

In this repository you can explenations on the algorithms used, full implementation code, results and how to reproduce the results shown.

The base code of this repository is from: https://github.com/berkeleydeeprlcourse/homework_fall2019. The code written here heavely relies on that repository. The homework included in the repo:

1. Behavioral Cloning

# Results

## HW1

Result from the HW1 on **Dagger**. Explanation about **Dagger** with mahematical notations can be found here - http://rail.eecs.berkeley.edu/deeprlcourse/ 

Iteration 5(Left) - 40(Right)

![](https://github.com/FelipeMarcelino/CS285-Berkeley-Reinforcement-Learning/blob/master/hw1/results/gifs/dagger_40_iter.gif)

* Run 1 experiment 
``` sh
$ python cs285/scripts/run_hw1_behavior_cloning.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name test_dagger_ant --n_iter 10 --do_dagger --expert_data cs285/expert_data/expert_data_Ant-v2.pkl
```

* Run N experiments (execute_experiment.py, -ext [N])
``` sh
python cs285/experiments/execute_experiment.py --env_name humanoidv2 --exp_name humanoid_bc_last_version --n_iter 1 -ext 10 --video_log_freq -1 -ats 200000 --use_gpu --batch_size 5000 --train_batch_size 1000 --eval_batch_size 1000
```

- Tensboard logs will be created inside cs285/data. 
- Executions logs will be crated inside cs285/experiments/logs 
- The file experiment_data.csv, inside cs285/experiments/, has a table with experimets executed until now. 

# Contact

Felipe Marcelino - <felipe.ggmarcelino@gmail.com>
