import subprocess
import time
import shlex
import sys
import os
import pandas as pd
import glob
import argparse


def add_results(dict_commands, params, exec_params):
    df = pd.read_csv("cs285/experiments/experiments_data.csv", sep=";")

    row_dict = dict()
    log_folder = "cs285/experiments/logs/" + params["exp_name"]
    for name in glob.glob(log_folder + "/*"):
        row_dict["exp_name"] = name.split("/")[-1][:-4]
        if params["do_dagger"] == False:
            row_dict["algorithm"] = "bc"
        else:
            row_dict["algorithm"] = "dagger"
        row_dict["env_name"] = params["env_name"]
        row_dict["num_agent_train_steps_per_iter"] = params["num_agent_train_steps_per_iter"]
        row_dict["n_iter"] = params["n_iter"]
        row_dict["batch_size"] = params["batch_size"]
        row_dict["eval_batch_size"] = params["eval_batch_size"]
        row_dict["train_batch_size"] = params["train_batch_size"]
        row_dict["n_layers"] = params["n_layers"]
        row_dict["size"] = params["size"]
        row_dict["learning_rate"] = params["learning_rate"]
        row_dict["max_replay_buffer_size"] = params["max_replay_buffer_size"]

        log_file = open(name, "r")

        best_reward_mean = 0
        best_reward_std = 0
        best_expert = 0
        best_ep_len = 0
        get_value = False
        lines = log_file.readlines()
        for line in lines:
            if "Eval_AverageReturn" in line:
                value = float(line.strip("\n ").split(":")[-1])

                if value > best_reward_mean:
                    best_reward_mean = round(value, 2)
                    get_value = True

            if "Eval_StdReturn" in line and get_value is True:
                best_reward_std = round(float(line.strip("\n ").split(":")[-1]), 2)

            if "Eval_AverageEpLen" in line and get_value is True:
                best_ep_len = int(float(line.strip("\n ").split(":")[-1]))

            if "Initial_DataCollection_AverageReturn" in line and get_value is True:
                best_expert = round(float(line.strip("\n ").split(":")[-1]), 2)
                get_value = False

        row_dict["ep_len"] = best_ep_len
        row_dict["reward_mean"] = best_reward_mean
        row_dict["reward_std"] = best_reward_std
        row_dict["expert_reward"] = best_expert

        df = df.append(row_dict, ignore_index=True)

        log_file.close()

    df.to_csv("cs285/experiments/experiments_data.csv", sep=";", index=False)


def execute_comands(command, params):

    log_files = []
    dict_commands = dict()
    os.makedirs("cs285/experiments/logs/" + params["exp_name"])
    try:
        processes = []
        for index in range(params["execute_times"]):
            log_filename = (
                "cs285/experiments/logs/"
                + params["exp_name"]
                + "/"
                + params["exp_name"]
                + "_"
                + str(index + 1)
                + ".log"
            )
            log_file = open(log_filename, "w", 1)
            log_files.append(log_file)
            splited_command = shlex.split(command)
            index_name_list = splited_command.index("--exp_name")
            splited_command[index_name_list + 1] = splited_command[index_name_list + 1] + str("_") + str(index + 1)
            splited_command.append("--seed")
            splited_command.append(str(index + 1))
            tracking_command = dict()
            tracking_command["command"] = splited_command
            tracking_command["terminated"] = 0
            dict_commands[splited_command[index_name_list + 1]] = tracking_command
            proc = subprocess.Popen(splited_command, stdout=log_file)
            processes.append(proc)

        count = 0
        while count < len(processes):
            for proc in processes:
                return_code = proc.poll()
                if return_code is not None:
                    count += 1

            time.sleep(5)

    finally:
        print("Training season finished...")
        for file in log_files:
            file.flush()
            file.close()

    return dict_commands


def create_command(exec_params, param):

    command = "python cs285/scripts/run_hw1_behavior_cloning.py"
    for key, value in exec_params.items():
        command += " " + str(key) + " " + str(value)

    return command


def treat_params(params):

    exec_params = dict()

    if params["env_name"] == "antv2":
        exec_params["--env_name"] = "Ant-v2"
        exec_params["--expert_policy_file"] = "cs285/policies/experts/Ant.pkl"
        exec_params["--expert_data"] = "cs285/expert_data/expert_data_Ant-v2.pkl"

    if params["env_name"] == "humanoidv2":
        exec_params["--env_name"] = "Humanoid-v2"
        exec_params["--expert_policy_file"] = "cs285/policies/experts/Humanoid.pkl"
        exec_params["--expert_data"] = "cs285/expert_data/expert_data_Humanoid-v2.pkl"

    if params["env_name"] == "halfcheetahv2":
        exec_params["--env_name"] = "HalfCheetah-v2"
        exec_params["--expert_policy_file"] = "cs285/policies/experts/HalCheetah.pkl"
        exec_params["--expert_data"] = "cs285/expert_data/expert_data_HalfCheetah-v2.pkl"

    if params["env_name"] == "hopperv2":
        exec_params["--env_name"] = "Hopper-v2"
        exec_params["--expert_policy_file"] = "cs285/policies/experts/Hopper.pkl"
        exec_params["--expert_data"] = "cs285/expert_data/expert_data_Hopper-v2.pkl"

    exec_params["--exp_name"] = params["exp_name"]

    if params["do_dagger"] is not False:
        exec_params["--do_dagger"] = ""

    if params["ep_len"] is not None:
        exec_params["--ep_len"] = params["ep_len"]

    exec_params["--num_agent_train_steps_per_iter"] = params["num_agent_train_steps_per_iter"]
    exec_params["--n_iter"] = params["n_iter"]
    exec_params["--batch_size"] = params["batch_size"]
    exec_params["--eval_batch_size"] = params["eval_batch_size"]
    exec_params["--train_batch_size"] = params["train_batch_size"]
    exec_params["--n_layers"] = params["n_layers"]
    exec_params["--size"] = params["size"]
    exec_params["--learning_rate"] = params["learning_rate"]
    exec_params["--video_log_freq"] = params["video_log_freq"]
    exec_params["--scalar_log_freq"] = params["scalar_log_freq"]

    if params["use_gpu"] is not False:
        exec_params["--use_gpu"] = ""

    exec_params["--which_gpu"] = params["which_gpu"]
    exec_params["--max_replay_buffer_size"] = params["max_replay_buffer_size"]

    return exec_params


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute_times",
        "-ext",
        type=int,
        help="How much times the experiment runs. Each run a new  \
                          random seed is setted",
        default=1,
    )
    parser.add_argument(
        "--env_name", "-env", type=str, choices=["antv2", "humanoidv2", "halfcheetahv2", "hopperv2"], required=True,
    )
    parser.add_argument("--exp_name", "-exp", type=str, default="pick an experiment name", required=True)
    parser.add_argument("--do_dagger", action="store_true")
    parser.add_argument("--ep_len", type=int)

    parser.add_argument(
        "--num_agent_train_steps_per_iter", "-ats", type=int, default=1000
    )  # number of gradient steps for training policy (per iter in n_iter)
    parser.add_argument("--n_iter", "-n", type=int, default=1)

    parser.add_argument(
        "--batch_size", type=int, default=1000
    )  # training data collected (in the env) during each iteration
    parser.add_argument(
        "--eval_batch_size", type=int, default=200
    )  # eval data collected (in the env) for logging metrics
    parser.add_argument(
        "--train_batch_size", type=int, default=100
    )  # number of sampled data points to be used per gradient/train step

    parser.add_argument("--n_layers", type=int, default=2)  # depth, of policy to be learned
    parser.add_argument("--size", type=int, default=64)  # width of each layer, of policy to be learned
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-3)  # LR for supervised learning

    parser.add_argument("--video_log_freq", type=int, default=5)
    parser.add_argument("--scalar_log_freq", type=int, default=1)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--which_gpu", type=int, default=0)
    parser.add_argument("--max_replay_buffer_size", type=int, default=1000000)
    args = parser.parse_args()

    # convert args to dictionary
    params = vars(args)

    exec_params = treat_params(params)

    command = create_command(exec_params, params)

    dict_commands = execute_comands(command, params)

    add_results(dict_commands, params, exec_params)


if __name__ == "__main__":
    main()
