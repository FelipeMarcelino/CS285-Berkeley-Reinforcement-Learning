import sys
import seaborn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go


seaborn.set()


def create_vis_2_2(dataframe):

    humanoid_dagger_output = open("./output_best_humanoid_2.txt")

    lines = humanoid_dagger_output.readlines()

    mean_iterations = []
    std_iterations = []
    expert_reward = []

    for line in lines:
        if "Eval_AverageReturn" in line:
            mean_iterations.append(float(line.strip("\n ").split(":")[-1]))
        if "Eval_StdReturn" in line:
            std_iterations.append(float(line.strip("\n ").split(":")[-1]))
        if "Initial_DataCollection_AverageReturn" in line:
            expert_reward.append(float(line.strip("\n ").split(":")[-1]))

    bc_humanoid_mean = [166.37] * 40
    bc_humanoid_std = [116.14] * 40

    plotter_dagger = go.Scatter(
        x=np.arange(40),
        y=mean_iterations,
        error_y=dict(type="data", array=std_iterations, visible=True),
        name="Dagger",
    )

    plotter_expert = go.Scatter(x=np.arange(40), y=expert_reward, name="Expert",)

    plotter_imitation = go.Scatter(
        x=np.arange(40), y=bc_humanoid_mean, error_y=dict(type="data", array=bc_humanoid_std, visible=True), name="BC",
    )

    fig = go.Figure(data=(plotter_dagger, plotter_expert, plotter_imitation))

    fig.update_layout(
        title="Comparison between BC, Dagger and Expert agents on Humanoid-v2",
        xaxis_title="train_steps_per_iter",
        yaxis_title="reward",
        title_x=0.5,
    )
    fig.show()


def create_table_1_2(dataframe):

    humanoid_BC = dataframe[dataframe["algorithm"] == "bc"]
    biggest_BC = humanoid_BC[humanoid_BC["num_agent_train_steps_per_iter"] == 200000]
    print(biggest_BC.mean())
    print(biggest_BC.std())


def create_vis_1_3(dataframe):

    only_ants_exp = dataframe[dataframe["env_name"] == "antv2"]
    group_per_iter_number_reward_mean = only_ants_exp.groupby("num_agent_train_steps_per_iter").mean()
    group_per_iter_number_reward_std = only_ants_exp.groupby("num_agent_train_steps_per_iter").std()
    reward_mean = group_per_iter_number_reward_mean[["reward_mean"]]
    reward_std = group_per_iter_number_reward_std[["reward_mean"]]
    expert_reward_std = group_per_iter_number_reward_std[["expert_reward"]]
    expert_reward_mean = group_per_iter_number_reward_mean[["expert_reward"]]
    reward_mean.columns = ["mean"]
    reward_std.columns = ["std"]
    expert_reward_mean.columns = ["mean"]
    expert_reward_std.columns = ["std"]
    bc_results = pd.concat([reward_mean, reward_std], axis=1)
    expert_results = pd.concat([expert_reward_mean, expert_reward_std], axis=1)
    bc_results["agent"] = "BC"
    expert_results["agent"] = "Expert"
    results = pd.concat([bc_results, expert_results], axis=0)

    results = results.reset_index()

    plotter_bc = go.Scatter(
        x=results[results["agent"] == "BC"]["num_agent_train_steps_per_iter"].values,
        y=results[results["agent"] == "BC"]["mean"].values,
        error_y=dict(type="data", array=results[results["agent"] == "BC"]["std"], visible=True),
        name="BC Agent",
    )

    plotter_expert = go.Scatter(
        x=results[results["agent"] == "Expert"]["num_agent_train_steps_per_iter"].values,
        y=results[results["agent"] == "Expert"]["mean"].values,
        error_y=dict(type="data", array=results[results["agent"] == "expert"]["std"], visible=True),
        name="Expert Agent",
    )
    fig = go.Figure(data=(plotter_bc, plotter_expert))
    fig.update_layout(
        title="Comparison between BC and Expert agents on Ant-v2",
        xaxis_title="train_steps_per_iter",
        yaxis_title="reward",
        title_x=0.5,
    )
    fig.show()


def main():

    dataframe = pd.read_csv("./experiments_data.csv", sep=";")

    # create_vis_1_3(dataframe)

    # create_table_1_2(dataframe)

    create_vis_2_2(dataframe)


if __name__ == "__main__":
    main()
