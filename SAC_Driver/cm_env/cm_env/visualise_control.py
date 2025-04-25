import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from cones import *
import pandas as pd
import ast

def plot_vehicle_controls(control_data, show_steering=True, show_throttle=True, show_brake=True):
    """
    Plots vehicle control inputs with custom scaling as line graphs.
    """
    if not control_data:
        raise ValueError("Control data is empty")
    
    # Unpack and scale data
    control_data = np.array(control_data)[:200]
    steering = control_data[:, 0] * 21  # -1 to 1 → -21 to 21 degrees
    throttle = control_data[:, 1] * 400  # 0 to 1 → 0 to 400 N
    brake = control_data[:, 2] * 80      # 0 to 1 → 0 to 80 N
    
    time_steps = np.arange(len(control_data))
    
    # Determine how many plots to show
    num_plots = sum([show_steering, show_throttle, show_brake])
    if num_plots == 0:
        raise ValueError("At least one control must be selected to plot.")
    
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 2.5 * num_plots), sharex=True)
    if num_plots == 1:
        axs = [axs]
    
    idx = 0
    if show_steering:
        axs[idx].plot(time_steps, steering, label='Steering', color='blue', linewidth=0.5)
        axs[idx].set_ylabel('Steering (°)')
        axs[idx].set_ylim(-25, 25)
        axs[idx].grid(True, alpha=0.3)
        axs[idx].legend()
        idx += 1

    if show_throttle:
        axs[idx].plot(time_steps, throttle, label='Throttle', color='green')
        axs[idx].set_ylabel('Throttle (N)')
        axs[idx].set_ylim(0, 420)
        axs[idx].grid(True, alpha=0.3)
        axs[idx].legend()
        idx += 1

    if show_brake:
        axs[idx].plot(time_steps, brake, label='Brake', color='red')
        axs[idx].set_ylabel('Brake (N)')
        axs[idx].set_ylim(0, 85)
        axs[idx].grid(True, alpha=0.3)
        axs[idx].legend()
    
    axs[-1].set_xlabel('Time Steps')
    plt.tight_layout()
    return fig, axs

df = pd.read_csv("/Users/yasinetawfeek/Developer/Dynamic_SAC/control_bank_deployment")
df.drop(df.loc[df['1'] == False].index, inplace=True)

df = df.sort_values(by=['3'], ascending=True).drop(columns=['Unnamed: 0'])

for index, row in df.iterrows():
    input = ast.literal_eval(row[0])
    plot = plot_vehicle_controls(input, True, False, False)
    plot = plot_vehicle_controls(input, True, True, False)
    plot = plot_vehicle_controls(input, True, False, True)

    plt.show()