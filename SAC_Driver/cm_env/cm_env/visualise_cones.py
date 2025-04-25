import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from cones import *
import pandas as pd
import ast

def plot_track_with_speed(yellow_cones, blue_cones, trajectory_data, lap_time=None, cone_size=50):
    """
    Plots track with cones, speed-colored trajectory, and lap time
    
    Parameters:
    yellow_cones : List of (x,y) for right boundary
    blue_cones : List of (x,y) for left boundary  
    trajectory_data : List of [((x,y), speed_m/s), ...]
    lap_time : Float representing lap time in seconds (optional)
    cone_size : Size of cone markers
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # --- Plot Track Boundaries ---
    y_x, y_y = zip(*yellow_cones) if yellow_cones else ([], [])
    b_x, b_y = zip(*blue_cones) if blue_cones else ([], [])
    
    ax.plot(y_x, y_y, 'y-', alpha=0.8, linewidth=2, zorder=1)
    ax.plot(b_x, b_y, 'b-', alpha=0.8, linewidth=2, zorder=1)
    
    ax.scatter(y_x, y_y, c='gold', s=cone_size, marker='^',
              linewidths=0.5, zorder=2,
              label='Yellow Cones')
    ax.scatter(b_x, b_y, c='blue', s=cone_size, marker='^',
              linewidths=0.5, zorder=2,
              label='Blue Cones')
    
    # --- Plot Speed Heatmap ---
    if trajectory_data and len(trajectory_data) > 1:
        points = np.array([point for (point, speed) in trajectory_data])
        speeds = np.array([speed * 3.6 for (point, speed) in trajectory_data])  # Convert m/s to km/h
        
        segments = np.stack([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(speeds.min(), speeds.max())
        lc = LineCollection(segments, cmap='plasma', norm=norm, 
                          linewidth=3, zorder=3)
        lc.set_array(speeds[:-1])
        line = ax.add_collection(lc)
        
        cbar = fig.colorbar(line, ax=ax)
        cbar.set_label('Speed (m/s)', rotation=270, labelpad=15)
    
    # --- Lap Time Display ---
    if lap_time is not None:
        # Convert seconds to minutes:seconds.milliseconds format
        minutes = int(lap_time // 60)
        seconds = lap_time % 60
        time_text = f"Lap Time: {minutes}:{seconds:06.3f}"
        
        ax.text(0.05, 0.95, time_text, transform=ax.transAxes,
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # --- Plot Settings ---
    ax.set_title('Track Layout with Speed Heatmap', pad=20)
    ax.set_xlabel('X Position (meters)')
    ax.set_ylabel('Y Position (meters)')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend()
    plt.tight_layout()
    
    return fig, ax

df = pd.read_csv("/Users/yasinetawfeek/Developer/Dynamic_SAC/checkpoint_bank")
df.drop(df.loc[df['1'] == False].index, inplace=True)

df = df.sort_values(by=['3'], ascending=True).drop(columns=['Unnamed: 0'])

for index, row in df.iterrows():
    input = ast.literal_eval(row[0])
    plot = plot_track_with_speed(yellow_cones, blue_cones, input, row[3], 30)
    plt.show()