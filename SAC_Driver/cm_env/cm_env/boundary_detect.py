import numpy as np
from shapely.geometry import LineString, Point, Polygon
from typing import List, Tuple
import matplotlib.pyplot as plt
from cone_pos_real import yellow_cones, blue_cones
import time 
from scipy import spatial
from config import *

class TrackBoundary:
    def __init__(self, left_cones: List[Tuple[float, float]], right_cones: List[Tuple[float, float]]):
        """
        Initialize track boundaries using lists of cone coordinates.
        
        Args:
            left_cones: List of (x, y) coordinates for left boundary cones
            right_cones: List of (x, y) coordinates for right boundary cones
        """
        # Convert cone lists to numpy arrays for easier manipulation
        self.left_boundary = np.array(left_cones)
        self.right_boundary = np.array(right_cones)
        
        self.left_cones = left_cones
        self.right_cones = right_cones

        self.left_cones_tree = spatial.KDTree(self.left_cones)
        self.right_cones_tree = spatial.KDTree(self.right_cones)
        

        # Create LineString objects for both boundaries
        self.left_line = LineString(left_cones)
        self.right_line = LineString(right_cones)

        track_points = np.vstack((self.left_boundary, self.right_boundary[::-1]))
        self.track_polygon = Polygon(track_points)

        self.centerline = self._create_centerline()
        self.track_length = self.centerline.length

        self.max_progress = 0
        self.last_position = None
        self.total_reward = 0

    def _create_centerline(self) -> LineString:
        """Create a centerline between left and right boundaries."""
        center_points = []
        for i in range(len(self.left_boundary)):
            left_point = self.left_boundary[i]
            right_point = self.right_boundary[i]
            center_x = (left_point[0] + right_point[0]) / 2
            center_y = (left_point[1] + right_point[1]) / 2
            center_points.append((center_x, center_y))
        return LineString(center_points)
    
    def reset_progress(self):
        self.max_progress = 0
        self.last_position = None
        self.total_reward = 0

    def update_positon(self, position: Tuple[float, float]) -> dict:
        point = Point(position)

        current_progress = self.centerline.project(point, normalized=True)

        distance_along_track = current_progress * self.track_length

        progress_made = 0
        if current_progress > self.max_progress and current_progress < PROGRESS_FOR_COMPLETION:
            progress_made = current_progress - self.max_progress
            self.max_progress = current_progress

        if progress_made < MINIMUM_PROGRESS:
            reward = REWARD_WHEN_BELOW_MINIMUM_PROGRESS
        else:
            reward = (progress_made * REWARD_SCALE_FACTOR) ** EXPONENTIAL_INCREASE_OF_REWARD

        if reward > MAX_REWARD:
            reward = REWARD_WHEN_BELOW_MINIMUM_PROGRESS

        # reward = (progress_made * REWARD_SCALE_FACTOR) - (0.00001 * REWARD_SCALE_FACTOR)

        left_track = self.check_exit(position)
        if left_track and self.max_progress > 0.001:
            reward = -abs(NEGATIVE_REWARD)
        else:
            left_track = False

        collision = self.check_collision(position, buffer=COLLISION_BUFFER)
        if collision['collision']:
            reward = -abs(NEGATIVE_REWARD)

        if current_progress > PROGRESS_FOR_COMPLETION and (progress_made is not 0 and progress_made < 0.05):
            reward = abs(REWARD_WHEN_COMPLETING_TRACK)

        self.total_reward += reward
        self.last_position = position

        return {
            'current_progress': current_progress,
            'progress_made': progress_made,
            'distance_along_track': distance_along_track,
            'reward': reward,
            'total_reward': self.total_reward,
            'is_outside_track': left_track,
            'collision': collision
        }
        
    def check_collision(self, position: Tuple[float, float], buffer: float = 0.1) -> dict:
        """
        Check if a position collides with either boundary.
        
        Args:
            position: (x, y) coordinates to check
            buffer: Distance threshold for collision detection (meters)
            
        Returns:
            dict containing collision information
        """
        if position == (0,0):
            return {
            'collision': False,
            'side': 'left',
            'distance': 0
        }
        point = Point(position)
        
        # Check distance to both boundaries
        # left_distance = point.distance(self.left_line)
        # right_distance = point.distance(self.right_line)
        
        left_distance = self.left_cones_tree.query(position)[0]
        right_distance = self.right_cones_tree.query(position)[0]
        

        # Determine if there's a collision with either boundary
        left_collision = left_distance <= buffer
        right_collision = right_distance <= buffer
        
        return {
            'collision': left_collision or right_collision,
            'side': 'left' if left_distance < right_distance else 'right',
            'distance': min(left_distance, right_distance)
        }
    
    def check_exit(self, position: Tuple[float, float]):
        if position == (0,0):
            return False
        point = Point(position)
        return not self.track_polygon.contains(point)
    
    def get_nearest_boundary_points(self, position: Tuple[float, float]) -> dict:
        """
        Find the nearest points on both boundaries to the given position.
        Useful for visualization and debugging.
        
        Args:
            position: (x, y) coordinates to check
            
        Returns:
            dict containing nearest points on both boundaries
        """
        point = Point(position)
        
        # Project point onto both lines
        left_nearest = self.left_line.interpolate(self.left_line.project(point))
        right_nearest = self.right_line.interpolate(self.right_line.project(point))
        
        return {
            'left_point': (left_nearest.x, left_nearest.y),
            'right_point': (right_nearest.x, right_nearest.y)
        }

    def visualize(self, agent_position: Tuple[float, float] = None, collision_buffer: float = 0.1):
        """
        Visualize the track boundaries, cones, and optionally the agent position.
        
        Args:
            agent_position: Optional (x, y) coordinates of the agent
            collision_buffer: Buffer distance for collision detection visualization
        """
        plt.figure(figsize=(12, 8))
        
        # Plot boundary lines
        left_x, left_y = self.left_line.xy
        right_x, right_y = self.right_line.xy
        plt.plot(left_x, left_y, 'b-', label='Left Boundary', linewidth=2)
        plt.plot(right_x, right_y, 'r-', label='Right Boundary', linewidth=2)
        
        # Plot cones
        plt.scatter(self.left_boundary[:, 0], self.left_boundary[:, 1], 
                   color='blue', marker='^', s=100, label='Left Cones')
        plt.scatter(self.right_boundary[:, 0], self.right_boundary[:, 1], 
                   color='red', marker='^', s=100, label='Right Cones')
        
        if agent_position:
            # Plot agent
            plt.scatter(agent_position[0], agent_position[1], 
                       color='green', marker='o', s=200, label='Agent')
            
            # Get and plot nearest boundary points
            nearest_points = self.get_nearest_boundary_points(agent_position)
            left_point = nearest_points['left_point']
            right_point = nearest_points['right_point']
            
            plt.scatter(left_point[0], left_point[1], color='cyan', marker='x', s=100)
            plt.scatter(right_point[0], right_point[1], color='cyan', marker='x', s=100)
            
            # Draw lines to nearest points
            plt.plot([agent_position[0], left_point[0]], 
                    [agent_position[1], left_point[1]], 'c--', alpha=0.5)
            plt.plot([agent_position[0], right_point[0]], 
                    [agent_position[1], right_point[1]], 'c--', alpha=0.5)
            
            # Check and visualize collision
            collision_info = self.check_collision(agent_position, collision_buffer)
            if collision_info['collision']:
                circle = plt.Circle(agent_position, collision_buffer, 
                                  color='red', fill=False, linestyle='--',
                                  label='Collision Zone')
                plt.gca().add_patch(circle)
                plt.title(f"Collision detected on {collision_info['side']} boundary!")
            else:
                circle = plt.Circle(agent_position, collision_buffer, 
                                  color='green', fill=False, linestyle='--',
                                  label='Safe Zone')
                plt.gca().add_patch(circle)
                plt.title("No collision detected")
        
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()

# Example usage:
# def example_usage():
    
#     # Create track boundary
#     track = TrackBoundary(blue_cones, yellow_cones)
    
#     track.visualize(collision_buffer=0.2)

#     start_time = time.time()
#     # result = track.check_collision((1507, 148))
#     result = track.update_positon((10, 0))
#     print(f"result: {result}\ntime delta: {time.time() - start_time}")

#     result = track.update_positon((38, -3))
#     print(f"result: {result}\ntime delta: {time.time() - start_time}")
#     # plt.pause(2)  # Pause between visualizations

# if __name__ == "__main__":
#     example_usage()
