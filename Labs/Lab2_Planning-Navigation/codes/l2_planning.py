#!/usr/bin/env python
#Standard Libraries
from dis import dis
from lib2to3.pytree import convert
import numpy as np
import yaml
import pygame
import time
import pygame_utils
import matplotlib.image as mpimg
from skimage.draw import circle
from scipy.linalg import block_diag
from math import sin, cos, atan2, sqrt

#Map Handling Functions
def load_map(filename):
    im = mpimg.imread("../maps/" + filename)
    im_np = np.array(im)  #Whitespace is true, black is false
    #im_np = np.logical_not(im_np)    
    return im_np

def load_map_yaml(filename):
    with open("../maps/" + filename, "r") as stream:
            map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict

#Node for building a graph
class Node:
    def __init__(self, point, parent_id, cost):
        self.point = point # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.children_ids = [] # The children node ids of this node
        return

#Path Planner 
class PathPlanner:
    #A path planner capable of performing RRT and RRT*
    def __init__(self, map_filename, map_setings_filename, goal_point, stopping_dist):
        #Get map information
        self.occupancy_map = load_map(map_filename)
        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_setings_filename)

        #Get the metric bounds of the map
        self.bounds = np.zeros([2,2]) #m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[0] * self.map_settings_dict["resolution"]
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[1] * self.map_settings_dict["resolution"]

        if map_filename == "simple_map.png":
            self.map = self.occupancy_map[:, :, 0] 
        else:
            self.map = self.occupancy_map

        #Robot information
        self.robot_radius = 0.22 #m
        self.vel_max = 0.5 #m/s (Feel free to change!)
        self.rot_vel_max = 0.2 #rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m
        self.goal_reached = False

        #Trajectory Simulation Parameters
        self.timestep = 1.0 #s
        self.num_substeps = 10
        self.sim_traj_steps = 5
        self.traj_goal_reach = 0.2

        #Planning storage
        self.nodes = [Node(np.zeros((3,1)), -1, 0)]   
        # self.nodes.append(Node([[1], [1], [0.3]], -1, 10))
        # self.nodes.append(Node([[3], [2], [0.3]], -1, 10))

        #RRT Sampling Parameters
        self.samp_count     = 0     # counter that maintains how many points have been sampled
        self.samp_goal_freq = 7    # the nth sample will be goal position

        # for simple map
        # self.samp_x_min     = self.bounds[0, 0]
        # self.samp_x_max     = self.bounds[0, 1]
        # self.samp_y_min     = self.bounds[1, 0]
        # self.samp_y_max     = self.bounds[1, 1]
        
        # for willow garage map
        self.samp_x_min     = 0
        self.samp_x_max     = 44
        self.samp_y_min     = -45
        self.samp_y_max     = 10
        self.res_round      = 2     # rounding to 1 leads 10cm resolution, 2 leads to 1cm

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5
        self.search_radius = 0.4  # in m 
        self.nodes_added  = 0
        self.last_node_rewire_idx = None

        # controller parameters
        self.kP = 1.0
        self.kD = 0.1
        self.prev_err_head = 0.0
        
        #Pygame window for visualization
        self.window = pygame_utils.PygameWindow(
            "Path Planner", (1000, 1000), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist, map_filename)

        # point = np.array([[0],[-45]])
        # self.window.add_point(point.flatten())
        # pygame.image.save(self.window.screen, 'testing.png') 
        return

    #Functions required for RRT
    def sample_map_space(self):
        #Return an [x,y] coordinate to drive the robot towards
        # print("TO DO: Sample point to drive towards")
        self.samp_count += 1

        # every nth sample will be the goal itself
        if self.samp_count%self.samp_goal_freq==0:
            # print('Goal point sampled!')
            return self.goal_point

        x_sample = round((self.samp_x_max - self.samp_x_min)*np.random.random_sample() + self.samp_x_min, self.res_round)
        y_sample = round((self.samp_y_max - self.samp_y_min)*np.random.random_sample() + self.samp_y_min, self.res_round)

        return np.array([[x_sample], [y_sample]])
    
    def check_if_duplicate(self, point):
        '''Check if a point is a duplicate of an already existing node
        
        input: point (2x1 array)  -  robot point (x,y) expressed in origin (map) reference frame {I}
        output: [bool, None/index] (True if duplicate)
        '''
        for i in range(len(self.nodes)):
            if (self.nodes[i].point[:2] == point).all():
                return [True, i]

        return [False, None]
    
    def closest_node(self, point):
        #Returns the index of the closest node
        closest_dist     = 10000
        closest_node_idx = 0

        for i in range(len(self.nodes)):
            node_point = self.nodes[i].point
            dist = np.linalg.norm(point - node_point[:2])
            if dist < closest_dist:
                closest_dist     = dist
                closest_node_idx = i

        return closest_node_idx
    
    def simulate_trajectory(self, node_i, point_s):
        # print("\n-----FUNCTION STARTS------\n")
        # Simulates the non-holonomic motion of the robot.
        # This function drives the robot from node_i towards point_s. This function does have many solutions!
        # node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{VI} in course notation
        # point_s is the sampled point vector [x; y]
        # inputs:  node_i     (3x1 array) - current robot wrt map frame {I}
        #          point_s    (2x1 array) - goal point in map frame {I}
        # outputs: robot_traj (3xN array) - series of robot poses in map frame {I}

        # the goal point is already near the closest node in the tree
        if np.linalg.norm(point_s - node_i[:2])<self.traj_goal_reach:
            return np.array([[],[],[]])
        
        x, y, theta = node_i[0], node_i[1], node_i[2]         # pos., orient. of robot wrt inertial frame {I}

        # 1. Initialize robot_traj
        vel, rot_vel = self.robot_controller(node_i, point_s, self.vel_max, self.kP, self.kD, self.timestep) # initial velocities
        robot_traj = self.trajectory_rollout(vel, rot_vel, theta,
                                             self.timestep, self.num_substeps) + node_i # initial trajectory expressed in {V} frame
        
        cur_node = robot_traj[:, -1].reshape(3,1)
        dist_to_goal = np.linalg.norm(point_s - cur_node[:2])

        iter = 1

        while dist_to_goal>self.traj_goal_reach and iter<self.sim_traj_steps:
            #1. calculate initial vel, rot_vel
            # print("cur_node:\n", cur_node, "\npoint_s\n", point_s)
            # print("\ncur theta:", cur_node[2])
            vel, rot_vel = self.robot_controller(cur_node, point_s, self.vel_max, self.kP, self.kD, self.timestep)
            # print("\niter:", iter, "- vel, rot_vel", vel, rot_vel)

            #2. simulate trajectory for another  timestep and add to existing trajectory
            step_traj = self.trajectory_rollout(vel, rot_vel, cur_node[2], 
                                                self.timestep, self.num_substeps) + cur_node
            # print("\nstep_traj:\n", step_traj)
            robot_traj = np.hstack((robot_traj, step_traj))
            #print("\nrobot_traj:\n", robot_traj)

            #3. update current node and dist
            cur_node = robot_traj[:, -1].reshape(3,1)
            dist_to_goal = np.linalg.norm(point_s - cur_node[:2])
            # print("outside dist:", dist_to_goal)
            # print("\nupdated cur_node:\n", cur_node)

            iter += 1
   
        return robot_traj

    def normalize_angle(self, theta):
        return atan2(sin(theta), cos(theta))
    
    def robot_controller(self, node_i, point_s, max_vel, kP, kD, delta_t):
        # This controller determines the velocities that will nominally move the robot from node i to node s
        # Max velocities should be enforced
        # inputs: node_i (3x1 array)  - current point of robot wrt robot frame {I}
        #         point_s (2x1 array) - goal point of robot wrt robot frame {I}
        # outputs: vel (float)        - robot velocity wrt inertial frame {I}
        #         rot_vel (float)     - robot angular velocity wrt frame {I}

        # OPTION 1: PID CONTROL

        # calculate head error: angle b/w desired (theta_d) and actual headings (theta)
        theta_d = np.arctan2((point_s[1]-node_i[1]),(point_s[0]-node_i[0])) # desired heading {I} frame
        theta = node_i[2]                                                   # actual heading {I} frame
        err_head = np.around(self.normalize_angle(theta_d-theta),3)         # normalized heading error

        rot_vel = np.round(kP*(err_head) + kD*(err_head-self.prev_err_head)/(delta_t), 2)
        # print("rot_vel:", rot_vel)

        if rot_vel > max_vel:
            rot_vel = max_vel
        if rot_vel < -max_vel:
            rot_vel = -max_vel

        vel = np.around(max_vel/(6*abs(rot_vel)+1),2)
        # print("vel:", vel)
        # update controller error values
        self.prev_err_head = err_head                                       # previous error term for kD
        # self.cumul_error_head += error_heading*self.timestep              # cumulative error for kI

        return vel, rot_vel
    
    def trajectory_rollout(self, vel, rot_vel, theta_i, timestep, num_substeps):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points in {I} frame to check for collisions
        # inputs: vel (float)                  - robot velocity wrt inertial frame {I}
        #         rot_vel (float)              - robot angular velocity wrt frame {I}
        # output: self.trajectory (3x10 array) - robot pose for each time-substep expressed in {I} frame

        trajectory = np.array([[],[],[]])                          # initialize array
        t = np.linspace(0.1, timestep, num_substeps+1)
    
        if rot_vel == 0:
            x_I = [np.around((vel*t*np.cos(theta_i)),2)]
            y_I = [np.around((vel*t*np.sin(theta_i)),2)]
            theta_I = [np.zeros(num_substeps+1)]
        else:
            x_I = [np.around((vel/rot_vel)*(np.sin(rot_vel*t + theta_i)-np.sin(theta_i)), 4)]       # position in {V} frame
            y_I = [np.around((vel/rot_vel)*(np.cos(theta_i)-np.cos(rot_vel*t + theta_i)), 4)]
            #print("\nx_components: vel/rot_vel", vel/rot_vel, "np.sin(rot_vel*t)", np.sin(rot_vel*t), "-np.sin(theta):", -np.sin(theta_i))
            #print("y_components: vel/rot_vel", vel/rot_vel, "np.cos(theta)", np.cos(theta_i), "-np.sin(theta):", -np.cos(rot_vel*t))

            theta_I = [np.around(rot_vel*t, 4)]                          # orientation in {V}

        trajectory = np.vstack((x_I, y_I, theta_I))
        return trajectory
    
    def point_to_cell(self, point):
        # Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        # point is a 2 by N matrix of points of interest
        # input: point (2xN array)        - points of interest expressed in origin (map) reference frame {I}
        # output: map_indices (2xN array) - points converted to indices wrt top left
        
        # convert from map reference frame {I} to bottom-left ref frame {B}
        # position vector: r_B = r_I + r_BI = r_I - r_IB (offset vector from yaml file)
        x_B = point[0] - self.map_settings_dict["origin"][0] 
        y_B = point[1] - self.map_settings_dict["origin"][1]

        # need to convert to index by dividing by resolution (*1/0.05 = *20)
        height = self.map_shape[1]*self.map_settings_dict["resolution"]          # map height in meters
        x_idx = (x_B/self.map_settings_dict["resolution"]).astype(int)
        y_idx = ((height-y_B)/self.map_settings_dict["resolution"]).astype(int)  # y_B is wrt bottom left, while y_idx is wrt top left
        map_indices = np.vstack((x_idx,y_idx))

        return map_indices

    def points_to_robot_circle(self, points):
        # Convert a series of [x,y] points to robot map footprints for collision detection
        # input: point (2xN array)        - points of interest expressed in origin (map) reference frame {I}
        # output: point (2xM array)       - pixel coordinates occupied by the robot footprint, M<=45*N for each point
        # Hint: The disk function is included to help you with this function

        points_idx   = self.point_to_cell(points)         # convert to occupancy grid indexes (pixels)
        points_shape = np.shape(points)
        points_total = points_shape[1]
        pixel_radius = int(self.robot_radius*20)         # robot radius in pixels (using res of 0.05)
        footprint = [[],[]]

        for j in range(points_total):
            rr, cc = circle(int(points_idx[0,j]), int(points_idx[1,j]), pixel_radius, shape=(1600,1600))
            footprint = np.hstack((footprint,np.vstack((rr,cc))))
        
        return footprint

    def check_collision(self, robot_traj):
        '''
        about: checks collision for a point in the occupancy map
        
        input: robot_traj (3xN array) - series of robot poses in map frame {I}
        output: bool (True if collision occurs)
        '''
        # point = np.array([[6.5], [-17]])
        # self.window.add_point(point.flatten())
        footprint  = self.points_to_robot_circle(robot_traj[:2, :])
        points_shape = np.shape(footprint)
        points_total = points_shape[1]         
        
        for i in range(points_total):
            if self.map[int(footprint[1, i]), int(footprint[0, i])]==0:
                return True

        return False


    def add_nodes_to_graph(self, closest_node_id, robot_traj):
        '''
        about: adds nodes to the graph after doing the collision checking

        input:  closest_node_id (int)  - node id from which trajectory rollout is done
                robot_traj (3xN array) - series of robot poses in map frame {I}
        output: None, updates self.nodes
        '''
        points_shape   = np.shape(robot_traj)
        points_total   = points_shape[1]
        collision      = self.check_collision(robot_traj)
        
        # list to keep track of duplicate elements 
        dupl_list = {}
        self.nodes_added  = 0

        # no collision in the trajectory
        if not collision:
            for i in range(points_total):
                # check for duplicate node
                duplicate = self.check_if_duplicate(robot_traj[:2, i].reshape(2,1))

                if duplicate[0]:
                    dupl_list[i] = duplicate[1]            
                else:              
                    if i==0:   # closest_node is the parent
                        dist_to_parent = np.linalg.norm(robot_traj[:2, i].reshape(2,1) - self.nodes[closest_node_id].point[:2])
                        cost_to_come   = round(self.nodes[closest_node_id].cost + dist_to_parent, 2)
                        self.nodes.append(Node(robot_traj[:, i].reshape((3,1)), closest_node_id, cost_to_come))
                        self.nodes[closest_node_id].children_ids.append(len(self.nodes)-1)
                        self.window.add_line(self.nodes[closest_node_id].point[:2].flatten(), robot_traj[:2, i].flatten())
                        self.nodes_added += 1

                    else:      
                        if (i-1) in dupl_list.keys():  # the node was not added
                            prev_node_idx = dupl_list[i-1]
                        else:
                            prev_node_idx = -1

                        dist_to_parent = np.linalg.norm(robot_traj[:2, i].reshape(2,1) - self.nodes[prev_node_idx].point[:2])
                        cost_to_come   = round(self.nodes[prev_node_idx].cost + dist_to_parent, 2)
                        self.nodes.append(Node(robot_traj[:, i].reshape((3,1)), len(self.nodes)-1, cost_to_come))
                        self.nodes[-2].children_ids.append(len(self.nodes)-1)
                        self.window.add_line(self.nodes[-2].point[:2].flatten(), robot_traj[:2, i].flatten())
                        self.nodes_added += 1

                    dist_from_goal = np.linalg.norm(self.nodes[-1].point[:2] - self.goal_point)
                    if dist_from_goal <= self.stopping_dist:
                        self.goal_reached = True
                        print('Goal Reached!')
    
    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions
    def last_node_rewire(self):
        '''
        about: rewires the last added node, if required

        input:  None
        output: None
        '''
        last_node_idx   = len(self.nodes)-1
        exist_parent_id = self.nodes[last_node_idx].parent_id
        min_node_idx    = None
        min_node_cost   = self.nodes[last_node_idx].cost
        min_node_traj   = None

        for i in range(len(self.nodes)-self.nodes_added-1):
            dist_to_node = np.linalg.norm(self.nodes[last_node_idx].point[:2] - self.nodes[i].point[:2])

            # the node should be in the search radius
            if  dist_to_node<= self.search_radius:
                new_cost_to_come = self.nodes[i].cost + dist_to_node

                if new_cost_to_come < min_node_cost:
                    trajectory = self.connect_node_to_point(self.nodes[i].point, self.nodes[last_node_idx].point[:2])
                    collision  = self.check_collision(trajectory)

                    if not collision:
                        min_node_idx = i
                        min_node_cost = new_cost_to_come
                        min_node_traj = trajectory

        if min_node_idx!=None:
            print('Re-wiring of last node!', min_node_idx)
            
            # removing existing line from the plot
            self.window.remove_line(self.nodes[last_node_idx].point[:2].flatten(), self.nodes[exist_parent_id].point[:2].flatten())
                    
            # remove the last node from existing parent's children's list
            self.nodes[exist_parent_id].children_ids.remove(last_node_idx)

            # add the new parent to last node
            self.nodes[last_node_idx].parent_id = min_node_idx
            self.nodes[last_node_idx].cost = min_node_cost
                    
            # add last node as child to new parent
            self.nodes[min_node_idx].children_ids.append(last_node_idx)

            # add line plot
            self.window.add_line(self.nodes[last_node_idx].point[:2].flatten(), self.nodes[min_node_idx].point[:2].flatten())

            self.last_node_rewire_idx = last_node_idx
            # self.add_nodes_to_graph(min_node_idx, min_node_traj)

            # # adding last node as well
            # self.add_nodes_to_graph(-1, min_node_traj)


    def near_point_rewiring(self):
        """
        re-wires the near nodes after last node re-wiring

        input: None
        output: None
        """
        print('Rewiring near-point!')
        for i in range(len(self.nodes)-1):
            dist_to_node = np.linalg.norm(self.nodes[self.last_node_rewire_idx].point[:2] - self.nodes[i].point[:2])

            if  dist_to_node<= self.search_radius and i!=self.last_node_rewire_idx:
                new_cost_to_come = self.nodes[self.last_node_rewire_idx].cost + dist_to_node

                if new_cost_to_come < self.nodes[i].cost:
                    # the difference of cost
                    cost_diff = self.nodes[i].cost - new_cost_to_come
                    
                    cur_parent_id = self.nodes[i].parent_id
                    self.window.remove_line(self.nodes[i].point[:2].flatten(), self.nodes[cur_parent_id].point[:2].flatten())
                    
                    # remove the last node from existing parent's children's list
                    self.nodes[cur_parent_id].children_ids.remove(i)

                    # add the new parent to last node
                    self.nodes[i].parent_id = self.last_node_rewire_idx
                    self.nodes[i].cost = new_cost_to_come
                    
                    # add last node as child to new parent
                    self.nodes[self.last_node_rewire_idx].children_ids.append(i)

                    # add line plot
                    self.window.add_line(self.nodes[self.last_node_rewire_idx].point[:2].flatten(), self.nodes[i].point[:2].flatten())

                    # update near-point index
                    self.last_node_rewire_idx = i

                    self.update_children(self.last_node_rewire_idx, cost_diff)



    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
    
    def connect_node_to_point(self, node_i, point_f):
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node
        #point is a 2 by 1 point
        # return np.zeros((3, self.num_substeps))
        return self.simulate_trajectory(node_i, point_f)
    
    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle 
        print("TO DO: Implement a cost to come metric")
        return 0
    
    def update_children(self, node_id, cost_diff):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        
        child_ids = self.nodes[node_id].children_ids
        for i in range(len(child_ids)):            
            self.nodes[child_ids[i]].cost -= cost_diff 
            self.update_children(child_ids[i], cost_diff)

        return

    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work

        while not self.goal_reached: #Most likely need more iterations than this to complete the map!
        # for i in range(50):
            #Sample map space
            point = self.sample_map_space()
            # point = np.array([[10],[-5]])
            
            #*** Plotting for sampling ***#
            # print('Sampled point: ', point)
            # self.window.add_point(point.flatten())

            #Get the closest point
            closest_node_id = self.closest_node(point)
            # self.window.add_line(point.flatten(), [4, 4])
            # print('Closest Node: ', self.nodes[closest_node_id].point)

            #Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)
            # print(trajectory_o)

            #Check for collisions
            self.add_nodes_to_graph(closest_node_id, trajectory_o)

            # raw_input("Press Enter for loop to continue!")
        return self.nodes
    
    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot        
        while not self.goal_reached:
        # for i in range(1): #Most likely need more iterations than this to complete the map!
            #Sample
            point = self.sample_map_space()            

            #Closest Node
            closest_node_id = self.closest_node(point)

            #Simulate trajectory
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for collision and 
            self.add_nodes_to_graph(closest_node_id, trajectory_o)
            self.last_node_rewire_idx = None

            #Last node rewire
            self.last_node_rewire()

            #Close node rewire
            if self.last_node_rewire_idx!=None:
                self.near_point_rewiring()

            # raw_input("Press Enter for loop to continue!")

        return self.nodes

        

    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        self.window.add_line_start_to_goal(self.nodes[node_id].point[:2].flatten(), self.nodes[current_node_id].point[:2].flatten())
        while current_node_id > 0:
            path.append(self.nodes[current_node_id].point)
            parent_id = self.nodes[current_node_id].parent_id
            self.window.add_line_start_to_goal(self.nodes[current_node_id].point[:2].flatten(), self.nodes[parent_id].point[:2].flatten())
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path

def main():
    #Set map information
    map_filename = "simple_map.png"
    map_filename = "willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"

    #robot information
    goal_point = np.array([[42], [-44.5]]) #m
    stopping_dist = 0.5 #m

    start_time = time.time()
    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)
    nodes = path_planner.rrt_star_planning()

    node_path_metric = np.hstack(path_planner.recover_path())
    np.save("RRTstar_path_willowgarageworld.npy", node_path_metric)

    plot_name = 'RRTstar_willowgarageworld.png'
    pygame.image.save(path_planner.window.screen, plot_name)    
    print('Time taken to find the path: ', (time.time() - start_time)/60, ' min')

    # # Ensures that pygame window does not close unless keyboard exit (CTRL+C)    
    
    # running = True
    # while running:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False


    # # nodes = path_planner.rrt_star_planning()
    
    

if __name__ == '__main__':
    main()