import sys
import pathlib
import numpy as np

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)

from keypose.task.utils import _smooth
from diffusion_policy.env.aloha.constants import (
    DT, vx300s, REAL_LEFT_BASE_POSE, REAL_RIGHT_BASE_POSE,
    REAL_GRIPPER_EPSILON, REAL_EE_DIST_BOUND
)

'''
Configuration for heuristic keypose extraction
'''
DIST_MIN_screwdriver = 50 ## 1s
WINDOW_SIZE_screwdriver = 5 ## smoothing window size

''' 
Configuration for understanding video by VLM 
'''
TASK_NAME = "aloha2_screwdriver_fix"
OBJECT_LIST = ['screwdriver', 'box']
ROBOT_LIST = ['robot_left', 'robot_right']
SURFACE_LIST = ['table']
TIME_LENGTH = 18


def _find_keypose_idx_screwdriver(
    trajectory,
    side,
    window_size,
    gripper_epsilon=REAL_GRIPPER_EPSILON,
    z_thres=0.05,
):
    # return None, None, False
    gripper = trajectory[f"gripper_{side}"]
    gripper_z = trajectory[f"ee_pos_{side}"][:, 2]
    gripper_act = trajectory[f"gripper_act_{side}"]
    gripper_change_rate = np.diff(gripper) / DT

    T = len(gripper)  # 800
    keypose_indices = [0]  # init and final state are keyposes

    problem = False
    coordination = 0
    for i in range(50, 250): # 1s-5s
        if (
            side == "left" and
            gripper_act[i-2] > gripper_epsilon and
            gripper_act[i-1] > gripper_epsilon and
            gripper_act[i] <= gripper_epsilon
        ):
            keypose_indices.append(i)  # left grasp the screwdriver
            break
    for i in range(50, 300):
        if (
            side == "right" and
            gripper_z[i-2] > z_thres * 1.5 and
            gripper_z[i-1] > z_thres * 1.5 and
            gripper_z[i] <= z_thres * 1.5
        ):
            keypose_indices.append(i)  # right press the box
            break
    for i in range(250, 450):
        if (
            side == "left" and
            gripper_act[i-2] < gripper_epsilon and
            gripper_act[i-1] < gripper_epsilon and
            gripper_act[i] >= gripper_epsilon
        ):
            keypose_indices.append(i)  # left release the screwdriver
            break
    for i in range(650, 400, -1):
        if (
            side == "right" and
            gripper_z[i-2] < z_thres*1.8 and
            gripper_z[i-1] < z_thres*1.8 and
            gripper_z[i] >= z_thres*1.8
        ):
            keypose_indices.append(i)  # right flip the box cover

            break
    # for i in range(650, T-1):
    #     if (
    #         (side == "left" or side == "right") and
    #         gripper_z[i-2] > z_thres * 1.6 and
    #         gripper_z[i-1] > z_thres * 1.6 and
    #         gripper_z[i] <= z_thres * 1.6
    #     ):
    #         keypose_indices.append(i) # right tighten the box
    #         # coordination = i
    #         break

    keypose_indices.append(T-1)

    return keypose_indices, problem


PROMPT = f"""
    Task: {TASK_NAME}
    Objects in the scene: {OBJECT_LIST}
    Robots in the scene: {ROBOT_LIST}
    Surfaces in the scene: {SURFACE_LIST}
    Video duration: {TIME_LENGTH} seconds
    
    You will analyze a sequence of video frames to detect contact relationship changes. Each frame has a sequential index starting from 0.
    
    FRAME INDEXING GUIDE:
    - If analyzing a video: frames are sampled at the specified fps, frame indices correspond to the sequence order (0, 1, 2, ...)
    - If analyzing image files: the files are named as frame_000000.jpg, frame_000001.jpg, etc. The frame number corresponds to the number in the filename

    CRITICAL CONTACT DETECTION RULES:
    1. GRASP (robot-object): Only when robot gripper/fingers are PHYSICALLY CLOSED around the object AND the object moves with the robot
    2. APPROACH vs GRASP: Robot moving toward object ≠ grasping. Only report grasp when contact is ESTABLISHED and MAINTAINED
    3. VISUAL CONFIRMATION: Look for:
       - Both of gripper fingers actually touching/enclosing the object
       - Object displacement/movement caused by robot action
       - Clear physical contact, not just proximity
    
    Answer the following questions step by step based on the given video:
        1. Use objects, robots, and surface as nodes, and the contact relationship among them as edges, construct a graph to describe the initial contact relationship among them. The <initial_graph> is made up of a list of edges and nodes of the whole scene. Initially, all objects should connect to the surface.
        
        2. Starting from <initial_graph>, detect contact mode changes by analyzing each video frame in sequence. 

           IMPORTANT: Only report contact changes when you can CLEARLY observe:
           - Physical contact establishment or breaking
           - Object state changes (lifted, moved, released)
           
           For contact changes, use these primitives:
           - Primitive to ADD a new edge: grasp (robot-object), attach object (object-object), place (object-surface)
           - Primitive to REMOVE an edge: release (robot-object), detach object (object-object), pick (object-surface)

    Note: 
    1) Do not miss the contact between object and the surface. 
    2) For each contact change, only one <edge> should be added or removed from the graph, with the <operation> be "add" or "remove". 
    4) Distinguish between "approaching" and "contacting"
    5) No explaination needed.

    After that, provide the final output in a json format.
    Your final output should be like:
        ```json
        {{
            "initial_graph": "<initial_graph>",
            "ModeChangeDetection": [
                {{
                    "graph_number": 1,
                    "frame_idx": <frame_index>,
                    "contact_change": "[<edge>, <operation>]",
                    "description": "<primitive>"
                }},
                ...
            ]
        }}
        ```
    """
