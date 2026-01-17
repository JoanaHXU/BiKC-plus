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
DIST_MIN_cup_temporal = 50 ## 1s
WINDOW_SIZE_cup_temporal = 25 ## smoothing window size

''' 
Configuration for understanding video by VLM 
'''
TASK_NAME = "aloha2_cup_temporal"
OBJECT_LIST = ['cup']
ROBOT_LIST = ['robot_left', 'robot_right']
SURFACE_LIST = ['table']
TIME_LENGTH = 10


def _find_keypose_idx_cup_temporal(
    trajectory: dict,
    side: str="left",
    gripper_epsilon=REAL_GRIPPER_EPSILON,
):
    # return None, None, False
    # load data and initialization
    gripper = trajectory[f"gripper_{side}"]
    gripper_act = trajectory[f"gripper_act_{side}"]
    gripper_change_rate = np.diff(gripper) / DT
    ee_dist = trajectory["ee_dist"]
    T = len(gripper)
    keypose_indices = [0]  # init and final state are keyposes

    problem = False
    # coordination = 0
    for i in range(100, 300):
        if (
            gripper_change_rate[i-2] < -gripper_epsilon and
            gripper_change_rate[i-1] < -gripper_epsilon and
            gripper_change_rate[i] >= -gripper_epsilon
        ):
            keypose_indices.append(i) # from closing to closed
            break
    # for i in range(150, 400):
    #     if (
    #         ee_dist[i-2] > REAL_EE_DIST_BOUND + 0.02 and
    #         ee_dist[i-1] > REAL_EE_DIST_BOUND + 0.02 and
    #         ee_dist[i] <= REAL_EE_DIST_BOUND + 0.02
    #     ):
    #         keypose_indices.append(i) # from far to close
    #         # coordination = i
    #         break
    # for i in range(250, T-1):
    #     if (
    #         -gripper_epsilon < gripper_change_rate[i-2] < gripper_epsilon and
    #         -gripper_epsilon < gripper_change_rate[i-1] < gripper_epsilon and
    #         gripper_change_rate[i] >= gripper_epsilon
    #     ):
    #         keypose_indices.append(i) # from closed to opening

    keypose_indices.append(T-1)  # ensure the last frame is included

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
