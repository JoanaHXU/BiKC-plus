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
DIST_MIN_pants = 50 ## 1s
WINDOW_SIZE_pants = 25 ## smoothing window size

''' 
Configuration for understanding video by VLM 
'''
TASK_NAME = "aloha2_pants"
OBJECT_LIST = ['hanger', 'pants']
ROBOT_LIST = ['robot_left', 'robot_right']
SURFACE_LIST = ['table']
TIME_LENGTH = 18


def _find_keypose_idx_hang(
    trajectory,
    side,
    gripper_epsilon=REAL_GRIPPER_EPSILON, #0.2
    vel_ee_epsilon=0.05,
):
    # return None, None, None
    gripper = trajectory[f"gripper_{side}"]
    gripper_act = trajectory[f"gripper_act_{side}"]
    ee_vel = trajectory[f"ee_vel_norm_{side}"]
    ee_vel = _smooth(ee_vel, window_size=WINDOW_SIZE_pants, visualize=False)

    T = len(gripper)
    keypose_indices = [0, T-1]  # init and final state are keyposes

    # right grasps the hanger
    if side == 'right':
        for i in range(75, 300):  # 1.5-6s
            if (
                gripper[i-2] >= 0.75 * gripper_epsilon and
                gripper[i-1] >= 0.75 * gripper_epsilon and
                gripper[i] < 0.75 * gripper_epsilon
            ):
                keypose_indices.append(i)
                break

    if side == 'right':
        for i in range(375, 625): #7.5-12.5s
            if ee_vel[i] < 0.05:
                keypose_indices.append(i)
                break

    # left grasps the pants
    if side == 'left':
        for i in range(300, 625):  # 6.0-12.5s
            if (
                gripper[i-2] >= 0.4 * gripper_epsilon and
                gripper[i-1] >= 0.4 * gripper_epsilon and
                gripper[i] < 0.4 * gripper_epsilon
            ):
                keypose_indices.append(i)
                break

    if side == 'left':
        for i in range(500, 800): # 10-16s
            if (
                gripper[i-2] < 0.4 * gripper_epsilon and
                gripper[i-1] < 0.4 * gripper_epsilon and
                gripper[i] >= 0.4 * gripper_epsilon
            ):
                keypose_indices.append(i)
                break

    keypose_indices.append(T-1)

    return keypose_indices, False


PROMPT = f"""
    Task: {TASK_NAME}
    Objects in the scene: {OBJECT_LIST}
    Robots in the scene: {ROBOT_LIST}
    Surfaces in the scene: {SURFACE_LIST}
    Video duration: {TIME_LENGTH} seconds
    
    You will analyze a sequence of video frames to detect contact relationship changes. Each frame has a sequential index starting from 0.

    VIDEO DETAILS
    The video contains 2 simultaneous camera angles:
    1. **Top Camera**: Bird's eye view of the entire scene.
    2. **Front Camera**: Front view of the manipulation.
    
    FRAME INDEXING GUIDE:
    - If analyzing a video: frames are sampled at the specified fps, frame indices correspond to the sequence order (0, 1, 2, ...)
    - If analyzing image files: the files are named as frame_000000.jpg, frame_000001.jpg, etc. The frame number corresponds to the number in the filename

    PRIOR INFORMATION ABOUT TASK DESCRIPTION IN VIDEO:
    1. Right arm grasps the hanger.
    2. Left arm grasps the pants.
    3. Pants are hung on the hanger.
    4. Left arm releases the pants.
    5. Right arm lifts the hanger with the pants.

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
