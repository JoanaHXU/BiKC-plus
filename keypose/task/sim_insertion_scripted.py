import sys
import pathlib
import numpy as np

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)

from keypose.task.utils import _smooth
from diffusion_policy.env.aloha.constants import (
    DT, vx300s, LEFT_BASE_POSE, RIGHT_BASE_POSE,
    GRIPPER_EPSILON, EE_VEL_EPSILONE, EE_DIST_BOUND
)

'''
Configuration for heuristic keypose extraction
'''
DIST_MIN_insertion = 40 ## 0.8s
WINDOW_SIZE_insertion = 5 ## smoothing window size


''' 
Configuration for understanding video by VLM 
'''
TASK_NAME = "sim_insertion_scripted"
OBJECT_LIST = ['object_1', 'object_2']
ROBOT_LIST = ['robot_left', 'robot_right']
SURFACE_LIST = ['table']
TIME_LENGTH = 8



''' 
Function for extracting keypose indices accordingn to heuristic rules.
'''
def _find_keypose_idx_insertion(
    trajectory,
    side,
    window_size,
    gripper_epsilon=GRIPPER_EPSILON,
    vel_epsilon=EE_VEL_EPSILONE,
):
    # load data and initialization
    gripper = trajectory[f"gripper_{side}"]
    gripper_act = trajectory[f"gripper_act_{side}"]
    ee_vel = trajectory[f"ee_vel_norm_{side}"]
    gripper = _smooth(gripper, window_size=window_size)
    keypose_indices = [0]  # the init state is a keypose
    T = len(gripper)

    # smooth to remove noise
    gripper_change_rate = np.diff(gripper) / DT
    curr_state = "stable"  # opening, closing, stable
    problem = False
    # coordination = None
    for i in range(1, T-1):
        if curr_state == "stable":
            if gripper_change_rate[i] > gripper_epsilon:
                curr_state = "opening"
                keypose_indices.append(i)
            elif gripper_change_rate[i] < -gripper_epsilon:
                curr_state = "closing"
                # keypose_indices.append(i)
        elif curr_state == "opening":
            if abs(gripper_change_rate[i]) < gripper_epsilon:
                curr_state = "stable"
            elif gripper_change_rate[i] < -gripper_epsilon:
                print(f"why the gripper is closing when it is opening at {i}? ")
                problem = True
        elif curr_state == "closing":
            if abs(gripper_act[i]) < 0.1:
                curr_state = "stable"
                keypose_indices.append(i)
            elif gripper_change_rate[i] > gripper_epsilon:
                print(f"why the gripper is opening when it is closing at {i}?")
                problem = True

        if keypose_indices[-1] != i:
            ## gripper state is not key, check velocity
            ## if the EE is slow, consider it is a keypose
            if ee_vel[i-1] > vel_epsilon and ee_vel[i] < vel_epsilon:
                if side == "left":
                    keypose_indices.append(i)
                # coordination = i

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