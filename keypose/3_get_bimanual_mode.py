import click
import os
import sys
import pathlib
import json
import ast
import h5py
from tqdm import tqdm

ROOT = pathlib.Path(__file__).parent.parent
sys.path.append(str(ROOT))

from keypose.utils.visualize_sg_transfer import SceneGraphVisualizer_Transfer
from keypose.utils.visualize_sg_insertion import SceneGraphVisualizer_Insertion


def parse_graph_edges(graph_str):
    """
    Parse graph string to extract edges.
    
    Args:
        graph_str (str): Graph representation as string
        
    Returns:
        Set[Tuple[str, str]]: Set of edges
    """
    # Remove brackets and split by comma
    edges_str = graph_str.strip("[]")
    
    # Parse tuples from string
    try:
        edges_set = ast.literal_eval(edges_str)
        return set(edges_set)
    except:
        return set()
    

def get_graph_at_step(data, step):
    """
    Reconstruct graph at specific step by applying all changes up to that step.

    Args:
        data (dict): Parsed JSON data containing initial graph and mode changes
        step (int): Step number to reconstruct the graph
        
    Returns:
        Set[Tuple[str, str]]: Graph edges at the specified step
    """
    initial_graph = parse_graph_edges(data['initial_graph'])
    mode_changes = data['ModeChangeDetection']

    current_graph = initial_graph.copy() ## type(current_graph) is set
    
    # Apply changes up to the target step
    for change in mode_changes:
        if change['graph_number'] <= step:
            edge_str = change['contact_change']
            # Parse edge and operation
            edge_data = ast.literal_eval(edge_str)
            edge = tuple(edge_data[0])
            operation = edge_data[1]
            
            if operation == 'add':
                current_graph.add(edge)
            elif operation == 'remove':
                current_graph.discard(edge)
    
    return current_graph


def detect_coordination_mode_1(G, graph, robots, objects):
    """
    Detect coordination mode 1: Both robots contact the same object simultaneously.
    At the same moment, the left robot and the right robot contact the same object.
    {G_i: <r_l, obj>, <r_r, obj>}
    """
    pattern = dict()
    
    for obj in objects:
        contacting_robots = []
        for robot in robots:
            if (robot, obj) in graph or (obj, robot) in graph:
                contacting_robots.append(robot)
        
        if len(contacting_robots) >= 2:
            pattern["co"] = True
        else:
            pattern["co"] = False
        
        pattern["ts"] = G['timestep_orig']
    
    return pattern


def detect_coordination_mode_2(G_prev, G_curr, graph_prev, graph_curr, robots, objects):
    """
    Detect coordination mode 2: Sequential object transfer between robots.
    At timestep t, object contacts with left robot; at timestep t+1, same object contacts with right robot.
    {G_i: <r_l, obj>} and {G_{i+1}: <obj, r_r>}
    """

    if G_prev is None:
        return {"co": False, "timestep": G_curr['timestep_orig']}

    pattern = dict()
    pattern["co"] = False
    pattern["ts"] = G_curr['timestep_orig']
    
    # Find left and right robots
    left_robot = None
    right_robot = None
    for robot in robots:
        if 'left' in robot.lower():
            left_robot = robot
        elif 'right' in robot.lower():
            right_robot = robot

    assert left_robot is not None and right_robot is not None, "Both left and right robots must be defined."
    
    for obj in objects:
        ### --- left_hand to right_hand transfer --- ###

        # Check if object was with left robot in previous state
        left_contact_prev = (left_robot, obj) in graph_prev or (obj, left_robot) in graph_prev
        # Check if object was NOT with right robot in previous state
        right_contact_prev = (right_robot, obj) in graph_prev or (obj, right_robot) in graph_prev
        # Check if object is with right robot in current state
        right_contact_curr = (right_robot, obj) in graph_curr or (obj, right_robot) in graph_curr
        # Check if object is NOT with left robot in current state
        left_contact_curr = (left_robot, obj) in graph_curr or (obj, left_robot) in graph_curr

        if left_contact_prev and right_contact_curr and not left_contact_curr and not right_contact_prev:
            pattern["co"] = True

        ### --- right_hand to left_hand transfer --- ###

        # Check if object was with right robot in previous state
        right_contact_prev = (right_robot, obj) in graph_prev or (obj, right_robot) in graph_prev
        # Check if object was NOT with left robot in previous state
        left_contact_prev = (left_robot, obj) in graph_prev or (obj, left_robot) in graph_prev
        # Check if object is with left robot in current state
        left_contact_curr = (left_robot, obj) in graph_curr or (obj, left_robot) in graph_curr
        # Check if object is NOT with right robot in current state
        right_contact_curr = (right_robot, obj) in graph_curr or (obj, right_robot) in graph_curr

        if right_contact_prev and left_contact_curr and not right_contact_curr and not left_contact_prev:
            pattern["co"] = True
    
    return pattern


def detect_coordination_mode_3(G_prev, G_curr, graph_prev, graph_curr, robots, objects):
    """
    Detect coordination mode 3: Object-object connection after robot-object contacts.
    At timestep t: object A contacts left robot, object B contacts right robot.
    At timestep t+1: object A contacts object B (while maintaining robot contacts).
    {G_i: <r_l, obj_A>, <r_r, obj_B>} and {G_{i+1}: <r_l, obj_A>, <r_r, obj_B>, <obj_A, obj_B>}
    """
    pattern = dict()
    pattern["co"] = False
    pattern["ts"] = G_curr['timestep_orig']

    # Find left and right robots
    left_robot = None
    right_robot = None
    for robot in robots:
        if 'left' in robot.lower():
            left_robot = robot
        elif 'right' in robot.lower():
            right_robot = robot
    
    assert left_robot is not None and right_robot is not None, "Both left and right robots must be defined."
    
    # Find objects connected to each robot in previous state
    left_objects_prev = []
    right_objects_prev = []
    
    for obj in objects:
        if (left_robot, obj) in graph_prev or (obj, left_robot) in graph_prev:
            left_objects_prev.append(obj)
        if (right_robot, obj) in graph_prev or (obj, right_robot) in graph_prev:
            right_objects_prev.append(obj)

    # Check for object-object connections in current state
    for obj_a in left_objects_prev:
        for obj_b in right_objects_prev:
            if obj_a != obj_b:
                # Check if robots still contact their respective objects
                left_contact_curr = (left_robot, obj_a) in graph_curr or (obj_a, left_robot) in graph_curr # <r1, a>
                right_contact_curr = (right_robot, obj_b) in graph_curr or (obj_b, right_robot) in graph_curr #<r2, b>
                
                # Check if objects are now connected
                obj_obj_contact = (obj_a, obj_b) in graph_curr or (obj_b, obj_a) in graph_curr # <a, b>
                
                # Check if this connection was NOT in previous state
                obj_obj_contact_prev = (obj_a, obj_b) in graph_prev or (obj_b, obj_a) in graph_prev 
                
                if left_contact_curr and right_contact_curr and obj_obj_contact and not obj_obj_contact_prev:
                    pattern["co"] = True
    
    return pattern
            

def analyze_coordination_modes(json_file_path):
    """
    Analyze coordination modes from JSON data and reconstruct contact graphs.
    Args:
        json_file_path (str): Path to the JSON file containing initial graph and mode changes.
    Returns:
        dict: Dictionary containing coordination modes and reconstructed graphs.
    """
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file {json_file_path} does not exist.")
    
    ## Load JSON data, and parse the initial graph and mode changes
    with open(json_file_path, 'r', encoding='utf-8') as f:
        """data: {
        'initial_graph': '[(agent1, agent2), (agent2, agent3)]', 
        'ModeChangeDetection': [{'mode': 'mode1', 'start_time': 0, 'end_time': 10}, ...]
        }"""
        data = json.load(f)
    
    initial_graph = parse_graph_edges(data['initial_graph'])
    mode_changes = data['ModeChangeDetection']

    ## check if (robot, table) pair in initial graph
    if ('robot_left', 'table') not in initial_graph:
        initial_graph.add(('robot_left', 'table'))
    if ('robot_right', 'table') not in initial_graph:
        initial_graph.add(('robot_right', 'table'))

    # Extract robots and objects from initial graph
    robots = set()
    objects = set()
    surfaces = set()
    
    for edge in initial_graph:
        node1, node2 = edge
        if 'robot' in node1.lower():
            robots.add(node1)
        elif 'table' in node1.lower() or 'surface' in node1.lower():
            surfaces.add(node1)
        else:
            objects.add(node1)
            
        if 'robot' in node2.lower():
            robots.add(node2)
        elif 'table' in node2.lower() or 'surface' in node2.lower():
            surfaces.add(node2)
        else:
            objects.add(node2)
    
    robots = list(robots)
    objects = list(objects)
    
    # 确定需要分析的图状态数量
    if not mode_changes:
        # 如果没有模式变化，只有初始图
        num_graphs = 1
    else:
        # 图状态数量 = 初始图 + 所有变化后的图
        max_graph_number = max([change['graph_number'] for change in mode_changes])
        num_graphs = max_graph_number + 1  # +1 包含初始图（G0）

    co_mode1 = dict()
    co_mode2 = dict()
    co_mode3 = dict()
    Graph = dict()

    # 处理初始图（G0 或不包含在 Mode 中，只用于参考）
    initial_graph_key = "G0"
    Graph[initial_graph_key] = initial_graph.copy()
    print(f"G0: {initial_graph}")

    
    for i in range(1, num_graphs):  # 图编号从1开始
        graph_name = f"G{i}"
        
        if i == 1:
            # 第一个图状态（初始图 + 第一个变化）
            curr_changes = [change for change in mode_changes if change['graph_number'] == 1]
            G_curr = curr_changes[0] if curr_changes else None
            G_prev = None
            
            # 第一个图状态应该是应用第一个变化后的图
            if G_curr is not None:
                curr_graph = get_graph_at_step(data, 1)
            else:
                curr_graph = initial_graph.copy()
            prev_graph = initial_graph.copy()  # 前一个图是初始图
        else:
            # 后续图状态
            curr_changes = [change for change in mode_changes if change['graph_number'] == i]
            prev_changes = [change for change in mode_changes if change['graph_number'] == i - 1]
            
            G_curr = curr_changes[0] if curr_changes else None
            G_prev = prev_changes[0] if prev_changes else None
            
            curr_graph = get_graph_at_step(data, i)
            prev_graph = get_graph_at_step(data, i - 1)

        ## Construct graph representation
        Graph[graph_name] = curr_graph
        print(f"{graph_name}: {curr_graph}")

        # Detect coordination modes
        if G_curr is not None:
            co_mode1[graph_name] = detect_coordination_mode_1(G_curr, curr_graph, robots, objects)
            co_mode2[graph_name] = detect_coordination_mode_2(G_prev, G_curr, prev_graph, curr_graph, robots, objects)
            co_mode3[graph_name] = detect_coordination_mode_3(G_prev, G_curr, prev_graph, curr_graph, robots, objects)
        else:
            pass
            # 处理没有对应变化的图状态
            default_ts = None
            co_mode1[graph_name] = {"co": False, "ts": default_ts}
            co_mode2[graph_name] = {"co": False, "ts": default_ts}
            co_mode3[graph_name] = {"co": False, "ts": default_ts}

    ## Final Mode considering three coordination modes
    Mode = dict()
    for i in range(1, num_graphs):

        if co_mode1[f"G{i}"]["ts"] is None:
            continue # skip if no graph data

        Mode[f"G{i}"] = dict()
        if co_mode1[f"G{i}"]["co"] or co_mode2[f"G{i}"]["co"] or co_mode3[f"G{i}"]["co"] == True:
            Mode[f"G{i}"]["co"] = True
        else:
            Mode[f"G{i}"]["co"] = False

        Mode[f"G{i}"]["ts"] = co_mode1[f"G{i}"]['ts']

    print("\nMode:", Mode, "\n")

    ## Final Result Construction
    Result = dict()
    Result['Mode'] = Mode
    Result['Graph'] = Graph

    return Result


@click.command()
@click.option('--task_name', default='sim_transfer_cube_scripted', help='Name of the task to process.')
@click.option('--file_dir', default='/home/xuhang/TASE/dataset/', help='Directory containing the HDF5 files.')
@click.option('--num_episodes', default=1, type=int, help='Index of the episode to process.')
@click.option('--if_visualize', default=True, type=bool, help='Whether to visualize the results.')
def main(task_name, file_dir, num_episodes, if_visualize):

    ## input file paths
    json_path = os.path.join(file_dir, f"{task_name}", 'json_sg_vlm')

    ## output dataset paths
    hdf5_path = os.path.join(file_dir, f"{task_name}", 'dataset_mode')
    if not os.path.exists(hdf5_path):
        os.makedirs(hdf5_path, exist_ok=True)
    else:
        print(f"Directory {hdf5_path} already exists. Overwriting files...")

    ## output image paths
    image_path = os.path.join(file_dir, f"{task_name}", 'image_graph')
    if not os.path.exists(image_path):
        os.makedirs(image_path, exist_ok=True)

    print(f"\nProcessing {num_episodes} episodes for task '{task_name}'...")

    ## process each episode
    for idx in tqdm(range(num_episodes), desc=f"Processing {task_name} episodes"):
        json_file = os.path.join(json_path, f"episode_{idx}.json")

        ## Get bimanual modes & contact graphs
        Result = analyze_coordination_modes(json_file_path=json_file)

        ## Save Result to hdf5 file
        hdf5_file = os.path.join(hdf5_path, f"mode_{idx}.hdf5")
        with h5py.File(hdf5_file, 'w') as f:
            # Save Mode
            mode_group = f.create_group('Mode')
            for key, value in Result["Mode"].items():
                # 为每个图状态创建子组
                state_group = mode_group.create_group(key)
                # 分别存储字典中的每个键值对
                for sub_key, sub_value in value.items():
                    state_group.create_dataset(sub_key, data=sub_value)

            # Save Graph
            graph_group = f.create_group('Graph')
            for key, value in Result["Graph"].items():
                graph_group.create_dataset(key, data=list(value))

        ## Visualize the contact-relationship if requested
        visualize_scene_graphs(if_visualize, Result, task_name, json_file, idx, image_path)
            
    print("\nDone!")
    print(f"\nAll episodes processed and saved to {hdf5_path} \nand visualizations saved to {image_path}.\n")


def visualize_scene_graphs(if_visualize, Result, task_name, json_file, idx, image_path):
    if if_visualize and task_name == 'sim_transfer_cube_scripted':
        visualizer = SceneGraphVisualizer_Transfer()
        visualizer.create_comprehensive_visualization(
            Result['Graph'], 
            save_prefix=os.path.join(image_path, f"episode_{idx}"))
        
    if if_visualize and task_name == 'sim_insertion_scripted':
        visualizer = SceneGraphVisualizer_Insertion()
        visualizer.create_comprehensive_visualization(
            episode_json_path=json_file,
            save_prefix=os.path.join(image_path, f"episode_{idx}")
        )
    # if if_visualize and task_name == 'aloha2_screwdriver_fix':
    #     visualizer = SceneGraphVisualizer_Screwdriver()
    #     visualizer.create_comprehensive_visualization(
    #         episode_json_path=json_file,
    #         save_prefix=os.path.join(image_path, f"episode_{idx}")
    #     )

        

if __name__ == '__main__':
    main()