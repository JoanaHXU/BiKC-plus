import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from typing import Dict, Set, Tuple, List, Optional
import ast

class SceneGraphVisualizer_Insertion:
    """
    基于 episode JSON 文件的场景图可视化器
    专门处理从 episode.json 文件生成的场景图可视化，包含时间线功能
    """
    
    def __init__(self):
        # 定义不同节点类型的颜色
        self.colors = {
            'robot_left': '#FF6B6B',     # 红色 - 左机械臂
            'robot_right': '#FF9F43',    # 橙色 - 右机械臂
            'object_1': '#4ECDC4',       # 青色 - 物体1
            'object_2': '#A8E6CF',       # 绿色 - 物体2
            'object': '#95A5A6',         # 灰色 - 其他物体
            'table': '#45B7D1',          # 蓝色 - 桌子
        }
        
        # 定义节点形状
        self.shapes = {
            'robot': 's',    # 方形 - 机械臂
            'object': 'o',   # 圆形 - 物体
            'table': '^',    # 三角形 - 桌子
        }
        
        # 定义节点大小
        self.sizes = {
            'robot': 2400,
            'object': 2400,
            'table': 2800,
        }
    
    def _parse_contact_change(self, contact_change_str: str) -> Tuple[Tuple[str, str], str]:
        """解析 contact_change 字符串"""
        try:
            parsed = ast.literal_eval(contact_change_str)
            edge, action = parsed
            return (edge, action)
        except Exception as e:
            print(f"Error parsing contact change: {contact_change_str}, Error: {e}")
            return (("", ""), "")
    
    def _parse_initial_graph(self, initial_graph_str: str) -> Set[Tuple[str, str]]:
        """解析初始图字符串"""
        try:
            edges_list = ast.literal_eval(initial_graph_str)
            return set(edges_list)
        except Exception as e:
            print(f"Error parsing initial graph: {initial_graph_str}, Error: {e}")
            return set()
    
    def _categorize_node(self, node: str) -> str:
        """根据节点名称分类节点"""
        node_lower = node.lower()
        if 'robot_left' in node_lower:
            return 'robot_left'
        elif 'robot_right' in node_lower:
            return 'robot_right'
        elif 'object_1' in node_lower:
            return 'object_1'
        elif 'object_2' in node_lower:
            return 'object_2'
        elif 'object' in node_lower:
            return 'object'
        elif 'table' in node_lower:
            return 'table'
        else:
            return 'object'
    
    def _get_node_color(self, node: str) -> str:
        """获取节点颜色"""
        category = self._categorize_node(node)
        return self.colors.get(category, self.colors['object'])
    
    def _get_node_shape(self, node: str) -> str:
        """获取节点形状"""
        category = self._categorize_node(node)
        if 'robot' in category:
            return self.shapes['robot']
        elif category == 'table':
            return self.shapes['table']
        else:
            return self.shapes['object']
    
    def _get_node_size(self, node: str) -> int:
        """获取节点大小"""
        category = self._categorize_node(node)
        if 'robot' in category:
            return self.sizes['robot']
        elif category == 'table':
            return self.sizes['table']
        else:
            return self.sizes['object']
    
    def generate_graphs_from_episode(self, episode_json_path: str) -> Dict[str, Set[Tuple[str, str]]]:
        """从 episode JSON 文件生成所有图状态"""
        with open(episode_json_path, 'r') as f:
            episode_data = json.load(f)
        
        # 解析初始图
        initial_edges = self._parse_initial_graph(episode_data['initial_graph'])
        
        # 创建图状态字典
        graphs_data = {}
        
        # 添加初始图
        graphs_data['Initial'] = initial_edges.copy()
        
        # 当前边集合（从初始图开始）
        current_edges = initial_edges.copy()
        
        # 处理每个模式变化
        for change in episode_data['ModeChangeDetection']:
            graph_number = change['graph_number']
            frame_idx = change['frame_idx']
            timestep = change.get('timestep_orig', 0)
            description = change.get('description', '')
            
            # 解析接触变化
            (node1, node2), action = self._parse_contact_change(change['contact_change'])
            
            if node1 and node2:  # 确保解析成功
                edge = (node1, node2)
                
                if action == 'add':
                    current_edges.add(edge)
                elif action == 'remove':
                    current_edges.discard(edge)
            
            # 创建图名称，包含更多信息
            graph_name = f"G_{graph_number}\nFrame {frame_idx}\n{description}"
            graphs_data[graph_name] = current_edges.copy()
        
        return graphs_data
    
    def _filter_edges(self, edges: Set[Tuple[str, str]], 
                     filter_robot_table: bool = True) -> Set[Tuple[str, str]]:
        """过滤边，可选择是否过滤机械臂-桌子连接"""
        if not filter_robot_table:
            return edges
        
        filtered_edges = set()
        for edge in edges:
            node1, node2 = edge
            # 检查是否为机械臂-桌子连接
            is_robot_table_connection = (
                ('robot' in node1.lower() and 'table' in node2.lower()) or
                ('robot' in node2.lower() and 'table' in node1.lower())
            )
            
            # 跳过机械臂-桌子连接，保留其他所有连接（包括 object_1 和 object_2 之间的连接）
            if not is_robot_table_connection:
                filtered_edges.add(edge)
        
        return filtered_edges
    
    def _filter_edges_for_timeline(self, edges: Set[Tuple[str, str]], 
                                filter_robot_table: bool = True,
                                filter_object_table: bool = True) -> Set[Tuple[str, str]]:
        """
        专门为时间线可视化过滤边，可选择过滤机械臂-桌子连接和物体-桌子连接
        
        Args:
            edges: 边集合
            filter_robot_table: 是否过滤机械臂-桌子连接
            filter_object_table: 是否过滤物体-桌子连接
            
        Returns:
            过滤后的边集合
        """
        filtered_edges = set()
        for edge in edges:
            node1, node2 = edge
            
            # 检查是否为机械臂-桌子连接
            is_robot_table_connection = (
                ('robot' in node1.lower() and 'table' in node2.lower()) or
                ('robot' in node2.lower() and 'table' in node1.lower())
            )
            
            # 检查是否为物体-桌子连接
            is_object_table_connection = (
                ('object' in node1.lower() and 'table' in node2.lower() and 'robot' not in node1.lower()) or
                ('object' in node2.lower() and 'table' in node1.lower() and 'robot' not in node2.lower())
            )
            
            # 应用过滤条件
            should_filter = False
            
            if filter_robot_table and is_robot_table_connection:
                should_filter = True
                
            if filter_object_table and is_object_table_connection:
                should_filter = True
            
            # 保留未被过滤的边
            if not should_filter:
                filtered_edges.add(edge)
        
        return filtered_edges

    def create_timeline_visualization(self, graphs_data: Dict[str, Set[Tuple[str, str]]], 
                                    save_path: Optional[str] = None,
                                    filter_robot_table: bool = True,
                                    filter_object_table: bool = True,  # 新增参数
                                    highlight_object_connections: bool = True) -> None:
        """
        创建时间线可视化，显示有意义的交互变化，特别突出 object_1 和 object_2 的接触关系
        
        Args:
            graphs_data: 图数据字典
            save_path: 保存路径
            filter_robot_table: 是否过滤机械臂-桌子连接
            filter_object_table: 是否过滤物体-桌子连接
            highlight_object_connections: 是否突出显示物体间连接
        """
        # 过滤边并获取所有唯一边
        all_edges = set()
        filtered_graphs = {}
        object_connections = set()  # 专门记录物体间连接
        
        for graph_name, edges in graphs_data.items():
            # 使用新的过滤方法
            filtered_edges = self._filter_edges_for_timeline(edges, filter_robot_table, filter_object_table)
            filtered_graphs[graph_name] = filtered_edges
            
            for edge in filtered_edges:
                # 标准化边表示（排序以避免重复）
                normalized_edge = tuple(sorted(edge))
                all_edges.add(normalized_edge)
                
                # 检查是否为物体间连接（特别是 object_1 和 object_2）
                if self._is_object_connection(edge):
                    object_connections.add(normalized_edge)
        
        if not all_edges:
            print("No meaningful interactions found for timeline visualization after filtering.")
            return
        
        # 按重要性排序边：物体间连接放在前面
        all_edges_sorted = self._sort_edges_by_importance(list(all_edges), object_connections)
        num_graphs = len(filtered_graphs)
        num_edges = len(all_edges_sorted)
        
        # 创建图形，确保有足够的空间
        fig_width = max(num_graphs * 1.5, 12)
        fig_height = max(num_edges * 0.8, 8)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # 创建矩阵显示哪些边在哪些图中存在
        edge_matrix = np.zeros((num_edges, num_graphs))
        graph_names = list(filtered_graphs.keys())
        
        for graph_idx, (graph_name, edges) in enumerate(filtered_graphs.items()):
            for edge in edges:
                normalized_edge = tuple(sorted(edge))
                if normalized_edge in all_edges_sorted:
                    edge_idx = all_edges_sorted.index(normalized_edge)
                    edge_matrix[edge_idx][graph_idx] = 1
        
        # 绘制矩阵背景
        im = ax.imshow(edge_matrix, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1, alpha=0.3)
        
        # 设置刻度和标签
        ax.set_xticks(range(num_graphs))
        ax.set_yticks(range(num_edges))
        
        # 简化图名称以便显示
        simplified_graph_names = []
        for name in graph_names:
            if '\n' in name:
                parts = name.split('\n')
                simplified = f"{parts[0]}\n{parts[2]}" if len(parts) > 2 else parts[0]
            else:
                simplified = name
            simplified_graph_names.append(simplified)
        
        ax.set_xticklabels(simplified_graph_names, fontsize=10, fontweight='bold', rotation=45)
        
        # 格式化边标签，特别突出物体间连接
        edge_labels = []
        for edge in all_edges_sorted:
            node1, node2 = edge
            label1 = self._format_node_name(node1)
            label2 = self._format_node_name(node2)
            
            # 特别标记物体间连接
            if edge in object_connections:
                if 'object_1' in edge and 'object_2' in edge:
                    edge_label = f"{label1} ↔ {label2}"
                else:
                    edge_label = f"{label1} ↔ {label2}"
            else:
                edge_label = f"{label1} ↔ {label2}"
            
            edge_labels.append(edge_label)
        
        ax.set_yticklabels(edge_labels, fontsize=10)
        
        # 添加增强的文本注释
        for i in range(num_edges):
            for j in range(num_graphs):
                edge = all_edges_sorted[i]
                
                if edge_matrix[i, j] == 1:
                    # 根据连接类型选择不同的符号和颜色
                    if edge in object_connections:
                        if 'object_1' in edge and 'object_2' in edge:
                            symbol = '★'  # 特别突出 object_1 和 object_2 的连接
                            color = '#E74C3C'  # 亮红色
                            size = 20
                        else:
                            symbol = '●'  # 其他物体间连接
                            color = '#F39C12'  # 橙色
                            size = 16
                    else:
                        symbol = '●'  # 普通连接（主要是机械臂-物体连接）
                        color = '#2C3E50'  # 深灰色
                        size = 14
                else:
                    symbol = '○'
                    color = '#BDC3C7'
                    size = 10
                
                ax.text(j, i, symbol, ha="center", va="center", 
                    color=color, fontsize=size, fontweight='bold')
        
        # 增强样式
        filter_info = ""
        if filter_robot_table and filter_object_table:
            filter_info = "(Excluding Robot-Table & Object-Table Connections)"
        elif filter_robot_table:
            filter_info = "(Excluding Robot-Table Connections)"
        elif filter_object_table:
            filter_info = "(Excluding Object-Table Connections)"
        
        title = f'Meaningful Interaction Timeline\n{filter_info}'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=30, color='#2C3E50')
        ax.set_xlabel('Graph States (Temporal Progression)', fontsize=12, fontweight='bold', color='#2C3E50')
        ax.set_ylabel('Meaningful Connections', fontsize=12, fontweight='bold', color='#2C3E50')
        
        # 添加网格以提高可读性
        ax.set_xticks(np.arange(-0.5, num_graphs, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_edges, 1), minor=True)
        ax.grid(which="minor", color="lightgray", linestyle='-', linewidth=0.5, alpha=0.7)
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C', 
                    markersize=12, label='★ Object₁ ↔ Object₂ Connection', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#F39C12', 
                    markersize=10, label='● Other Object Connections', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2C3E50', 
                    markersize=8, label='● Robot-Object Connections', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#BDC3C7', 
                    markersize=6, label='○ No Connection', markeredgecolor='gray')
        ]
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
                fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        # 添加文本说明
        if object_connections:
            object_connection_count = sum(1 for edge in object_connections if 'object_1' in edge and 'object_2' in edge)
            if object_connection_count > 0:
                ax.text(0.02, 0.98, f'Critical Object-Object Connections Found: {object_connection_count}', 
                    transform=ax.transAxes, fontsize=12, fontweight='bold', 
                    color='#E74C3C', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # 添加过滤信息说明
        filtered_count = sum(len(edges) for edges in graphs_data.values()) - sum(len(edges) for edges in filtered_graphs.values())
        if filtered_count > 0:
            ax.text(0.02, 0.02, f'Filtered out {filtered_count} table connections', 
                transform=ax.transAxes, fontsize=10, style='italic', 
                color='#7F8C8D', bbox=dict(boxstyle="round,pad=0.2", facecolor='#ECF0F1', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
            print(f"Timeline visualization saved to: {save_path}")
        
        plt.show()
        plt.close()
    
    def _is_object_connection(self, edge: Tuple[str, str]) -> bool:
        """判断是否为物体间连接"""
        node1, node2 = edge
        return ('object' in node1.lower() and 'object' in node2.lower() and 
                'robot' not in node1.lower() and 'robot' not in node2.lower() and
                'table' not in node1.lower() and 'table' not in node2.lower())
    
    def _sort_edges_by_importance(self, edges: List[Tuple[str, str]], 
                                object_connections: Set[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """按重要性排序边，物体间连接优先"""
        # 分类边
        object_object_edges = []  # object_1 和 object_2 之间
        other_object_edges = []   # 其他物体间连接
        robot_object_edges = []   # 机械臂-物体连接
        other_edges = []          # 其他连接
        
        for edge in edges:
            if edge in object_connections:
                if 'object_1' in edge and 'object_2' in edge:
                    object_object_edges.append(edge)
                else:
                    other_object_edges.append(edge)
            elif any('robot' in node and 'object' in node for node in edge if 'table' not in node):
                robot_object_edges.append(edge)
            else:
                other_edges.append(edge)
        
        # 按重要性顺序返回
        return (sorted(object_object_edges) + sorted(other_object_edges) + 
                sorted(robot_object_edges) + sorted(other_edges))
    
    def _format_node_name(self, node: str) -> str:
        """格式化节点名称以便显示"""
        formatted = node.replace('robot_', '').replace('_', ' ').title()
        if 'Object 1' in formatted:
            return 'Object₁'
        elif 'Object 2' in formatted:
            return 'Object₂'
        return formatted
    
    def visualize_episode_scene_graphs(self, episode_json_path: str, 
                                     save_path: Optional[str] = None,
                                     filter_robot_table: bool = True,
                                     figsize_per_graph: Tuple[int, int] = (4, 5)) -> None:
        """可视化episode的所有场景图"""
        # 从 episode 文件生成图数据
        graphs_data = self.generate_graphs_from_episode(episode_json_path)
        
        # 过滤图数据
        filtered_graphs = {}
        for graph_name, edges in graphs_data.items():
            filtered_edges = self._filter_edges(edges, filter_robot_table)
            filtered_graphs[graph_name] = filtered_edges
        
        num_graphs = len(filtered_graphs)
        fig_width = figsize_per_graph[0] * num_graphs
        fig_height = figsize_per_graph[1]
        
        # 创建子图
        fig, axes = plt.subplots(1, num_graphs, figsize=(fig_width, fig_height))
        if num_graphs == 1:
            axes = [axes]
        
        for idx, (graph_name, edges) in enumerate(filtered_graphs.items()):
            ax = axes[idx]
            
            # 创建 networkx 图
            G = nx.Graph()
            if edges:
                G.add_edges_from(edges)
            
            # 获取所有节点
            nodes = list(G.nodes())
            if not nodes:
                ax.text(0.5, 0.5, 'No meaningful\ninteractions',
                       ha='center', va='center', fontsize=12,
                       transform=ax.transAxes)
                ax.set_title(f'{graph_name}', fontsize=12, fontweight='bold', pad=20)
                ax.axis('off')
                continue
            
            # 创建布局
            if len(nodes) <= 2:
                pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
            else:
                pos = nx.spring_layout(G, k=4, iterations=200, seed=42)
            
            # 绘制边 - 特别突出物体间连接
            for edge in G.edges():
                if self._is_object_connection(edge):
                    if 'object_1' in edge and 'object_2' in edge:
                        # 特别突出 object_1 和 object_2 的连接
                        nx.draw_networkx_edges(G, pos, edgelist=[edge], ax=ax, 
                                             edge_color='#E74C3C', width=5, alpha=0.9)
                    else:
                        # 其他物体间连接
                        nx.draw_networkx_edges(G, pos, edgelist=[edge], ax=ax, 
                                             edge_color='#F39C12', width=4, alpha=0.8)
                else:
                    # 普通连接
                    nx.draw_networkx_edges(G, pos, edgelist=[edge], ax=ax, 
                                         edge_color='#2C3E50', width=3, alpha=0.7)
            
            # 按类别绘制节点
            for node in nodes:
                color = self._get_node_color(node)
                shape = self._get_node_shape(node)
                size = self._get_node_size(node)
                
                nx.draw_networkx_nodes(G, pos, nodelist=[node],
                                     node_color=color, node_shape=shape,
                                     node_size=size, ax=ax, alpha=0.9,
                                     edgecolors='black', linewidths=2)
            
            # 绘制标签
            labels = {node: node.replace('_', '\n') for node in nodes}
            nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=9,
                                   font_weight='bold', font_color='white')
            
            # 设置标题
            ax.set_title(f'{graph_name}', fontsize=11, fontweight='bold',
                        pad=20, color='#2C3E50')
            ax.axis('off')
        
        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='s', color='w',
                  markerfacecolor=self.colors['robot_left'],
                  markersize=12, label='Left Robot', markeredgecolor='black'),
            Line2D([0], [0], marker='s', color='w',
                  markerfacecolor=self.colors['robot_right'],
                  markersize=12, label='Right Robot', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=self.colors['object_1'],
                  markersize=12, label='Object 1', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=self.colors['object_2'],
                  markersize=12, label='Object 2', markeredgecolor='black'),
            Line2D([0], [0], marker='^', color='w',
                  markerfacecolor=self.colors['table'],
                  markersize=12, label='Table', markeredgecolor='black')
        ]
        
        fig.legend(handles=legend_elements, loc='lower center',
                  bbox_to_anchor=(0.5, 0.95), ncol=5, fontsize=11,
                  frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Scene graph visualization saved to: {save_path}")
        
        plt.show()
        plt.close()
    
    def create_comprehensive_visualization(self, episode_json_path: str, 
                                         save_prefix: str = "episode_scene_graph") -> None:
        """
        创建所有可视化并保存
        
        Args:
            episode_json_path: episode.json 文件路径
            save_prefix: 保存文件的前缀
        """
        print("Creating comprehensive scene graph visualizations...")
        print("Highlighting object-object connections, especially Object₁ ↔ Object₂")
        
        # 从 episode 文件生成图数据
        graphs_data = self.generate_graphs_from_episode(episode_json_path)
        
        # 1. 网络图可视化
        print("1. Generating network graph visualization...")
        self.visualize_episode_scene_graphs(
            episode_json_path, 
            save_path=f'{save_prefix}_networks.png'
        )
        
        # 2. 时间线可视化
        print("2. Generating timeline visualization...")
        self.create_timeline_visualization(
            graphs_data, 
            save_path=f'{save_prefix}_timeline.png'
        )
        
        print("All visualizations completed successfully!")
    
    def print_episode_summary(self, episode_json_path: str) -> None:
        """打印 episode 的摘要信息"""
        graphs_data = self.generate_graphs_from_episode(episode_json_path)
        
        print(f"Episode Scene Graph Summary:")
        print(f"Total graphs: {len(graphs_data)}")
        print("\nGraph progression:")
        
        for graph_name, edges in graphs_data.items():
            # 检查是否包含 object_1 和 object_2 之间的连接
            object_connection = any(
                (('object_1' in edge[0] and 'object_2' in edge[1]) or
                 ('object_2' in edge[0] and 'object_1' in edge[1]))
                for edge in edges
            )
            
            connection_indicator = " [OBJECT₁↔OBJECT₂ CONNECTION!]" if object_connection else ""
            print(f"{graph_name}: {len(edges)} edges{connection_indicator}")
            for edge in sorted(edges):
                edge_indicator = "" if (('object_1' in edge[0] and 'object_2' in edge[1]) or 
                                          ('object_2' in edge[0] and 'object_1' in edge[1])) else ""
                print(f"  - {edge[0]} ↔ {edge[1]}{edge_indicator}")
            print()

# 使用示例
if __name__ == "__main__":
    # 创建可视化器实例
    visualizer = EpisodeSceneGraphVisualizer()
    
    # episode.json 文件路径
    episode_json_path = "/home/xuhang/TASE/kp_dataset/sim_insertion_scripted/json_sg/episode_0.json"
    
    # 1. 打印 episode 摘要（显示 object_1 和 object_2 的连接）
    print("=== Episode Analysis ===")
    visualizer.print_episode_summary(episode_json_path)
    
    # 2. 创建完整的可视化（包括时间线）
    print("=== Generating Complete Visualizations ===")
    visualizer.create_comprehensive_visualization(
        episode_json_path=episode_json_path,
        save_prefix="episode_0_complete"
    )
    
    # 3. 单独创建时间线可视化
    print("=== Generating Timeline Visualization Only ===")
    graphs_data = visualizer.generate_graphs_from_episode(episode_json_path)
    visualizer.create_timeline_visualization(
        graphs_data=graphs_data,
        save_path="episode_0_timeline_highlight_objects.png",
        filter_robot_table=True,
        highlight_object_connections=True
    )