import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from typing import Dict, Set, Tuple, List, Optional

class SceneGraphVisualizer_Transfer:
    """
    Scene Graph Visualizer for robot manipulation tasks.
    Excludes robot-table connections from visualization to focus on meaningful interactions.
    Keeps object-table connections to show support relationships.
    """
    
    def __init__(self):
        # Define colors for different node types
        self.colors = {
            'robot_left': '#FF6B6B',     # Red for left robot
            'robot_right': '#FF9F43',    # Orange for right robot
            'object': '#4ECDC4',         # Teal for objects
            'robot': '#FF6B6B',          # Default robot color
            'table': '#45B7D1',          # Blue for table/surface
            'surface': '#45B7D1'         # Blue for surface
        }
        
        # Define node shapes
        self.shapes = {
            'robot': 's',    # Square for robots
            'object': 'o',   # Circle for objects
            'table': '^',    # Triangle for table/surface
            'surface': '^'   # Triangle for surface
        }
        
        # Define node sizes
        self.sizes = {
            'robot': 2400,
            'object': 2400,
            'table': 2800,
            'surface': 2800
        }
    
    def _filter_edges(self, edges: Set[Tuple[str, str]]) -> Set[Tuple[str, str]]:
        """
        Filter out edges that involve robot-table connections only.
        Keep object-table connections to show support relationships.
        
        Args:
            edges: Set of edge tuples
            
        Returns:
            Filtered set of edges excluding robot-table connections
        """
        filtered_edges = set()
        
        for edge in edges:
            node1, node2 = edge
            # Check if this is a robot-table connection
            is_robot_table_connection = (
                ('robot' in node1.lower() and ('table' in node2.lower() or 'surface' in node2.lower())) or
                ('robot' in node2.lower() and ('table' in node1.lower() or 'surface' in node1.lower()))
            )
            
            # Skip only robot-table connections, keep everything else
            if is_robot_table_connection:
                continue
            filtered_edges.add(edge)
        
        return filtered_edges
    
    def _categorize_node(self, node: str) -> str:
        """
        Categorize a node based on its name.
        
        Args:
            node: Node name
            
        Returns:
            Node category ('robot_left', 'robot_right', 'object', 'table', 'surface')
        """
        node_lower = node.lower()
        if 'robot' in node_lower:
            if 'left' in node_lower:
                return 'robot_left'
            elif 'right' in node_lower:
                return 'robot_right'
            else:
                return 'robot'
        elif 'table' in node_lower:
            return 'table'
        elif 'surface' in node_lower:
            return 'surface'
        else:
            return 'object'
    
    def _get_node_color(self, node: str) -> str:
        """Get color for a specific node."""
        category = self._categorize_node(node)
        return self.colors.get(category, self.colors['object'])
    
    def _get_node_shape(self, node: str) -> str:
        """Get shape for a specific node."""
        category = self._categorize_node(node)
        if 'robot' in category:
            return self.shapes['robot']
        elif category in ['table', 'surface']:
            return self.shapes[category]
        else:
            return self.shapes['object']
    
    def _get_node_size(self, node: str) -> int:
        """Get size for a specific node."""
        category = self._categorize_node(node)
        if 'robot' in category:
            return self.sizes['robot']
        elif category in ['table', 'surface']:
            return self.sizes[category]
        else:
            return self.sizes['object']
    
    def visualize_scene_graphs(self, graphs_data: Dict[str, Set[Tuple[str, str]]], 
                             save_path: Optional[str] = None, 
                             figsize_per_graph: Tuple[int, int] = (4, 5)) -> None:
        """
        Visualize multiple scene graphs showing meaningful interactions.
        
        Args:
            graphs_data: Dictionary with graph names as keys and sets of edges as values
            save_path: Optional path to save the visualization
            figsize_per_graph: Size of each individual graph subplot
        """
        # Filter out robot-table edges from all graphs
        filtered_graphs = {}
        for graph_name, edges in graphs_data.items():
            filtered_edges = self._filter_edges(edges)
            filtered_graphs[graph_name] = filtered_edges  # Include all graphs, even with no filtered edges
        
        num_graphs = len(filtered_graphs)
        fig_width = figsize_per_graph[0] * num_graphs
        fig_height = figsize_per_graph[1]
        
        # Create subplots
        fig, axes = plt.subplots(1, num_graphs, figsize=(fig_width, fig_height))
        if num_graphs == 1:
            axes = [axes]
        
        for idx, (graph_name, edges) in enumerate(filtered_graphs.items()):
            ax = axes[idx]
            
            # Create networkx graph
            G = nx.Graph()
            if edges:
                G.add_edges_from(edges)
            
            # Get all nodes
            nodes = list(G.nodes())
            if not nodes:
                ax.text(0.5, 0.5, 'No meaningful\ninteractions', # type: ignore
                       ha='center', va='center', fontsize=12, 
                       transform=ax.transAxes)# type: ignore
                ax.set_title(f'{graph_name}', fontsize=14, fontweight='bold', pad=20)# type: ignore
                ax.axis('off')# type: ignore
                continue
            
            # Create layout
            if len(nodes) <= 2:
                # Simple layout for small graphs
                pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
            else:
                # More spread out layout for larger graphs
                pos = nx.spring_layout(G, k=4, iterations=200, seed=42)
            
            # Draw edges with enhanced styling
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#2C3E50', 
                                  width=3, alpha=0.7, style='-')
            
            # Draw nodes by category
            for node in nodes:
                color = self._get_node_color(node)
                shape = self._get_node_shape(node)
                size = self._get_node_size(node)
                
                nx.draw_networkx_nodes(G, pos, nodelist=[node],
                                     node_color=color, node_shape=shape,
                                     node_size=size, ax=ax, alpha=0.9,
                                     edgecolors='black', linewidths=2)
            
            # Draw labels with better styling
            labels = {node: node.replace('robot_', '').replace('_', '\n') for node in nodes}
            nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=10, 
                                   font_weight='bold', font_color='white')
            
            # Set title with enhanced styling
            ax.set_title(f'{graph_name}', fontsize=16, fontweight='bold', # type: ignore
                        pad=20, color='#2C3E50')
            ax.axis('off') # type: ignore
        
        # Add enhanced legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', 
                      markerfacecolor=self.colors['robot_left'], 
                      markersize=12, label='Left Robot', markeredgecolor='black'),
            Line2D([0], [0], marker='s', color='w', 
                      markerfacecolor=self.colors['robot_right'], 
                      markersize=12, label='Right Robot', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=self.colors['object'], 
                      markersize=12, label='Object', markeredgecolor='black'),
            Line2D([0], [0], marker='^', color='w', 
                      markerfacecolor=self.colors['table'], 
                      markersize=12, label='Table/Surface', markeredgecolor='black')
        ]
        
        fig.legend(handles=legend_elements, loc='lower center', 
                  bbox_to_anchor=(0.5, 0.95), ncol=4, fontsize=12,
                  frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        
        plt.show()
        plt.close()
    

    
    def create_timeline_visualization(self, graphs_data: Dict[str, Set[Tuple[str, str]]], 
                                    save_path: Optional[str] = None) -> None:
        """
        Create a timeline visualization showing meaningful interaction changes over time.
        
        Args:
            graphs_data: Dictionary with graph names as keys and sets of edges as values
            save_path: Optional path to save the visualization
        """
        # Filter out robot-table edges and get all unique edges
        all_edges = set()
        filtered_graphs = {}
        
        for graph_name, edges in graphs_data.items():
            filtered_edges = self._filter_edges(edges)
            filtered_graphs[graph_name] = filtered_edges
            for edge in filtered_edges:
                # Normalize edge representation (sort to avoid duplicates)
                normalized_edge = tuple(sorted(edge))
                all_edges.add(normalized_edge)
        
        if not all_edges:
            print("No meaningful interactions found for timeline visualization.")
            return
        
        all_edges_sorted = sorted(list(all_edges))
        num_graphs = len(filtered_graphs)
        num_edges = len(all_edges_sorted)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(num_graphs*1.5, 8), max(num_edges*0.8, 6)))
        
        # Create matrix showing which edges exist in which graphs
        edge_matrix = np.zeros((num_edges, num_graphs))
        
        graph_names = list(filtered_graphs.keys())
        
        for graph_idx, (graph_name, edges) in enumerate(filtered_graphs.items()):
            for edge in edges:
                normalized_edge = tuple(sorted(edge))
                if normalized_edge in all_edges_sorted:
                    edge_idx = all_edges_sorted.index(normalized_edge)
                    edge_matrix[edge_idx][graph_idx] = 1
        
        # Plot the matrix with enhanced styling
        im = ax.imshow(edge_matrix, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1) # RdYlBu_r
        
        # Set ticks and labels
        ax.set_xticks(range(num_graphs))
        ax.set_yticks(range(num_edges))
        ax.set_xticklabels(graph_names, fontsize=12, fontweight='bold')
        
        # Format edge labels
        edge_labels = []
        for edge in all_edges_sorted:
            node1, node2 = edge
            label1 = node1.replace('robot_', '').replace('_', ' ').title()
            label2 = node2.replace('robot_', '').replace('_', ' ').title()
            edge_labels.append(f"{label1} ↔ {label2}")
        
        ax.set_yticklabels(edge_labels, fontsize=11)
        
        # Add enhanced text annotations
        for i in range(num_edges):
            for j in range(num_graphs):
                if edge_matrix[i, j] == 1:
                    symbol = '●'
                    color = '#2C3E50'
                    size = 14
                else:
                    symbol = '○'
                    color = '#BDC3C7'
                    size = 12
                
                ax.text(j, i, symbol, ha="center", va="center", 
                       color=color, fontsize=size, fontweight='bold')
        
        # Enhanced styling
        ax.set_title('Meaningful Interaction Timeline', 
                    fontsize=16, fontweight='bold', pad=25, color='#2C3E50')
        ax.set_xlabel('Graph States', fontsize=14, fontweight='bold', color='#2C3E50')
        ax.set_ylabel('Meaningful Connections', fontsize=14, fontweight='bold', color='#2C3E50')
        
        # Add grid for better readability
        ax.set_xticks(np.arange(-0.5, num_graphs, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_edges, 1), minor=True)
        ax.grid(which="minor", color="lightgray", linestyle='-', linewidth=0.5, alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        plt.show()
        plt.close()
    
    def create_comprehensive_visualization(self, graphs_data: Dict[str, Set[Tuple[str, str]]], 
                                         save_prefix: str = "scene_graph") -> None:
        """
        Create all three visualizations and save them.
        
        Args:
            graphs_data: Dictionary with graph names as keys and sets of edges as values
            save_prefix: Prefix for saved files
        """
        # print("Creating comprehensive scene graph visualizations...")
        # print("Excluding robot-table connections to focus on meaningful interactions")
        
        # 1. Network graph visualization
        # print("1. Generating network graph visualization...")
        self.visualize_scene_graphs(graphs_data, save_path=f'{save_prefix}_networks.png')
        
        # 2 Timeline visualization
        # print("2. Generating timeline visualization...")
        self.create_timeline_visualization(graphs_data, save_path=f'{save_prefix}_timeline.png')
        
        # print("All visualizations completed successfully!")

# Example usage
if __name__ == "__main__":
    # Your graph data (robot-table connections will be automatically filtered out)
    graphs_data = {
        'G_1': {('robot_left', 'table'), ('robot_right', 'table'), ('cube', 'table')},
        'G_2': {('robot_right', 'cube'), ('robot_left', 'table'), ('robot_right', 'table'), ('cube', 'table')},
        'G_3': {('robot_right', 'cube'), ('robot_left', 'table'), ('robot_right', 'table')},
        'G_4': {('robot_right', 'table'), ('robot_left', 'cube'), ('robot_left', 'table'), ('robot_right', 'cube')},
        'G_5': {('robot_right', 'table'), ('robot_left', 'cube'), ('robot_left', 'table')}
    }
    
    # Create visualizer instance
    visualizer = SceneGraphVisualizer()
    
    # Option 1: Create all visualizations at once
    visualizer.create_comprehensive_visualization(graphs_data, save_prefix="meaningful_interactions")
    
    # Option 2: Create individual visualizations
    # visualizer.visualize_scene_graphs(graphs_data, save_path='networks_only.png')
    # visualizer.create_interaction_matrix(graphs_data, save_path='matrices_only.png')
    # visualizer.create_timeline_visualization(graphs_data, save_path='timeline_only.png')