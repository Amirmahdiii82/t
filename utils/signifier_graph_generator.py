import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
import sys

def create_publication_graph(json_path, agent_name):
    """Create a clean, publication-ready network visualization from the JSON data"""
    
    # Create output directory in agent folder
    agent_dir = os.path.dirname(json_path)
    output_dir = os.path.join(agent_dir, 'graphs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output paths
    output_path = os.path.join(output_dir, f"signifier_graph_publication.png")
    
    try:
        # Load JSON data
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # Check if data contains signifier_graph
        if "signifier_graph" not in data or not data["signifier_graph"]:
            print(f"⚠️ No signifier graph found in {json_path}")
            return False
            
        # Create directed graph
        G = nx.DiGraph()

        # Add nodes with attributes
        for node in data["signifier_graph"]["nodes"]:
            G.add_node(
                node["id"],
                activation=node.get("activation", 0.5),
                repression=node.get("repressed", 0.5)
            )

        # Add edges with attributes
        for edge in data["signifier_graph"]["edges"]:
            G.add_edge(
                edge["source"],
                edge["target"],
                weight=edge.get("weight", 1.0),
                edge_type=edge.get("type", "neutral")
            )

        # Create figure with white background
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
        ax.set_facecolor('white')
        
        # Use Kamada-Kawai layout for better node distribution
        pos = nx.kamada_kawai_layout(G)
        
        # Create a professional color scheme
        # Blues for neutral connections, reds for displacements
        neutral_color = '#1f77b4'  # Standard matplotlib blue
        displacement_color = '#d62728'  # Standard matplotlib red
        
        # Node colormap - professional blue gradient
        node_cmap = plt.cm.Blues
        
        # Draw edges
        for (u, v, d) in G.edges(data=True):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            # Edge styling based on type
            if d['edge_type'] == 'displacement':
                edge_color = displacement_color
                edge_style = '--'
            else:
                edge_color = neutral_color
                edge_style = '-'
            
            # Calculate appropriate curvature
            rad = 0.1 if d['edge_type'] == 'displacement' else 0
            
            # Draw edge with appropriate width based on weight
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                      arrowprops=dict(arrowstyle='->', 
                                    connectionstyle=f"arc3,rad={rad}",
                                    color=edge_color,
                                    lw=1.0 + d['weight'],
                                    linestyle=edge_style))
        
        # Prepare node properties
        node_sizes = []
        node_colors = []
        node_edges = []
        
        for node in G.nodes():
            activation = G.nodes[node]['activation']
            repression = G.nodes[node]['repression']
            
            # Size based on degree centrality
            degree = G.degree(node)
            node_sizes.append(300 + 100 * degree)
            
            # Color based on activation level (blue gradient)
            node_colors.append(node_cmap(activation))
            
            # Edge color based on repression
            if repression > 0.7:
                node_edges.append(displacement_color)
            else:
                node_edges.append('black')
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                             node_size=node_sizes,
                             node_color=node_colors,
                             edgecolors=node_edges,
                             linewidths=1.5,
                             alpha=0.9)
        
        # Draw labels with better readability
        label_pos = {node: (coords[0], coords[1] - 0.03) for node, coords in pos.items()}
        nx.draw_networkx_labels(G, label_pos, 
                              font_size=10,
                              font_weight='bold',
                              font_color='black')
        
        # Add a title
        plt.title(f"{agent_name}'s Signifier Network", fontsize=16, fontweight='bold')
        
        # Add a legend
        legend_elements = [
            plt.Line2D([0], [0], color=neutral_color, lw=2, label='Neutral Connection'),
            plt.Line2D([0], [0], color=displacement_color, lw=2, linestyle='--', label='Displacement'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=node_cmap(0.8), 
                       markersize=10, label='High Activation', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=node_cmap(0.2), 
                       markersize=10, label='Low Activation', markeredgecolor='black')
        ]
        
        plt.legend(handles=legend_elements, 
                 loc='lower right',
                 frameon=True,
                 fancybox=False,
                 shadow=False,
                 fontsize=9)
        
        # Remove axes
        plt.axis('off')
        
        # Save with high DPI
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Publication graph saved to {output_path}")
        
        # Create hierarchical and community graphs
        create_hierarchical_graph(data, agent_name, output_dir)
        create_community_graph(data, agent_name, output_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {json_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_hierarchical_graph(data, agent_name, output_dir):
    """Create a hierarchical layout visualization suitable for publication"""
    
    output_path = os.path.join(output_dir, f"signifier_graph_hierarchical.png")
    
    try:
        G = nx.DiGraph()
        
        # Add nodes and edges
        for node in data["signifier_graph"]["nodes"]:
            G.add_node(node["id"], **node)
        
        for edge in data["signifier_graph"]["edges"]:
            G.add_edge(edge["source"], edge["target"], **edge)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
        ax.set_facecolor('white')
        
        # Compute node levels for hierarchical layout
        # First, find root nodes (nodes with no incoming edges or lowest in-degree)
        in_degrees = dict(G.in_degree())
        min_in_degree = min(in_degrees.values()) if in_degrees else 0
        roots = [n for n, d in in_degrees.items() if d == min_in_degree]
        
        # Assign levels based on distance from roots
        levels = {}
        for root in roots:
            # BFS to assign levels
            queue = [(root, 0)]
            visited = set([root])
            while queue:
                node, level = queue.pop(0)
                # Update level to maximum distance from any root
                levels[node] = max(levels.get(node, 0), level)
                # Add unvisited neighbors to queue
                for neighbor in G.successors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, level + 1))
        
        # For any nodes not reached, assign to level 0
        for node in G.nodes():
            if node not in levels:
                levels[node] = 0
        
        # Create a hierarchical layout
        # Group nodes by level
        hierarchy = {}
        for node, level in levels.items():
            if level not in hierarchy:
                hierarchy[level] = []
            hierarchy[level].append(node)
        
        # Position nodes by level
        pos = {}
        level_count = len(hierarchy)
        for level, nodes in hierarchy.items():
            y = 1.0 - (level / max(1, level_count - 1))  # Normalize to 0-1 range
            node_count = len(nodes)
            for i, node in enumerate(nodes):
                x = (i + 0.5) / max(1, node_count)  # Evenly space nodes horizontally
                pos[node] = (x, y)
        
        # Professional color scheme
        neutral_color = '#1f77b4'  # Blue
        displacement_color = '#d62728'  # Red
        node_cmap = plt.cm.Blues
        
        # Draw edges
        for (u, v, d) in G.edges(data=True):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            # Edge styling
            if d.get('type') == 'displacement':
                edge_color = displacement_color
                edge_style = '--'
            else:
                edge_color = neutral_color
                edge_style = '-'
            
            # Draw edge
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                      arrowprops=dict(arrowstyle='->', 
                                    connectionstyle="arc3,rad=0.1",
                                    color=edge_color,
                                    lw=1.0 + d['weight'],
                                    linestyle=edge_style))
        
        # Prepare node properties
        node_sizes = []
        node_colors = []
        node_edges = []
        
        for node in G.nodes():
            activation = G.nodes[node].get('activation', 0.5)
            repression = G.nodes[node].get('repressed', 0.5)
            
            # Size based on connections
            connections = len(list(G.predecessors(node))) + len(list(G.successors(node)))
            node_sizes.append(300 + 50 * connections)
            
            # Color based on activation
            node_colors.append(node_cmap(activation))
            
            # Edge color based on repression
            if repression > 0.7:
                node_edges.append(displacement_color)
            else:
                node_edges.append('black')
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                             node_size=node_sizes,
                             node_color=node_colors,
                             edgecolors=node_edges,
                             linewidths=1.5,
                             alpha=0.9)
        
        # Draw labels
        label_pos = {node: (coords[0], coords[1] - 0.02) for node, coords in pos.items()}
        nx.draw_networkx_labels(G, label_pos, 
                              font_size=10,
                              font_weight='bold',
                              font_color='black')
        
        # Add level labels
        for level, y_pos in {level: 1.0 - (level / max(1, level_count - 1)) for level in hierarchy.keys()}.items():
            ax.text(-0.05, y_pos, f"Level {level}", 
                   fontsize=10, 
                   ha='right', 
                   va='center',
                   fontweight='bold')
        
        # Add title
        plt.title(f"{agent_name}'s Hierarchical Signifier Network", fontsize=16, fontweight='bold')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color=neutral_color, lw=2, label='Neutral Connection'),
            plt.Line2D([0], [0], color=displacement_color, lw=2, linestyle='--', label='Displacement'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=node_cmap(0.8), 
                       markersize=10, label='High Activation', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=node_cmap(0.2), 
                       markersize=10, label='Low Activation', markeredgecolor='black')
        ]
        
        plt.legend(handles=legend_elements, 
                 loc='best',
                 frameon=True,
                 fancybox=False,
                 shadow=False,
                 fontsize=9)
        
        # Remove axes
        plt.axis('off')
        
        # Save with high DPI
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Hierarchical graph saved to {output_path}")
        
    except Exception as e:
        print(f"❌ Error creating hierarchical graph: {e}")

def create_community_graph(data, agent_name, output_dir):
    """Create a publication-ready visualization highlighting community structure"""
    
    output_path = os.path.join(output_dir, f"signifier_graph_communities.png")
    
    try:
        G = nx.DiGraph()
        
        # Add nodes and edges
        for node in data["signifier_graph"]["nodes"]:
            G.add_node(node["id"], **node)
        
        for edge in data["signifier_graph"]["edges"]:
            G.add_edge(edge["source"], edge["target"], **edge)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(9, 7), facecolor='white')
        ax.set_facecolor('white')
        
        # Use spring layout with higher repulsion for clearer communities
        pos = nx.spring_layout(G, k=0.4, seed=42)
        
        # Simple community detection based on node connectivity
        # Group nodes by their connection patterns
        connection_patterns = {}
        for node in G.nodes():
            # Create a signature based on connections
            successors = set(G.successors(node))
            predecessors = set(G.predecessors(node))
            # Use the number of connections as a simple community identifier
            key = len(successors) + len(predecessors)
            if key not in connection_patterns:
                connection_patterns[key] = []
            connection_patterns[key].append(node)
        
        # Assign community IDs
        partition = {}
        for community_id, (_, nodes) in enumerate(connection_patterns.items()):
            for node in nodes:
                partition[node] = community_id
        
        # Get number of communities
        community_count = len(set(partition.values()))
        
        # Create a colormap for communities - use colorblind-friendly palette
        community_colors = plt.cm.Set3(np.linspace(0, 1, community_count))
        
        # Edge colors
        neutral_color = '#1f77b4'  # Blue
        displacement_color = '#d62728'  # Red
        
        # Draw edges
        for (u, v, d) in G.edges(data=True):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            # Edge styling
            if d.get('type') == 'displacement':
                edge_color = displacement_color
                edge_style = '--'
            else:
                edge_color = neutral_color
                edge_style = '-'
            
            # Draw edge with weight determining width
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                      arrowprops=dict(arrowstyle='->', 
                                    connectionstyle="arc3,rad=0.1",
                                    color=edge_color,
                                    lw=0.8 + d['weight'] * 0.8,
                                    linestyle=edge_style,
                                    alpha=0.8))
        
        # Draw nodes with community colors
        for node, (x, y) in pos.items():
            activation = G.nodes[node].get('activation', 0.5)
            repression = G.nodes[node].get('repressed', 0.5)
            community = partition[node]
            
            # Node size based on connections
            connections = len(list(G.predecessors(node))) + len(list(G.successors(node)))
            size = 300 + 50 * connections
            
            # Node color based on community
            color = community_colors[community]
            
            # Draw node
            nx.draw_networkx_nodes(G, {node: (x, y)}, 
                                 nodelist=[node],
                                 node_size=size,
                                 node_color=[color],
                                 edgecolors='black',
                                 linewidths=1.0,
                                 alpha=0.9)
            
            # Draw label
            nx.draw_networkx_labels(G, {node: (x, y-0.02)}, 
                                  labels={node: node},
                                  font_size=9,
                                  font_weight='bold',
                                  font_color='black')
        
        # Add title
        plt.title(f"Community Structure in {agent_name}'s Network", fontsize=16, fontweight='bold')
        
        # Create legend for communities
        community_names = {}
        for community_id in range(community_count):
            # Get nodes in this community
            nodes = [node for node, comm_id in partition.items() if comm_id == community_id]
            # Use first node as representative or create generic name
            if nodes:
                name = f"Group {community_id+1}"
                community_names[community_id] = name
        
        # Legend for communities and edge types
        legend_elements = [
            plt.Line2D([0], [0], color=neutral_color, lw=2, label='Neutral Connection'),
            plt.Line2D([0], [0], color=displacement_color, lw=2, linestyle='--', label='Displacement')
        ]
        
        # Add community elements to legend
        for comm_id, name in community_names.items():
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', 
                         markerfacecolor=community_colors[comm_id], 
                         markersize=10, 
                         label=name,
                         markeredgecolor='black')
            )
        
        plt.legend(handles=legend_elements, 
                 loc='lower right',
                 frameon=True,
                 fancybox=False,
                 shadow=False,
                 fontsize=8)
        
        # Remove axes
        plt.axis('off')
        
        # Save with high DPI
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Community graph saved to {output_path}")
        
    except Exception as e:
        print(f"❌ Error creating community graph: {e}")

if __name__ == "__main__":
    # Get agent name from command line or use default
    if len(sys.argv) > 1:
        agent_name = sys.argv[1]
    else:
        agent_name = "Nancy"  # Default agent
    
    # Build path to agent's unconscious memory file
    base_path = 'base_agents'
    json_path = os.path.join(base_path, agent_name, 'unconscious_memory.json')
    
    print(f"Generating signifier graphs for agent: {agent_name}")
    
    # Check if file exists
    if not os.path.exists(json_path):
        print(f"Error: File not found at {json_path}")
        print("Please provide a valid agent name or check file path.")
        sys.exit(1)
    
    # Create graphs
    create_publication_graph(json_path, agent_name)
    
    print(f"\nAll graphs created for {agent_name} in {os.path.join(base_path, agent_name, 'graphs')}")