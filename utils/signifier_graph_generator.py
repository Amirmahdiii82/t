import networkx as nx
import matplotlib.pyplot as plt
import json
import numpy as np
import os
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

def create_publication_ready_signifier_graph(json_path, agent_name):
    """Create a clean, publication-ready signifier network visualization."""
    
    # Create output directory
    agent_dir = os.path.dirname(json_path)
    output_dir = os.path.join(agent_dir, 'graphs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "signifier_network.png")
    
    try:
        # Load JSON data
        with open(json_path, "r") as f:
            data = json.load(f)
        
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
                repressed=node.get("repressed", False),
                type=node.get("type", "symbolic")
            )
        
        # Add edges
        for edge in data["signifier_graph"]["edges"]:
            G.add_edge(
                edge["source"],
                edge["target"],
                weight=edge.get("weight", 0.5),
                edge_type=edge.get("type", "neutral")
            )
        
        # Create figure with white background
        fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
        ax.set_facecolor('white')
        
        # Use force-directed layout with optimal parameters
        pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
        
        # Define color scheme
        node_colors = {
            'symbolic': '#2E86AB',    # Deep blue
            'imaginary': '#A23B72',   # Purple
            'real': '#F18F01'         # Orange
        }
        
        edge_colors = {
            'neutral': '#666666',
            'displacement': '#E63946',
            'condensation': '#06D6A0'
        }
        
        # Draw edges first (so they appear behind nodes)
        for (u, v, d) in G.edges(data=True):
            edge_type = d.get('edge_type', 'neutral')
            edge_color = edge_colors.get(edge_type, '#666666')
            edge_width = 1 + d.get('weight', 0.5) * 2
            
            # Draw edge
            ax.annotate('', xy=pos[v], xytext=pos[u],
                       arrowprops=dict(
                           arrowstyle='-|>',
                           connectionstyle="arc3,rad=0.1",
                           color=edge_color,
                           lw=edge_width,
                           alpha=0.7
                       ))
        
        # Calculate node sizes based on degree centrality
        degree_centrality = nx.degree_centrality(G)
        node_sizes = [300 + 2000 * degree_centrality[node] for node in G.nodes()]
        
        # Draw nodes
        for i, (node, (x, y)) in enumerate(pos.items()):
            node_data = G.nodes[node]
            node_type = node_data.get('type', 'symbolic')
            activation = node_data.get('activation', 0.5)
            repressed = node_data.get('repressed', False)
            
            # Node color based on type
            color = node_colors.get(node_type, '#2E86AB')
            
            # Adjust alpha based on activation
            alpha = 0.4 + 0.6 * activation
            
            # Draw node
            circle = plt.Circle((x, y), radius=np.sqrt(node_sizes[i])/150, 
                               color=color, alpha=alpha, ec='black', 
                               linewidth=3 if repressed else 1.5)
            ax.add_patch(circle)
            
            # Add label
            ax.text(x, y, node, fontsize=11, ha='center', va='center',
                   weight='bold', color='white' if activation > 0.5 else 'black')
        
        # Add title
        ax.text(0.5, 1.05, f"{agent_name}'s Unconscious Signifier Network",
               transform=ax.transAxes, ha='center', fontsize=18, weight='bold')
        
        # Create legend
        legend_elements = []
        
        # Node type legend
        for node_type, color in node_colors.items():
            legend_elements.append(
                mpatches.Circle((0, 0), 0.1, facecolor=color, edgecolor='black',
                               label=f'{node_type.capitalize()} signifier')
            )
        
        # Edge type legend
        legend_elements.extend([
            plt.Line2D([0], [0], color=edge_colors['displacement'], lw=2,
                      label='Displacement'),
            plt.Line2D([0], [0], color=edge_colors['condensation'], lw=2,
                      label='Condensation'),
            plt.Line2D([0], [0], color='black', lw=3,
                      label='Repressed (thick border)')
        ])
        
        ax.legend(handles=legend_elements, loc='best', fontsize=10,
                 frameon=True, fancybox=False, edgecolor='black')
        
        # Add subtle grid
        ax.grid(True, alpha=0.1, linestyle='--')
        
        # Set axis limits with padding
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(xlim[0] - 0.1 * (xlim[1] - xlim[0]), 
                    xlim[1] + 0.1 * (xlim[1] - xlim[0]))
        ax.set_ylim(ylim[0] - 0.1 * (ylim[1] - ylim[0]), 
                    ylim[1] + 0.1 * (ylim[1] - ylim[0]))
        
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add subtle frame
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(1)
        
        # Save with high quality
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Publication-ready signifier graph saved to {output_path}")
        
        # Generate graph statistics
        stats = {
            "Total signifiers": G.number_of_nodes(),
            "Total connections": G.number_of_edges(),
            "Average connections per signifier": f"{2 * G.number_of_edges() / G.number_of_nodes():.2f}",
            "Most central signifier": max(degree_centrality, key=degree_centrality.get),
            "Network density": f"{nx.density(G):.3f}",
            "Repressed signifiers": sum(1 for n, d in G.nodes(data=True) if d.get('repressed', False))
        }
        
        print("\n📊 Graph Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating signifier graph: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    agent_name = sys.argv[1] if len(sys.argv) > 1 else "Nancy"
    json_path = os.path.join('base_agents', agent_name, 'unconscious_memory.json')
    
    if os.path.exists(json_path):
        create_publication_ready_signifier_graph(json_path, agent_name)
    else:
        print(f"Error: File not found at {json_path}")