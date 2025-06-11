import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from datetime import datetime
import os
from typing import Dict
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import seaborn as sns
from matplotlib.patheffects import withStroke

class AgentVisualizer:
    """
    Publication-quality visualizations for psychoanalytic AI agents.
    
    Creates clean, professional visualizations that demonstrate the sophisticated
    emotional and unconscious dynamics of the psychoanalytic architecture.
    """
    
    def __init__(self, agent_name: str, base_path: str = "base_agents"):
        self.agent_name = agent_name
        self.agent_path = os.path.join(base_path, agent_name)
        self.plots_dir = os.path.join(self.agent_path, "plots")
        
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Publication-ready style settings
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'axes.linewidth': 1.5,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.size': 7,
            'xtick.minor.size': 4,
            'ytick.major.size': 7,
            'ytick.minor.size': 4,
            'figure.dpi': 300
        })
        
    def create_all_visualizations(self) -> Dict[str, bool]:
        """Create all publication-quality visualizations."""
        results = {}
        results['pad_journey'] = self.create_pad_journey()
        results['signifier_graph'] = self.create_signifier_graph()
        return results
    
    def create_pad_journey(self) -> bool:
        """Create a simple, powerful PAD emotional journey showing only agent changes."""
        try:
            neuroproxy_path = os.path.join(self.agent_path, "neuroproxy_state.json")
            if not os.path.exists(neuroproxy_path):
                return False
            
            with open(neuroproxy_path, 'r') as f:
                data = json.load(f)
            
            emotional_history = data.get('emotional_history', [])
            if len(emotional_history) < 2:
                return False
            
            # Filter to show ONLY agent emotional changes
            agent_data = []
            step_counter = 0
            for entry in emotional_history:
                if entry.get('context') == 'agent_response':  # Only agent responses
                    pad_state = entry.get('resulting_pad', {})
                    agent_data.append({
                        'step': step_counter,
                        'pleasure': pad_state.get('pleasure', 0),
                        'arousal': pad_state.get('arousal', 0),
                        'dominance': pad_state.get('dominance', 0)
                    })
                    step_counter += 1
            
            if len(agent_data) < 2:
                return False
            
            df = pd.DataFrame(agent_data)
            
            # Single, focused plot
            fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
            
            # Draw trajectory path with clear lines
            for i in range(len(df) - 1):
                x1, y1 = df.iloc[i]['pleasure'], df.iloc[i]['arousal']
                x2, y2 = df.iloc[i+1]['pleasure'], df.iloc[i+1]['arousal']
                
                # Thick, visible trajectory line
                ax.plot([x1, x2], [y1, y2], color='#2C3E50', linewidth=4, alpha=0.8, zorder=1)
                
                # Add arrow to show direction
                if i % 2 == 0:  # Every 2nd step to avoid clutter
                    dx, dy = x2 - x1, y2 - y1
                    if abs(dx) > 0.001 or abs(dy) > 0.001:  # Only if there's movement
                        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                                   arrowprops=dict(arrowstyle='->', color='#2C3E50', 
                                                 lw=3, alpha=0.9))
            
            # Plot agent emotional states
            for i, row in df.iterrows():
                # Size based on dominance (confidence)
                size = 300 + row['dominance'] * 1000
                
                # Color gradient based on emotional trajectory
                if i == 0:
                    color = '#27AE60'  # Green for start
                elif i == len(df) - 1:
                    color = '#E74C3C'  # Red for end
                else:
                    # Blue gradient for middle points
                    intensity = i / (len(df) - 1)
                    color = plt.cm.Blues(0.4 + intensity * 0.5)
                
                ax.scatter(row['pleasure'], row['arousal'], c=color, s=size, 
                          alpha=0.9, edgecolors='black', linewidth=2, zorder=5)
                
                # Add step numbers for clarity
                ax.text(row['pleasure'], row['arousal'], str(i+1), 
                       ha='center', va='center', fontsize=12, fontweight='bold',
                       color='white', zorder=10,
                       path_effects=[withStroke(linewidth=3, foreground='black')])
            
            # Add quadrant labels to show emotional space
            ax.text(0.8, 0.8, 'High Pleasure\nHigh Arousal\n(Excited)', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, alpha=0.6, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.3))
            
            ax.text(0.2, 0.8, 'Low Pleasure\nHigh Arousal\n(Stressed)', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, alpha=0.6, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.3))
            
            ax.text(0.8, 0.2, 'High Pleasure\nLow Arousal\n(Calm)', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, alpha=0.6, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.3))
            
            ax.text(0.2, 0.2, 'Low Pleasure\nLow Arousal\n(Sad)', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, alpha=0.6, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.3))
            
            # Clean styling
            ax.set_xlabel('Pleasure (Valence)', fontsize=16, fontweight='bold')
            ax.set_ylabel('Arousal (Activation)', fontsize=16, fontweight='bold')
            ax.set_title(f'{self.agent_name}: Agent Emotional Evolution in PAD Space', 
                        fontsize=18, fontweight='bold', pad=20)
            
            # Grid and center lines
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.axhline(df['arousal'].mean(), color='red', linestyle=':', alpha=0.5, linewidth=2)
            ax.axvline(df['pleasure'].mean(), color='red', linestyle=':', alpha=0.5, linewidth=2)
            
            # Single legend positioned to avoid overlap
            legend_elements = [
                plt.scatter([], [], c='#27AE60', s=300, marker='o', 
                           label='Start State', edgecolors='black'),
                plt.scatter([], [], c='#E74C3C', s=300, marker='o', 
                           label='End State', edgecolors='black'),
                plt.Line2D([0], [0], color='#2C3E50', linewidth=4, 
                          label='Emotional Trajectory'),
                plt.scatter([], [], c='gray', s=150, marker='o', 
                           label='Small = Low Confidence', edgecolors='black', alpha=0.6),
                plt.scatter([], [], c='gray', s=400, marker='o', 
                           label='Large = High Confidence', edgecolors='black', alpha=0.6)
            ]
            
            ax.legend(handles=legend_elements, loc='upper right', 
                     frameon=True, fancybox=True, shadow=True, fontsize=11)
            
            # Add emotional statistics box in bottom left to avoid overlap
            stats_text = f"Agent Emotional Evolution:\n"
            stats_text += f"Response Steps: {len(df)}\n"
            stats_text += f"Pleasure Δ: {df['pleasure'].iloc[-1] - df['pleasure'].iloc[0]:+.3f}\n"
            stats_text += f"Arousal Δ: {df['arousal'].iloc[-1] - df['arousal'].iloc[0]:+.3f}\n"
            stats_text += f"Avg Confidence: {df['dominance'].mean():.2f}"
            
            ax.text(0.02, 0.35, stats_text, transform=ax.transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor='black', alpha=0.9))
            
            plt.tight_layout()
            output_path = os.path.join(self.plots_dir, 'emotional_journey.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            return True
            
        except Exception as e:
            return False
    
    def create_signifier_graph(self) -> bool:
        """Create a clean, publication-quality signifier network."""
        try:
            unconscious_path = os.path.join(self.agent_path, "unconscious_memory.json")
            if not os.path.exists(unconscious_path):
                return False
            
            with open(unconscious_path, 'r') as f:
                data = json.load(f)
            
            signifier_graph = data.get("signifier_graph")
            if not signifier_graph or not signifier_graph.get("nodes"):
                return False
            
            # Create clean NetworkX graph
            G = nx.DiGraph()
            
            # Add nodes and edges
            for node in signifier_graph["nodes"]:
                G.add_node(node["id"], 
                          node_type=node.get("node_type", "S2"),
                          activation=node.get("activation", 0.5))
            
            for edge in signifier_graph.get("edges", []):
                G.add_edge(edge["source"], edge["target"],
                          weight=edge.get("weight", 0.5))
            
            if G.number_of_nodes() == 0:
                return False
            
            # Create elegant visualization
            fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
            
            # Use hierarchical layout for better structure
            pos = self._create_hierarchical_layout(G)
            
            # Clean color scheme
            node_colors = {'S1': '#E74C3C', 'S2': '#3498DB'}
            
            # Draw edges with clean styling
            self._draw_clean_edges(ax, G, pos)
            
            # Draw nodes with elegant design
            self._draw_clean_nodes(ax, G, pos, node_colors)
            
            # Clean title and styling
            ax.set_title(f'{self.agent_name}: Unconscious Signifier Network', 
                        fontsize=16, fontweight='bold', pad=20)
            
            # Remove axes completely for clean look
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.axis('off')
            
            # Minimal legend
            legend_elements = [
                Circle((0, 0), 0.1, facecolor='#E74C3C', edgecolor='black',
                       label='Master Signifiers (S1)'),
                Circle((0, 0), 0.1, facecolor='#3498DB', edgecolor='black',
                       label='Knowledge Signifiers (S2)')
            ]
            
            ax.legend(handles=legend_elements, loc='upper right', 
                     frameon=True, fancybox=True, shadow=True)
            
            plt.tight_layout()
            output_path = os.path.join(self.plots_dir, 'signifier_network.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            return True
            
        except Exception as e:
            return False
    
    def _create_hierarchical_layout(self, G):
        """Create a clean hierarchical layout."""
        # Separate S1 and S2 nodes
        s1_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'S1']
        s2_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'S2']
        
        pos = {}
        
        # Place S1 nodes in inner circle
        if s1_nodes:
            angles = np.linspace(0, 2*np.pi, len(s1_nodes), endpoint=False)
            for i, node in enumerate(s1_nodes):
                pos[node] = (0.4 * np.cos(angles[i]), 0.4 * np.sin(angles[i]))
        
        # Place S2 nodes in outer circle
        if s2_nodes:
            angles = np.linspace(0, 2*np.pi, len(s2_nodes), endpoint=False)
            for i, node in enumerate(s2_nodes):
                pos[node] = (0.9 * np.cos(angles[i]), 0.9 * np.sin(angles[i]))
        
        return pos
    
    def _draw_clean_edges(self, ax, G, pos):
        """Draw edges with clean, minimal styling."""
        for u, v, d in G.edges(data=True):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            # Clean arrows
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color='gray', 
                                     lw=1.5, alpha=0.7,
                                     connectionstyle="arc3,rad=0.1"))
    
    def _draw_clean_nodes(self, ax, G, pos, node_colors):
        """Draw nodes with clean, professional styling."""
        degree_centrality = nx.degree_centrality(G)
        
        for node in G.nodes():
            node_data = G.nodes[node]
            node_type = node_data.get('node_type', 'S2')
            
            x, y = pos[node]
            color = node_colors[node_type]
            size = 800 + 1200 * degree_centrality[node]
            
            # Clean circle
            circle = Circle((x, y), radius=np.sqrt(size)/150, 
                           color=color, alpha=0.8, 
                           edgecolor='black', linewidth=2)
            ax.add_patch(circle)
            
            # Clean label
            fontsize = 10 + 4 * degree_centrality[node]
            ax.text(x, y, node, fontsize=int(fontsize), ha='center', va='center',
                   fontweight='bold', color='white')

def create_agent_visualizations(agent_name: str, base_path: str = "base_agents") -> bool:
    """Create publication-quality visualizations for an agent."""
    visualizer = AgentVisualizer(agent_name, base_path)
    results = visualizer.create_all_visualizations()
    return all(results.values())

if __name__ == "__main__":
    import sys
    agent_name = sys.argv[1] if len(sys.argv) > 1 else "Nancy"
    create_agent_visualizations(agent_name)