import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

def create_publication_ready_pad_journey(file_path):
    """Create a clean, publication-ready emotional journey visualization."""
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(file_path), 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'emotional_journey.png')
    
    try:
        # Load data
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract PAD history
        df = extract_pad_history(data)
        
        if df.empty or len(df) < 2:
            print("Insufficient data for visualization")
            return False
        
        # Create figure with white background
        fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
        ax.set_facecolor('white')
        
        # Define emotion colors (PAD octants)
        emotion_colors = {
            'elated': '#FFD700',      # Gold - high pleasure, high arousal, high dominance
            'excited': '#FF6B6B',     # Light red - high pleasure, high arousal, low dominance
            'content': '#4ECDC4',     # Teal - high pleasure, low arousal, high dominance
            'relaxed': '#95E1D3',     # Mint - high pleasure, low arousal, low dominance
            'angry': '#E74C3C',       # Red - low pleasure, high arousal, high dominance
            'anxious': '#FF7979',     # Pink - low pleasure, high arousal, low dominance
            'bored': '#74B9FF',       # Light blue - low pleasure, low arousal, high dominance
            'sad': '#A29BFE',         # Purple - low pleasure, low arousal, low dominance
            'neutral': '#BDC3C7',     # Gray
            'alert': '#F39C12',       # Orange
            'tired': '#5F9EA0'        # Cadet blue
        }
        
        # Create the journey path with gradient
        points = np.array([[row['pleasure'], row['arousal']] for _, row in df.iterrows()])
        
        # Create line segments
        segments = []
        colors_for_gradient = []
        
        for i in range(len(points) - 1):
            segments.append([points[i], points[i + 1]])
            # Color based on emotional state at that point
            emotion = df.iloc[i]['emotion_category']
            colors_for_gradient.append(emotion_colors.get(emotion, '#BDC3C7'))
        
        # Create LineCollection with varying colors
        lc = LineCollection(segments, colors=colors_for_gradient, linewidths=3, alpha=0.7)
        ax.add_collection(lc)
        
        # Plot points with emotion colors
        for i, row in df.iterrows():
            color = emotion_colors.get(row['emotion_category'], '#BDC3C7')
            
            # Size based on dominance (third dimension)
            size = 100 + 200 * (row['dominance'] + 1) / 2  # Scale from 100 to 300
            
            # Plot point
            ax.scatter(row['pleasure'], row['arousal'], c=color, s=size, 
                      edgecolors='black', linewidth=1.5, alpha=0.8, zorder=10)
        
        # Mark start and end points
        start_point = df.iloc[0]
        end_point = df.iloc[-1]
        
        # Start marker
        ax.scatter(start_point['pleasure'], start_point['arousal'], 
                  marker='o', s=400, c='none', edgecolors='green', 
                  linewidth=3, zorder=15, label='Start')
        
        # End marker
        ax.scatter(end_point['pleasure'], end_point['arousal'], 
                  marker='s', s=400, c='none', edgecolors='red', 
                  linewidth=3, zorder=15, label='End')
        
        # Add PAD space quadrant labels
        ax.text(0.5, 0.5, 'Happy\nEnergetic', ha='center', va='center', 
               fontsize=12, alpha=0.3, weight='bold')
        ax.text(-0.5, 0.5, 'Stressed\nAgitated', ha='center', va='center', 
               fontsize=12, alpha=0.3, weight='bold')
        ax.text(0.5, -0.5, 'Calm\nPeaceful', ha='center', va='center', 
               fontsize=12, alpha=0.3, weight='bold')
        ax.text(-0.5, -0.5, 'Sad\nDepressed', ha='center', va='center', 
               fontsize=12, alpha=0.3, weight='bold')
        
        # Set axis properties
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel('Pleasure (Valence)', fontsize=14, weight='bold')
        ax.set_ylabel('Arousal (Activation)', fontsize=14, weight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='black', linewidth=0.5, alpha=0.5)
        
        # Title
        agent_name = data.get('agent_name', 'Agent')
        ax.set_title(f"{agent_name}'s Emotional Journey During Interaction", 
                    fontsize=18, weight='bold', pad=20)
        
        # Create custom legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                      markeredgecolor='green', markersize=12, markeredgewidth=2,
                      label='Conversation Start'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='none',
                      markeredgecolor='red', markersize=12, markeredgewidth=2,
                      label='Conversation End'),
            plt.Line2D([0], [0], color='gray', linewidth=3, alpha=0.7,
                      label='Emotional Trajectory')
        ]
        
        # Add size legend for dominance
        legend_elements.extend([
            plt.scatter([], [], s=100, c='gray', alpha=0.6, edgecolors='black',
                       label='Low Dominance'),
            plt.scatter([], [], s=300, c='gray', alpha=0.6, edgecolors='black',
                       label='High Dominance')
        ])
        
        # Add emotion color samples to legend
        sample_emotions = ['excited', 'content', 'anxious', 'sad']
        for emotion in sample_emotions:
            legend_elements.append(
                plt.Circle((0, 0), 0.1, facecolor=emotion_colors[emotion],
                          edgecolor='black', label=emotion.capitalize())
            )
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
                 fontsize=10, frameon=True, fancybox=False, edgecolor='black')
        
        # Add timestamp annotations for key emotional shifts
        # Find significant emotional shifts
        significant_shifts = find_significant_shifts(df)
        
        for shift in significant_shifts[:3]:  # Show top 3 shifts
            idx = shift['index']
            point = df.iloc[idx]
            ax.annotate(f"{shift['from']}→{shift['to']}", 
                       xy=(point['pleasure'], point['arousal']),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                edgecolor='black', alpha=0.8),
                       fontsize=9, ha='left',
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
        
        # Add statistics box
        stats_text = generate_journey_statistics(df)
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        edgecolor='black', alpha=0.9))
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Publication-ready emotional journey saved to {output_path}")
        
        # Also create a neurochemical influence diagram
        create_neurochemical_diagram(data, output_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating emotional journey: {e}")
        import traceback
        traceback.print_exc()
        return False

def extract_pad_history(data):
    """Extract PAD states and timestamps from emotional history."""
    history = data.get('emotional_history', [])
    pad_data = []
    
    for entry in history:
        timestamp = entry.get('timestamp')
        pad_state = entry.get('resulting_pad', {})
        emotion_category = entry.get('resulting_emotion', 'unknown')
        context = entry.get('context', 'unknown')
        
        pad_data.append({
            'timestamp': pd.to_datetime(timestamp),
            'pleasure': pad_state.get('pleasure', 0),
            'arousal': pad_state.get('arousal', 0),
            'dominance': pad_state.get('dominance', 0),
            'emotion_category': emotion_category,
            'context': context
        })
    
    # Add current state
    current_pad_state = data.get('derived_pad_state', {})
    current_emotion_category = data.get('current_emotion', 'unknown')
    current_timestamp = data.get('timestamp')
    
    if current_pad_state and current_timestamp:
        pad_data.append({
            'timestamp': pd.to_datetime(current_timestamp),
            'pleasure': current_pad_state.get('pleasure', 0),
            'arousal': current_pad_state.get('arousal', 0),
            'dominance': current_pad_state.get('dominance', 0),
            'emotion_category': current_emotion_category,
            'context': 'current_state'
        })
    
    return pd.DataFrame(pad_data)

def find_significant_shifts(df):
    """Find significant emotional shifts in the journey."""
    shifts = []
    
    for i in range(1, len(df)):
        prev_emotion = df.iloc[i-1]['emotion_category']
        curr_emotion = df.iloc[i]['emotion_category']
        
        if prev_emotion != curr_emotion:
            # Calculate magnitude of PAD change
            pad_change = np.sqrt(
                (df.iloc[i]['pleasure'] - df.iloc[i-1]['pleasure'])**2 +
                (df.iloc[i]['arousal'] - df.iloc[i-1]['arousal'])**2 +
                (df.iloc[i]['dominance'] - df.iloc[i-1]['dominance'])**2
            )
            
            shifts.append({
                'index': i,
                'from': prev_emotion,
                'to': curr_emotion,
                'magnitude': pad_change
            })
    
    # Sort by magnitude and return
    return sorted(shifts, key=lambda x: x['magnitude'], reverse=True)

def generate_journey_statistics(df):
    """Generate statistics about the emotional journey."""
    # Calculate total distance traveled in PAD space
    total_distance = 0
    for i in range(1, len(df)):
        dist = np.sqrt(
            (df.iloc[i]['pleasure'] - df.iloc[i-1]['pleasure'])**2 +
            (df.iloc[i]['arousal'] - df.iloc[i-1]['arousal'])**2 +
            (df.iloc[i]['dominance'] - df.iloc[i-1]['dominance'])**2
        )
        total_distance += dist
    
    # Calculate average PAD values
    avg_pleasure = df['pleasure'].mean()
    avg_arousal = df['arousal'].mean()
    avg_dominance = df['dominance'].mean()
    
    # Count emotion categories
    emotion_counts = df['emotion_category'].value_counts()
    dominant_emotion = emotion_counts.idxmax()
    
    stats_text = f"Journey Statistics:\n"
    stats_text += f"Total emotional distance: {total_distance:.2f}\n"
    stats_text += f"Average valence: {avg_pleasure:.2f}\n"
    stats_text += f"Average arousal: {avg_arousal:.2f}\n"
    stats_text += f"Dominant emotion: {dominant_emotion}\n"
    stats_text += f"Emotional states: {len(emotion_counts)}"
    
    return stats_text

def create_neurochemical_diagram(data, output_dir):
    """Create a supplementary diagram showing neurochemical influences."""
    output_path = os.path.join(output_dir, 'neurochemical_influence.png')
    
    neurochemicals = data.get('neurochemical_state', {})
    if not neurochemicals:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
    
    # Left panel: Neurochemical levels
    neurotransmitters = list(neurochemicals.keys())
    levels = list(neurochemicals.values())
    
    # Define colors based on function
    colors = {
        'dopamine': '#E74C3C',      # Red - reward
        'serotonin': '#3498DB',     # Blue - mood
        'oxytocin': '#E91E63',      # Pink - bonding
        'cortisol': '#F39C12',      # Orange - stress
        'norepinephrine': '#27AE60', # Green - alertness
        'gaba': '#9B59B6'           # Purple - relaxation
    }
    
    bar_colors = [colors.get(nt.lower(), '#95A5A6') for nt in neurotransmitters]
    
    bars = ax1.bar(neurotransmitters, levels, color=bar_colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=2)
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel('Level', fontsize=12, weight='bold')
    ax1.set_title('Current Neurochemical State', fontsize=14, weight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Rotate x-axis labels
    ax1.set_xticklabels(neurotransmitters, rotation=45, ha='right')
    
    # Right panel: PAD derivation diagram
    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylim(-0.5, 3.5)
    ax2.axis('off')
    
    # Draw connections from neurochemicals to PAD
    connections = [
        # Pleasure connections
        ('Dopamine\nSerotonin\nOxytocin', 0.3, 2.5, 'Pleasure', 1.5, 2.5, '#4ECDC4'),
        ('Cortisol', 0.3, 2.2, 'Pleasure', 1.5, 2.5, '#E74C3C'),
        # Arousal connections
        ('Norepinephrine\nCortisol\nDopamine', 0.3, 1.5, 'Arousal', 1.5, 1.5, '#FFD700'),
        ('GABA', 0.3, 1.2, 'Arousal', 1.5, 1.5, '#9B59B6'),
        # Dominance connections
        ('Dopamine\nNorepinephrine', 0.3, 0.5, 'Dominance', 1.5, 0.5, '#27AE60'),
        ('Cortisol', 0.3, 0.2, 'Dominance', 1.5, 0.5, '#E74C3C')
    ]
    
    for source, sx, sy, target, tx, ty, color in connections:
        # Draw arrow
        ax2.annotate('', xy=(tx-0.1, ty), xytext=(sx+0.4, sy),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
        # Add source label
        ax2.text(sx, sy, source, fontsize=10, va='center')
    
    # Draw PAD boxes
    pad_values = data.get('derived_pad_state', {})
    pad_labels = ['Pleasure', 'Arousal', 'Dominance']
    pad_y_positions = [2.5, 1.5, 0.5]
    
    for label, y_pos in zip(pad_labels, pad_y_positions):
        value = pad_values.get(label.lower(), 0)
        # Draw box
        box = plt.Rectangle((1.3, y_pos-0.15), 0.6, 0.3, 
                           facecolor='lightblue', edgecolor='black', linewidth=2)
        ax2.add_patch(box)
        # Add text
        ax2.text(1.6, y_pos, f'{label}\n{value:.2f}', ha='center', va='center',
                fontsize=11, weight='bold')
    
    ax2.set_title('Neurochemical → PAD Mapping', fontsize=14, weight='bold')
    
    # Add description
    description = ("Neurochemical levels modulate emotional experience through PAD dimensions:\n"
                  "• Positive contributors shown with colored arrows\n"
                  "• Negative contributors shown with red arrows")
    ax2.text(0.5, -0.3, description, transform=ax2.transAxes, 
            fontsize=10, ha='center', style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ Neurochemical influence diagram saved to {output_path}")

def main(file_path):
    """Main function to create publication-ready visualizations."""
    print(f"Creating publication-ready emotional journey visualization...")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return
    
    success = create_publication_ready_pad_journey(file_path)
    
    if success:
        output_dir = os.path.join(os.path.dirname(file_path), 'visualizations')
        print(f"\n📊 Visualizations created successfully in {output_dir}:")
        print("  - emotional_journey.png: Agent's emotional trajectory in PAD space")
        print("  - neurochemical_influence.png: Neurochemical-to-PAD mapping")
        print("\nThese visualizations demonstrate the psychoanalytic architecture's")
        print("ability to track and model complex emotional dynamics during interaction.")

if __name__ == '__main__':
    import sys
    file_to_process = sys.argv[1] if len(sys.argv) > 1 else 'base_agents/Nancy/neuroproxy_state.json'
    main(file_to_process)