import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from matplotlib.collections import LineCollection
import os

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

def load_emotional_data(file_path):
    """Load emotional state data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_pad_history(data):
    """Extract PAD states with psychoanalytic context."""
    history = data.get('emotional_history', [])
    pad_data = []
    
    for entry in history:
        timestamp = entry.get('timestamp')
        pad_state = entry.get('resulting_pad', {})
        emotion_category = entry.get('resulting_emotion', 'unknown')
        context = entry.get('context', 'unknown')
        
        # Extract psychoanalytic context if available
        input_summary = entry.get('input_summary', '')
        emotional_analysis = entry.get('emotional_analysis', {})
        
        pad_data.append({
            'timestamp': pd.to_datetime(timestamp),
            'pleasure': pad_state.get('pleasure', 0),
            'arousal': pad_state.get('arousal', 0),
            'dominance': pad_state.get('dominance', 0),
            'emotion_category': emotion_category,
            'context': context,
            'input_summary': input_summary,
            'dominant_emotion': emotional_analysis.get('dominant_emotion', 'neutral'),
            'intensity': emotional_analysis.get('intensity', 0.5)
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
            'context': 'current_state',
            'input_summary': 'Current emotional state',
            'dominant_emotion': current_emotion_category,
            'intensity': 0.8
        })
    
    return pd.DataFrame(pad_data)

def plot_psychoanalytic_pad_journey(df, output_file='pad_journey_psychoanalytic.png'):
    """Create a sophisticated PAD journey visualization with psychoanalytic interpretation."""
    if df.empty or len(df) < 2:
        print(f"Skipping {output_file}: Insufficient data")
        return
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], width_ratios=[2, 1])
    
    # Main PAD space plot
    ax_main = fig.add_subplot(gs[0, :])
    
    # Create the journey path with varying line properties
    points = df[['pleasure', 'arousal']].values
    segments = []
    colors = []
    linewidths = []
    
    for i in range(len(points) - 1):
        segments.append([points[i], points[i + 1]])
        # Color based on dominance
        colors.append(plt.cm.coolwarm((df.iloc[i]['dominance'] + 1) / 2))
        # Width based on intensity
        linewidths.append(1 + df.iloc[i]['intensity'] * 3)
    
    lc = LineCollection(segments, colors=colors, linewidths=linewidths, alpha=0.8)
    ax_main.add_collection(lc)
    
    # Add emotion category labels with sophisticated styling
    emotion_colors = {
        'joy': '#FFD700', 'elated': '#FF69B4', 'excited': '#FF4500',
        'content': '#98FB98', 'relaxed': '#87CEEB', 'calm': '#E0E0E0',
        'sad': '#4682B4', 'anxious': '#FF6B6B', 'angry': '#DC143C',
        'bored': '#708090', 'neutral': '#C0C0C0'
    }
    
    # Plot points with emotion-specific styling
    for idx, row in df.iterrows():
        emotion = row['emotion_category']
        color = emotion_colors.get(emotion, '#808080')
        
        # Size based on intensity
        size = 100 + row['intensity'] * 200
        
        # Add glow effect for high intensity
        if row['intensity'] > 0.7:
            ax_main.scatter(row['pleasure'], row['arousal'], 
                          s=size*2, c=color, alpha=0.2, zorder=1)
        
        ax_main.scatter(row['pleasure'], row['arousal'], 
                      s=size, c=color, edgecolors='black', 
                      linewidth=1, zorder=2, alpha=0.9)
    
    # Add psychoanalytic quadrant labels
    ax_main.text(0.5, 0.5, 'Manic Defense', fontsize=12, alpha=0.3, 
                ha='center', va='center', style='italic')
    ax_main.text(-0.5, 0.5, 'Anxiety/\nActing Out', fontsize=12, alpha=0.3, 
                ha='center', va='center', style='italic')
    ax_main.text(-0.5, -0.5, 'Depression/\nWithdrawal', fontsize=12, alpha=0.3, 
                ha='center', va='center', style='italic')
    ax_main.text(0.5, -0.5, 'Sublimation', fontsize=12, alpha=0.3, 
                ha='center', va='center', style='italic')
    
    # Add start and end markers with psychoanalytic significance
    ax_main.annotate('Session Start\n(Initial Defense)', 
                    xy=(df.iloc[0]['pleasure'], df.iloc[0]['arousal']),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7),
                    fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
    
    ax_main.annotate('Current State\n(Working Through)', 
                    xy=(df.iloc[-1]['pleasure'], df.iloc[-1]['arousal']),
                    xytext=(10, -10), textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='salmon', alpha=0.7),
                    fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3'))
    
    # Identify and mark significant transitions
    for i in range(1, len(df) - 1):
        pleasure_change = abs(df.iloc[i]['pleasure'] - df.iloc[i-1]['pleasure'])
        arousal_change = abs(df.iloc[i]['arousal'] - df.iloc[i-1]['arousal'])
        
        if pleasure_change > 0.3 or arousal_change > 0.3:
            ax_main.plot(df.iloc[i]['pleasure'], df.iloc[i]['arousal'], 
                       'k*', markersize=15, alpha=0.6)
            ax_main.text(df.iloc[i]['pleasure'] + 0.05, df.iloc[i]['arousal'] + 0.05,
                       'Defensive\nShift', fontsize=8, alpha=0.6)
    
    # Configure main plot
    ax_main.set_xlim(-1.2, 1.2)
    ax_main.set_ylim(-1.2, 1.2)
    ax_main.set_xlabel('Pleasure (Libidinal Economy)', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Arousal (Psychic Energy)', fontsize=12, fontweight='bold')
    ax_main.set_title('Psychoanalytic Journey Through Affective Space', fontsize=16, fontweight='bold')
    ax_main.grid(True, alpha=0.3)
    ax_main.axhline(0, color='black', linewidth=0.5)
    ax_main.axvline(0, color='black', linewidth=0.5)
    
    # Dominance timeline
    ax_dom = fig.add_subplot(gs[1, 0])
    time_points = range(len(df))
    ax_dom.plot(time_points, df['dominance'], 'b-', linewidth=2, alpha=0.8)
    ax_dom.fill_between(time_points, df['dominance'], alpha=0.3)
    ax_dom.set_ylabel('Dominance\n(Ego Strength)', fontsize=10, fontweight='bold')
    ax_dom.set_ylim(-1.1, 1.1)
    ax_dom.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax_dom.grid(True, alpha=0.3)
    
    # Intensity timeline
    ax_int = fig.add_subplot(gs[2, 0])
    ax_int.plot(time_points, df['intensity'], 'r-', linewidth=2, alpha=0.8)
    ax_int.fill_between(time_points, df['intensity'], alpha=0.3, color='red')
    ax_int.set_ylabel('Affect\nIntensity', fontsize=10, fontweight='bold')
    ax_int.set_xlabel('Time Points', fontsize=10)
    ax_int.set_ylim(0, 1.1)
    ax_int.grid(True, alpha=0.3)
    
    # Emotion distribution (psychoanalytic interpretation)
    ax_dist = fig.add_subplot(gs[1:, 1])
    emotion_counts = df['emotion_category'].value_counts()
    colors_list = [emotion_colors.get(e, '#808080') for e in emotion_counts.index]
    
    wedges, texts, autotexts = ax_dist.pie(emotion_counts.values, 
                                           labels=emotion_counts.index,
                                           colors=colors_list,
                                           autopct='%1.1f%%',
                                           startangle=90)
    ax_dist.set_title('Emotional State Distribution\n(Defensive Positions)', fontsize=12, fontweight='bold')
    
    # Add legend for line colors (dominance)
    dom_legend = [
        plt.Line2D([0], [0], color=plt.cm.coolwarm(0), lw=4, label='Low Dominance\n(Submissive)'),
        plt.Line2D([0], [0], color=plt.cm.coolwarm(0.5), lw=4, label='Neutral\nDominance'),
        plt.Line2D([0], [0], color=plt.cm.coolwarm(1), lw=4, label='High Dominance\n(Assertive)')
    ]
    ax_main.legend(handles=dom_legend, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Psychoanalytic PAD journey saved to {output_file}")

def plot_defensive_patterns(df, output_file='defensive_patterns.png'):
    """Visualize defensive patterns in emotional transitions."""
    if df.empty or len(df) < 3:
        print(f"Skipping {output_file}: Insufficient data")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Calculate defensive shifts
    df['pleasure_change'] = df['pleasure'].diff()
    df['arousal_change'] = df['arousal'].diff()
    df['defensive_shift'] = np.sqrt(df['pleasure_change']**2 + df['arousal_change']**2)
    
    # Plot 1: Defensive shift intensity over time
    time_points = range(len(df))
    ax1.plot(time_points[1:], df['defensive_shift'][1:], 'k-', linewidth=2, alpha=0.8)
    ax1.fill_between(time_points[1:], df['defensive_shift'][1:], alpha=0.3, color='gray')
    
    # Mark significant defensive maneuvers
    threshold = df['defensive_shift'].quantile(0.75)
    significant_shifts = df[df['defensive_shift'] > threshold]
    
    for idx, row in significant_shifts.iterrows():
        if idx > 0:  # Skip first row
            ax1.axvline(idx, color='red', alpha=0.5, linestyle='--')
            ax1.text(idx, row['defensive_shift'] + 0.05, 
                    f"{df.iloc[idx-1]['emotion_category']}→\n{row['emotion_category']}", 
                    fontsize=8, rotation=45, va='bottom')
    
    ax1.set_title('Defensive Shift Intensity (Psychic Mobility)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Shift Magnitude', fontsize=12)
    ax1.set_xlabel('Time Points', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Defense mechanism patterns
    ax2.set_title('Defensive Trajectory Patterns', fontsize=14, fontweight='bold')
    
    # Create vector field showing defensive movements
    for i in range(1, len(df)):
        p1 = df.iloc[i-1]
        p2 = df.iloc[i]
        
        # Color based on type of shift
        if p2['pleasure'] > p1['pleasure'] and p2['arousal'] > p1['arousal']:
            color = 'green'  # Manic defense
            label = 'Manic'
        elif p2['pleasure'] < p1['pleasure'] and p2['arousal'] > p1['arousal']:
            color = 'orange'  # Anxious defense
            label = 'Anxious'
        elif p2['pleasure'] < p1['pleasure'] and p2['arousal'] < p1['arousal']:
            color = 'blue'  # Depressive defense
            label = 'Depressive'
        else:
            color = 'purple'  # Sublimation
            label = 'Sublimation'
        
        ax2.annotate('', xy=(p2['pleasure'], p2['arousal']), 
                    xytext=(p1['pleasure'], p1['arousal']),
                    arrowprops=dict(arrowstyle='->', color=color, alpha=0.6, lw=2))
        
        # Add label at midpoint
        mid_x = (p1['pleasure'] + p2['pleasure']) / 2
        mid_y = (p1['arousal'] + p2['arousal']) / 2
        ax2.text(mid_x, mid_y, label[0], fontsize=8, color=color, fontweight='bold')
    
    # Add quadrant labels
    ax2.text(0.7, 0.7, 'Manic\nDefense', fontsize=12, alpha=0.5, ha='center')
    ax2.text(-0.7, 0.7, 'Anxious\nDefense', fontsize=12, alpha=0.5, ha='center')
    ax2.text(-0.7, -0.7, 'Depressive\nDefense', fontsize=12, alpha=0.5, ha='center')
    ax2.text(0.7, -0.7, 'Sublimation', fontsize=12, alpha=0.5, ha='center')
    
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_xlabel('Pleasure', fontsize=12)
    ax2.set_ylabel('Arousal', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Defensive patterns saved to {output_file}")

def plot_transference_heat_map(df, data, output_file='transference_heatmap.png'):
    """Create a heat map showing transference patterns."""
    if df.empty:
        print(f"Skipping {output_file}: No data")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a 2D histogram of PAD positions
    H, xedges, yedges = np.histogram2d(df['pleasure'], df['arousal'], bins=20, 
                                       range=[[-1, 1], [-1, 1]])
    
    # Apply Gaussian smoothing
    from scipy.ndimage import gaussian_filter
    H_smooth = gaussian_filter(H.T, sigma=1.5)
    
    # Create heat map
    im = ax.imshow(H_smooth, extent=[-1, 1, -1, 1], origin='lower', 
                   cmap='YlOrRd', alpha=0.8, interpolation='bilinear')
    
    # Add trajectory on top
    ax.plot(df['pleasure'], df['arousal'], 'k-', alpha=0.5, linewidth=1)
    ax.scatter(df['pleasure'], df['arousal'], c='black', s=30, alpha=0.6, zorder=5)
    
    # Add psychoanalytic interpretation
    ax.text(0, 1.15, 'Transference Heat Map: Where Psychic Energy Accumulates', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Mark areas of high concentration
    max_indices = np.where(H_smooth > H_smooth.max() * 0.7)
    for i in range(len(max_indices[0])):
        y_idx = max_indices[0][i]
        x_idx = max_indices[1][i]
        x_pos = xedges[x_idx] + (xedges[x_idx + 1] - xedges[x_idx]) / 2
        y_pos = yedges[y_idx] + (yedges[y_idx + 1] - yedges[y_idx]) / 2
        
        circle = Circle((x_pos, y_pos), 0.15, fill=False, edgecolor='white', 
                       linewidth=2, linestyle='--')
        ax.add_patch(circle)
    
    # Add colorbar with psychoanalytic label
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Libidinal Investment\n(Transference Intensity)', fontsize=12)
    
    ax.set_xlabel('Pleasure (Object Relation)', fontsize=12)
    ax.set_ylabel('Arousal (Psychic Activation)', fontsize=12)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Transference heat map saved to {output_file}")

def main(file_path):
    """Main function to generate psychoanalytic visualizations."""
    # Load and process data
    data = load_emotional_data(file_path)
    df = extract_pad_history(data)
    
    if df.empty:
        print("No data to plot. Exiting.")
        return
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(file_path), 'psychoanalytic_visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all visualizations
    plot_psychoanalytic_pad_journey(df, os.path.join(output_dir, 'pad_journey_psychoanalytic.png'))
    plot_defensive_patterns(df, os.path.join(output_dir, 'defensive_patterns.png'))
    plot_transference_heat_map(df, data, os.path.join(output_dir, 'transference_heatmap.png'))
    
    # Also generate the original visualizations with enhanced styling
    plot_pad_timeline(df, os.path.join(output_dir, 'pad_timeline_enhanced.png'))
    plot_emotion_heatmap(df, os.path.join(output_dir, 'emotion_heatmap_enhanced.png'))
    
    print("\n✨ Psychoanalytic visualizations complete!")
    print(f"📊 Generated files in {output_dir}:")
    print("  - pad_journey_psychoanalytic.png: Complete affective journey with defenses")
    print("  - defensive_patterns.png: Analysis of defensive shifts")
    print("  - transference_heatmap.png: Where psychic energy accumulates")
    print("  - pad_timeline_enhanced.png: Enhanced timeline view")
    print("  - emotion_heatmap_enhanced.png: Enhanced emotion patterns")

# Keep original functions but enhance them
def plot_pad_timeline(df, output_file='pad_timeline_enhanced.png'):
    """Enhanced PAD timeline with psychoanalytic annotations."""
    if df.empty:
        print(f"Skipping {output_file}: DataFrame is empty.")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    time_points = range(len(df))
    
    # Pleasure timeline with gradient background
    ax1 = axes[0]
    ax1.plot(time_points, df['pleasure'], 'b-', linewidth=2.5, alpha=0.8, label='Pleasure')
    ax1.fill_between(time_points, df['pleasure'], alpha=0.3, color='blue')
    ax1.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
    ax1.set_ylabel('Pleasure\n(Libido)', fontsize=11, fontweight='bold')
    ax1.set_ylim(-1.1, 1.1)
    ax1.grid(True, alpha=0.3)
    
    # Add shaded regions for positive/negative
    ax1.axhspan(0, 1.1, alpha=0.1, color='green', label='Positive')
    ax1.axhspan(-1.1, 0, alpha=0.1, color='red', label='Negative')
    
    # Arousal timeline
    ax2 = axes[1]
    ax2.plot(time_points, df['arousal'], 'r-', linewidth=2.5, alpha=0.8, label='Arousal')
    ax2.fill_between(time_points, df['arousal'], alpha=0.3, color='red')
    ax2.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
    ax2.set_ylabel('Arousal\n(Activation)', fontsize=11, fontweight='bold')
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, alpha=0.3)
    
    # Dominance timeline
    ax3 = axes[2]
    ax3.plot(time_points, df['dominance'], 'g-', linewidth=2.5, alpha=0.8, label='Dominance')
    ax3.fill_between(time_points, df['dominance'], alpha=0.3, color='green')
    ax3.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
    ax3.set_ylabel('Dominance\n(Control)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Session Progress', fontsize=12, fontweight='bold')
    ax3.set_ylim(-1.1, 1.1)
    ax3.grid(True, alpha=0.3)
    
    # Add emotion category annotations
    for i, (idx, row) in enumerate(df.iterrows()):
        if i % max(1, len(df) // 10) == 0:  # Show every 10th or so
            ax1.annotate(row['emotion_category'][:3], 
                        (i, row['pleasure']), 
                        fontsize=8, alpha=0.6,
                        rotation=45)
    
    plt.suptitle('PAD Dimensions Over Time: Tracking Psychic Energy', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Enhanced PAD timeline saved to {output_file}")

def plot_emotion_heatmap(df, output_file='emotion_heatmap_enhanced.png'):
    """Enhanced emotion heatmap with psychoanalytic interpretation."""
    if df.empty or len(df) < 2:
        print(f"Skipping {output_file}: Insufficient data.")
        return
    
    # Prepare data for heatmap
    pad_values = df[['pleasure', 'arousal', 'dominance']].T
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Create custom colormap
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
    # Plot heatmap
    sns.heatmap(pad_values, 
                cmap=cmap, 
                center=0,
                annot=False, 
                fmt='.2f',
                cbar_kws={'label': 'Intensity'},
                xticklabels=False,
                yticklabels=['Pleasure\n(Libido)', 'Arousal\n(Energy)', 'Dominance\n(Control)'],
                linewidths=0.5,
                linecolor='gray',
                ax=ax)
    
    # Add emotion categories as text above
    for i, emotion in enumerate(df['emotion_category']):
        if i % max(1, len(df) // 20) == 0:  # Show subset
            ax.text(i, -0.5, emotion, rotation=45, fontsize=8, ha='right')
    
    ax.set_title('Psychic Energy Distribution Across PAD Dimensions', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Time Progression →', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Enhanced emotion heatmap saved to {output_file}")

if __name__ == '__main__':
    import sys
    default_file = 'base_agents/Nancy/neuroproxy_state.json'
    file_to_process = sys.argv[1] if len(sys.argv) > 1 else default_file
    
    if os.path.exists(file_to_process):
        main(file_to_process)
    else:
        print(f"Error: File not found at '{file_to_process}'.")