import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import os

# Set the style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_emotional_data(file_path):
    """Load emotional state data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_pad_history(data):
    """Extract PAD states and timestamps from emotional history in neuroproxy_state.json."""
    history = data.get('emotional_history', [])
    pad_data = []
    for entry in history:
        timestamp = entry.get('timestamp')
        # In neuroproxy_state.json, PAD state for history entries is under 'resulting_pad'
        pad_state = entry.get('resulting_pad', {}) 
        # Emotion category is under 'resulting_emotion'
        emotion_category = entry.get('resulting_emotion', 'unknown') 
        context = entry.get('context', 'unknown') # Context seems to be available directly
        
        pad_data.append({
            'timestamp': pd.to_datetime(timestamp),
            'pleasure': pad_state.get('pleasure', 0),
            'arousal': pad_state.get('arousal', 0),
            'dominance': pad_state.get('dominance', 0),
            'emotion_category': emotion_category,
            'context': context
        })
        
    # Add current affective state from the top level of neuroproxy_state.json
    # Current PAD state is under 'derived_pad_state'
    current_pad_state = data.get('derived_pad_state', {}) 
    # Current emotion category is under 'current_emotion'
    current_emotion_category = data.get('current_emotion', 'unknown')
    # Timestamp for the current state can be the file's save timestamp
    current_timestamp = data.get('timestamp') 
    
    if current_pad_state and current_timestamp:
        pad_data.append({
            'timestamp': pd.to_datetime(current_timestamp),
            'pleasure': current_pad_state.get('pleasure', 0),
            'arousal': current_pad_state.get('arousal', 0),
            'dominance': current_pad_state.get('dominance', 0),
            'emotion_category': current_emotion_category,
            'context': 'current_state' # Assign a context for the current snapshot
        })
        
    if not pad_data:
        print("Warning: No PAD data extracted. Check the structure of neuroproxy_state.json.")
        # Return an empty DataFrame with expected columns to prevent errors in plotting functions
        return pd.DataFrame(columns=['timestamp', 'pleasure', 'arousal', 'dominance', 'emotion_category', 'context'])

    return pd.DataFrame(pad_data)

def plot_pad_timeline(df, output_file='pad_timeline.png'):
    """Plot Pleasure, Arousal, Dominance over time."""
    if df.empty:
        print(f"Skipping {output_file} generation: DataFrame is empty.")
        return

    plt.figure(figsize=(15, 8))
    
    plt.subplot(3, 1, 1)
    sns.lineplot(x='timestamp', y='pleasure', data=df, label='Pleasure', color='skyblue', marker='o')
    plt.title('Pleasure Over Time')
    plt.ylabel('Pleasure')
    plt.xticks(rotation=45)
    plt.legend()

    plt.subplot(3, 1, 2)
    sns.lineplot(x='timestamp', y='arousal', data=df, label='Arousal', color='salmon', marker='o')
    plt.title('Arousal Over Time')
    plt.ylabel('Arousal')
    plt.xticks(rotation=45)
    plt.legend()

    plt.subplot(3, 1, 3)
    sns.lineplot(x='timestamp', y='dominance', data=df, label='Dominance', color='lightgreen', marker='o')
    plt.title('Dominance Over Time')
    plt.ylabel('Dominance')
    plt.xlabel('Timestamp')
    plt.xticks(rotation=45)
    plt.legend()

    plt.suptitle('PAD States Over Time', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_emotion_heatmap(df, output_file='emotion_heatmap.png'):
    """Plot a heatmap of PAD states over time, colored by emotion category."""
    if df.empty or len(df) < 2: # Heatmap needs at least 2 data points for proper visualization
        print(f"Skipping {output_file} generation: DataFrame is empty or has insufficient data.")
        return

    # Pivot data for heatmap
    pad_values = df[['pleasure', 'arousal', 'dominance']].copy()
    pad_values.index = df['timestamp']
    
    plt.figure(figsize=(18, 6))
    sns.heatmap(pad_values.transpose(), cmap="viridis", annot=False, cbar=True) # Annot can be True for smaller datasets
    
    # Add emotion category as text annotations above the heatmap cells if desired (can get crowded)
    # For simplicity, this example doesn't add emotion category text directly on heatmap cells.
    # Instead, ensure your timeline or journey plot shows emotion categories.

    plt.title('PAD State Heatmap Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Timestamp')
    plt.ylabel('PAD Dimension')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_emotion_radar(df, output_file='emotion_radar.png'):
    """Create a radar chart for average PAD states per emotion category."""
    if df.empty:
        print(f"Skipping {output_file} generation: DataFrame is empty.")
        return
        
    avg_pad = df.groupby('emotion_category')[['pleasure', 'arousal', 'dominance']].mean().reset_index()
    
    if avg_pad.empty:
        print(f"Skipping {output_file} generation: No data after grouping by emotion category.")
        return

    labels = ['Pleasure', 'Arousal', 'Dominance']
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Complete the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for i, row in avg_pad.iterrows():
        values = row[['pleasure', 'arousal', 'dominance']].values.flatten().tolist()
        values += values[:1] # Complete the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['emotion_category'], alpha=0.7)
        ax.fill(angles, values, alpha=0.2)

    ax.set_yticklabels([]) # Hide radial ticks if PAD values are not on a common scale for radar
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    # Optional: Set y-axis limits if your PAD values are consistently in a certain range (e.g., -1 to 1)
    ax.set_ylim(-1, 1)


    plt.title('Average PAD States by Emotion Category (Radar Chart)', size=16, y=1.1, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_emotion_journey(df, output_file='emotion_journey.png'):
    """Plot the journey of emotions in the PAD space."""
    if df.empty or len(df) < 2: # Journey plot needs at least 2 data points
        print(f"Skipping {output_file} generation: DataFrame is empty or has insufficient data.")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Scatter plot for PAD states, colored by emotion category
    categories = pd.Categorical(df['emotion_category'])
    category_codes = categories.codes
    
    scatter = ax.scatter(df['pleasure'], df['arousal'], c=category_codes, cmap='viridis', s=100, alpha=0.7, edgecolors='w', linewidth=0.5)
    
    # Create a legend
    handles = [mpatches.Patch(color=scatter.cmap(scatter.norm(code)), label=cat) for cat, code in zip(categories.categories, range(len(categories.categories)))]
    ax.legend(handles=handles, title="Emotion Category", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Draw lines connecting consecutive points to show the journey
    ax.plot(df['pleasure'], df['arousal'], color='grey', linestyle='-', linewidth=1, alpha=0.5, zorder=0)

    # Annotate start and end points
    ax.annotate('Start', xy=(df.iloc[0]['pleasure'], df.iloc[0]['arousal']),
               xytext=(10, 10), textcoords='offset points',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
               fontsize=10, fontweight='bold')
    ax.annotate('End', xy=(df.iloc[-1]['pleasure'], df.iloc[-1]['arousal']),
               xytext=(10, -10), textcoords='offset points',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.7),
               fontsize=10, fontweight='bold')
    
    # Set labels and title
    ax.set_xlabel('Pleasure')
    ax.set_ylabel('Arousal')
    ax.set_title('Emotional Journey in PAD Space (Pleasure vs Arousal)', fontsize=16, fontweight='bold')
    
    # Add grid and standardize axes for PAD space
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_neurochemical_state(data, output_file='neurochemical_state.png'):
    """Plot a bar chart of the current neurochemical state."""
    if not data or 'neurochemical_state' not in data:
        print(f"Skipping {output_file} generation: No neurochemical data available.")
        return
        
    neurochemicals = data.get('neurochemical_state', {})
    if not neurochemicals:
        print(f"Skipping {output_file} generation: Empty neurochemical state.")
        return
        
    # Create lists for plotting
    neurotransmitters = list(neurochemicals.keys())
    levels = list(neurochemicals.values())
    
    # Define colors based on neurotransmitter function
    colors = {
        'dopamine': '#FF9999',     # Reward/motivation - pinkish red
        'serotonin': '#99CCFF',    # Mood regulation - light blue
        'oxytocin': '#FF99CC',     # Social bonding - pink
        'cortisol': '#FFCC99',     # Stress - light orange
        'norepinephrine': '#CCFF99', # Alertness - light green
        'gaba': '#CC99FF'          # Relaxation - light purple
    }
    
    # Get colors for each neurotransmitter (with fallback)
    bar_colors = [colors.get(nt.lower(), '#AAAAAA') for nt in neurotransmitters]
    
    plt.figure(figsize=(12, 8))
    
    # Create the bar chart
    bars = plt.bar(neurotransmitters, levels, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Add reference line for baseline levels
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Baseline (0.5)')
    
    # Customize the chart
    plt.ylim(0, 1.1)  # Set y-axis limits with a bit of padding
    plt.ylabel('Level', fontsize=14)
    plt.title('Current Neurochemical State', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add description of what each neurotransmitter does
    descriptions = {
        'dopamine': 'Reward/motivation',
        'serotonin': 'Mood/well-being',
        'oxytocin': 'Social bonding',
        'cortisol': 'Stress response',
        'norepinephrine': 'Alertness/arousal',
        'gaba': 'Relaxation/inhibition'
    }
    
    # Add annotations for neurotransmitter functions
    for i, nt in enumerate(neurotransmitters):
        if nt.lower() in descriptions:
            plt.annotate(
                descriptions[nt.lower()],
                xy=(i, 0.05),  # Position at bottom of bar
                ha='center',
                fontsize=9,
                color='black'
            )
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def main(file_path):
    """Main function to process data and generate plots."""
    # Load and process data
    data = load_emotional_data(file_path)
    df = extract_pad_history(data)
    
    if df.empty:
        print("No data to plot. Exiting.")
        return

    # Create output directory next to the data file
    output_dir = os.path.join(os.path.dirname(file_path), 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    plot_pad_timeline(df, os.path.join(output_dir, 'pad_timeline.png'))
    plot_emotion_heatmap(df, os.path.join(output_dir, 'emotion_heatmap.png')) 
    plot_emotion_radar(df, os.path.join(output_dir, 'emotion_radar.png'))
    plot_emotion_journey(df, os.path.join(output_dir, 'emotion_journey.png'))
    
    # Plot current neurochemical state (which drives the PAD values)
    plot_neurochemical_state(data, os.path.join(output_dir, 'neurochemical_state.png'))
    
    print("✨ Plots saved successfully!")
    print(f"📊 Generated files in {output_dir}:")
    print("  - pad_timeline.png: Timeline view of PAD states")
    print("  - emotion_heatmap.png: Heatmap visualization")
    print("  - emotion_radar.png: Radar chart of average PAD by emotion")
    print("  - emotion_journey.png: 2D plot of emotional journey in PAD space")
    print("  - neurochemical_state.png: Current neurochemical levels (the foundation of emotional state)")

if __name__ == '__main__':
    # Default file path
    default_file = 'base_agents/Nancy/neuroproxy_state.json' 
    
    # Allow command-line specification of file path
    import sys
    file_to_process = sys.argv[1] if len(sys.argv) > 1 else default_file
    
    # Basic check if the file exists before running
    if os.path.exists(file_to_process):
        main(file_to_process)
    else:
        print(f"Error: File not found at '{file_to_process}'.")
        print("Please ensure 'neuroproxy_state.json' is in the correct location or specify the path.")
        
        # Try to find neuroproxy_state.json files in the current directory structure
        print("\nSearching for neuroproxy_state.json files...")
        found_files = []
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file == 'neuroproxy_state.json':
                    found_path = os.path.join(root, file)
                    found_files.append(found_path)
                    print(f"  - Found: {found_path}")
        
        if found_files:
            print("\nYou can run the script with one of these files:")
            for i, file in enumerate(found_files):
                print(f"  python t2.py {file}")