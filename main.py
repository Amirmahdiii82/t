import os
import argparse
from dotenv import load_dotenv
import google.generativeai as genai
from phase1.agent_extractor import AgentExtractor
from utils.file_utils import ensure_directory
from utils.signifier_graph_generator import create_publication_graph
from utils.pad_visualizer import main as generate_pad_visualizations

def setup_project_structure():
    """Set up the initial project structure."""
    directories = [
        "phase1/prompts",
        "phase2/prompts", 
        "base_agents",
        "utils",
        "config",
        "interfaces"
    ]
    
    for directory in directories:
        ensure_directory(directory)

def extract_agent(dream_file_path):
    """Extract an agent from a dream file (Phase 1)."""
    print(f"Starting agent extraction from {dream_file_path}...")
    
    # Check if file exists
    if not os.path.exists(dream_file_path):
        # Try with dataset prefix
        dataset_path = os.path.join("dataset", dream_file_path)
        if os.path.exists(dataset_path):
            dream_file_path = dataset_path
        else:
            print(f"Error: Dream file not found: {dream_file_path}")
            return None
    
    try:
        extractor = AgentExtractor()
        agent = extractor.extract_agent_from_dreams(dream_file_path)
        
        if agent:
            agent_name = agent['name']
            print(f"\n✅ Successfully extracted agent: {agent_name}")
            print(f"Agent data saved to base_agents/{agent_name}/")
            
            # Generate signifier graph visualization after extraction
            print(f"\n📊 Generating unconscious signifier graphs for {agent_name}...")
            unconscious_path = os.path.join("base_agents", agent_name, "unconscious_memory.json")
            if os.path.exists(unconscious_path):
                success = create_publication_graph(unconscious_path, agent_name)
                if success:
                    print(f"✅ Signifier graphs generated in base_agents/{agent_name}/graphs/")
                else:
                    print("⚠️ Could not generate signifier graphs")
            else:
                print("⚠️ Unconscious memory file not found")
            
            return agent_name
        else:
            print("❌ Failed to extract agent")
            return None
            
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return None

def list_available_agents():
    """List all available agents."""
    agents = []
    base_agents_dir = "base_agents"
    
    if os.path.exists(base_agents_dir):
        for item in os.listdir(base_agents_dir):
            agent_path = os.path.join(base_agents_dir, item)
            if os.path.isdir(agent_path):
                # Check if it has the required files
                conscious_path = os.path.join(agent_path, "conscious_memory.json")
                if os.path.exists(conscious_path):
                    agents.append(item)
    
    return agents

def generate_agent_visualizations(agent_name):
    """Generate PAD visualizations for an agent."""
    print(f"\n📈 Generating PAD visualizations for {agent_name}...")
    neuroproxy_path = os.path.join("base_agents", agent_name, "neuroproxy_state.json")
    
    if os.path.exists(neuroproxy_path):
        try:
            generate_pad_visualizations(neuroproxy_path)
            viz_dir = os.path.join("base_agents", agent_name, "visualizations")
            print(f"✅ PAD visualizations saved to {viz_dir}/")
            return True
        except Exception as e:
            print(f"⚠️ Error generating PAD visualizations: {e}")
            return False
    else:
        print("⚠️ Neuroproxy state file not found")
        return False

def interactive_mode(agent_name):
    """Run interactive mode with an agent."""
    print("=== Interactive Mode ===")
    
    # List available agents
    available_agents = list_available_agents()
    if not available_agents:
        print("❌ No agents available. Please extract an agent first using:")
        print("   python main.py --extract <dream_file>")
        return
    
    print(f"Available agents: {', '.join(available_agents)}")
    
    # Validate agent name
    if agent_name:
        if agent_name not in available_agents:
            print(f"❌ Agent '{agent_name}' not found.")
            print(f"Available agents: {', '.join(available_agents)}")
            return
    else:
        # If no agent specified, use the first available one
        agent_name = available_agents[0]
        print(f"No agent specified, using: {agent_name}")
    
    # Initialize the agent components step by step to avoid circular imports
    try:
        print(f"\n🧠 Initializing {agent_name}...")
        
        # Step 1: Initialize memory manager
        from phase2.memory_manager import MemoryManager
        memory_manager = MemoryManager(agent_name, "base_agents")
        
        # Step 2: Initialize conscious processor
        from phase2.conscious_processor import ConsciousProcessor
        conscious_processor = ConsciousProcessor(agent_name, memory_manager)
        
        # Step 3: Initialize dream generator (needed for unconscious processor)
        from phase2.dream_generator import DreamGenerator
        dream_generator = DreamGenerator(agent_name, memory_manager, "base_agents")
        
        # Step 4: Initialize unconscious processor with dream generator
        from phase2.unconscious_processor import UnconsciousProcessor
        unconscious_processor = UnconsciousProcessor(agent_name, memory_manager, dream_generator)
        
        print(f"✅ {agent_name} initialized successfully!")
        
        # Get agent stats
        stats = memory_manager.get_memory_stats()
        print(f"\n📊 Agent Stats:")
        print(f"   Conscious memories: {stats['conscious_memories']}")
        print(f"   Relationships: {stats['conscious_relationships']}")
        print(f"   Unconscious signifiers: {stats['unconscious_signifiers']}")
        print(f"   Current emotion: {stats['emotional_state']}")
        
        # Start interaction loop
        current_mode = "wake"
        print(f"\n💬 You are now chatting with {agent_name} (Mode: {current_mode})")
        print("\nCommands:")
        print("  'exit' - Quit the conversation")
        print("  'sleep' - Put agent to sleep")
        print("  'wake' - Wake the agent")
        print("  'dream' - Generate a dream (only in sleep mode)")
        print("  'stats' - Show agent statistics")
        print("  'visualize' - Generate PAD visualizations")
        print("  'help' - Show this help message")
        print("\nStart chatting!")
        
        while True:
            try:
                user_input = input(f"\n[{current_mode.upper()}] You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'exit':
                    print(f"\n👋 Goodbye! {agent_name} is going to rest.")
                    break
                
                elif user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  'exit' - Quit the conversation")
                    print("  'sleep' - Put agent to sleep")
                    print("  'wake' - Wake the agent")
                    print("  'dream' - Generate a dream (only in sleep mode)")
                    print("  'stats' - Show agent statistics")
                    print("  'visualize' - Generate PAD visualizations")
                    print("  'help' - Show this help message")
                
                elif user_input.lower() == 'visualize':
                    generate_agent_visualizations(agent_name)
                
                elif user_input.lower() == 'stats':
                    stats = memory_manager.get_memory_stats()
                    emotional_state = memory_manager.get_emotional_state()
                    print(f"\n📊 {agent_name} Stats:")
                    print(f"   Mode: {current_mode}")
                    print(f"   Conscious memories: {stats['conscious_memories']}")
                    print(f"   Relationships: {stats['conscious_relationships']}")
                    print(f"   Short-term entries: {stats['short_term_entries']}")
                    print(f"   Current emotion: {stats['emotional_state']}")
                    print(f"   Emotional values: P={emotional_state.get('pleasure', 0):.2f}, A={emotional_state.get('arousal', 0):.2f}, D={emotional_state.get('dominance', 0):.2f}")
                
                elif user_input.lower() == 'sleep':
                    if current_mode == "sleep":
                        print(f"💤 {agent_name} is already asleep.")
                    else:
                        current_mode = "sleep"
                        print(f"💤 {agent_name} is now sleeping...")
                        print("   (You can generate dreams or wake them up)")
                
                elif user_input.lower() == 'wake':
                    if current_mode == "wake":
                        print(f"👁️ {agent_name} is already awake.")
                    else:
                        current_mode = "wake"
                        print(f"👁️ {agent_name} is now awake!")
                        print("   (Ready for conversation)")
                
                elif user_input.lower() == 'dream':
                    if current_mode != "sleep":
                        print(f"❌ {agent_name} must be asleep to dream. Type 'sleep' first.")
                    else:
                        print(f"🌙 Generating dream for {agent_name}...")
                        try:
                            dream = dream_generator.generate_dream("sleep")
                            if dream:
                                print(f"\n✨ Dream Generated: {dream['id']}")
                                
                                # Fix: Access narrative from manifest_content
                                manifest = dream.get('manifest_content', {})
                                if isinstance(manifest, dict):
                                    narrative = manifest.get('overall_narrative', 'No narrative available')
                                else:
                                    narrative = str(manifest)[:200] if manifest else 'No narrative available'
                                
                                print(f"📝 Narrative: {narrative[:200]}...")
                                
                                if dream.get('images'):
                                    print(f"🖼️ Images: {len(dream['images'])} generated")
                                    
                                # Show dream scenes if available
                                if isinstance(manifest, dict) and manifest.get('scenes'):
                                    print(f"\n🎬 Dream Scenes:")
                                    for i, scene in enumerate(manifest['scenes'][:3]):
                                        print(f"  Scene {i+1}: {scene.get('setting', 'Unknown setting')}")
                                        print(f"    {scene.get('narrative', 'No description')[:100]}...")
                                
                                # Show analysis if available
                                if dream.get('analysis'):
                                    analysis = dream['analysis']
                                    if isinstance(analysis, dict):
                                        interp = analysis.get('interpretation', analysis.get('clinical_interpretation', ''))
                                    else:
                                        interp = str(analysis)
                                    
                                    if interp:
                                        print(f"\n🔍 Analysis: {interp[:200]}...")
                                
                                print(f"\n💾 Dream saved to: base_agents/{agent_name}/dreams/{dream['id']}.json")
                            else:
                                print("❌ Failed to generate dream")
                        except Exception as e:
                            print(f"❌ Error generating dream: {e}")
                            import traceback
                            traceback.print_exc()
                
                else:
                    # Regular conversation
                    if current_mode == "sleep":
                        print(f"💤 {agent_name} is sleeping and cannot respond. Wake them up first or generate a dream.")
                    else:
                        try:
                            # Add to short-term memory
                            memory_manager.add_to_short_term_memory(user_input, "user_interaction")
                            
                            # Process with conscious processor
                            response = conscious_processor.process_input(user_input)
                            
                            if response:
                                print(f"\n{agent_name}: {response}")
                                # Add response to short-term memory
                                memory_manager.add_to_short_term_memory(response, "agent_response")
                            else:
                                print(f"\n{agent_name}: I'm having trouble processing that right now.")
                        
                        except Exception as e:
                            print(f"❌ Error processing message: {e}")
                            print(f"{agent_name}: I'm sorry, I'm having some technical difficulties.")
                            import traceback
                            traceback.print_exc()
            
            except KeyboardInterrupt:
                print(f"\n\n👋 Interrupted. {agent_name} is going to rest.")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                continue
        
        # Save state before exit
        try:
            memory_manager.save_state()
            print("💾 Agent state saved.")
            
            # Generate PAD visualizations after interaction session
            generate_agent_visualizations(agent_name)
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save agent state: {e}")
            
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🧠 PsyAgent - AI with Conscious and Unconscious Memory")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Initialize Gemini API
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        print("✅ Gemini API configured")
    else:
        print("⚠️ Warning: GEMINI_API_KEY not found in environment variables.")
    
    # Set up project structure
    setup_project_structure()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI Agent with Conscious and Unconscious Memory")
    parser.add_argument("--extract", metavar="DREAM_FILE", help="Extract an agent from a dream file (Phase 1)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode (Phase 2)")
    parser.add_argument("--agent", metavar="AGENT_NAME", help="Specify agent name for interactive mode")
    parser.add_argument("--list", action="store_true", help="List available agents")
    parser.add_argument("--visualize", metavar="AGENT_NAME", help="Generate visualizations for a specific agent")
    
    args = parser.parse_args()
    
    if args.list:
        agents = list_available_agents()
        if agents:
            print(f"\n📋 Available agents: {', '.join(agents)}")
        else:
            print("\n📋 No agents available. Extract an agent first.")
    
    elif args.visualize:
        # Standalone visualization command
        agents = list_available_agents()
        if args.visualize in agents:
            print(f"\n📊 Generating visualizations for {args.visualize}...")
            
            # Generate signifier graphs
            unconscious_path = os.path.join("base_agents", args.visualize, "unconscious_memory.json")
            if os.path.exists(unconscious_path):
                create_publication_graph(unconscious_path, args.visualize)
            
            # Generate PAD visualizations
            generate_agent_visualizations(args.visualize)
        else:
            print(f"❌ Agent '{args.visualize}' not found.")
            if agents:
                print(f"Available agents: {', '.join(agents)}")
    
    elif args.extract:
        print(f"\n🔄 Starting Phase 1: Agent Extraction")
        agent_name = extract_agent(args.extract)
        
        if agent_name:
            print(f"\n🎉 Extraction completed successfully!")
            # If interactive flag is also set, go straight to interactive mode
            if args.interactive:
                print(f"\n🚀 Starting Phase 2: Interactive Mode")
                interactive_mode(agent_name)
        else:
            print(f"\n💥 Extraction failed!")
    
    elif args.interactive:
        print(f"\n🚀 Starting Phase 2: Interactive Mode")
        interactive_mode(args.agent)
    
    else:
        print("\n📖 Usage Examples:")
        print("  python main.py --extract joan.json                    # Extract agent from dataset")
        print("  python main.py --interactive --agent Joan             # Chat with Joan")
        print("  python main.py --extract joan.json --interactive      # Extract and chat")
        print("  python main.py --list                                 # List available agents")
        print("  python main.py --visualize Joan                       # Generate all visualizations for Joan")
        print("\n")
        parser.print_help()

if __name__ == "__main__":
    main()