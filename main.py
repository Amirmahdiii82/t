import os
import argparse
from dotenv import load_dotenv
import google.generativeai as genai
from phase1.agent_extractor import AgentExtractor
from utils.file_utils import ensure_directory

def setup_project_structure():
    """Initialize project directory structure."""
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
    """Extract agent from dream dataset (Phase 1)."""
    if not os.path.exists(dream_file_path):
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
            print(f"✅ Successfully extracted agent: {agent_name}")
            return agent_name
        else:
            print("❌ Failed to extract agent")
            return None
            
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        return None

def list_available_agents():
    """List all extracted agents."""
    agents = []
    base_agents_dir = "base_agents"
    
    if os.path.exists(base_agents_dir):
        for item in os.listdir(base_agents_dir):
            agent_path = os.path.join(base_agents_dir, item)
            if os.path.isdir(agent_path):
                conscious_path = os.path.join(agent_path, "conscious_memory.json")
                if os.path.exists(conscious_path):
                    agents.append(item)
    
    return agents

def interactive_mode(agent_name):
    """Run interactive session with agent (Phase 2)."""
    print("=== DreamerAgent Interactive Mode ===")
    
    available_agents = list_available_agents()
    if not available_agents:
        print("❌ No agents available. Extract an agent first.")
        return
    
    if agent_name and agent_name not in available_agents:
        print(f"❌ Agent '{agent_name}' not found.")
        print(f"Available agents: {', '.join(available_agents)}")
        return
    
    if not agent_name:
        agent_name = available_agents[0]
    
    print(f"🧠 Initializing {agent_name}...")
    
    try:
        from phase2.agent_brain import AgentBrain
        agent_brain = AgentBrain(agent_name, "base_agents")
        
        print(f"✅ {agent_name} initialized successfully!")
        
        # Show agent statistics
        state = agent_brain.get_state()
        stats = state.get('memory_statistics', {})
        print(f"\n📊 Agent Statistics:")
        print(f"   Memories: {stats.get('conscious_memories', 0)}")
        print(f"   Relationships: {stats.get('conscious_relationships', 0)}")
        print(f"   Signifiers: {stats.get('unconscious_signifiers', 0)}")
        print(f"   Emotion: {stats.get('emotional_state', 'neutral')}")
        
        current_mode = "wake"
        print(f"\n💬 Chatting with {agent_name} (Mode: {current_mode})")
        print("\nCommands: 'exit', 'sleep', 'wake', 'dream', 'stats', 'help'")
        
        while True:
            try:
                user_input = input(f"\n[{current_mode.upper()}] You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'exit':
                    print(f"👋 Goodbye! {agent_name} is resting.")
                    break
                
                elif user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  'exit' - Quit conversation")
                    print("  'sleep' - Put agent to sleep")
                    print("  'wake' - Wake the agent")
                    print("  'dream' - Generate dream (sleep mode only)")
                    print("  'stats' - Show agent statistics")
                
                elif user_input.lower() == 'stats':
                    state = agent_brain.get_state()
                    stats = state.get('memory_statistics', {})
                    print(f"\n📊 {agent_name} Statistics:")
                    print(f"   Mode: {current_mode}")
                    print(f"   Memories: {stats.get('conscious_memories', 0)}")
                    print(f"   Interactions: {state.get('session_data', {}).get('interactions', 0)}")
                    print(f"   Dreams: {state.get('session_data', {}).get('dreams_generated', 0)}")
                
                elif user_input.lower() == 'sleep':
                    if current_mode == "sleep":
                        print(f"💤 {agent_name} is already asleep.")
                    else:
                        result = agent_brain.switch_mode("sleep")
                        current_mode = "sleep"
                        print(f"💤 {agent_name} is now sleeping...")
                
                elif user_input.lower() == 'wake':
                    if current_mode == "wake":
                        print(f"👁️ {agent_name} is already awake.")
                    else:
                        result = agent_brain.switch_mode("wake")
                        current_mode = "wake"
                        print(f"👁️ {agent_name} is now awake!")
                
                elif user_input.lower() == 'dream':
                    if current_mode != "sleep":
                        print(f"❌ {agent_name} must be asleep to dream.")
                    else:
                        print(f"🌙 Generating dream for {agent_name}...")
                        result = agent_brain.generate_dream()
                        
                        if result.get('dream'):
                            dream = result['dream']
                            print(f"\n✨ Dream: {dream.get('title', 'Untitled')}")
                            print(f"📝 {dream.get('narrative', 'No narrative')[:200]}...")
                            
                            if dream.get('activated_signifiers'):
                                print(f"🧠 Activated signifiers: {', '.join(dream['activated_signifiers'][:5])}")
                            
                            if dream.get('images'):
                                print(f"🖼️ Generated {len(dream['images'])} dream images")
                        else:
                            print("❌ Failed to generate dream")
                
                else:
                    # Regular conversation
                    if current_mode == "sleep":
                        print(f"💤 {agent_name} is sleeping. Wake them up first.")
                    else:
                        result = agent_brain.process_message(user_input)
                        
                        if result.get('response'):
                            print(f"\n{agent_name}: {result['response']}")
                            
                            # Show unconscious activity if significant
                            unconscious_state = result.get('unconscious_state', {})
                            active_signifiers = unconscious_state.get('active_signifiers', [])
                            if active_signifiers:
                                print(f"[Unconscious: {', '.join(active_signifiers[:3])}]")
                        else:
                            print(f"\n{agent_name}: I'm having trouble processing that.")
            
            except KeyboardInterrupt:
                print(f"\n\n👋 Interrupted. {agent_name} is resting.")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
        
        # Save state before exit
        try:
            agent_brain.save_state()
            print("💾 Agent state saved.")
        except Exception as e:
            print(f"⚠️ Warning: Could not save state: {e}")
            
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")

def main():
    """Main entry point for DreamerAgent system."""
    print("🧠 DreamerAgent - Psychoanalytic AI System")
    print("=" * 40)
    
    # Load environment variables
    load_dotenv()
    
    # Configure APIs
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        print("✅ Gemini API configured")
    else:
        print("⚠️ Warning: GEMINI_API_KEY not found")
    
    # Set up project structure
    setup_project_structure()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="DreamerAgent - Psychoanalytic AI")
    parser.add_argument("--extract", metavar="DREAM_FILE", 
                       help="Extract agent from dream file (Phase 1)")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run interactive mode (Phase 2)")
    parser.add_argument("--agent", metavar="AGENT_NAME", 
                       help="Specify agent for interactive mode")
    parser.add_argument("--list", action="store_true", 
                       help="List available agents")
    
    args = parser.parse_args()
    
    if args.list:
        agents = list_available_agents()
        if agents:
            print(f"\n📋 Available agents: {', '.join(agents)}")
        else:
            print("\n📋 No agents available. Extract an agent first.")
    
    elif args.extract:
        print(f"\n🔄 Phase 1: Agent Extraction")
        agent_name = extract_agent(args.extract)
        
        if agent_name:
            print(f"\n🎉 Extraction completed successfully!")
            if args.interactive:
                print(f"\n🚀 Phase 2: Interactive Mode")
                interactive_mode(agent_name)
        else:
            print(f"\n💥 Extraction failed!")
    
    elif args.interactive:
        print(f"\n🚀 Phase 2: Interactive Mode")
        interactive_mode(args.agent)
    
    else:
        print("\n📖 Usage Examples:")
        print("  python main.py --extract nancy.json")
        print("  python main.py --interactive --agent Nancy")
        print("  python main.py --extract nancy.json --interactive")
        print("  python main.py --list")
        print("\n")
        parser.print_help()

if __name__ == "__main__":
    main()