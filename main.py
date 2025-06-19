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
            print(f"❌ Error: Dream file not found: {dream_file_path}")
            return None
    
    try:
        print(f"\n🔄 Phase 1: Agent Extraction from {dream_file_path}")
        print("=" * 60)
        
        extractor = AgentExtractor()
        agent = extractor.extract_agent_from_dreams(dream_file_path)
        
        if agent:
            agent_name = agent['name']
            
            # Enhanced status reporting
            conscious_status = agent.get('conscious_extraction_status', 'unknown')
            unconscious_status = agent.get('unconscious_extraction_status', 'unknown')
            
            print(f"\n📊 EXTRACTION RESULTS FOR {agent_name.upper()}:")
            print(f"   Conscious Memory: {'✅ Success' if conscious_status == 'completed' else '❌ Failed'}")
            print(f"   Unconscious Memory: {'✅ Success' if unconscious_status == 'completed' else '⚠️ Partial' if unconscious_status == 'failed' else '✅ Success'}")
            
            if conscious_status == 'completed':
                conscious_data = agent.get('conscious', {})
                print(f"   └─ Memories: {len(conscious_data.get('memories', []))}")
                print(f"   └─ Relationships: {len(conscious_data.get('relationships', []))}")
                
            if unconscious_status in ['completed', 'failed']:  # Even failed attempts may have fallback data
                unconscious_data = agent.get('unconscious', {})
                print(f"   └─ Signifiers: {len(unconscious_data.get('signifiers', []))}")
                print(f"   └─ Chains: {len(unconscious_data.get('signifying_chains', []))}")
                
                if unconscious_data.get('metadata', {}).get('fallback_mode'):
                    print(f"   └─ ⚠️ Note: Using fallback unconscious structure")
            
            # Check for errors
            if agent.get('conscious_extraction_error'):
                print(f"   ⚠️ Conscious extraction issue: {agent['conscious_extraction_error']}")
            if agent.get('unconscious_extraction_error'):
                print(f"   ⚠️ Unconscious extraction issue: {agent['unconscious_extraction_error']}")
            
            overall_status = "✅ SUCCESS" if conscious_status == 'completed' else "⚠️ PARTIAL"
            print(f"\n🎯 Overall Status: {overall_status}")
            print(f"📁 Agent saved to: base_agents/{agent_name}/")
            
            return agent_name
        else:
            print("❌ Failed to extract agent - no data returned")
            return None
            
    except Exception as e:
        print(f"❌ Critical error during extraction: {e}")
        import traceback
        traceback.print_exc()
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
    print("\n🚀 Phase 2: Interactive Mode")
    print("=" * 40)
    
    available_agents = list_available_agents()
    if not available_agents:
        print("❌ No agents available. Extract an agent first using:")
        print("   python main.py --extract <dream_file.json>")
        return
    
    if agent_name and agent_name not in available_agents:
        print(f"❌ Agent '{agent_name}' not found.")
        print(f"Available agents: {', '.join(available_agents)}")
        return
    
    if not agent_name:
        agent_name = available_agents[0]
        print(f"🎯 Auto-selecting: {agent_name}")
    
    print(f"🧠 Initializing {agent_name}...")
    
    try:
        from phase2.agent_brain import AgentBrain
        agent_brain = AgentBrain(agent_name, "base_agents")
        
        print(f"✅ {agent_name} initialized successfully!")
        
        # Show agent statistics
        state = agent_brain.get_state()
        stats = state.get('memory_statistics', {})
        print(f"\n📊 Agent Statistics:")
        print(f"   Conscious Memories: {stats.get('conscious_memories', 0)}")
        print(f"   Relationships: {stats.get('conscious_relationships', 0)}")
        print(f"   Unconscious Signifiers: {stats.get('unconscious_signifiers', 0)}")
        print(f"   Current Emotion: {stats.get('emotional_state', 'neutral')}")
        
        current_mode = "wake"
        print(f"\n💬 Ready to chat with {agent_name}")
        print(f"Mode: {current_mode.upper()}")
        print("\n" + "="*50)
        print("COMMANDS:")
        print("  'exit' - Quit conversation")
        print("  'sleep' - Put agent to sleep")
        print("  'wake' - Wake the agent")
        print("  'dream' - Generate dream (sleep mode only)")
        print("  'stats' - Show agent statistics")
        print("  'help' - Show this help")
        print("="*50)
        
        while True:
            try:
                user_input = input(f"\n[{current_mode.upper()}] You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'exit':
                    print(f"👋 Goodbye! {agent_name} is resting.")
                    break
                
                elif user_input.lower() == 'help':
                    print("\n📖 Available Commands:")
                    print("  'exit' - Quit conversation")
                    print("  'sleep' - Put agent to sleep")
                    print("  'wake' - Wake the agent")
                    print("  'dream' - Generate dream (sleep mode only)")
                    print("  'stats' - Show agent statistics")
                    print("  'help' - Show this help")
                
                elif user_input.lower() == 'stats':
                    state = agent_brain.get_state()
                    stats = state.get('memory_statistics', {})
                    session_data = state.get('session_data', {})
                    
                    print(f"\n📊 {agent_name} Statistics:")
                    print(f"   Current Mode: {current_mode}")
                    print(f"   Session Interactions: {session_data.get('interactions', 0)}")
                    print(f"   Dreams Generated: {session_data.get('dreams_generated', 0)}")
                    print(f"   Conscious Memories: {stats.get('conscious_memories', 0)}")
                    print(f"   Relationships: {stats.get('conscious_relationships', 0)}")
                    print(f"   Short-term Memory: {stats.get('short_term_entries', 0)} entries")
                    print(f"   Current Emotion: {stats.get('emotional_state', 'neutral')}")
                
                elif user_input.lower() == 'sleep':
                    if current_mode == "sleep":
                        print(f"💤 {agent_name} is already asleep.")
                    else:
                        result = agent_brain.switch_mode("sleep")
                        current_mode = "sleep"
                        print(f"💤 {agent_name} is now sleeping... Ready to dream.")
                
                elif user_input.lower() == 'wake':
                    if current_mode == "wake":
                        print(f"👁️ {agent_name} is already awake.")
                    else:
                        result = agent_brain.switch_mode("wake")
                        current_mode = "wake"
                        print(f"👁️ {agent_name} is now awake and ready to chat!")
                
                elif user_input.lower() == 'dream':
                    if current_mode != "sleep":
                        print(f"❌ {agent_name} must be asleep to dream. Type 'sleep' first.")
                    else:
                        print(f"🌙 Generating dream for {agent_name}...")
                        print("   (This may take a moment...)")
                        
                        result = agent_brain.generate_dream()
                        
                        if result.get('dream'):
                            dream = result['dream']
                            print(f"\n✨ Dream Generated!")
                            print(f"📝 Title: {dream.get('title', 'Untitled Dream')}")
                            print(f"📖 Narrative: {dream.get('narrative', 'No narrative')[:300]}...")
                            
                            if dream.get('activated_signifiers'):
                                signifiers = dream['activated_signifiers'][:5]
                                print(f"🧠 Activated Signifiers: {', '.join(signifiers)}")
                            
                            if dream.get('images'):
                                print(f"🖼️ Generated {len(dream['images'])} dream images")
                            
                            # Show dream count
                            dream_count = result.get('dream_count', 1)
                            print(f"💭 Total dreams this session: {dream_count}")
                        else:
                            error = result.get('error', 'Unknown error')
                            print(f"❌ Failed to generate dream: {error}")
                
                else:
                    # Regular conversation
                    if current_mode == "sleep":
                        print(f"💤 {agent_name} is sleeping. Wake them up first with 'wake'.")
                    else:
                        result = agent_brain.process_message(user_input)
                        
                        if result.get('response'):
                            response = result['response']
                            print(f"\n{agent_name}: {response}")
                            
                            # Show unconscious activity if significant
                            unconscious_state = result.get('unconscious_state', {})
                            active_signifiers = unconscious_state.get('active_signifiers', [])
                            if active_signifiers:
                                print(f"[Unconscious activity: {', '.join(active_signifiers[:3])}]")
                        else:
                            error = result.get('error', 'Unknown error')
                            print(f"\n❌ {agent_name}: {error}")
            
            except KeyboardInterrupt:
                print(f"\n\n👋 Interrupted. {agent_name} is resting.")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
        
        # Save state before exit
        try:
            agent_brain.save_state()
            print("💾 Agent state saved successfully.")
        except Exception as e:
            print(f"⚠️ Warning: Could not save state: {e}")
            
    except ImportError as e:
        print(f"❌ Error importing Phase 2 components: {e}")
        print("Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        import traceback
        traceback.print_exc()

def validate_environment():
    """Validate that the environment is properly configured."""
    issues = []
    
    # Check API keys
    if not os.environ.get("GROQ_API_KEY"):
        issues.append("GROQ_API_KEY not found in environment variables")
    
    if not os.environ.get("GEMINI_API_KEY"):
        issues.append("GEMINI_API_KEY not found in environment variables")
    
    # Check required directories
    required_dirs = ["phase1/prompts", "phase2/prompts", "utils"]
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            issues.append(f"Required directory missing: {dir_path}")
    
    # Check critical template files
    critical_templates = [
        "phase1/prompts/extract_unconscious_signifiers.mustache",
        "phase1/prompts/extract_relationships.mustache",
        "phase1/prompts/extract_persona.mustache",
        "phase2/prompts/agent_response.mustache"
    ]
    
    for template in critical_templates:
        if not os.path.exists(template):
            issues.append(f"Critical template missing: {template}")
    
    return issues

def main():
    """Main entry point for DreamerAgent system."""
    print("🧠 DreamerAgent - Psychoanalytic AI System")
    print("🔬 Production Release v1.0")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Validate environment
    print("🔍 Validating environment...")
    issues = validate_environment()
    
    if issues:
        print("❌ Environment validation failed:")
        for issue in issues:
            print(f"   • {issue}")
        print("\n💡 Please fix these issues before continuing.")
        return
    
    # Configure APIs
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    groq_api_key = os.environ.get("GROQ_API_KEY")
    
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        print("✅ Gemini API configured")
    else:
        print("⚠️ Warning: GEMINI_API_KEY not found")
    
    if groq_api_key:
        print("✅ Groq API configured")
    else:
        print("⚠️ Warning: GROQ_API_KEY not found")
    
    # Set up project structure
    setup_project_structure()
    print("✅ Project structure validated")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="DreamerAgent - Psychoanalytic AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --extract nancy.json                    # Extract agent from dream file
  python main.py --interactive --agent Nancy             # Chat with specific agent
  python main.py --extract nancy.json --interactive      # Extract then chat
  python main.py --list                                  # List available agents
        """
    )
    
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
            print(f"\n📋 Available agents ({len(agents)}):")
            for i, agent in enumerate(agents, 1):
                print(f"   {i}. {agent}")
        else:
            print("\n📋 No agents available.")
            print("💡 Extract an agent first using: python main.py --extract <dream_file.json>")
    
    elif args.extract:
        agent_name = extract_agent(args.extract)
        
        if agent_name:
            print(f"\n🎉 Extraction completed!")
            if args.interactive:
                interactive_mode(agent_name)
        else:
            print(f"\n💥 Extraction failed!")
    
    elif args.interactive:
        interactive_mode(args.agent)
    
    else:
        print("\n📖 DreamerAgent Usage Guide:")
        print("=" * 30)
        print("1. Extract an agent from dream data:")
        print("   python main.py --extract your_dream_file.json")
        print()
        print("2. Chat with an extracted agent:")
        print("   python main.py --interactive --agent AgentName")
        print()
        print("3. List available agents:")
        print("   python main.py --list")
        print()
        print("4. Extract and immediately start chatting:")
        print("   python main.py --extract dreams.json --interactive")
        print()
        parser.print_help()

if __name__ == "__main__":
    main()