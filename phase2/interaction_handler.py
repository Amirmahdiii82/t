import os
from phase2.agent_brain import AgentBrain

class InteractionHandler:
    def __init__(self):
        self.agents = {}
        self.current_agent = None
    
    def list_available_agents(self):
        """List available agents in the base_agents directory."""
        agents = []
        if os.path.exists("base_agents"):
            for agent_dir in os.listdir("base_agents"):
                agent_path = os.path.join("base_agents", agent_dir)
                if os.path.isdir(agent_path):
                    agents.append(agent_dir)
        return agents
    
    def load_agent(self, agent_name):
        """Load an agent by name."""
        agent_dir = f"base_agents/{agent_name}"
        
        if not os.path.exists(agent_dir):
            raise ValueError(f"Agent '{agent_name}' not found.")
        
        # Check if agent is already loaded
        if agent_name in self.agents:
            self.current_agent = agent_name
            return True
        
        # Load agent
        self.agents[agent_name] = AgentBrain(agent_name, agent_dir)
        self.current_agent = agent_name
        
        return True
    
    def send_message(self, message):
        """Send a message to the current agent."""
        if not self.current_agent:
            return {
                "error": "No agent selected.",
                "agent": None,
                "response": None
            }
        
        agent = self.agents[self.current_agent]
        result = agent.process_message(message)
        
        return result
    
    def switch_mode(self, mode):
        """Switch the current agent's mode (wake/sleep)."""
        if not self.current_agent:
            return {
                "error": "No agent selected.",
                "mode": None
            }
        
        agent = self.agents[self.current_agent]
        result = agent.switch_mode(mode)
        
        return result
    
    def generate_dream(self):
        """Generate a dream for the current agent."""
        if not self.current_agent:
            return {
                "error": "No agent selected.",
                "dream": None
            }
        
        agent = self.agents[self.current_agent]
        
        # Make sure agent is in sleep mode
        if agent.state["mode"] != "sleep":
            agent.switch_mode("sleep")
        
        # Generate dream
        print("Generating dream...")
        try:
            result = agent.generate_dream()
            print("Dream generation complete")
            return result
        except Exception as e:
            print(f"Error generating dream: {e}")
            return {
                "error": f"Error generating dream: {str(e)}",
                "dream": None
            }
    
    def get_agent_state(self):
        """Get the current state of the agent."""
        if not self.current_agent:
            return {
                "error": "No agent selected.",
                "state": None
            }
        
        agent = self.agents[self.current_agent]
        return agent.get_state()