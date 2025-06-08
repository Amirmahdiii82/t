from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

class PsychicIntegration:
    """Manages the dynamic interaction between conscious and unconscious processes."""
    
    def __init__(self, conscious_processor, unconscious_processor, memory_manager):
        self.conscious = conscious_processor
        self.unconscious = unconscious_processor
        self.memory = memory_manager
        
        # Set bidirectional references
        self.conscious.set_unconscious_processor(self.unconscious)
        
        # Track psychic energy distribution
        self.psychic_energy = {
            'conscious': 0.6,  # Ego energy
            'unconscious': 0.4,  # Id energy
            'defensive': 0.0   # Energy bound in defenses
        }
        
        # Track working through process
        self.working_through = {
            'repetitions': {},
            'insights': [],
            'resistance_patterns': []
        }
    
    def process_interaction(self, user_input: str, context: str = "dialogue") -> Dict[str, Any]:
        """Process interaction through full psychoanalytic apparatus."""
        
        # 1. Initial unconscious processing
        unconscious_response = self.unconscious.process_input(user_input, context)
        
        # 2. Calculate psychic energy distribution
        self._update_psychic_energy(unconscious_response)
        
        # 3. Check for repetition compulsion
        repetition = self._check_repetition(user_input, unconscious_response)
        
        # 4. Process through conscious with unconscious influence
        conscious_response = self.conscious.process_input(user_input, context)
        
        # 5. Check for breakthrough moments
        breakthrough = self._check_breakthrough(unconscious_response)
        
        # 6. Apply reality principle modulation
        final_response = self._apply_reality_principle(conscious_response, unconscious_response)
        
        # 7. Track working through
        self._track_working_through(user_input, final_response, unconscious_response)
        
        # 8. Update memory with full psychic state
        self._update_psychic_memory(user_input, final_response, unconscious_response)
        
        return {
            'response': final_response,
            'psychic_state': {
                'energy_distribution': self.psychic_energy.copy(),
                'active_defenses': self._get_active_defenses(unconscious_response),
                'transference_state': unconscious_response.get('transference', {}),
                'repetition_active': repetition,
                'breakthrough_potential': breakthrough,
                'discourse_position': self._get_discourse_position(unconscious_response)
            },
            'interpretation_hints': self._generate_interpretation_hints(unconscious_response)
        }
    
    def _update_psychic_energy(self, unconscious_response: Dict) -> None:
        """Update distribution of psychic energy based on unconscious activity."""
        # High resistance binds energy in defenses
        resistance = unconscious_response.get('resistance', {})
        if resistance.get('present', False):
            defense_energy = resistance.get('intensity', 0.5) * 0.3
            self.psychic_energy['defensive'] = defense_energy
            # Reduce available conscious energy
            self.psychic_energy['conscious'] = 0.6 - defense_energy
        else:
            self.psychic_energy['defensive'] = 0.0
            self.psychic_energy['conscious'] = 0.6
        
        # High jouissance increases unconscious energy
        jouissance = unconscious_response.get('jouissance_effects', {})
        if jouissance.get('level', 0) > 0.5:
            self.psychic_