import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from interfaces.llm_interface import LLMInterface

class NeuroProxyEngine:
    def __init__(self, agent_name: str, base_path: str = "base_agents"):
        """Initialize NeuroProxy Engine with unified emotional-neurochemical simulation."""
        self.agent_name = agent_name
        self.agent_path = os.path.join(base_path, agent_name)
        
        # Initialize LLM interface for emotional analysis
        self.llm_interface = LLMInterface()
        
        # Unified neurochemical-emotional state
        self.neurochemical_state = {
            "dopamine": 0.5,      # Reward/motivation
            "serotonin": 0.5,     # Mood/well-being
            "norepinephrine": 0.5, # Attention/arousal
            "cortisol": 0.3,      # Stress
            "oxytocin": 0.4,      # Social bonding
            "gaba": 0.6           # Relaxation/inhibition
        }
        
        # Derived PAD values (calculated from neurochemicals, not stored separately)
        self.pad_state = {
            "pleasure": 0.0,
            "arousal": 0.0,
            "dominance": 0.0
        }
        
        # Emotional history for tracking patterns
        self.emotional_history = []
        self.last_update = datetime.now()
        
        # Load persistent state if exists
        self._load_persistent_state()
        
        # Derive initial affective state
        self.current_affective_state = self._derive_affective_state()
        
        # Clean up any old duplicate files
        self._cleanup_duplicate_files()
        
        print(f"NeuroProxy Engine initialized for {self.agent_name}")
        print(f"Initial neurochemical state: {self.neurochemical_state}")
        print(f"Derived PAD state: P={self.pad_state['pleasure']:.2f}, A={self.pad_state['arousal']:.2f}, D={self.pad_state['dominance']:.2f}")

    def _cleanup_duplicate_files(self) -> None:
        """Remove duplicate emotional state files, keeping only neuroproxy_state.json."""
        try:
            old_emotional_file = os.path.join(self.agent_path, "emotional_state.json")
            if os.path.exists(old_emotional_file):
                os.remove(old_emotional_file)
                print(f"Removed duplicate file: {old_emotional_file}")
        except Exception as e:
            print(f"Error cleaning up duplicate files: {e}")

    def _load_persistent_state(self) -> None:
        """Load persistent state from unified neuroproxy_state.json file."""
        state_file = os.path.join(self.agent_path, "neuroproxy_state.json")
        
        try:
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    data = json.load(f)
                
                # Load neurochemical state (primary data)
                if 'neurochemical_state' in data and isinstance(data['neurochemical_state'], dict):
                    self.neurochemical_state.update(data['neurochemical_state'])
                
                # Load emotional history
                if 'emotional_history' in data and isinstance(data['emotional_history'], list):
                    self.emotional_history = data['emotional_history'][-20:]
                
                # Load last update time
                if 'last_update' in data:
                    try:
                        self.last_update = datetime.fromisoformat(data['last_update'])
                    except ValueError:
                        self.last_update = datetime.now()
                
                print(f"Loaded persistent state from {state_file}")
            else:
                # Try to migrate from old emotional_state.json if it exists
                self._migrate_from_old_format()
                
        except Exception as e:
            print(f"Error loading persistent state: {e}")
            print("Using default neurochemical state.")

    def _migrate_from_old_format(self) -> None:
        """Migrate from old emotional_state.json to new format if it exists."""
        old_file = os.path.join(self.agent_path, "emotional_state.json")
        
        if os.path.exists(old_file):
            try:
                with open(old_file, 'r') as f:
                    data = json.load(f)
                
                if 'neurochemical_state' in data:
                    self.neurochemical_state.update(data['neurochemical_state'])
                    print(f"Migrated neurochemical state from old format")
                
            except Exception as e:
                print(f"Error migrating from old format: {e}")

    def _save_persistent_state(self) -> None:
        """Save current state to unified neuroproxy_state.json file."""
        state_file = os.path.join(self.agent_path, "neuroproxy_state.json")
        
        try:
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            
            # Save complete state including derived values for reference
            data_to_save = {
                "agent_name": self.agent_name,
                "neurochemical_state": self.neurochemical_state,
                "derived_pad_state": self.pad_state,  # For reference only
                "current_emotion": self.current_affective_state.get("emotion_category", "neutral"),
                "emotional_history": self.emotional_history[-30:],
                "last_update": self.last_update.isoformat(),
                "timestamp": datetime.now().isoformat()
            }
            
            with open(state_file, 'w') as f:
                json.dump(data_to_save, f, indent=2)
                
        except Exception as e:
            print(f"Error saving persistent state: {e}")

    def _derive_affective_state(self) -> Dict[str, Any]:
        """Derive affective state (PAD values and emotion category) from neurochemical levels."""
        try:
            # Enhanced PAD calculation with more nuanced neurochemical interactions
            
            # Pleasure: dopamine and serotonin positive, cortisol negative, oxytocin mildly positive
            pleasure = (
                (self.neurochemical_state["dopamine"] * 0.4) + 
                (self.neurochemical_state["serotonin"] * 0.35) + 
                (self.neurochemical_state["oxytocin"] * 0.15) -
                (self.neurochemical_state["cortisol"] * 0.6)
            )
            
            # Arousal: norepinephrine and cortisol positive, GABA negative
            arousal = (
                (self.neurochemical_state["norepinephrine"] * 0.5) + 
                (self.neurochemical_state["cortisol"] * 0.3) + 
                (self.neurochemical_state["dopamine"] * 0.1) -
                (self.neurochemical_state["gaba"] * 0.4)
            )
            
            # Dominance: dopamine and norepinephrine positive, cortisol negative, serotonin stabilizing
            dominance = (
                (self.neurochemical_state["dopamine"] * 0.35) + 
                (self.neurochemical_state["norepinephrine"] * 0.25) + 
                (self.neurochemical_state["serotonin"] * 0.15) -
                (self.neurochemical_state["cortisol"] * 0.4) +
                (self.neurochemical_state["oxytocin"] * 0.05)
            )
            
            # Normalize to [-1, 1] range with smooth clamping
            self.pad_state["pleasure"] = max(-1.0, min(1.0, pleasure))
            self.pad_state["arousal"] = max(-1.0, min(1.0, arousal))
            self.pad_state["dominance"] = max(-1.0, min(1.0, dominance))
            
            # Categorize emotion based on PAD octants
            emotion_category = self._categorize_emotion(
                self.pad_state["pleasure"], 
                self.pad_state["arousal"], 
                self.pad_state["dominance"]
            )
            
            # Determine response style
            response_style = self._determine_response_style(
                self.pad_state["pleasure"], 
                self.pad_state["arousal"], 
                self.pad_state["dominance"]
            )
            
            # Create comprehensive affective state
            affective_state = {
                "pleasure": self.pad_state["pleasure"],
                "arousal": self.pad_state["arousal"],
                "dominance": self.pad_state["dominance"],
                "emotion_category": emotion_category,
                "response_style": response_style,
                "neurochemical_state": self.neurochemical_state.copy(),
                "emotional_description": self._generate_emotional_description(),
                "timestamp": datetime.now().isoformat()
            }
            
            return affective_state
            
        except Exception as e:
            print(f"Error deriving affective state: {e}")
            return {
                "pleasure": 0.0, "arousal": 0.0, "dominance": 0.0,
                "emotion_category": "neutral", "response_style": "balanced",
                "neurochemical_state": self.neurochemical_state.copy(),
                "emotional_description": "feeling neutral",
                "timestamp": datetime.now().isoformat()
            }

    def _generate_emotional_description(self) -> str:
        """Generate natural language description of current emotional state."""
        p, a, d = self.pad_state["pleasure"], self.pad_state["arousal"], self.pad_state["dominance"]
        
        # Base descriptors
        if p > 0.5 and a > 0.5:
            base = "energized and happy"
        elif p > 0.5 and a < -0.5:
            base = "calm and content"
        elif p < -0.5 and a > 0.5:
            base = "agitated and distressed"
        elif p < -0.5 and a < -0.5:
            base = "sad and withdrawn"
        elif p > 0.3:
            base = "generally positive"
        elif p < -0.3:
            base = "somewhat troubled"
        else:
            base = "relatively neutral"
        
        # Dominance modifier
        if d > 0.5:
            modifier = "confident and "
        elif d < -0.5:
            modifier = "uncertain and "
        else:
            modifier = ""
        
        return modifier + base

    def _categorize_emotion(self, pleasure: float, arousal: float, dominance: float) -> str:
        """Enhanced emotion categorization using PAD octants."""
        # Define emotion mappings for PAD octants
        if pleasure > 0.3:
            if arousal > 0.3:
                if dominance > 0.3:
                    return "elated"  # +P+A+D
                else:
                    return "excited"  # +P+A-D
            else:
                if dominance > 0.3:
                    return "content"  # +P-A+D
                else:
                    return "relaxed"  # +P-A-D
        else:
            if arousal > 0.3:
                if dominance > 0.3:
                    return "angry"    # -P+A+D
                else:
                    return "anxious"  # -P+A-D
            else:
                if dominance > 0.3:
                    return "bored"    # -P-A+D
                else:
                    return "sad"      # -P-A-D
        
        # Near-neutral states
        if abs(pleasure) < 0.3 and abs(arousal) < 0.3:
            return "neutral"
        elif arousal > 0.5:
            return "alert"
        elif arousal < -0.5:
            return "tired"
        
        return "neutral"

    def _determine_response_style(self, pleasure: float, arousal: float, dominance: float) -> str:
        """Determine response style based on PAD values."""
        # Primary style based on most extreme dimension
        styles = []
        
        if abs(dominance) > abs(pleasure) and abs(dominance) > abs(arousal):
            if dominance > 0.6:
                return "assertive"
            elif dominance < -0.6:
                return "submissive"
        
        if abs(arousal) > abs(pleasure) and abs(arousal) > abs(dominance):
            if arousal > 0.6:
                return "energetic"
            elif arousal < -0.6:
                return "passive"
        
        if abs(pleasure) >= abs(arousal) and abs(pleasure) >= abs(dominance):
            if pleasure > 0.6:
                return "optimistic"
            elif pleasure < -0.6:
                return "pessimistic"
        
        return "balanced"

    def update_emotional_state(self, text_input: str, context: str = "interaction") -> Dict[str, Any]:
        """Update emotional state based on text input and context."""
        try:
            print(f"\n=== Updating emotional state ===")
            print(f"Context: {context}")
            print(f"Input: '{text_input[:100]}...'")
            
            # Apply homeostasis before update
            self._apply_homeostasis()
            
            # Analyze emotional content using LLM
            emotional_analysis = self._analyze_emotional_content(text_input)
            print(f"LLM Analysis: {emotional_analysis}")
            
            # Update neurochemical state based on analysis
            self._update_neurochemical_state(emotional_analysis, context)
            
            # Derive new affective state from updated neurochemicals
            self.current_affective_state = self._derive_affective_state()
            
            # Record in history
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "context": context,
                "input_summary": text_input[:100] + "..." if len(text_input) > 100 else text_input,
                "emotional_analysis": {
                    "dominant_emotion": emotional_analysis.get("dominant_emotion"),
                    "intensity": emotional_analysis.get("intensity"),
                    "valence": emotional_analysis.get("valence")
                },
                "resulting_pad": self.pad_state.copy(),
                "resulting_emotion": self.current_affective_state["emotion_category"]
            }
            self.emotional_history.append(history_entry)
            self.emotional_history = self.emotional_history[-50:]
            
            # Update timestamp and save
            self.last_update = datetime.now()
            self._save_persistent_state()
            
            print(f"Updated emotional state: {self.current_affective_state['emotion_category']}")
            print(f"PAD: P={self.pad_state['pleasure']:.2f}, A={self.pad_state['arousal']:.2f}, D={self.pad_state['dominance']:.2f}")
            print(f"Neurochemicals: {', '.join([f'{k[:3]}={v:.2f}' for k, v in self.neurochemical_state.items()])}")
            
            return self.current_affective_state
            
        except Exception as e:
            print(f"ERROR in update_emotional_state: {e}")
            import traceback
            traceback.print_exc()
            return self.current_affective_state

    def _apply_homeostasis(self) -> None:
        """Apply homeostasis - gradual return to baseline emotional state."""
        try:
            current_time = datetime.now()
            time_elapsed = (current_time - self.last_update).total_seconds()
            
            if time_elapsed <= 0:
                return

            # Baseline values for each neurochemical
            baselines = {
                "dopamine": 0.5,
                "serotonin": 0.5,
                "norepinephrine": 0.5,
                "cortisol": 0.3,
                "oxytocin": 0.4,
                "gaba": 0.6
            }
            
            # Adaptive decay rate based on distance from baseline
            # Faster decay when far from baseline, slower when close
            for neurotransmitter, baseline in baselines.items():
                current = self.neurochemical_state[neurotransmitter]
                distance = abs(current - baseline)
                
                # Decay rate increases with distance from baseline
                decay_rate = 0.001 + (distance * 0.002)  # 0.1% to 0.3% per second
                decay_amount = decay_rate * time_elapsed
                
                # Apply decay towards baseline
                if current > baseline:
                    self.neurochemical_state[neurotransmitter] = max(
                        baseline, 
                        current - decay_amount
                    )
                else:
                    self.neurochemical_state[neurotransmitter] = min(
                        baseline, 
                        current + decay_amount
                    )
            
            # Ensure bounds
            for key in self.neurochemical_state:
                self.neurochemical_state[key] = max(0.0, min(1.0, self.neurochemical_state[key]))
                
        except Exception as e:
            print(f"Error in homeostasis: {e}")

    def _analyze_emotional_content(self, text: str) -> Dict[str, Any]:
        """Analyze emotional content using LLM."""
        try:
            # Call LLM for emotional analysis
            llm_result = self._llm_emotional_analysis(text)
            
            # Ensure all required fields are present
            return {
                "dominant_emotion": llm_result.get("dominant_emotion", "neutral"),
                "intensity": float(llm_result.get("intensity", 0.5)),
                "valence": float(llm_result.get("valence", 0.0)),
                "arousal": float(llm_result.get("arousal", 0.0)),
                "secondary_emotions": llm_result.get("secondary_emotions", []),
                "emotional_indicators": llm_result.get("emotional_indicators", []),
                "analysis": llm_result.get("analysis", "")
            }
            
        except Exception as e:
            print(f"Error in emotional analysis: {e}")
            return self._default_emotion_analysis()

    def _llm_emotional_analysis(self, text: str) -> Dict[str, Any]:
        """Call LLM for emotional analysis."""
        try:
            prompt_data = {"text": text}
            response = self.llm_interface.generate("utils", "analyze_emotion", prompt_data)
            
            if response:
                return self._parse_emotion_response(response)
            else:
                return self._default_emotion_analysis()
                
        except Exception as e:
            print(f"LLM analysis error: {e}")
            return self._default_emotion_analysis()

    def _parse_emotion_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM emotion analysis response."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Clean up common JSON issues
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                
                data = json.loads(json_str)
                
                # Validate and normalize data
                result = {
                    "dominant_emotion": str(data.get("dominant_emotion", "neutral")).lower(),
                    "intensity": max(0.0, min(1.0, float(data.get("intensity", 0.5)))),
                    "valence": max(-1.0, min(1.0, float(data.get("valence", 0.0)))),
                    "secondary_emotions": data.get("secondary_emotions", []),
                    "emotional_indicators": data.get("emotional_indicators", []),
                    "analysis": data.get("analysis", "")
                }
                
                # Derive arousal if not provided
                if "arousal" in data:
                    result["arousal"] = max(-1.0, min(1.0, float(data["arousal"])))
                else:
                    # Infer arousal from emotion
                    emotion = result["dominant_emotion"]
                    if emotion in ["excited", "angry", "anxious", "elated"]:
                        result["arousal"] = 0.6
                    elif emotion in ["sad", "bored", "tired", "relaxed"]:
                        result["arousal"] = -0.4
                    else:
                        result["arousal"] = 0.0
                
                return result
                
        except Exception as e:
            print(f"Parse error: {e}")
        
        return self._default_emotion_analysis()

    def _default_emotion_analysis(self) -> Dict[str, Any]:
        """Return default neutral emotional analysis."""
        return {
            "dominant_emotion": "neutral",
            "intensity": 0.3,
            "valence": 0.0,
            "arousal": 0.0,
            "secondary_emotions": [],
            "emotional_indicators": [],
            "analysis": "Unable to determine specific emotional content"
        }

    def _update_neurochemical_state(self, emotional_analysis: Dict[str, Any], context: str) -> None:
        """Update neurochemical state based on emotional analysis."""
        try:
            intensity = emotional_analysis.get("intensity", 0.3)
            valence = emotional_analysis.get("valence", 0.0)
            arousal = emotional_analysis.get("arousal", 0.0)
            emotion = emotional_analysis.get("dominant_emotion", "neutral")
            
            # Scale factor based on intensity (non-linear for more realistic response)
            scale = (intensity ** 0.8) * 0.2  # Max 20% change
            
            # Dopamine: reward, motivation, pleasure
            if valence > 0.1:
                self.neurochemical_state["dopamine"] += scale * valence * 0.7
                if emotion in ["excited", "elated", "happy"]:
                    self.neurochemical_state["dopamine"] += scale * 0.2
            elif valence < -0.1:
                self.neurochemical_state["dopamine"] -= scale * abs(valence) * 0.4
            
            # Serotonin: mood regulation, contentment
            if valence > 0.2:
                self.neurochemical_state["serotonin"] += scale * valence * 0.5
            elif valence < -0.2:
                self.neurochemical_state["serotonin"] -= scale * abs(valence) * 0.6
            
            # Norepinephrine: alertness, arousal
            if arousal != 0:
                self.neurochemical_state["norepinephrine"] += scale * arousal * 0.6
            elif emotion in ["anxious", "angry", "excited"]:
                self.neurochemical_state["norepinephrine"] += scale * 0.4
            elif emotion in ["sad", "bored", "tired"]:
                self.neurochemical_state["norepinephrine"] -= scale * 0.3
            
            # Cortisol: stress response
            if emotion in ["anxious", "angry", "fearful", "stressed"]:
                self.neurochemical_state["cortisol"] += scale * 0.7
            elif valence > 0.3 and arousal < 0.3:  # Positive and calm reduces stress
                self.neurochemical_state["cortisol"] -= scale * 0.4
            
            # Oxytocin: social bonding, trust
            if context.lower() in ["social_interaction", "user_interaction", "conversation"]:
                if valence > 0:
                    self.neurochemical_state["oxytocin"] += scale * 0.5
                else:
                    self.neurochemical_state["oxytocin"] += scale * 0.1  # Even negative interactions can increase oxytocin
            
            # GABA: relaxation, inhibition
            if arousal < -0.3 or emotion in ["calm", "relaxed", "content"]:
                self.neurochemical_state["gaba"] += scale * 0.5
            elif arousal > 0.5 or emotion in ["anxious", "excited", "angry"]:
                self.neurochemical_state["gaba"] -= scale * 0.4
            
            # Ensure all values stay within bounds [0, 1]
            for key in self.neurochemical_state:
                self.neurochemical_state[key] = max(0.0, min(1.0, self.neurochemical_state[key]))
            
        except Exception as e:
            print(f"Error updating neurochemical state: {e}")

    def get_current_state(self) -> Dict[str, Any]:
        """Get current emotional state."""
        return self.current_affective_state.copy()

    def get_emotional_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent emotional history."""
        return self.emotional_history[-limit:]

    def get_response_style_guidance(self) -> Dict[str, str]:
        """Get guidance for response style based on current emotional state."""
        state = self.current_affective_state
        
        return {
            "style": state.get("response_style", "balanced"),
            "emotion": state.get("emotion_category", "neutral"),
            "emotional_description": state.get("emotional_description", "feeling neutral"),
            "pad_pleasure": f"{state.get('pleasure', 0.0):.2f}",
            "pad_arousal": f"{state.get('arousal', 0.0):.2f}",
            "pad_dominance": f"{state.get('dominance', 0.0):.2f}",
            "tone": self._get_tone_guidance(state.get('pleasure', 0.0)),
            "energy": self._get_energy_guidance(state.get('arousal', 0.0)),
            "approach": self._get_approach_guidance(state.get('dominance', 0.0)),
            "dopamine_level": f"{self.neurochemical_state['dopamine']:.2f}",
            "serotonin_level": f"{self.neurochemical_state['serotonin']:.2f}",
            "oxytocin_level": f"{self.neurochemical_state['oxytocin']:.2f}",
            "cortisol_level": f"{self.neurochemical_state['cortisol']:.2f}",
            "norepinephrine_level": f"{self.neurochemical_state['norepinephrine']:.2f}"
        }

    def _get_tone_guidance(self, pleasure: float) -> str:
        """Get tone guidance based on pleasure dimension."""
        if pleasure > 0.6:
            return "warm, enthusiastic, and positive"
        elif pleasure > 0.3:
            return "friendly and generally positive"
        elif pleasure > 0:
            return "mildly positive"
        elif pleasure > -0.3:
            return "neutral to slightly cautious"
        elif pleasure > -0.6:
            return "somewhat subdued or concerned"
        else:
            return "serious or troubled"

    def _get_energy_guidance(self, arousal: float) -> str:
        """Get energy guidance based on arousal dimension."""
        if arousal > 0.6:
            return "high energy, animated, and expressive"
        elif arousal > 0.3:
            return "energetic and engaged"
        elif arousal > 0:
            return "moderately energetic"
        elif arousal > -0.3:
            return "calm and measured"
        elif arousal > -0.6:
            return "quite calm and deliberate"
        else:
            return "very low energy, possibly tired"

    def _get_approach_guidance(self, dominance: float) -> str:
        """Get approach guidance based on dominance dimension."""
        if dominance > 0.6:
            return "confident, assertive, and direct"
        elif dominance > 0.3:
            return "self-assured and proactive"
        elif dominance > 0:
            return "balanced and cooperative"
        elif dominance > -0.3:
            return "somewhat cautious or deferential"
        elif dominance > -0.6:
            return "hesitant and seeking guidance"
        else:
            return "very submissive or uncertain"

    def reset_to_baseline(self) -> None:
        """Reset neurochemical state to baseline."""
        print(f"Resetting emotional state to baseline for {self.agent_name}...")
        
        self.neurochemical_state = {
            "dopamine": 0.5,
            "serotonin": 0.5,
            "norepinephrine": 0.5,
            "cortisol": 0.3,
            "oxytocin": 0.4,
            "gaba": 0.6
        }
        
        self.current_affective_state = self._derive_affective_state()
        self.last_update = datetime.now()
        self._save_persistent_state()
        
        print(f"Reset complete. PAD: P={self.pad_state['pleasure']:.2f}, A={self.pad_state['arousal']:.2f}, D={self.pad_state['dominance']:.2f}")

    def __str__(self) -> str:
        """String representation of current emotional state."""
        state = self.current_affective_state
        return (f"NeuroProxy({self.agent_name}): "
                f"{state.get('emotion_category', 'neutral')} - "
                f"{state.get('emotional_description', 'feeling neutral')} "
                f"[P:{state.get('pleasure', 0.0):.2f}, A:{state.get('arousal', 0.0):.2f}, D:{state.get('dominance', 0.0):.2f}]")