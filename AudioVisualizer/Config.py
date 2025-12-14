import json
import time
from pathlib import Path
from typing import Optional, Tuple


from AudioVisualizer.DataClasses import *
from AudioVisualizer.constants import *



class ConfigManager:
    """Manages configuration loading and updates."""
    
    DEFAULT_CONFIG_NAME = "basic-config"
    
    def __init__(self, config_path: str = "visualizer_config.json"):
        self.config_path = Path(config_path)
        self.last_check_time = time.time()
        self.active_config_name = self.DEFAULT_CONFIG_NAME
        self._ensure_config_exists()
        
    def _get_default_config_structure(self) -> dict:
        """Returns the default configuration structure."""
        return {
            "config-to-use": self.DEFAULT_CONFIG_NAME,
            "configs": {
                self.DEFAULT_CONFIG_NAME: {
                    "window": vars(WindowConfig()),
                    "colors": vars(ColorConfig()),
                    "line": vars(LineConfig()),
                    "glow": vars(GlowConfig()),
                    "gradient": vars(GradientConfig()),
                    "effects": vars(EffectsConfig()),
                    "audio": vars(AudioConfig())
                }
            }
        }
    
    def _ensure_config_exists(self) -> None:
        """Creates config file if it doesn't exist."""
        if not self.config_path.exists():
            with open(self.config_path, 'w') as f:
                json.dump(self._get_default_config_structure(), f, indent=4)
            print(f"✓ Created config file: {self.config_path}")
    
    def load_config(self) -> dict:
        """Loads configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            config_name = config_data.get("config-to-use", self.DEFAULT_CONFIG_NAME)
            
            if "configs" in config_data and config_name in config_data["configs"]:
                self.active_config_name = config_name
                config = config_data["configs"][config_name]
                print(f"✓ Loaded config '{config_name}' from: {self.config_path}")
                return self._flatten_config(config)
            else:
                print(f"⚠ Config '{config_name}' not found, using defaults")
                return self._flatten_config(
                    self._get_default_config_structure()["configs"][self.DEFAULT_CONFIG_NAME]
                )
                
        except Exception as e:
            print(f"⚠ Error loading config: {e}, using defaults")
            return self._flatten_config(
                self._get_default_config_structure()["configs"][self.DEFAULT_CONFIG_NAME]
            )
    
    def _flatten_config(self, nested_config: dict) -> dict:
        """Flattens nested configuration structure."""
        flat = {}
        
        # Map nested structure to flat keys
        section_mappings = {
            "window": ["width", "height", "x", "y"],
            "colors": ["background", "line", "glow", "status_text"],
            "line": ["width", "alpha", "style"],
            "glow": ["width", "alpha"],
            "gradient": ["enabled", "low_color", "mid_color", "high_color"],
            "effects": ["fill_under_curve", "fill_alpha", "mirror_mode", 
                       "symmetry_mode", "thick_on_bass"],
            "audio": ["smoothing", "amplitude_scale", "amplitude_boost", 
                     "rotation_angle", "y_offset"]
        }
        
        for section, keys in section_mappings.items():
            if section in nested_config:
                section_data = nested_config[section]
                prefix = f"{section}_" if section != "window" and section != "colors" else ""
                
                if section == "window":
                    prefix = "window_"
                elif section == "colors":
                    for key in keys:
                        flat[f"{key}_color" if key != "background" else "background_color"] = \
                            section_data.get(key, getattr(ColorConfig(), key))
                    continue
                
                for key in keys:
                    flat_key = f"{prefix}{key}"
                    default_value = getattr(eval(f"{section.capitalize()}Config")(), key)
                    flat[flat_key] = section_data.get(key, default_value)
        
        return flat
    
    def check_for_updates(self) -> Optional[dict]:
        """Checks if config has been updated. Returns new config if changed."""
        current_time = time.time()
        if current_time - self.last_check_time < CONFIG_CHECK_INTERVAL:
            return None
            
        self.last_check_time = current_time
        
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            new_config_name = config_data.get("config-to-use", self.DEFAULT_CONFIG_NAME)
            
            if "configs" in config_data and new_config_name in config_data["configs"]:
                if new_config_name != self.active_config_name:
                    print(f"✓ Config switched to: '{new_config_name}'")
                    self.active_config_name = new_config_name
                    return self._flatten_config(config_data["configs"][new_config_name])
                    
        except Exception as e:
            print(f"⚠ Error checking config: {e}")
        
        return None

