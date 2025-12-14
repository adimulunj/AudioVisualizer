"""
Taskbar Audio Visualizer
A real-time audio visualization tool that displays system audio output.
"""

import json
import os
import queue
import threading
import time
from typing import Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import soundcard as sc
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection

from AudioVisualizer.DataClasses import *
from AudioVisualizer.constants import *
from AudioVisualizer.Config import *



class AudioCapture:
    """Handles audio capture from system output."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, sample_rate: int = SAMPLE_RATE):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.running = False
        self.capture_thread = None
        self.microphone = self._setup_microphone()
        
    def _setup_microphone(self):
        """Sets up the audio input device (loopback)."""
        self._print_available_devices()
        
        try:
            default_speaker = sc.default_speaker()
            print(f"Default speaker: {default_speaker.name}")
            
            mic = sc.get_microphone(
                id=str(default_speaker.name), 
                include_loopback=True
            )
            print(f"✓ Using loopback: {mic.name}\n")
            return mic
            
        except Exception as e:
            print(f"Error getting loopback: {e}")
            return self._find_loopback_device()
    
    def _print_available_devices(self) -> None:
        """Prints all available audio devices."""
        print("\n" + "=" * 60)
        print("AVAILABLE SPEAKERS:")
        print("=" * 60)
        for i, speaker in enumerate(sc.all_speakers()):
            print(f"[{i}] {speaker.name}")
        
        print("\n" + "=" * 60)
        print("AVAILABLE MICROPHONES (including loopback):")
        print("=" * 60)
        for i, mic in enumerate(sc.all_microphones(include_loopback=True)):
            print(f"[{i}] {mic.name}")
        print("=" * 60 + "\n")
    
    def _find_loopback_device(self):
        """Attempts to find a loopback device."""
        print("\nSearching for loopback device...")
        
        loopback_keywords = ['loopback', 'stereo mix', 'what u hear']
        mics = sc.all_microphones(include_loopback=True)
        
        for mic in mics:
            if any(keyword in mic.name.lower() for keyword in loopback_keywords):
                print(f"✓ Found loopback device: {mic.name}\n")
                return mic
        
        print("\nNo loopback device found!")
        print("Using default microphone (will only capture mic input)")
        mic = sc.default_microphone()
        print(f"Using: {mic.name}\n")
        return mic
    
    def start(self) -> None:
        """Starts audio capture in a separate thread."""
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
    
    def stop(self) -> None:
        """Stops audio capture."""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
    
    def _capture_loop(self) -> None:
        """Main audio capture loop."""
        try:
            with self.microphone.recorder(
                samplerate=self.sample_rate, 
                blocksize=self.chunk_size
            ) as recorder:
                
                while self.running:
                    try:
                        data = recorder.record(numframes=self.chunk_size)
                        data = self._process_raw_audio(data)
                        self._enqueue_audio(data)
                        
                    except Exception as e:
                        print(f"Recording error: {e}")
                        time.sleep(0.1)
                        
        except Exception as e:
            print(f"❌ Audio capture failed: {e}")
    
    def _process_raw_audio(self, data: np.ndarray) -> np.ndarray:
        """Processes raw audio data to mono."""
        if len(data.shape) > 1 and data.shape[1] > 1:
            return data.mean(axis=1)
        return data.flatten()
    
    def _enqueue_audio(self, data: np.ndarray) -> None:
        """Adds audio data to queue, dropping old data if full."""
        try:
            self.audio_queue.put(data, block=False)
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put(data, block=False)
            except:
                pass
    
    def get_audio(self) -> Optional[np.ndarray]:
        """Gets the latest audio data from queue."""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None


class AudioProcessor:
    """Processes audio data with various effects."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE):
        self.chunk_size = chunk_size
    
    def process(self, audio_data: np.ndarray, config: dict) -> np.ndarray:
        """Applies all configured audio processing."""
        # Ensure correct length
        audio_data = self._normalize_length(audio_data)
        
        # Apply transformations in order
        audio_data = self._apply_scaling(audio_data, config)
        audio_data = self._apply_smoothing(audio_data, config)
        audio_data = self._apply_mirror_mode(audio_data, config)
        audio_data = self._apply_symmetry_mode(audio_data, config)
        audio_data = self._apply_rotation(audio_data, config)
        audio_data = self._apply_offset(audio_data, config)
        
        return audio_data
    
    def _normalize_length(self, data: np.ndarray) -> np.ndarray:
        """Ensures audio data is the correct length."""
        if len(data) < self.chunk_size:
            return np.pad(data, (0, self.chunk_size - len(data)))
        elif len(data) > self.chunk_size:
            return data[:self.chunk_size]
        return data
    
    def _apply_scaling(self, data: np.ndarray, config: dict) -> np.ndarray:
        """Applies amplitude scaling and boost."""
        scale = config.get("amplitude_scale", 1.0)
        boost = config.get("amplitude_boost", 0.0)
        return data * scale + boost
    
    def _apply_smoothing(self, data: np.ndarray, config: dict) -> np.ndarray:
        """Applies smoothing filter."""
        smoothing = config.get("smoothing", 0)
        if smoothing > 1:
            window = np.ones(smoothing) / smoothing
            return np.convolve(data, window, mode='same')
        return data
    
    def _apply_mirror_mode(self, data: np.ndarray, config: dict) -> np.ndarray:
        """Applies mirror mode (absolute values)."""
        if config.get("mirror_mode", False):
            return np.abs(data)
        return data
    
    def _apply_symmetry_mode(self, data: np.ndarray, config: dict) -> np.ndarray:
        """Applies symmetry mode (mirror second half)."""
        if config.get("symmetry_mode", False):
            half = len(data) // 2
            data[half:] = data[:half][::-1]
        return data
    
    def _apply_rotation(self, data: np.ndarray, config: dict) -> np.ndarray:
        """Applies rotation/phase shift."""
        angle = config.get("rotation_angle", 0)
        if angle != 0:
            return np.roll(data, int(angle))
        return data
    
    def _apply_offset(self, data: np.ndarray, config: dict) -> np.ndarray:
        """Applies vertical offset."""
        offset = config.get("y_offset", 0.0)
        return data + offset


class ColorGradient:
    """Handles color gradient calculations."""
    
    @staticmethod
    def hex_to_rgb(hex_color: str):
        """Converts hex color to RGB tuple (0-1 range)."""
        hex_color = hex_color.lstrip('#')
        tmp = (int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))
        return tuple(tmp)
    
    @staticmethod
    def get_gradient_color(value: float, min_val: float, max_val: float, 
                          low_color: str, mid_color: str, high_color: str) -> Tuple[float, float, float]:
        """Calculates color based on value using three-point gradient."""
        # Normalize value to 0-1 range
        normalized = np.clip(
            (value - min_val) / (max_val - min_val) if max_val != min_val else 0,
            0, 1
        )
        
        low_rgb = ColorGradient.hex_to_rgb(low_color)
        mid_rgb = ColorGradient.hex_to_rgb(mid_color)
        high_rgb = ColorGradient.hex_to_rgb(high_color)
        
        if normalized < 0.5:
            # Interpolate between low and mid
            t = normalized * 2
            return tuple(low_rgb[i] * (1 - t) + mid_rgb[i] * t for i in range(3))
        else:
            # Interpolate between mid and high
            t = (normalized - 0.5) * 2
            return tuple(mid_rgb[i] * (1 - t) + high_rgb[i] * t for i in range(3))


class Visualizer:
    """Main visualization class."""
    
    LINESTYLE_MAP = {
        "solid": "-",
        "dashed": "--",
        "dotted": ":",
        "dashdot": "-."
    }
    
    def __init__(self, config: dict, root: tk.Tk, chunk_size: int = CHUNK_SIZE):
        self.config = config
        self.root = root
        self.chunk_size = chunk_size
        self.x = np.arange(0, chunk_size)
        
        self.fig = None
        self.ax = None
        self.canvas = None
        self.line = None
        self.line_collection = None
        self.glow_line = None
        self.fill = None
        self.status_text = None
        
        self._setup_plot()
    
    def _setup_plot(self):
        """Sets up the matplotlib plot."""
        bg_color = self.config.get("background_color", "#000000")
        
        self.fig, self.ax = plt.subplots(figsize=(8, 0.4), facecolor=bg_color)
        self.ax.set_facecolor(bg_color)
        self.ax.set_ylim(-0.5, 0.5)
        self.ax.set_xlim(0, self.chunk_size)
        self.ax.axis('off')
        
        plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)
        
        self._create_plot_elements()
        self._embed_in_tkinter()
    
    def _create_plot_elements(self):
        """Creates all plot elements."""
        # Main line
        line_style = self.LINESTYLE_MAP.get(
            self.config.get("line_style", "solid"), "-"
        )
        self.line, = self.ax.plot(
            self.x, np.zeros(self.chunk_size),
            color=self.config.get("line_color", "#00ff88"),
            linewidth=self.config.get("line_width", 4),
            linestyle=line_style,
            antialiased=True,
            alpha=self.config.get("line_alpha", 1)
        )
        
        # Line collection for gradient mode
        points = np.array([self.x, np.zeros(self.chunk_size)]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        self.line_collection = LineCollection(
            segments.tolist(),
            linewidths=self.config.get("line_width", 4),
            alpha=self.config.get("line_alpha", 1)
        )
        self.ax.add_collection(self.line_collection)
        self.line_collection.set_visible(False)
        
        # Glow effect
        self.glow_line, = self.ax.plot(
            self.x, np.zeros(self.chunk_size),
            color=self.config.get("glow_color", "#00ff88"),
            linewidth=self.config.get("glow_width", 20),
            antialiased=True,
            alpha=self.config.get("glow_alpha", 0.6)
        )
        
        # Status text
        self.status_text = self.ax.text(
            self.chunk_size / 2, 0, 'Waiting...',
            ha='center', va='center',
            color=self.config.get("status_text_color", "#00ff88"),
            fontsize=7, alpha=0.5
        )
    
    def _embed_in_tkinter(self) -> None:
        """Embeds matplotlib figure in tkinter window."""
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update(self, audio_data: np.ndarray) -> None:
        """Updates visualization with new audio data."""
        # Check if there's actual audio
        level = np.abs(audio_data).max()
        self.status_text.set_text('' if level > AUDIO_THRESHOLD else 'Waiting...')
        
        # Determine line width (with bass effect if enabled)
        line_width = self._calculate_line_width(audio_data)
        
        # Update visualization based on gradient mode
        if self.config.get("gradient_enabled", True):
            self._update_gradient_mode(audio_data, line_width)
        else:
            self._update_solid_mode(audio_data, line_width)
        
        # Update glow effect
        self._update_glow(audio_data, line_width)
        
        # Handle fill under curve
        self._update_fill(audio_data)
        
        # Force redraw
        self.canvas.draw_idle()
    
    def _calculate_line_width(self, audio_data: np.ndarray) -> float:
        """Calculates line width, applying bass effect if enabled."""
        base_width = self.config.get("line_width", 4)
        
        if self.config.get("thick_on_bass", False):
            avg_amplitude = np.abs(audio_data).mean()
            return base_width * (1 + avg_amplitude * 3)
        
        return base_width
    
    def _update_gradient_mode(self, audio_data: np.ndarray, line_width: float) -> None:
        """Updates visualization in gradient mode."""
        self.line.set_visible(False)
        self.line_collection.set_visible(True)
        
        # Create colored segments
        points = np.array([self.x, audio_data]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Calculate colors
        abs_data = np.abs(audio_data)
        max_val = max(abs_data.max(), 0.1)
        
        colors = [
            ColorGradient.get_gradient_color(
                abs(val), 0, max_val,
                self.config.get("gradient_low_color", "#0000ff"),
                self.config.get("gradient_mid_color", "#00ff00"),
                self.config.get("gradient_high_color", "#ff0000")
            )
            for val in audio_data
        ]
        
        self.line_collection.set_segments(segments.tolist())
        self.line_collection.set_colors(colors)
        self.line_collection.set_linewidth(line_width)
    
    def _update_solid_mode(self, audio_data: np.ndarray, line_width: float) -> None:
        """Updates visualization in solid color mode."""
        self.line.set_visible(True)
        self.line_collection.set_visible(False)
        self.line.set_ydata(audio_data)
        self.line.set_linewidth(line_width)
    
    def _update_glow(self, audio_data: np.ndarray, line_width: float) -> None:
        """Updates glow effect."""
        glow_width = self.config.get("glow_width", 20)
        if self.config.get("thick_on_bass", False):
            avg_amplitude = np.abs(audio_data).mean()
            glow_width = glow_width * (1 + avg_amplitude * 3)
        
        self.glow_line.set_ydata(audio_data)
        self.glow_line.set_linewidth(glow_width)
    
    def _update_fill(self, audio_data: np.ndarray) -> None:
        """Updates fill under curve."""
        if self.config.get("fill_under_curve", False):
            if self.fill is not None:
                self.fill.remove()
            
            fill_color = self.config.get("line_color", "#00ff88")
            if self.config.get("gradient_enabled", True):
                fill_color = self.config.get("gradient_mid_color", "#00ff00")
            
            self.fill = self.ax.fill_between(
                self.x, 0, audio_data,
                color=fill_color,
                alpha=self.config.get("fill_alpha", 0.3)
            )
        else:
            if self.fill is not None:
                self.fill.remove()
                self.fill = None
    
    def apply_config_update(self, config: dict):
        """Applies configuration updates to visual elements."""
        self.config = config
        
        try:
            bg_color = config.get("background_color", "#000000")
            self.fig.set_facecolor(bg_color)
            self.ax.set_facecolor(bg_color)
            
            self.status_text.set_color(config.get("status_text_color", "#00ff88"))
            
            if not config.get("gradient_enabled", True):
                self.line.set_color(config.get("line_color", "#00ff88"))
                self.line.set_linewidth(config.get("line_width", 4))
                self.line.set_alpha(config.get("line_alpha", 1))
            
            self.glow_line.set_color(config.get("glow_color", "#00ff88"))
            self.glow_line.set_linewidth(config.get("glow_width", 20))
            self.glow_line.set_alpha(config.get("glow_alpha", 0.6))
            
            print("✓ Visual properties updated!")
        except Exception as e:
            print(f"⚠ Error applying config: {e}")


class TaskbarWindow:
    """Manages the taskbar window."""
    
    def __init__(self, config: dict):
        self.config = config
        self.root = tk.Tk()
        self.is_pinned = True
        self._setup_window()
        self._bind_events()
        self._keep_on_top()
    
    def _setup_window(self) -> None:
        """Sets up the window properties."""
        self.root.title("Audio Visualizer")
        
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        width, height, x, y = self._calculate_geometry(
            screen_width, screen_height
        )
        
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Window attributes
        self.root.attributes('-topmost', 1)
        self.root.overrideredirect(True)
        self.root.attributes('-toolwindow', True)
        self.root.attributes('-transparentcolor', '#000000')
    
    def _calculate_geometry(self, screen_width: int, screen_height: int) -> Tuple[int, int, int, int]:
        """Calculates window geometry."""
        width = self.config.get("window_width", 600)
        height = self.config.get("window_height", 40)
        
        x_config = self.config.get("window_x", "auto")
        y_config = self.config.get("window_y", "auto")
        
        # Calculate X position
        if x_config == "auto":
            x = (screen_width - width) // 2 - 100
        else:
            x = int(str(x_config).replace("+", ""))
        
        # Calculate Y position
        if y_config == "auto":
            taskbar_height = -880
            y = screen_height - taskbar_height - height - 5
        else:
            y = int(str(y_config).replace("+", ""))
        
        return width, height, x, y
    
    def _bind_events(self) -> None:
        """Binds keyboard events."""
        self.root.bind('<Escape>', lambda e: self.close())
    
    def _keep_on_top(self) -> None:
        """Forces window to stay on top."""
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after(100, self._keep_on_top)
    
    def update_geometry(self, config: dict) -> None:
        """Updates window geometry from config."""
        self.config = config
        
        try:
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            
            width, height, x, y = self._calculate_geometry(
                screen_width, screen_height
            )
            
            self.root.geometry(f'{width}x{height}+{x}+{y}')
            print(f"✓ Window geometry updated: {width}x{height}+{x}+{y}")
        except Exception as e:
            print(f"⚠ Error updating window geometry: {e}")
    
    def close(self) -> None:
        """Closes the window."""
        self.root.quit()


class TaskbarAudioVisualizer:
    """Main application class coordinating all components."""
    
    def __init__(self):
        # Initialize configuration
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()
        
        # Initialize window
        self.window = TaskbarWindow(self.config)
        
        # Initialize audio capture
        self.audio_capture = AudioCapture()
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor()
        
        # Initialize visualizer
        self.visualizer = Visualizer(self.config, self.window.root)
        
        # Start audio capture
        self.audio_capture.start()
        
        # Start animation
        self.animation = animation.FuncAnimation(
            self.visualizer.fig,
            self._update_frame,
            interval=UPDATE_INTERVAL,
            blit=False,
            cache_frame_data=False
        )
        
        self._print_startup_info()
    
    def _update_frame(self, frame: int) -> None:
        """Updates visualization frame."""
        # Check for config updates
        new_config = self.config_manager.check_for_updates()
        if new_config:
            self._handle_config_update(new_config)
        
        # Get and process audio
        audio_data = self.audio_capture.get_audio()
        if audio_data is not None:
            processed_audio = self.audio_processor.process(audio_data, self.config)
            self.visualizer.update(processed_audio)
    
    def _handle_config_update(self, new_config: dict) -> None:
        """Handles configuration updates."""
        old_config = self.config
        self.config = new_config
        
        # Apply visual updates
        self.visualizer.apply_config_update(new_config)
        
        # Check if window geometry changed
        geometry_keys = ["window_width", "window_height", "window_x", "window_y"]
        if any(old_config.get(key) != new_config.get(key) for key in geometry_keys):
            self.window.update_geometry(new_config)
    
    def _print_startup_info(self) -> None:
        """Prints startup information."""
        print("=" * 60)
        print("✓ TASKBAR AUDIO VISUALIZER RUNNING!")
        print("=" * 60)
        print(f"Active config: '{self.config_manager.active_config_name}'")
        print(f"Edit '{self.config_manager.config_path}' to change configs")
        print("Change 'config-to-use' to switch between configs")
        print("Changes will be applied automatically every 2 seconds")
        print("\nWaveform options:")
        print("  - fill_under_curve: Fill area under waveform")
        print("  - fill_alpha: Transparency of fill (0-1)")
        print("  - line_style: solid, dashed, dotted, dashdot")
        print("  - symmetry_mode: Mirror the waveform")
        print("  - thick_on_bass: Line gets thicker with bass")
        print("  - rotation_angle: Rotate waveform (in samples)")
        print("  - y_offset: Shift waveform up/down")
        print("\nPress ESC to close")
        print("=" * 60 + "\n")
    
    def run(self) -> None:
        """Starts the application."""
        try:
            self.window.root.mainloop()
        except KeyboardInterrupt:
            print("\n\nShutting down...")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Cleans up resources."""
        print("\n✓ Shutting down cleanly...")
        self.audio_capture.stop()


def main():
    """Entry point for the application."""
    try:
        visualizer = TaskbarAudioVisualizer()
        visualizer.run()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()