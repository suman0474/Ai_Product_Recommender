"""
Auto Log Watcher
================
Automatically captures terminal/log data and stores it in TL1.md - TL5.md+ files.

Features:
- Stores output word-by-word (or line-by-line) to markdown files.
- Automatically handles file rotation (rollover) when a file reaches 5000 lines.
- Can wrap a command to capture its output in real-time.
- Can watch an existing log file for changes.

Usage:
  1. Wrap a command (Recommended):
     python auto_log_watcher.py --wrap "python main.py"
  
  2. Watch a log file:
     python auto_log_watcher.py --watch app.log
     
  3. Help:
     python auto_log_watcher.py --help
"""

import os
import sys
import time
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, List

class AutoLogWatcher:
    """Automatically watches and stores log data into TL*.md files."""
    
    MAX_LINES_PER_FILE = 5000
    FILE_PREFIX = "TL"
    FILE_EXTENSION = ".md"
    DEFAULT_LOG_FILE = "app_output.log"
    
    def __init__(self, base_dir: Optional[Path] = None):
        # Determine storage directory (parent of 'backend' or current)
        script_dir = Path(__file__).parent.resolve()
        if base_dir:
            self.base_dir = Path(base_dir)
        elif script_dir.name == 'backend':
            self.base_dir = script_dir.parent
        else:
            self.base_dir = script_dir
            
        self.log_file_path = script_dir / self.DEFAULT_LOG_FILE
        self.current_file_index = 1
        self.running = True
        self.last_position = 0
        
        # Ensure initial state
        self._find_active_file()
        
        print(f"📁 Log storage directory: {self.base_dir}")
        print(f"📝 Active log file: {self._get_file_path(self.current_file_index).name}")
        print("=" * 50)
    
    def _get_file_path(self, index: int) -> Path:
        return self.base_dir / f"{self.FILE_PREFIX}{index}{self.FILE_EXTENSION}"
    
    def _create_file(self, file_path: Path, file_num: int):
        """Create a new TL file with header."""
        header = f"""# Terminal Log {file_num}

> Auto-generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> Max lines: {self.MAX_LINES_PER_FILE}

---

"""
        try:
            file_path.write_text(header, encoding='utf-8')
            print(f"✨ Created new log file: {file_path.name}")
        except Exception as e:
            print(f"⚠️ Failed to create file {file_path}: {e}")
    
    def _get_line_count(self, file_path: Path) -> int:
        """Count lines in a file safely."""
        try:
            if not file_path.exists():
                return 0
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return sum(1 for _ in f)
        except:
            return 0
    
    def _find_active_file(self):
        """Find the first file that has space, creating new ones if needed."""
        while True:
            file_path = self._get_file_path(self.current_file_index)
            
            # If file doesn't exist, create it
            if not file_path.exists():
                self._create_file(file_path, self.current_file_index)
                return

            # If file is full, move to next
            if self._get_line_count(file_path) >= self.MAX_LINES_PER_FILE:
                self.current_file_index += 1
                continue
            
            # File exists and has space
            return
    
    def _append_lines(self, lines: List[str]):
        """Append lines to the current TL file, handling rotation."""
        if not lines:
            return
            
        # Filter out empty lines if desired, or keep structure
        # User requested "store each word" or "terminal data". 
        # Keeping lines structure is usually best for readability.
        
        for line in lines:
            file_path = self._get_file_path(self.current_file_index)
            
            # Check capacity
            if self._get_line_count(file_path) >= self.MAX_LINES_PER_FILE:
                self.current_file_index += 1
                file_path = self._get_file_path(self.current_file_index)
                self._create_file(file_path, self.current_file_index)
            
            # Append
            try:
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(line + '\n')
            except Exception as e:
                print(f"⚠️ Error writing to {file_path.name}: {e}")

    def watch_file(self, target_file: str, interval: float = 1.0):
        """Watch a specific log file for new content."""
        target_path = Path(target_file)
        print(f"👀 Watching: {target_path} (Ctrl+C to stop)")
        
        # Ensure target exists
        if not target_path.exists():
            print(f"Creating empty target file: {target_path}")
            target_path.touch()
            
        while self.running:
            try:
                with open(target_path, 'r', encoding='utf-8', errors='replace') as f:
                    f.seek(self.last_position)
                    new_content = f.read()
                    self.last_position = f.tell()
                    
                    if new_content:
                        lines = new_content.splitlines()
                        self._append_lines(lines)
                        sys.stdout.write(f"\r📝 Captured {len(lines)} lines...")
                        sys.stdout.flush()
                
                time.sleep(interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n⚠️ Error: {e}")
                time.sleep(interval)
        print("\nStopped.")

    def wrap_command(self, command: str):
        """Run a command and capture its output in real-time."""
        print(f"🚀 Running wrapped command: {command}")
        print("📝 Output will be mirrored to console and stored in TL files.")
        
        # Log start
        self._append_lines([f"\n## [{datetime.now()}] Command Start: {command}\n"])
        
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1, # Line buffered
            encoding='utf-8',
            errors='replace'
        )
        
        try:
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    # Echo to console
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    # Store
                    self._append_lines([line.rstrip()])
        except KeyboardInterrupt:
            print("\n🛑 Execution interrupted.")
            process.terminate()
        
        print(f"\n✅ Command finished with return code: {process.returncode}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Auto Log Watcher")
    parser.add_argument('--wrap', '-w', type=str, help='Command to run and capture')
    parser.add_argument('--watch', '-f', type=str, help='File to watch (tail)')
    parser.add_argument('--interval', '-i', type=float, default=1.0, help='Watch interval')
    
    args = parser.parse_args()
    
    watcher = AutoLogWatcher()
    
    if args.wrap:
        watcher.wrap_command(args.wrap)
    elif args.watch:
        watcher.watch_file(args.watch, args.interval)
    else:
        # Default behavior: Print help
        parser.print_help()
        print("\nExample:")
        print('  python auto_log_watcher.py --wrap "python main.py"')

if __name__ == "__main__":
    main()
