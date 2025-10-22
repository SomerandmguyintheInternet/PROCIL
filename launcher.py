"""
Eggscript Live File Web Server

This script launches a simple HTTP server that serves the processed content
of a specified .egg or .eggless file on port 9696.

Features:
1. Imports and uses the run_egg_file function from eggscript.py.
2. Implements live reloading: Checks the file modification time on every request (do_GET).
   If the file is updated on disk, it re-runs the eggscript execution automatically.
3. Graceful handling of file not found and execution errors.
"""
import sys
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# IMPORTANT: Ensure eggscript.py is in the same directory or on the Python path
try:
    from eggscript import run_egg_file
except ImportError:
    print("FATAL: Cannot find 'eggscript.py'. Please ensure it is in the same directory.")
    sys.exit(1)

# --- Configuration ---
HOST_NAME = "localhost"
PORT_NUMBER = 9696
SERVED_FILE_PATH = None

# --- Cache & State ---
# Store the last output and modification time to check for changes
FILE_CACHE = {
    "content": None,
    "mtime": 0.0,
    "path": None
}

class EggscriptHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for the Eggscript Server."""

    def log_message(self, format, *args):
        """Suppress logging GET requests to keep the terminal clean, only log errors."""
        if self.command != 'GET':
            super().log_message(format, *args)
    
    def do_GET(self):
        """
        Handles GET requests, implementing the core live-reloading logic.
        """
        global SERVED_FILE_PATH

        if self.path != '/':
            # Ignore requests for favicon.ico or other assets
            self.send_response(404)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(f"<html><head><title>404 Not Found</title></head><body><h1>404 Not Found</h1><p>Server only serves the root path (/).</p></body></html>".encode('utf-8'))
            return

        print(f"[SERVER] Request received. Checking file: {SERVED_FILE_PATH}")

        try:
            current_mtime = os.path.getmtime(SERVED_FILE_PATH)
        except FileNotFoundError:
            # File is missing
            self.send_response(404)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            error_msg = f"File Not Found: {SERVED_FILE_PATH}."
            print(f"[ERROR] {error_msg}")
            self.wfile.write(f"<html><body><h1>404 Error</h1><p>{error_msg}</p></body></html>".encode('utf-8'))
            return
        
        # --- Live Reloading Check ---
        if current_mtime > FILE_CACHE["mtime"]:
            # File has been modified, re-run the script
            print(f"[RELOAD] File changed ({time.strftime('%H:%M:%S', time.localtime(current_mtime))}). Re-executing script...")
            
            try:
                # Execute the eggscript file and get the generated HTML output
                generated_content = run_egg_file(SERVED_FILE_PATH)
                
                # Update cache
                FILE_CACHE["content"] = generated_content
                FILE_CACHE["mtime"] = current_mtime
                print("[RELOAD] Execution successful. New content cached.")
                
            except Exception as e:
                # Catch any errors during script execution (beyond what eggscript handles)
                print(f"[RUNTIME FATAL] Failed to execute eggscript: {e}")
                
                # Serve an error page based on the previous cache or a default error message
                error_html = f"<html><body><h1>Eggscript Execution Error</h1><p>A fatal error occurred in the execution engine:</p><pre>{e}</pre></body></html>"
                
                # Send 500 Internal Server Error
                self.send_response(500)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(error_html.encode('utf-8'))
                return

        # --- Serve Cached Content ---
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        
        # Ensure we always have content, even if the cache update failed or the file was just loaded
        content_to_serve = FILE_CACHE["content"] if FILE_CACHE["content"] is not None else "<html><body><h1>Loading...</h1><p>Attempting to run script...</p></body></html>"
        self.wfile.write(content_to_serve.encode('utf-8'))


def run_server(file_path=None):
    """
    Initializes and runs the HTTP server.
    If file_path is None (when called without an argument), prompts the user interactively.
    """
    global SERVED_FILE_PATH
    
    # --- INTERACTIVE PROMPT LOGIC ---
    if file_path is None:
        while True:
            try:
                # Prompt the user for the file path
                target_file = input("Enter path to .egg or .eggless file (or 'quit'): ").strip()
                if target_file.lower() == 'quit':
                    sys.exit(0)
                    
                target_path = Path(target_file)
                
                # Validate the path
                if not target_path.exists():
                    print(f"Error: File '{target_file}' not found. Please try again.")
                    continue
                
                # File is found, break the loop and use this path
                file_path = str(target_path)
                break
            except KeyboardInterrupt:
                # Handle Ctrl+C during input
                print("\nServer launch cancelled.")
                return # Exit the function if user interrupts
    
    SERVED_FILE_PATH = file_path
    
    # Run the file once to populate the initial cache
    try:
        FILE_CACHE["content"] = run_egg_file(SERVED_FILE_PATH)
        FILE_CACHE["mtime"] = os.path.getmtime(SERVED_FILE_PATH)
    except FileNotFoundError:
        print(f"[FATAL] File not found: {file_path}. Server cannot start.")
        return
    except Exception as e:
        print(f"[FATAL] Initial script execution failed: {e}. Server cannot start.")
        return

    server_address = (HOST_NAME, PORT_NUMBER)
    httpd = HTTPServer(server_address, EggscriptHandler)
    
    print("-" * 50)
    print(f"[{Path(SERVED_FILE_PATH).name}] Eggscript Live Server running!")
    print(f"→ Serving: {SERVED_FILE_PATH}")
    print(f"→ URL: http://{HOST_NAME}:{PORT_NUMBER}")
    print(f"→ Live Reload: Enabled (Refresh browser to see changes)")
    print("→ Press Ctrl+C to stop the server.")
    print("-" * 50)
    
    try:
        # This function loops forever until Ctrl+C is pressed
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

    httpd.server_close()
    print("\n[SERVER] Server stopped successfully.")

if __name__ == "__main__":
    
    # Determine the file path: argument provided or None
    target_file_path = sys.argv[1] if len(sys.argv) >= 2 else None
    
    # The run_server function now handles both the argument and the interactive prompt
    run_server(target_file_path)
