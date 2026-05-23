import subprocess
import time
import sys
import os
from pyngrok import ngrok

def main():
    print("=======================================================")
    print("🚀 Exposing Dashboard to the Public Web via Ngrok")
    print("=======================================================")
    
    # Start Streamlit in a subprocess
    print("Starting Streamlit server on local port 8501...")
    streamlit_path = os.path.join(".venv", "Scripts", "streamlit.exe")
    if not os.path.exists(streamlit_path):
        streamlit_path = "streamlit"  # Fallback to global if venv not found

    try:
        streamlit_proc = subprocess.Popen(
            [streamlit_path, "run", "dashboard.py", "--server.port", "8501", "--server.headless", "true", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f"❌ Failed to start Streamlit: {e}")
        print("Please ensure your virtual environment is activated and streamlit is installed.")
        sys.exit(1)

    # Give Streamlit a moment to start
    time.sleep(3)

    # Start ngrok
    print("Starting ngrok tunnel...")
    try:
        # Create tunnel on port 8501
        public_url = ngrok.connect(8501)
        print("\n=======================================================")
        print(f"🎯 SUCCESS! Your dashboard is now live at:\n   {public_url}")
        print("=======================================================\n")
        print("Press Ctrl+C to stop sharing and shut down the dashboard.")
        
        # Keep the script running to maintain the tunnel
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down servers...")
    except Exception as e:
        print(f"\n❌ Ngrok Error: {e}")
        print("Make sure you don't have another ngrok instance running.")
    finally:
        # Clean up processes
        try:
            ngrok.kill()
        except Exception:
            pass
        try:
            streamlit_proc.terminate()
        except Exception:
            pass
        print("Done. Goodbye!")

if __name__ == "__main__":
    main()
