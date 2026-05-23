import subprocess
import time
import sys
import urllib.request
import os

def main():
    print("=======================================================")
    print("🌐 Exposing Dashboard to the Public Web")
    print("=======================================================")
    
    # 1. Fetch public IP (used as localtunnel password)
    print("Fetching your public IP (this is your tunnel password)...")
    try:
        public_ip = urllib.request.urlopen('https://ipv4.icanhazip.com', timeout=5).read().decode('utf8').strip()
    except Exception:
        try:
            public_ip = urllib.request.urlopen('https://api.ipify.org', timeout=5).read().decode('utf8').strip()
        except Exception:
            public_ip = "Could not fetch IP automatically. Please find your public IP using a site like whatismyip.com"

    print("\n=======================================================")
    print(f"🔑 YOUR TUNNEL PASSWORD (PUBLIC IP): {public_ip}")
    print("=======================================================\n")
    
    # 2. Start Streamlit in a subprocess
    print("Starting Streamlit server on local port 8501...")
    streamlit_path = os.path.join(".venv", "Scripts", "streamlit.exe")
    if not os.path.exists(streamlit_path):
        streamlit_path = "streamlit"  # Fallback to global if venv not found

    try:
        streamlit_proc = subprocess.Popen(
            [streamlit_path, "run", "dashboard.py", "--server.port", "8501", "--server.address", "127.0.0.1", "--server.headless", "true", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f"❌ Failed to start Streamlit: {e}")
        print("Please ensure your virtual environment is activated and streamlit is installed.")
        sys.exit(1)

    # Give Streamlit a moment to start
    time.sleep(3)

    # 3. Start localtunnel
    print("Starting localtunnel via npx...")
    print("Please wait for your public URL to appear below...\n")
    try:
        # Use shell=True since npx is a batch/cmd script on Windows
        localtunnel_proc = subprocess.Popen(
            ["npx", "localtunnel", "--port", "8501", "--local-host", "127.0.0.1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=True
        )
    except Exception as e:
        print(f"❌ Failed to start localtunnel: {e}")
        streamlit_proc.terminate()
        sys.exit(1)

    # Read and print the localtunnel URL (non-blocking simulation)
    # npx localtunnel prints the URL as its first line of output
    try:
        for line in iter(localtunnel_proc.stdout.readline, ''):
            clean_line = line.strip()
            if clean_line:
                print(clean_line)
                if "url is" in clean_line.lower() or "loca.lt" in clean_line:
                    print("\n🎯 Click the URL above, and enter your public IP (printed above) as the password to access the app.")
                    print("Press Ctrl+C to terminate sharing and shut down the dashboard.")
            if localtunnel_proc.poll() is not None:
                break
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down servers...")
        try:
            streamlit_proc.terminate()
        except Exception:
            pass
        try:
            localtunnel_proc.terminate()
        except Exception:
            pass
        print("Done. Goodbye!")

if __name__ == "__main__":
    main()
