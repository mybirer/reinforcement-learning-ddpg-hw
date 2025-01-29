from unityagents import UnityEnvironment
import time
import os
import logging
import glob  # For searching files

# Clean environment variables
if 'HTTP_PROXY' in os.environ:
    del os.environ['HTTP_PROXY']
if 'HTTPS_PROXY' in os.environ:
    del os.environ['HTTPS_PROXY']

# Enable logging
logging.basicConfig(level=logging.DEBUG)

# List existing log files
print("\nChecking for existing log files...")
log_files = glob.glob("unity_logs/*.log")
for log_file in log_files:
    print(f"\nFound log file: {log_file}")
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            print(f"Content of {log_file}:")
            print(content)
    except Exception as e:
        print(f"Error reading log file: {e}")

print("\nStarting basic Unity test...")

try:
    # Fix file path
    exe_path = os.path.abspath("Reacher_Windows_x86_64/Reacher.exe")
    print(f"Executable path: {exe_path}")
    
    if not os.path.exists(exe_path):
        raise FileNotFoundError(f"Unity executable not found at: {exe_path}")
    
    print("Attempting to connect to Unity environment...")
    env = UnityEnvironment(
        file_name=exe_path,
        no_graphics=True,  # Turn off graphics
        seed=0  # Set random seed
    )
    print("Successfully connected!")
    
    # Try to get basic info
    print("Getting environment info...")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    print(f"Found brain name: {brain_name}")
    
    # Try to reset
    print("Attempting environment reset...")
    env_info = env.reset(train_mode=True)[brain_name]
    print("Reset successful!")
    
    # Close properly
    env.close()
    print("Environment closed successfully!")
    
except Exception as e:
    print(f"\nError occurred:")
    print(f"Type: {type(e).__name__}")
    print(f"Message: {str(e)}")
    
    # Check environment variables
    print("\nEnvironment variables:")
    print(f"HTTP_PROXY: {os.environ.get('HTTP_PROXY', 'Not set')}")
    print(f"HTTPS_PROXY: {os.environ.get('HTTPS_PROXY', 'Not set')}")
    
    # Check Unity executable
    print("\nUnity executable check:")
    if os.path.exists(exe_path):
        print(f"Executable exists at: {exe_path}")
        print("Directory contents:")
        dir_path = os.path.dirname(exe_path)
        for item in os.listdir(dir_path):
            print(f"  {item}")
    else:
        print(f"Executable not found at: {exe_path}")
    
    # Check Unity process
    import subprocess
    result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq Reacher.exe'], capture_output=True, text=True)
    print("\nUnity process check:")
    print(result.stdout)
    
    # Check for new log files
    print("\nChecking for new log files...")
    new_log_files = glob.glob("unity_logs/*.log")
    for log_file in new_log_files:
        if log_file not in log_files:  # Only show new files
            print(f"\nNew log file: {log_file}")
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    print(f"Content of {log_file}:")
                    print(content)
            except Exception as e:
                print(f"Error reading log file: {e}")
    
    raise
finally:
    print("\nTest completed.")