
import os
import subprocess
import sys

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgs", type=str, default="./configs/deepgaitv2/DeepGaitV2_gait3d_custom.yaml")
    parser.add_argument("--phase", type=str, default="train")
    args, unknown = parser.parse_known_args()
    
    config_path = args.cfgs
    
    # Check if config exists
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return

    cmd = [
        sys.executable, "-m", "torch.distributed.launch",
        "--nproc_per_node=4",
        "opengait/main.py",
        "--cfgs", config_path,
        "--phase", args.phase
    ]
    
    # Forward any other arguments given to this script
    if unknown:
        cmd.extend(unknown)

    # Environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    print(f"Starting training with command: {' '.join(cmd)}")
    print(f"CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}")

    try:
        # Run the command and stream output
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end='')

        process.wait()

        if process.returncode != 0:
            print(f"Training failed with return code {process.returncode}")
        else:
            print("Training finished successfully.")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        process.terminate()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
