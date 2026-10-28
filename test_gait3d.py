
import argparse
import os
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Test DeepGaitV2 on Gait3D")
    parser.add_argument("--iter", type=int, default=60000, help="Iteration number to restore (default: 60000)")
    args = parser.parse_args()

    config_path = "./configs/deepgaitv2/DeepGaitV2_gait3d_custom.yaml"
    
    # Check if config exists
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return

    # Command to run distributed evaluation
    cmd = [
        sys.executable, "-m", "torch.distributed.launch",
        "--nproc_per_node=4",
        "opengait/main.py",
        "--cfgs", config_path,
        "--phase", "test",
        "--iter", str(args.iter)
    ]

    # Environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    print(f"Starting evaluation with command: {' '.join(cmd)}")
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
            print(f"Evaluation failed with return code {process.returncode}")
        else:
            print("Evaluation finished successfully.")

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        process.terminate()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
