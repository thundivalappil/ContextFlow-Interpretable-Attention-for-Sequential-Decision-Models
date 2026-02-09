"""One-command entry point.

Example:
  python run.py train
  python run.py infer --ckpt runs/contextflow_ckpt.pt
"""
import sys
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py [train|infer] <args...>")
        raise SystemExit(2)

    cmd = sys.argv[1]
    rest = sys.argv[2:]

    if cmd == "train":
        subprocess.check_call([sys.executable, "train.py", *rest])
    elif cmd == "infer":
        subprocess.check_call([sys.executable, "infer.py", *rest])
    else:
        print("Unknown command:", cmd)
        raise SystemExit(2)

if __name__ == "__main__":
    main()
