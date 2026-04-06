import os
import time
import sys

DEADLINE_FILE = os.path.join(os.path.dirname(__file__), ".deadline")


def set_deadline(minutes: int):
    deadline = time.time() + minutes * 60
    with open(DEADLINE_FILE, "w") as f:
        f.write(f"{deadline}\n{minutes}\n")
    print(f"Deadline set: {minutes} minutes from now")


def check_deadline() -> bool:
    if not os.path.exists(DEADLINE_FILE):
        return True
    with open(DEADLINE_FILE) as f:
        lines = f.readlines()
    deadline = float(lines[0].strip())
    remaining = deadline - time.time()
    if remaining <= 0:
        print(f"DEADLINE REACHED. Session over.")
        return False
    print(f"Time remaining: {remaining / 60:.1f} minutes")
    return True


def minutes_remaining() -> float:
    if not os.path.exists(DEADLINE_FILE):
        return float("inf")
    with open(DEADLINE_FILE) as f:
        deadline = float(f.readline().strip())
    return max(0, (deadline - time.time()) / 60)


def clear_deadline():
    if os.path.exists(DEADLINE_FILE):
        os.remove(DEADLINE_FILE)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python deadline.py set <minutes> | check | remaining | clear")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "set":
        set_deadline(int(sys.argv[2]))
    elif cmd == "check":
        sys.exit(0 if check_deadline() else 1)
    elif cmd == "remaining":
        m = minutes_remaining()
        print(f"{m:.1f} minutes remaining" if m < float("inf") else "No deadline set")
    elif cmd == "clear":
        clear_deadline()
        print("Deadline cleared")
