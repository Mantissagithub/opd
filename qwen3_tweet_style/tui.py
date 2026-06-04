import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_PATH = SCRIPT_DIR / "training.log"

console = Console()

MENU = {
    "1": ("Train Qwen3-4B-Base", ["train_qwen3_4b.py", "--push"]),
    "2": ("Train Qwen3-32B-Base", ["train_qwen3_32b.py", "--push"]),
    "3": ("Run eval benchmark", ["eval_benchmark.py"]),
}

def collect_secrets():
    console.print(Panel.fit("Qwen3 tweet-style trainer", style="bold cyan"))
    env = os.environ.copy()
    env["HF_USERNAME"] = Prompt.ask("HF username")
    env["HF_TOKEN"] = Prompt.ask("HF token", password=True)
    env["OPENROUTER_API_KEY"] = Prompt.ask("OpenRouter API key", password=True)
    return env

def run_script(script_args, env):
    cmd = [sys.executable, str(SCRIPT_DIR / script_args[0]), *script_args[1:]]
    console.rule(f"[bold green]{script_args[0]}")
    with open(LOG_PATH, "a") as log:
        log.write(f"\n===== {datetime.now().isoformat(timespec='seconds')} :: {' '.join(script_args)} =====\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, cwd=SCRIPT_DIR, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        # tee the child's output to the screen and training.log line by line
        for line in proc.stdout:
            console.out(line.rstrip())
            log.write(line)
            log.flush()
        proc.wait()
    status = "[green]done[/]" if proc.returncode == 0 else f"[red]exit {proc.returncode}[/]"
    console.print(f"{script_args[0]} {status} — logged to {LOG_PATH}")

def main():
    env = collect_secrets()
    while True:
        console.print()
        for key, (label, _) in MENU.items():
            console.print(f"  [bold]{key}[/]. {label}")
        console.print("  [bold]q[/]. Quit")
        choice = Prompt.ask("Select", choices=[*MENU.keys(), "q"], default="q")
        if choice == "q":
            break
        run_script(MENU[choice][1], env)

if __name__ == "__main__":
    main()
