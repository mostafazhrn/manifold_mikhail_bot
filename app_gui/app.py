"""
CustomTkinter GUI for Manifold Mikhail bot
- File: gui/app.py
- Usage: python -m gui.app or python gui/app.py (run from repo root)

Features:
- Start / Stop bot (runs Trader.run_once repeatedly in background thread)
- Paper / Live toggle
- Mode selector (smart / super) -> calls smart_strategy_v3.set_mode
- Creator username, max_pages, max_markets overrides
- Delay (seconds) between runs
- Live log console capturing stdout prints from bot
- Basic stats (markets evaluated in session, trades simulated)

Notes:
- This script expects your package importable as `src` (run from repo root)
- Requires customtkinter installed in the active Python environment

"""

import threading
import queue
import sys
import time
import traceback
import subprocess

import os, sys
# --- Fix Python path so 'src' is always importable ---
ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import customtkinter as ctk
from tkinter import scrolledtext

# Import your bot components
from src.mikhail_bot.trader import Trader
from src.mikhail_bot.smart_strategy_v3 import set_mode
from src.mikhail_bot import config as cfg_module


class StdoutRedirector:
    def __init__(self, queue: queue.Queue):
        self._queue = queue
        self._orig = sys.stdout

    def write(self, s):
        # push strings to queue
        if s and not s.isspace():
            self._queue.put(str(s))

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass


class BotRunner(threading.Thread):
    def __init__(self, trader: Trader, delay: int, stop_event: threading.Event, log_queue: queue.Queue):
        super().__init__(daemon=True)
        self.trader = trader
        self.stop_event = stop_event
        self.log_queue = log_queue

    def run(self):
        try:
            self.log_queue.put("[RUNNER] Starting single run...")
            self.trader.run_once()
            self.log_queue.put("[RUNNER] run_once finished. Bot is idle.")
        except Exception as e:
            self.log_queue.put(f"[RUNNER] Exception: {e}\n{traceback.format_exc()}")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.title("MikhailTal Manifold Bot — GUI")
        self.geometry("1000x700")

        # Log queue and stdout redirector
        self.log_queue = queue.Queue()
        self.stdout_redirector = StdoutRedirector(self.log_queue)
        self.orig_stdout = sys.stdout

        # Runner controls
        self.runner_thread = None
        self.runner_stop_event = None

        # Stats
        self.session_markets = 0
        self.session_trades = 0

        self._build_ui()
        # start polling for logs
        self.after(100, self._poll_log_queue)

    def _build_ui(self):
        # Create a tabview so we can add Training Tools without changing the Bot UI logic
        tabview = ctk.CTkTabview(self, width=0)
        tabview.pack(side="left", expand=True, fill="both", padx=12, pady=12)

        # ---------------------
        # Tab: Bot (existing UI)
        # ---------------------
        tabview.add("Bot")
        bot_tab = tabview.tab("Bot")

        # Left panel: controls (moved into Bot tab)
        control_frame = ctk.CTkFrame(bot_tab, width=320)
        control_frame.pack(side="left", fill="y", padx=12, pady=12)

        ctk.CTkLabel(control_frame, text="Settings", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(4, 8))

        # Creator username
        ctk.CTkLabel(control_frame, text="Creator username:").pack(anchor="w", padx=8)
        self.entry_creator = ctk.CTkEntry(control_frame)
        self.entry_creator.insert(0, cfg_module.config.designated_username)
        self.entry_creator.pack(fill="x", padx=8, pady=(0, 8))

        # Max pages
        ctk.CTkLabel(control_frame, text="Max pages (override):").pack(anchor="w", padx=8)
        self.entry_max_pages = ctk.CTkEntry(control_frame)
        self.entry_max_pages.insert(0, "")
        self.entry_max_pages.pack(fill="x", padx=8, pady=(0, 8))

        # Max markets
        ctk.CTkLabel(control_frame, text="Max markets (override):").pack(anchor="w", padx=8)
        self.entry_max_markets = ctk.CTkEntry(control_frame)
        self.entry_max_markets.insert(0, "")
        self.entry_max_markets.pack(fill="x", padx=8, pady=(0, 8))

        # Delay (disabled – bot runs once)
        #ctk.CTkLabel(control_frame, text="Delay between runs (disabled):").pack(anchor="w", padx=8)
        #self.entry_delay = ctk.CTkEntry(control_frame, state="disabled")
        #self.entry_delay.insert(0, "N/A")
        #self.entry_delay.pack(fill="x", padx=8, pady=(0, 8))

        # Paper mode
        self.paper_var = ctk.BooleanVar(value=cfg_module.config.paper_mode)
        self.paper_switch = ctk.CTkCheckBox(control_frame, text="Paper mode (no real trades)", variable=self.paper_var)
        self.paper_switch.pack(padx=8, pady=(6, 6))

        # Strategy mode: smart / super
        ctk.CTkLabel(control_frame, text="Strategy mode:").pack(anchor="w", padx=8)
        self.mode_var = ctk.StringVar(value="smart")
        mode_frame = ctk.CTkFrame(control_frame)
        mode_frame.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkRadioButton(mode_frame, text="smart", variable=self.mode_var, value="smart").pack(side="left", padx=6)
        ctk.CTkRadioButton(mode_frame, text="super", variable=self.mode_var, value="super").pack(side="left", padx=6)

        # Start / Stop buttons
        btn_frame = ctk.CTkFrame(control_frame)
        btn_frame.pack(fill="x", padx=8, pady=(12, 8))
        self.btn_start = ctk.CTkButton(btn_frame, text="Start Bot", command=self.start_bot)
        self.btn_start.pack(side="left", expand=True, fill="x", padx=(0, 6))
        self.btn_stop = ctk.CTkButton(btn_frame, text="Stop Bot", command=self.stop_bot, fg_color="#a83232")
        self.btn_stop.pack(side="left", expand=True, fill="x", padx=(6, 0))

        # Stats
        stats_frame = ctk.CTkFrame(control_frame)
        stats_frame.pack(fill="x", padx=8, pady=(12, 0))
        ctk.CTkLabel(stats_frame, text="Session stats:").pack(anchor="w")
        self.lbl_markets = ctk.CTkLabel(stats_frame, text="Markets evaluated: 0")
        self.lbl_markets.pack(anchor="w")
        self.lbl_trades = ctk.CTkLabel(stats_frame, text="Trades simulated: 0")
        self.lbl_trades.pack(anchor="w")

        # Right panel: logs (in Bot tab)
        right_frame = ctk.CTkFrame(bot_tab)
        right_frame.pack(side="right", expand=True, fill="both", padx=12, pady=12)

        ctk.CTkLabel(right_frame, text="Bot Console", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(4, 6))

        # Use a scrolledtext for logs (tkinter.Text)
        self.txt_logs = scrolledtext.ScrolledText(right_frame, wrap="word", state="disabled", height=30)
        self.txt_logs.pack(expand=True, fill="both", padx=6, pady=6)

        # Bottom: quick commands
        bottom_frame = ctk.CTkFrame(right_frame)
        bottom_frame.pack(fill="x", padx=6, pady=(6, 0))
        self.btn_clear = ctk.CTkButton(bottom_frame, text="Clear Logs", command=self.clear_logs)
        self.btn_clear.pack(side="left")
        self.btn_set_mode = ctk.CTkButton(bottom_frame, text="Apply Mode", command=self.apply_mode)
        self.btn_set_mode.pack(side="left", padx=8)

        # ---------------------------
        # Tab: Training Tools
        # ---------------------------
        tabview.add("Training Tools")
        train_tab = tabview.tab("Training Tools")

        # Training controls left column
        train_left = ctk.CTkFrame(train_tab, width=320)
        train_left.pack(side="left", fill="y", padx=12, pady=12)

        self.btn_syscheck = ctk.CTkButton(
            control_frame,
            text="Run System Check",
            fg_color="#4A7AFF",
            command=self.run_system_check
        )
        self.btn_syscheck.pack(fill="x", padx=8, pady=(10, 10))


        ctk.CTkLabel(train_left, text="Training Tools", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(4, 8))

        # Fetch resolved markets button
        self.btn_fetch = ctk.CTkButton(
            train_left,
            text="Fetch Resolved Markets",
            command=self.start_fetch_resolved
        )
        self.btn_fetch.pack(fill="x", padx=8, pady=(6, 6))

        # Check bot stats button  ✅ NEW
        self.btn_stats = ctk.CTkButton(
            train_left,
            text="Check Bot Stats",
            fg_color="#2E8B57",
            command=self.run_simple_stats
        )
        self.btn_stats.pack(fill="x", padx=8, pady=(6, 6))

        # Max to fetch override
        ctk.CTkLabel(
            train_left,
            text="Max resolved to fetch (leave blank = .env)"
        ).pack(anchor="w", padx=8)

        self.entry_max_resolved = ctk.CTkEntry(train_left)
        self.entry_max_resolved.insert(0, "")
        self.entry_max_resolved.pack(fill="x", padx=8, pady=(0, 8))


        # Train model button
        self.btn_train = ctk.CTkButton(train_left, text="Train Model", command=self.start_train_model)
        self.btn_train.pack(fill="x", padx=8, pady=(6, 6))

        # Train options
        ctk.CTkLabel(train_left, text="Train test split (float 0..1):").pack(anchor="w", padx=8)
        self.entry_test_frac = ctk.CTkEntry(train_left)
        self.entry_test_frac.insert(0, "0.15")
        self.entry_test_frac.pack(fill="x", padx=8, pady=(0, 8))

        # Right side: training logs
        train_right = ctk.CTkFrame(train_tab)
        train_right.pack(side="right", expand=True, fill="both", padx=12, pady=12)

        ctk.CTkLabel(train_right, text="Training Console", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(4, 6))
        self.txt_train_logs = scrolledtext.ScrolledText(train_right, wrap="word", state="disabled", height=30)
        self.txt_train_logs.pack(expand=True, fill="both", padx=6, pady=6)

        # Redirector placeholder: we'll redirect only when training/fetching runs
        self.train_log_queue = queue.Queue()
        self.after(100, self._poll_train_log_queue)

    def clear_logs(self):
        self.txt_logs.config(state="normal")
        self.txt_logs.delete("1.0", "end")
        self.txt_logs.config(state="disabled")

    def _poll_log_queue(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                self._append_log(item)
        except queue.Empty:
            pass
        finally:
            # update stats labels occasionally (they are managed by the Trader prints too)
            self.after(100, self._poll_log_queue)

    def _append_log(self, text: str):
        # Ensure newline
        if not text.endswith("\n"):
            text = text + "\n"
        self.txt_logs.config(state="normal")
        self.txt_logs.insert("end", text)
        self.txt_logs.see("end")
        self.txt_logs.config(state="disabled")

    def apply_mode(self):
        mode = self.mode_var.get()
        try:
            set_mode(mode)
            self.log_queue.put(f"[GUI] Strategy mode set to '{mode}'")
        except Exception as e:
            self.log_queue.put(f"[GUI] Failed to set mode: {e}")

    # -------------------------
    # Training & Fetch runners
    # -------------------------
    def _poll_train_log_queue(self):
        try:
            while True:
                item = self.train_log_queue.get_nowait()
                # append to train log textbox
                if not str(item).endswith("\n"):
                    item = str(item) + "\n"
                self.txt_train_logs.config(state="normal")
                self.txt_train_logs.insert("end", str(item))
                self.txt_train_logs.see("end")
                self.txt_train_logs.config(state="disabled")
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_train_log_queue)

    def _run_fetch_resolved(self, max_resolved_override=None):
        # Import dynamically so GUI starts even if modules missing
        try:
            try:
                from src import fetch_resolved_markets as fetch_mod
                fetch_fn = fetch_mod.fetch_all_resolved
            except Exception:
                try:
                    # fallback to module at repo root
                    import fetch_resolved_markets as fetch_mod
                    fetch_fn = fetch_mod.fetch_all_resolved
                except Exception:
                    self.train_log_queue.put("[TRAIN] Could not import fetch_all_resolved. Ensure script exists.")
                    return

            # Optionally set env override
            if max_resolved_override is not None and str(max_resolved_override).strip():
                os.environ["MAX_RESOLVED_MARKETS"] = str(max_resolved_override)
                self.train_log_queue.put(f"[TRAIN] Overriding MAX_RESOLVED_MARKETS = {max_resolved_override}")

            # Redirect prints to train_log_queue
            orig_stdout = sys.stdout
            class QueueWriter:
                def __init__(self, q):
                    self.q = q
                def write(self, s):
                    if s and not str(s).isspace():
                        self.q.put(str(s))
                def flush(self):
                    pass
            sys.stdout = QueueWriter(self.train_log_queue)

            try:
                fetch_fn()
            except Exception as e:
                self.train_log_queue.put(f"[TRAIN] fetch_all_resolved raised: {e}\n")
            finally:
                sys.stdout = orig_stdout

        except Exception as e:
            self.train_log_queue.put(f"[TRAIN] Unexpected error in fetch runner: {e}\n")

    def start_fetch_resolved(self):
        # prevent double start
        if hasattr(self, "fetch_thread") and self.fetch_thread and self.fetch_thread.is_alive():
            self.train_log_queue.put("[TRAIN] Fetch already running")
            return
        maxr = self.entry_max_resolved.get().strip()
        self.fetch_thread = threading.Thread(target=self._run_fetch_resolved, args=(maxr,), daemon=True)
        self.fetch_thread.start()
        self.train_log_queue.put("[TRAIN] Fetch started")
    
    def run_system_check(self):
        """
        Runs scripts/system_check.py and prints its output to the bot console.
        """
        script_path = os.path.join(ROOT, "scripts", "check_system.py")

        if not os.path.exists(script_path):
            self.log_queue.put(f"[SystemCheck] ERROR: {script_path} not found.")
            return

        self.log_queue.put("[SystemCheck] Running system check...\n")

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True
            )

            if result.stdout:
                for line in result.stdout.splitlines():
                    self.log_queue.put("[SystemCheck] " + line)

            if result.stderr:
                for line in result.stderr.splitlines():
                    self.log_queue.put("[SystemCheck-ERR] " + line)

        except Exception as e:
            self.log_queue.put(f"[SystemCheck] Exception: {e}")
    ####
    def run_simple_stats(self):
        """
        Runs simple_stats.py and prints its output to the training console.
        """
        script_path = os.path.join(
            ROOT,
            "src",
            "mikhail_bot",
            "simple_stats.py"
        )

        if not os.path.exists(script_path):
            self.train_log_queue.put("[STATS] ERROR: simple_stats.py not found.")
            return

        self.train_log_queue.put("[STATS] Running bot performance stats...\n")

        def _runner():
            try:
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True
                )

                if result.stdout:
                    for line in result.stdout.splitlines():
                        self.train_log_queue.put("[STATS] " + line)

                if result.stderr:
                    for line in result.stderr.splitlines():
                        self.train_log_queue.put("[STATS-ERR] " + line)

            except Exception as e:
                self.train_log_queue.put(f"[STATS] Exception: {e}")

        threading.Thread(target=_runner, daemon=True).start()



    def _run_train_model(self, test_frac_override=None):
        try:
            try:
                from src import train_model as train_mod
                train_fn = train_mod.train_model
            except Exception:
                try:
                    import train_model as train_mod
                    train_fn = train_mod.train_model
                except Exception:
                    self.train_log_queue.put("[TRAIN] Could not import train_model.train_model. Ensure script exists at src/train_model.py or train_model.py")
                    return

            # Optionally set test fraction override via env (train_model reads constant; we set env var if desired)
            if test_frac_override is not None and str(test_frac_override).strip():
                os.environ["TEST_FRACTION"] = str(test_frac_override)
                self.train_log_queue.put(f"[TRAIN] Overriding TEST_FRACTION = {test_frac_override}")

            # Redirect prints
            orig_stdout = sys.stdout
            class QueueWriter2:
                def __init__(self, q):
                    self.q = q
                def write(self, s):
                    if s and not str(s).isspace():
                        self.q.put(str(s))
                def flush(self):
                    pass
            sys.stdout = QueueWriter2(self.train_log_queue)

            try:
                train_fn()
            except Exception as e:
                self.train_log_queue.put(f"[TRAIN] train_model raised: {e}\n")
            finally:
                sys.stdout = orig_stdout

        except Exception as e:
            self.train_log_queue.put(f"[TRAIN] Unexpected error in train runner: {e}\n")

    def start_train_model(self):
        if hasattr(self, "train_thread") and self.train_thread and self.train_thread.is_alive():
            self.train_log_queue.put("[TRAIN] Training already running")
            return
        tf = self.entry_test_frac.get().strip()
        self.train_thread = threading.Thread(target=self._run_train_model, args=(tf,), daemon=True)
        self.train_thread.start()
        self.train_log_queue.put("[TRAIN] Training started")

    def start_bot(self):
        # prevent double start
        if self.runner_thread and self.runner_thread.is_alive():
            self.log_queue.put("[GUI] Bot already running")
            return

        # read inputs
        creator = self.entry_creator.get().strip() or None
        max_pages = None
        max_markets = None
        try:
            mp = self.entry_max_pages.get().strip()
            if mp:
                max_pages = int(mp)
        except Exception:
            self.log_queue.put("[GUI] Invalid max_pages; ignoring")
        try:
            mm = self.entry_max_markets.get().strip()
            if mm:
                max_markets = int(mm)
        except Exception:
            self.log_queue.put("[GUI] Invalid max_markets; ignoring")

        #delay = 30
        #try:
        #    delay = int(self.entry_delay.get().strip())
        #except Exception:
        #    self.log_queue.put("[GUI] Invalid delay; using 30s")

        paper = bool(self.paper_var.get())
        mode = self.mode_var.get()

        # 🔒 HARD SOURCE OF TRUTH (used by smart_strategy_v4)
        os.environ["STRATEGY_MODE"] = mode

        self.log_queue.put(f"[GUI] STRATEGY_MODE={mode}")

        # instantiate Trader with overrides
        trader = Trader(paper=paper, strategy=None, max_pages=max_pages, max_markets=max_markets)

        # redirect stdout
        sys.stdout = self.stdout_redirector

        self.runner_stop_event = threading.Event()
        self.runner_thread = BotRunner(trader=trader, delay=0, stop_event=self.runner_stop_event, log_queue=self.log_queue)
        self.runner_thread.start()
        self.log_queue.put("[GUI] Bot started")

    def stop_bot(self):
        if not self.runner_thread:
            self.log_queue.put("[GUI] Bot not running")
            return
        if not self.runner_thread.is_alive():
            self.log_queue.put("[GUI] Bot thread not alive")
            return

        # signal stop
        self.runner_stop_event.set()
        # restore stdout
        sys.stdout = self.orig_stdout
        self.log_queue.put("[GUI] Stop signal sent. Waiting for thread to join...")
        # wait briefly
        self.runner_thread.join(timeout=5)
        if self.runner_thread.is_alive():
            self.log_queue.put("[GUI] Thread did not exit within timeout; it will be killed on process exit")
        else:
            self.log_queue.put("[GUI] Bot stopped")
        self.runner_thread = None
        self.runner_stop_event = None


if __name__ == "__main__":
    app = App()
    app.mainloop()
