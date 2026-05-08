#!/usr/bin/env python3
"""
Tegrastats Monitor
- Waits for SIGUSR1 to start recording
- Stops and prints results on SIGUSR2
- Prints live RAM/GPU usage
- Tracks peak and average for RAM and GPU
- Uses 300ms interval
- Writes PID to /tmp/tegrastats_monitor.pid for signaling
"""

import subprocess
import re
import signal
import os

PID_FILE = "/tmp/tegrastats_monitor.pid"


class TegrastatsMonitor:
    def __init__(self):
        # RAM tracking
        self.peak_ram_mb = 0
        self.total_ram_mb = 0
        self.ram_sum = 0
        self.ram_samples = 0

        # GPU tracking
        self.peak_gpu_percent = 0
        self.gpu_sum = 0
        self.gpu_samples = 0

        self.running = True
        self.recording = False

    def parse_ram(self, line):
        match = re.search(r'RAM (\d+)/(\d+)MB', line)
        if match:
            used_mb = int(match.group(1))
            total_mb = int(match.group(2))

            if used_mb > self.peak_ram_mb:
                self.peak_ram_mb = used_mb
                self.total_ram_mb = total_mb

            # update average tracking
            self.ram_sum += used_mb
            self.ram_samples += 1

            return used_mb, total_mb
        return None, None

    def parse_gpu(self, line):
        match = re.search(r'GR3D_FREQ (\d+)%', line)
        if match:
            gpu_percent = int(match.group(1))

            if gpu_percent > self.peak_gpu_percent:
                self.peak_gpu_percent = gpu_percent

            self.gpu_sum += gpu_percent
            self.gpu_samples += 1
            return gpu_percent
        return None

    def start_recording(self, sig, frame):
        print("\n[Monitor] START signal received. Recording...")
        self.recording = True

    def stop_recording(self, sig, frame):
        print("\n[Monitor] STOP signal received. Stopping...")
        self.running = False

    def monitor(self):
        # Write PID for other scripts to signal
        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))

        signal.signal(signal.SIGUSR1, self.start_recording)
        signal.signal(signal.SIGUSR2, self.stop_recording)
        signal.signal(signal.SIGINT, self.stop_recording)

        cmd = ["tegrastats", "--interval", "300"]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        print("Waiting for START signal (SIGUSR1)...")

        for line in process.stdout:
            if not self.running:
                process.terminate()
                break

            line = line.strip()
            if not line:
                continue

            if self.recording:
                used, total = self.parse_ram(line)
                gpu = self.parse_gpu(line)

                if used is not None and gpu is not None:
                    avg_ram = self.ram_sum / self.ram_samples if self.ram_samples else 0
                    avg_gpu = self.gpu_sum / self.gpu_samples if self.gpu_samples else 0

                    print(
                        f"\rRAM: {used}/{total} MB | GPU: {gpu}% | "
                        f"PEAK RAM: {self.peak_ram_mb}/{self.total_ram_mb} MB | "
                        f"AVG RAM: {avg_ram:.1f} MB | "
                        f"PEAK GPU: {self.peak_gpu_percent}% | AVG GPU: {avg_gpu:.1f}%",
                        end="",
                        flush=True,
                    )

        process.wait()

        avg_ram = self.ram_sum / self.ram_samples if self.ram_samples else 0
        avg_gpu = self.gpu_sum / self.gpu_samples if self.gpu_samples else 0

        print("\n" + "=" * 50)
        print(f"PEAK RAM: {self.peak_ram_mb} MB / {self.total_ram_mb} MB")
        if self.total_ram_mb:
            print(f"RAM %: {(self.peak_ram_mb/self.total_ram_mb*100):.1f}%")
        print(f"AVG RAM: {avg_ram:.1f} MB")
        print(f"PEAK GPU: {self.peak_gpu_percent}%")
        print(f"AVG GPU: {avg_gpu:.1f}%")
        print("=" * 50)

        # Cleanup PID file
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)


if __name__ == "__main__":
    monitor = TegrastatsMonitor()
    monitor.monitor()
