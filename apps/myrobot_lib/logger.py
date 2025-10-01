from __future__ import annotations
from typing import List, Optional
import os
import csv
from datetime import datetime


def make_header(has_adc: bool, ldc_addrs: List[int], has_enc: bool) -> List[str]:
    header = ['ms', 'valve_a_pct', 'valve_b_pct', 'q_des', 'pid_u']
    if has_adc:
        header += ['adc0_raw','adc0_volt','adc0_kpa','adc1_raw','adc1_volt','adc1_kpa']
    for addr in ldc_addrs:
        header.append(f'tension_0x{addr:02X}')
    if has_enc:
        header.append('enc_deg')
    return header


class DataLogger:
    def __init__(self, base_dir: str, filename_hint: str, header: List[str]):
        os.makedirs(base_dir, exist_ok=True)
        self.base_dir = base_dir
        self.filename_hint = filename_hint
        self.header = header
        self.fp = None
        self.writer = None
        self.path = ''

    def open_file(self, run_index: int | None = None) -> str:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        # If a filename hint is provided, use sequential suffixes by scanning existing files.
        if self.filename_hint:
            if run_index is None:
                # find existing files that match the hint and extract numeric suffixes
                try:
                    files = [f for f in os.listdir(self.base_dir) if f.startswith(self.filename_hint + "_") and f.lower().endswith('.csv')]
                except Exception:
                    files = []
                max_idx = 0
                for f in files:
                    try:
                        suffix = f[len(self.filename_hint) + 1:]
                        suffix = suffix.rsplit('.', 1)[0]
                        idx = int(suffix)
                        if idx > max_idx:
                            max_idx = idx
                    except Exception:
                        # ignore files that don't follow the numeric suffix pattern
                        pass
                run_index = max_idx + 1
            filename = f'{self.filename_hint}_{run_index}.csv'
        else:
            filename = f'data_{ts}.csv'
        self.path = os.path.join(self.base_dir, filename)
        self.fp = open(self.path, 'w', buffering=1, newline='')
        self.writer = csv.writer(self.fp)
        self.writer.writerow(self.header)
        return self.path

    def write_row(self, row: List[str | float | int]) -> None:
        if self.writer is None:
            raise RuntimeError('Logger file not open')
        self.writer.writerow(row)

    def close(self) -> None:
        if self.fp:
            try:
                self.fp.close()
            except Exception:
                pass
            self.fp = None
            self.writer = None


class TerminalPrinter:
    def __init__(self, verbose: int = 0):
        self.verbose = verbose
        self._printed_header = False

    def print_row(self, header: List[str], row: List[str | float | int]) -> None:
        if self.verbose:
            if not self._printed_header:
                print(','.join(header), flush=True)
                self._printed_header = True
            print(','.join(str(x) for x in row), flush=True)
