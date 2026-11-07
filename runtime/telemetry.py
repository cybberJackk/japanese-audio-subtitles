import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional


class ResourceProfiler:
    """Real-time background hardware telemetry and LLM token usage profiler."""

    def __init__(self, sample_interval_sec: float = 1.0):
        self.interval = sample_interval_sec
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Telemetry samples
        self.vram_samples: List[float] = []
        self.gpu_util_samples: List[float] = []
        self.gpu_name: str = 'Unknown GPU'
        self.total_vram_mib: float = 0.0

        # LLM request stats
        self.requests: List[Dict[str, float]] = []

        self._init_gpu_info()

    def _init_gpu_info(self):
        try:
            raw = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=gpu_name,memory.total', '--format=csv,noheader,nounits'],
                text=True, stderr=subprocess.DEVNULL
            ).strip()
            if raw:
                parts = [p.strip() for p in raw.split('\n')[0].split(',')]
                self.gpu_name = parts[0]
                self.total_vram_mib = float(parts[1])
        except Exception:
            pass

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _sample_loop(self):
        while self.running:
            try:
                raw = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                    text=True, stderr=subprocess.DEVNULL
                ).strip()
                if raw:
                    parts = [p.strip() for p in raw.split('\n')[0].split(',')]
                    util = float(parts[0])
                    vram = float(parts[1])
                    with self._lock:
                        self.gpu_util_samples.append(util)
                        self.vram_samples.append(vram)
            except Exception:
                pass
            time.sleep(self.interval)

    def record_request(self, stage: str, prompt_tokens: int, completion_tokens: int, elapsed_sec: float):
        with self._lock:
            self.requests.append({
                'stage': stage,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
                'elapsed_sec': max(0.001, elapsed_sec),
                'tok_per_sec': round(completion_tokens / max(0.001, elapsed_sec), 2),
            })

    def get_summary(self, configured_slots: int = 16, configured_slot_ctx: int = 5120) -> Dict:
        with self._lock:
            vram_peak = max(self.vram_samples) if self.vram_samples else 0.0
            vram_avg = sum(self.vram_samples) / len(self.vram_samples) if self.vram_samples else 0.0
            gpu_util_peak = max(self.gpu_util_samples) if self.gpu_util_samples else 0.0
            gpu_util_avg = sum(self.gpu_util_samples) / len(self.gpu_util_samples) if self.gpu_util_samples else 0.0

            p_tokens = [r['prompt_tokens'] for r in self.requests if r['prompt_tokens'] > 0]
            c_tokens = [r['completion_tokens'] for r in self.requests if r['completion_tokens'] > 0]
            total_gen_tokens = sum(c_tokens)
            total_duration = sum(r['elapsed_sec'] for r in self.requests)

            p_max = max(p_tokens) if p_tokens else 0
            p_avg = round(sum(p_tokens) / len(p_tokens), 1) if p_tokens else 0
            c_max = max(c_tokens) if c_tokens else 0
            c_avg = round(sum(c_tokens) / len(c_tokens), 1) if c_tokens else 0

        # Empirical calculations for hardware optimization:
        # Minimum safe slot context = max prompt + max completion + safety padding of 512 tokens
        empirical_min_slot_ctx = max(2048, int(((p_max + c_max + 512 + 255) // 256) * 256)) if p_max > 0 else 4096
        slot_headroom = max(0, configured_slot_ctx - (p_max + c_max))

        # Recommend maximum parallel concurrency based on empirical min slot context
        # Assuming Qwen3-14B KV slope ~152 MiB/1024 tok, base ~9185 MiB, safety reserve 2560 MiB
        usable_vram = max(0.0, self.total_vram_mib - 2560.0 - 9185.0)
        empirical_slot_kv_mib = 152.0 * (empirical_min_slot_ctx / 1024.0)
        rec_max_slots_14b = max(4, int(usable_vram // empirical_slot_kv_mib)) if empirical_slot_kv_mib > 0 else configured_slots

        # For 9B-Echo: base ~7700 MiB, KV slope ~32.5 MiB/1024 tok
        usable_vram_9b = max(0.0, self.total_vram_mib - 2560.0 - 7700.0)
        empirical_slot_kv_9b = 32.5 * (empirical_min_slot_ctx / 1024.0)
        rec_max_slots_9b = max(6, int(usable_vram_9b // empirical_slot_kv_9b)) if empirical_slot_kv_9b > 0 else configured_slots

        overall_throughput = round(total_gen_tokens / max(0.001, total_duration), 2) if total_duration > 0 else 0.0

        return {
            'hardware': {
                'gpu_name': self.gpu_name,
                'total_vram_mib': self.total_vram_mib,
                'peak_vram_mib': round(vram_peak, 1),
                'peak_vram_pct': round((vram_peak / self.total_vram_mib * 100), 1) if self.total_vram_mib > 0 else 0.0,
                'avg_vram_mib': round(vram_avg, 1),
                'peak_gpu_util_pct': round(gpu_util_peak, 1),
                'avg_gpu_util_pct': round(gpu_util_avg, 1),
            },
            'token_profile': {
                'total_requests': len(self.requests),
                'total_completion_tokens': total_gen_tokens,
                'prompt_tokens_max': p_max,
                'prompt_tokens_avg': p_avg,
                'completion_tokens_max': c_max,
                'completion_tokens_avg': c_avg,
                'slot_ctx_headroom': slot_headroom,
                'overall_throughput_tok_sec': overall_throughput,
            },
            'optimization_insights': {
                'configured_slots': configured_slots,
                'configured_slot_ctx': configured_slot_ctx,
                'empirical_min_safe_slot_ctx': empirical_min_slot_ctx,
                'recommended_max_concurrency_14b': min(64, rec_max_slots_14b),
                'recommended_max_concurrency_9b': min(80, rec_max_slots_9b),
            }
        }

    def format_report_text(self, summary: Dict) -> str:
        hw = summary['hardware']
        tp = summary['token_profile']
        opt = summary['optimization_insights']

        lines = [
            "============================================================",
            f"        GPU TELEMETRY & HARDWARE PROFILING REPORT           ",
            "============================================================",
            f"GPU Model:          {hw['gpu_name']}",
            f"Total VRAM:         {hw['total_vram_mib']:.0f} MiB",
            f"Peak VRAM Used:     {hw['peak_vram_mib']:.0f} MiB ({hw['peak_vram_pct']}%)",
            f"Avg GPU Compute:    {hw['avg_gpu_util_pct']}% (Peak: {hw['peak_gpu_util_pct']}%)",
            "------------------------------------------------------------",
            f"Total LLM Requests: {tp['total_requests']}",
            f"Tokens Generated:   {tp['total_completion_tokens']} tokens",
            f"Prompt Token Range: avg {tp['prompt_tokens_avg']} | max {tp['prompt_tokens_max']} tokens",
            f"Output Token Range: avg {tp['completion_tokens_avg']} | max {tp['completion_tokens_max']} tokens",
            f"Current Slot Margin: {tp['slot_ctx_headroom']} tokens free per slot",
            "------------------------------------------------------------",
            f"EMPIRICAL TUNING RECOMMENDATIONS:",
            f"- Observed Max Sentence: {tp['prompt_tokens_max']} (prompt) + {tp['completion_tokens_max']} (gen)",
            f"- Empirical Safe Slot Context: {opt['empirical_min_safe_slot_ctx']} tokens (no waste, zero 400)",
            f"- Theoretical Max 14B Slots:   {opt['recommended_max_concurrency_14b']} 并发 (keeping 2.5GB reserve)",
            f"- Theoretical Max 9B Slots:    {opt['recommended_max_concurrency_9b']} 并发 (keeping 2.5GB reserve)",
            "============================================================",
        ]
        return "\n".join(lines)

    def write_report(self, dest_path: Path, configured_slots: int = 16, configured_slot_ctx: int = 5120):
        summary = self.get_summary(configured_slots=configured_slots, configured_slot_ctx=configured_slot_ctx)
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        report_txt = self.format_report_text(summary)
        (dest_path.with_suffix('.txt')).write_text(report_txt, encoding='utf-8')
        return summary, report_txt
