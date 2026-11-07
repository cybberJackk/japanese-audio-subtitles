import unittest
import tempfile
from pathlib import Path
from runtime.telemetry import ResourceProfiler


class TestTelemetry(unittest.TestCase):
    def test_profiler_statistics_and_recommendations(self):
        profiler = ResourceProfiler()
        profiler.total_vram_mib = 40960.0
        profiler.gpu_name = 'NVIDIA A100-SXM4-40GB'

        # Fake some telemetry samples
        profiler.vram_samples = [10000.0, 15000.0, 18500.0, 14000.0]
        profiler.gpu_util_samples = [60.0, 75.0, 85.0, 70.0]

        # Fake some LLM request completions
        profiler.record_request('translate-zh', prompt_tokens=1850, completion_tokens=320, elapsed_sec=1.5)
        profiler.record_request('translate-zh', prompt_tokens=2100, completion_tokens=410, elapsed_sec=1.8)
        profiler.record_request('translate-zh', prompt_tokens=1400, completion_tokens=150, elapsed_sec=0.9)

        summary = profiler.get_summary(configured_slots=16, configured_slot_ctx=5120)

        # Check hardware stats
        self.assertEqual(summary['hardware']['peak_vram_mib'], 18500.0)
        self.assertEqual(summary['hardware']['peak_gpu_util_pct'], 85.0)

        # Check token stats
        self.assertEqual(summary['token_profile']['prompt_tokens_max'], 2100)
        self.assertEqual(summary['token_profile']['completion_tokens_max'], 410)

        # Check empirical recommendations
        # max_total = 2100 + 410 = 2510. Plus 512 = 3022. Round up to 256 = 3072 tokens.
        self.assertEqual(summary['optimization_insights']['empirical_min_safe_slot_ctx'], 3072)
        self.assertGreaterEqual(summary['optimization_insights']['recommended_max_concurrency_14b'], 36)

        # Test writing report
        with tempfile.TemporaryDirectory() as tmp:
            out_p = Path(tmp) / 'perf_profile.json'
            s, txt = profiler.write_report(out_p, configured_slots=16, configured_slot_ctx=5120)
            self.assertTrue(out_p.exists())
            self.assertTrue((Path(tmp) / 'perf_profile.txt').exists())
            self.assertIn('NVIDIA A100-SXM4-40GB', txt)


if __name__ == '__main__':
    unittest.main()
