import sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace

ROOT_REPO = Path(__file__).resolve().parent.parent
RUNTIME_DIR = ROOT_REPO / 'runtime'
sys.path.insert(0, str(RUNTIME_DIR))
sys.path.insert(0, str(ROOT_REPO))

import subtitle_utils as su
import pipeline
import colab_subtitles as cli
from drive_store import MountedDriveStore


class SubtitlePipelineTests(unittest.TestCase):
    def setUp(self):
        self.source = [dict(start=0.0, end=1.0, ja='こんにちは。'), dict(start=2.0, end=3.5, ja='元気ですか？')]

    def test_source_timestamps_authoritative(self):
        out = su.parse_translation([dict(ids=[1, 2], text='你好，最近好吗？', start=99999, end=-5)], self.source, 0)
        self.assertEqual((out[0]['start'], out[0]['end']), (0.0, 3.5))

    def test_missing_or_duplicate_ids_rejected(self):
        for ids in ([1, 1], [1], [1, 3], [2, 1]):
            with self.assertRaises(ValueError):
                su.parse_translation([dict(ids=ids, text='译文')], self.source, 0)

    def test_arbitration_structure(self):
        with self.assertRaises(ValueError):
            su.parse_arbitration([dict(seg_id=1, ja_golden='a')] * 2, [dict(seg_id=1), dict(seg_id=2)])

    def test_stage_completion_without_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.assertFalse(pipeline.is_stage_done(out, 'dual-asr', '01.wav'))
            su.atomic_write_json(out / 'dual_asr_chunks.json', [{'seg_id': 1, 'start': 0, 'end': 1, 'ja_anime': 'あ', 'ja_turbo': 'あ'}])
            self.assertTrue(pipeline.is_stage_done(out, 'dual-asr', '01.wav'))

            su.write_vtt(out / '01.wav.vtt', [{'start': 0.0, 'end': 1.2, 'zh': '测试字幕'}], 'zh')
            su.atomic_write_text(out / '01.wav.zh.txt', '测试字幕\n')
            self.assertTrue(pipeline.is_stage_done(out, 'translate-zh', '01.wav'))

    def test_compress_extreme_repeats(self):
        noisy = 'あはは、' * 20 + '大好きです'
        compressed = su.compress_extreme_repeats(noisy)
        self.assertLess(len(compressed), len(noisy))
        self.assertIn('大好きです', compressed)

    def test_punctuation_and_decimal_projection(self):
        units = [
            SimpleNamespace(text='確率', start_time=0.0, end_time=0.4),
            SimpleNamespace(text='0', start_time=0.4, end_time=0.5),
            SimpleNamespace(text='02', start_time=0.5, end_time=0.7),
        ]
        proj = su.project_punctuation('確率0.02%！', units)
        self.assertEqual(''.join(w['text'] for w in proj), '確率0.02%！')

    def test_long_silence_not_merged(self):
        class Aligner:
            def align(self, audio, text, language):
                return [[SimpleNamespace(text=text, start_time=0, end_time=1)]]

        result = pipeline.align_chunks_to_cues(
            Aligner(), [0] * 40, 1,
            [dict(start=0, end=1, ja='合成テスト。'), dict(start=30, end=31, ja='です。')],
            'ja',
        )
        self.assertEqual(len(result), 2)

    def test_mounted_store_sync_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            mount_parent = base / 'MyDrive'
            mount_parent.mkdir()
            store = MountedDriveStore(drive_root=mount_parent / 'japanese-subtitles')
            src = base / 'outputs'
            src.mkdir()
            su.write_vtt(src / '01.wav.vtt', [{'start': 0.0, 'end': 1.0, 'zh': '你好'}], 'zh')
            store.sync_tree_up(src, 'jobs/AlbumA/01.wav/outputs')
            restored = base / 'restored'
            self.assertTrue(store.sync_tree_down('jobs/AlbumA/01.wav/outputs', restored))
            su.validate_vtt(restored / '01.wav.vtt')
            self.assertTrue(store.delete_path('jobs/AlbumA/01.wav'))
            self.assertTrue(store.delete_folder_if_empty('jobs/AlbumA'))

    def test_discovery_and_clean_project_slugs(self):
        with tempfile.TemporaryDirectory() as tmp:
            album = Path(tmp) / 'DramaAlbum01'
            album.mkdir()
            f1 = album / '01.wav'
            f2 = album / '02.mp3'
            f1.write_bytes(b'RIFF')
            f2.write_bytes(b'ID3')
            found = cli.discover_audio_files([album])
            self.assertEqual(found, [f1.resolve(), f2.resolve()])
            items = [{'audio': p} for p in found]
            cli.assign_project_slugs(items)
            self.assertEqual(items[0]['remote_job_rel'], 'jobs/DramaAlbum01/01.wav')
            self.assertEqual(items[1]['drive_input_rel'], 'input/DramaAlbum01/02.mp3.flac')

    def test_analyze_asr_pair_and_name_normalization(self):
        hints = su.analyze_asr_pair('本日は晴天なり', '本日は聖天なり')
        self.assertTrue(any('AB差異' in h for h in hints))
        normalized = su.normalize_ja_names('ハナコとはな子とハナッコ', [{'ja': 'ハナ子', 'zh': '花子'}])
        self.assertEqual(normalized, 'ハナ子とハナ子とハナ子')

    def test_parse_arbitration_blocks_cross_segment_copy(self):
        sub = [
            {'seg_id': 21, 'ja_anime': '設定を確認してから作業を開始します。', 'ja_turbo': '設定を確認してから作業を開始します。'},
            {'seg_id': 22, 'ja_anime': 'では、次の工程へ進みましょう。', 'ja_turbo': 'では、次の工程へ進みましょう。'},
        ]
        arr = [
            {'seg_id': 21, 'ja_golden': 'では、次の工程へ進みましょう。', 'zh_anchor': '那么我们进入下一道工序吧。'},
            {'seg_id': 22, 'ja_golden': 'では、次の工程へ進みましょう。', 'zh_anchor': '那么我们进入下一道工序吧。'},
        ]
        with self.assertRaises(ValueError):
            su.parse_arbitration(arr, sub)

    def test_sanitize_cue_timestamps(self):
        cues = [
            {'start': 192.5, 'end': 201.94, 'ja': 'A'},
            {'start': 199.685, 'end': 206.3, 'ja': 'B'},
        ]
        cleaned = su.sanitize_cue_timestamps(cues)
        su.validate_cues([{'start': c['start'], 'end': c['end'], 'text': c['ja']} for c in cleaned])
        self.assertLessEqual(cleaned[0]['end'], cleaned[1]['start'])

    def test_dedup_boundary_echo(self):
        rows = [
            {'start': 52.0, 'end': 58.6, 'ja': '大変、嬉しく思います。'},
            {'start': 58.6, 'end': 60.1, 'ja': '嬉しく思います。'},
        ]
        deduped = su.dedup_boundary_echo(rows)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]['end'], 60.1)


if __name__ == '__main__':
    unittest.main()
