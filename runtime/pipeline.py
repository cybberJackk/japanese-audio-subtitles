"""4-Stage Dual-ASR + Guided LLM Subtitle Pipeline (Colab L4).

Supports single-track (--audio) and stage-wise multi-track batch execution (--batch-json)
so each model is loaded at most once per stage across an entire album or queue.

Stages:
  1. dual-asr      : 450ms pause-aware chunking + Anime-Whisper & guided Whisper-Turbo
  2. arbitrate-ja  : Qwen3-14B Album-Wide Shared Track Guide + Phonetic/Homophone Reasoning + Semantic Anchor (zh_anchor)
  3. align         : Qwen3-ForcedAligner-0.6B word-level timing + punctuation projection + boundary echo deduplication
  4. translate-zh  : ASMR-Qwen3.5-9B-Echo controlled polishing anchored by Qwen3-14B's zh_anchor & sliding context window
"""
import argparse, json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from subtitle_utils import (
    analyze_asr_pair, atomic_write_json, atomic_write_text, cap_guide_lines,
     compress_extreme_repeats, compute_span_pan, dedup_boundary_echo,
    is_nonverbal_only, normalize_ja_names, pan_to_pos_label,
    parse_arbitration, parse_translation, project_punctuation,
    read_nonempty_json, validate_cues, validate_vtt, write_vtt,
)

ROOT = Path('/content/japanese-subtitles')
STAGES = ['dual-asr', 'arbitrate-ja', 'align', 'translate-zh']


def _detect_gpu_parallel_config():
    """Dynamically scale concurrency and thread count based on available GPU VRAM (A100/H100 vs L4)."""
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if vram_gb >= 35.0:  # A100 (40GB/80GB) or H100 (80GB)
                return 10
    except Exception:
        pass
    return 6


DEFAULT_PARALLEL = _detect_gpu_parallel_config()

DEFAULT_METADATA = {
    'work_title': '',
    'track_title': '',
    'synopsis_ja': '日本語の音声作品（ボイスドラマ・ASMR）。',
    'whisper_prompt_ja': 'ご主人様、マスター、先輩、先生、お兄ちゃん、大好き、囁き、吐息。',
    'base_glossary': [
        {'ja': 'ご主人様', 'zh': '主人', 'note': '通用称呼'},
        {'ja': 'マスター', 'zh': '主人', 'note': '通用称呼'},
        {'ja': '先輩', 'zh': '前辈', 'note': '通用称呼'},
        {'ja': '先生', 'zh': '老师', 'note': '通用称呼'},
    ],
}


def load_metadata(meta_path):
    meta = dict(DEFAULT_METADATA)
    base_list = list(DEFAULT_METADATA['base_glossary'])
    if meta_path and Path(meta_path).is_file():
        user_meta = json.loads(Path(meta_path).read_text(encoding='utf-8'))
        for k, v in user_meta.items():
            if k == 'base_glossary' and isinstance(v, list):
                seen = {x['ja'].lower() for x in base_list if x.get('ja')}
                for item in v:
                    ja = (item.get('ja') or '').strip()
                    if ja and ja.lower() not in seen:
                        base_list.append(item)
                        seen.add(ja.lower())
            elif v not in (None, '', []):
                meta[k] = v
    meta['base_glossary'] = base_list
    if meta.get('whisper_prompt_ja') == DEFAULT_METADATA['whisper_prompt_ja']:
        extra_terms = []
        if meta.get('work_title'):
            extra_terms.append(meta['work_title'][:60])
        if meta.get('synopsis_ja') and meta['synopsis_ja'] != DEFAULT_METADATA['synopsis_ja']:
            extra_terms.append(meta['synopsis_ja'][:80])
        for g in meta.get('base_glossary', []):
            if g.get('ja') and g['ja'] not in meta['whisper_prompt_ja']:
                extra_terms.append(g['ja'])
        if extra_terms:
            meta['whisper_prompt_ja'] = '、'.join(extra_terms[:15]) + '。' + meta['whisper_prompt_ja']
    return meta


def is_stage_done(out_dir, stage, stem):
    out_dir = Path(out_dir)
    if stage == 'dual-asr':
        return bool(read_nonempty_json(out_dir / 'dual_asr_chunks.json'))
    if stage == 'arbitrate-ja':
        return bool(
            read_nonempty_json(out_dir / 'dual_asr_arbitrated.json')
            and read_nonempty_json(out_dir / 'track_guide.json')
        )
    if stage == 'align':
        if not read_nonempty_json(out_dir / 'aligned_cues.json'):
            return False
        try:
            validate_vtt(out_dir / f'{stem}.ja.vtt')
            return (out_dir / f'{stem}.ja.txt').is_file()
        except ValueError:
            return False
    if stage == 'translate-zh':
        try:
            validate_vtt(out_dir / f'{stem}.vtt')
            zh_txt = out_dir / f'{stem}.zh.txt'
            return zh_txt.is_file() and zh_txt.stat().st_size > 0
        except ValueError:
            return False
    return False


def segment_audio_by_pauses(audio, sr, min_sec=5.0, max_sec=14.0):
    """Split audio at true inter-sentence pauses (450ms window) to avoid slicing inside stop consonants (/k/t/p/っ)."""
    import numpy as np
    segments = []
    start = 0
    total = len(audio)
    win = int(0.45 * sr)
    step = int(0.05 * sr)
    while start < total:
        rem = (total - start) / sr
        if rem <= max_sec + 2.0:
            segments.append((start, total))
            break
        search_start = start + int(min_sec * sr)
        search_end = min(start + int(max_sec * sr), total - win)
        candidates = range(search_start, search_end, step)
        best = min(candidates, key=lambda idx: float(np.mean(audio[idx:idx + win] ** 2)))
        cut = best + win // 2
        segments.append((start, cut))
        start = cut
    return segments


def run_dual_asr_batch(jobs, batch_size=16):
    import numpy as np, torch, soundfile as sf
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    paths = json.loads((ROOT / 'models.json').read_text(encoding='utf-8'))
    prepared = []
    for job in jobs:
        out_dir = Path(job['out_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)
        audio, sr = sf.read(str(job['audio']), dtype='float32')
        if sr != 16000 or audio.ndim != 1:
            raise ValueError(f"Expected 16kHz mono audio for {job['audio']}")
        segs = segment_audio_by_pauses(audio, sr, min_sec=5.0, max_sec=14.0)
        valid_segs, clips = [], []
        for s, e in segs:
            clip = audio[s:e]
            if float(np.sqrt(np.mean(clip ** 2))) >= 0.0015:
                valid_segs.append((s, e))
                clips.append(clip)
        prepared.append(dict(
            job=job, sr=sr, valid_segs=valid_segs, clips=clips,
            anime_texts=[], turbo_texts=[], t_anime=0.0, t_turbo=0.0,
        ))

    # Stream A: litagin/anime-whisper
    mp_a = paths['litagin/anime-whisper']
    proc_a = AutoProcessor.from_pretrained(mp_a)
    mod_a = AutoModelForSpeechSeq2Seq.from_pretrained(
        mp_a, torch_dtype=torch.float16, attn_implementation='sdpa'
    ).to('cuda').eval()
    try:
        for item in prepared:
            t0 = time.monotonic()
            clips, sr = item['clips'], item['sr']
            for b_start in range(0, len(clips), batch_size):
                b_clips = clips[b_start:b_start + batch_size]
                enc = proc_a(b_clips, sampling_rate=sr, return_tensors='pt', padding='max_length', return_attention_mask=True)
                with torch.inference_mode():
                    ids = mod_a.generate(
                        enc.input_features.to('cuda', dtype=torch.float16),
                        attention_mask=enc.attention_mask.to('cuda'),
                        language='ja', task='transcribe',
                        max_new_tokens=256, do_sample=False,
                        no_repeat_ngram_size=5, repetition_penalty=1.05,
                    )
                item['anime_texts'].extend(t.strip() for t in proc_a.batch_decode(ids, skip_special_tokens=True))
            torch.cuda.synchronize()
            item['t_anime'] = round(time.monotonic() - t0, 2)
            print(f"STREAM_A_ANIME_WHISPER_DONE [{item['job']['stem']}]: {len(item['anime_texts'])} segs in {item['t_anime']}s", flush=True)
    finally:
        del mod_a
        torch.cuda.empty_cache()

    # Stream B: openai/whisper-large-v3-turbo (with acoustic prompt)
    mp_b = paths['openai/whisper-large-v3-turbo']
    proc_b = AutoProcessor.from_pretrained(mp_b)
    mod_b = AutoModelForSpeechSeq2Seq.from_pretrained(
        mp_b, torch_dtype=torch.float16, attn_implementation='sdpa'
    ).to('cuda').eval()
    try:
        for item in prepared:
            t0 = time.monotonic()
            meta = item['job']['meta']
            clips, sr = item['clips'], item['sr']
            whisper_prompt = meta.get('whisper_prompt_ja') or DEFAULT_METADATA['whisper_prompt_ja']
            prompt_ids = proc_b.get_prompt_ids(whisper_prompt, return_tensors='pt').to('cuda')
            if prompt_ids.ndim == 2:
                prompt_ids = prompt_ids.squeeze(0)
            if prompt_ids.shape[-1] > 150:
                prompt_ids = torch.cat([prompt_ids[:1], prompt_ids[-149:]])
            safe_max_new = min(224, 440 - int(prompt_ids.shape[-1]))

            for clip in clips:
                enc = proc_b(clip, sampling_rate=sr, return_tensors='pt', return_attention_mask=True)
                with torch.inference_mode():
                    ids = mod_b.generate(
                        enc.input_features.to('cuda', dtype=torch.float16),
                        attention_mask=enc.attention_mask.to('cuda'),
                        language='ja', task='transcribe',
                        max_new_tokens=safe_max_new, do_sample=False,
                        no_repeat_ngram_size=5, repetition_penalty=1.08,
                        prompt_ids=prompt_ids,
                    )
                t = proc_b.batch_decode(ids, skip_special_tokens=True)[0].strip()
                if whisper_prompt[:10] in t:
                    t = t.replace(whisper_prompt, '').strip()
                item['turbo_texts'].append(t)
            torch.cuda.synchronize()
            item['t_turbo'] = round(time.monotonic() - t0, 2)
            print(f"STREAM_B_WHISPER_TURBO_DONE [{item['job']['stem']}]: {len(item['turbo_texts'])} segs in {item['t_turbo']}s", flush=True)

            out_dir = Path(item['job']['out_dir'])
            paired = []
            for idx, ((s, e), ta, tb) in enumerate(zip(item['valid_segs'], item['anime_texts'], item['turbo_texts'])):
                if not ta and not tb:
                    continue
                paired.append(dict(
                    seg_id=idx + 1,
                    start=round(s / sr, 3),
                    end=round(e / sr, 3),
                    ja_anime=ta,
                    ja_turbo=tb,
                ))
            atomic_write_json(out_dir / 'dual_asr_chunks.json', paired)
            if item['job'].get('batch_mode'):
                print(f"BATCH_TRACK_STAGE_DONE {item['job']['index']} dual-asr {item['job']['stem']}", flush=True)
    finally:
        del mod_b
        torch.cuda.empty_cache()


def start_llama_server(model_path, out_dir, port, n_parallel=4, ctx_size=40960, log_name='llama.log'):
    import requests
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    binary = next((ROOT / 'llama').rglob('llama-server'))
    libs = {str(p.parent) for p in (ROOT / 'llama').rglob('*.so*')}
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = ':'.join(libs) + ':' + env.get('LD_LIBRARY_PATH', '')
    fa_variants = [['-fa', 'on'], ['-fa'], []]
    variant_idx = 0
    log_path = out_dir / log_name
    log_path.write_text('', encoding='utf-8')

    def launch(fa_args):
        cmd = [
            str(binary), '-m', str(model_path), '-ngl', '99',
            '-c', str(ctx_size), '--parallel', str(n_parallel), *fa_args,
            '--host', '127.0.0.1', '--port', str(port),
        ]
        return subprocess.Popen(
            cmd, env=env, stdin=subprocess.DEVNULL,
            stdout=log_path.open('a'), stderr=subprocess.STDOUT, close_fds=True,
        )

    proc = launch(fa_variants[0])
    for _ in range(180):
        if proc.poll() is not None:
            log_tail = log_path.read_text(errors='replace')[-1200:]
            if (variant_idx + 1 < len(fa_variants)
                    and any(k in log_tail.lower() for k in ('invalid', 'argument', 'flash', '-fa'))
                    and 'out of memory' not in log_tail.lower()):
                variant_idx += 1
                log_path.write_text('', encoding='utf-8')
                proc = launch(fa_variants[variant_idx])
                continue
            raise RuntimeError(f'llama-server failed to start: {log_tail}')
        try:
            if requests.get(f'http://127.0.0.1:{port}/health', timeout=2).ok:
                return proc
        except Exception:
            pass
        time.sleep(1)
    proc.kill()
    raise RuntimeError('llama-server startup timeout')


def stop_server(proc):
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def _extract_track_guide(stem, out_dir, meta, port):
    import requests
    out_dir = Path(out_dir)
    paired = json.loads((out_dir / 'dual_asr_chunks.json').read_text(encoding='utf-8'))
    if not paired:
        raise ValueError(f'No ASR segments detected in {stem}')

    existing = read_nonempty_json(out_dir / 'track_guide.json')
    if existing:
        return existing

    t_p1 = time.monotonic()
    pan_10hz = meta.get('stereo_pan_10hz') or []
    raw_lines = []
    for c in paired:
        if is_nonverbal_only(c['ja_anime']):
            continue
        seg_pan = compute_span_pan(pan_10hz, c['start'], c['end'])
        pos_tag = f"|{pan_to_pos_label(seg_pan)}" if pan_10hz and abs(seg_pan) >= 0.11 else ''
        line = f"[{c['seg_id']}{pos_tag}] A:{compress_extreme_repeats(c['ja_anime'])} | B:{compress_extreme_repeats(c['ja_turbo'])}"
        hints = analyze_asr_pair(c['ja_anime'], c['ja_turbo'])
        if hints:
            line += f"  <{'; '.join(hints)}>"
        raw_lines.append(line)
    if not raw_lines:
        raw_lines = [
            f"[{c['seg_id']}] A:{compress_extreme_repeats(c['ja_anime'])} | B:{compress_extreme_repeats(c['ja_turbo'])}"
            for c in paired[:30]
        ]
    compact_lines = cap_guide_lines(raw_lines, max_chars=9500)

    guide_schema = {
        'type': 'object',
        'properties': {
            'story_summary_zh': {'type': 'string'},
            'scene_constraints_zh': {'type': 'string'},
            'character_tone_zh': {'type': 'string'},
            'character_self_reference_ja': {'type': 'string'},
            'context_homophone_fixes': {
                'type': 'array',
                'maxItems': 10,
                'items': {
                    'type': 'object',
                    'properties': {
                        'wrong_ja': {'type': 'string'},
                        'correct_ja': {'type': 'string'},
                        'zh_meaning': {'type': 'string'},
                    },
                    'required': ['wrong_ja', 'correct_ja', 'zh_meaning'],
                    'additionalProperties': False,
                },
            },
            'discovered_terms': {
                'type': 'array',
                'maxItems': 10,
                'items': {
                    'type': 'object',
                    'properties': {
                        'ja': {'type': 'string'},
                        'zh': {'type': 'string'},
                        'note': {'type': 'string'},
                    },
                    'required': ['ja', 'zh', 'note'],
                    'additionalProperties': False,
                },
            },
        },
        'required': [
            'story_summary_zh', 'scene_constraints_zh', 'character_tone_zh',
            'character_self_reference_ja', 'context_homophone_fixes', 'discovered_terms',
        ],
        'additionalProperties': False,
    }
    title_line = f"{meta.get('work_title', '')} / {meta.get('track_title', '')}".strip(' /') or stem
    synopsis_clean = compress_extreme_repeats(meta.get('synopsis_ja', ''))
    p1_prompt = (
        '你是一位资深日语音声（ボイスドラマ / ASMR）字幕总监。请结合作品简介与整轨双路ASR初稿（A为Anime-Whisper，B为Whisper-Turbo），通过【场景推导 -> 同音错字排查】生成简洁准确的《全轨全局导引卡》。\n\n'
        f'【作品/音轨标题】{title_line}\n'
        f'【作品简介】{synopsis_clean}\n\n'
        '请严格基于本轨实际台词与作品简介提取（各项文字请精炼扼要）：\n'
        '1. story_summary_zh：按时间顺序用3~4句话概括本轨完整剧情脉络（开场人物处境、互动发展、结局）。\n'
        '2. scene_constraints_zh：明确（a）说话人的真实身份、职业/属性与利害关系；（b）本轨正在进行的核心物理动作域与互动方式。\n'
        '3. character_tone_zh：角色说话模式、敬语关系与反差特征（例如表面礼貌敬语与内心独白/小声吐槽的切换规律；若为多人对话请标明各自性格与声道位置）。\n'
        '4. character_self_reference_ja：仅填写角色使用的第一人称代词（如『私』『あたし』『僕』等，10字以内）。\n'
        '5. context_homophone_fixes：【基于本轨场景动态排查同音异义词与假名混写（最多8项）】请对照上方推导出的【人物身份】与【核心动作域】，扫描全轨初稿（特别是带有 `<AB差異:...>` 的句子或前后文语义不通的词），找出读音相同/相近但汉字写偏的词（例如日语ASR常见的同音异义词、清浊音近音混淆、或汉字与片假名混写错误），给出错误写法 `wrong_ja`、修正后写法 `correct_ja` 及中文含义 `zh_meaning`。\n'
        '6. discovered_terms：提取修正后的角色名、称呼及本轨核心专有名词对照表（最多8项，切勿罗列普通日常词或拟声词）。\n/no_think\n\n'
        '【双路ASR全文初稿】\n' + '\n'.join(compact_lines)
    )
    track_guide = None
    last_err = None
    for temp_try in (0.15, 0.35, 0.55):
        try:
            r_p1 = requests.post(
                f'http://127.0.0.1:{port}/v1/chat/completions',
                json={
                    'messages': [{'role': 'user', 'content': p1_prompt}],
                    'temperature': temp_try,
                    'max_tokens': 4096,
                    'frequency_penalty': 0.2,
                    'repeat_penalty': 1.18,
                    'chat_template_kwargs': {'enable_thinking': False},
                    'response_format': {'type': 'json_schema', 'json_schema': {'name': 'track_guide', 'strict': True, 'schema': guide_schema}},
                },
                timeout=180,
            )
            r_p1.raise_for_status()
            if r_p1.json()['choices'][0].get('finish_reason') == 'length':
                raise ValueError('Track guide generation truncated')
            c_p1 = r_p1.json()['choices'][0]['message']['content'].strip()
            if '</think>' in c_p1:
                c_p1 = c_p1.split('</think>', 1)[1].strip()
            track_guide = json.loads(c_p1)
            break
        except Exception as exc:
            last_err = exc
    if track_guide is None:
        print(f'PASS1_TRACK_GUIDE_FALLBACK [{stem}] ({last_err})', flush=True)
        track_guide = {
            'story_summary_zh': f'本轨为《{title_line}》的日语音声剧情（简介：{synopsis_clean[:120]}）。',
            'scene_constraints_zh': '请严格依据作品简介与台词上下文保持人物关系、表面敬语与内心独白反差的一致性。',
            'character_tone_zh': '结合作品简介还原角色的真实语气与情绪变化。',
            'character_self_reference_ja': '私',
            'context_homophone_fixes': [],
            'discovered_terms': [],
        }
    atomic_write_json(out_dir / 'track_guide.json', track_guide)
    pass1_sec = round(time.monotonic() - t_p1, 2)
    fixes_cnt = len(track_guide.get('context_homophone_fixes', []))
    print(f'PASS1_TRACK_GUIDE_DONE [{stem}] in {pass1_sec}s (fixes={fixes_cnt}): {track_guide["story_summary_zh"][:80]}', flush=True)
    return track_guide


def _merge_album_glossary(jobs, track_guides):
    """Broadcast character names, self-references, discovered terms, and scene homophone fixes across the batch."""
    shared_terms = []
    seen_ja = set()
    shared_fixes = []
    seen_wrong = set()
    self_refs = []
    ko_name_fixes = {}

    def _add_term(ja, zh, note):
        ja, zh = ja.strip(), zh.strip()
        if not ja or not zh:
            return
        if zh.endswith('子') and ja.endswith('コ') and len(ja) >= 3:
            kanji_ja = ja[:-1] + '子'
            ko_name_fixes[ja] = kanji_ja
            ja = kanji_ja
        elif ja.endswith('子') and zh.endswith('子') and len(ja) >= 3:
            ko_name_fixes[ja[:-1] + 'コ'] = ja
        if ja.lower() not in seen_ja:
            shared_terms.append({'ja': ja, 'zh': zh, 'note': note})
            seen_ja.add(ja.lower())

    for job in jobs:
        for g in job['meta'].get('base_glossary', []):
            _add_term(g['ja'], g['zh'], g.get('note', ''))

    for stem, tg in sorted(track_guides.items(), key=lambda x: x[0]):
        for item in tg.get('discovered_terms', []):
            _add_term(item.get('ja', ''), item.get('zh', ''), item.get('note', ''))
        for fx in tg.get('context_homophone_fixes', []):
            w = (fx.get('wrong_ja') or '').strip()
            c = (fx.get('correct_ja') or '').strip()
            m = (fx.get('zh_meaning') or '').strip()
            if w and c and w != c and w not in seen_wrong:
                shared_fixes.append({'wrong_ja': w, 'correct_ja': c, 'zh_meaning': m})
                seen_wrong.add(w)
                _add_term(c, m, f'原误识:{w}')

    # Find primary anchor track (prefer 1.wav / 01.wav or lowest index) to broadcast album-wide world view & scene constraints
    sorted_guides = sorted(track_guides.items(), key=lambda x: x[0])
    primary_stem = sorted_guides[0][0]
    for s, _ in sorted_guides:
        if re.search(r'(?:^|[\D])0?1(?:\.|\b)', s):
            primary_stem = s
            break
    primary_guide = track_guides[primary_stem]
    album_core_summary = primary_guide.get('story_summary_zh', '')
    album_core_scene = primary_guide.get('scene_constraints_zh', '')
    album_core_tone = primary_guide.get('character_tone_zh', '')

    for stem, tg in sorted(track_guides.items(), key=lambda x: x[0]):
        for k_field in ('story_summary_zh', 'scene_constraints_zh', 'character_self_reference_ja'):
            val = tg.get(k_field, '')
            for wrong_ko, right_ko in ko_name_fixes.items():
                val = val.replace(wrong_ko, right_ko)
            tg[k_field] = val
        sr = tg.get('character_self_reference_ja', '').strip()
        if sr and sr not in self_refs:
            self_refs.append(sr)

    for stem, tg in track_guides.items():
        tg['album_shared_terms'] = shared_terms[:30]
        tg['album_shared_fixes'] = shared_fixes[:20]
        tg['album_self_references'] = ', '.join(self_refs)
        # Deep heritage from 01.wav: ensure all tracks inherit album world view and emotional baseline
        tg['album_core_summary'] = album_core_summary
        tg['album_core_scene'] = album_core_scene
        tg['album_core_tone'] = album_core_tone
    return shared_terms[:30], ', '.join(self_refs)


def _arbitrate_one_track(stem, out_dir, meta, track_guide, port):
    import requests
    out_dir = Path(out_dir)
    paired = json.loads((out_dir / 'dual_asr_chunks.json').read_text(encoding='utf-8'))
    if not paired:
        raise ValueError(f'No ASR segments detected in {stem}')

    t_p2 = time.monotonic()
    shared_terms = track_guide.get('album_shared_terms') or track_guide.get('discovered_terms', [])
    shared_fixes = track_guide.get('album_shared_fixes') or track_guide.get('context_homophone_fixes', [])
    discovered_list = ', '.join(f"{t['ja']}->{t['zh']}({t.get('note', '')})" for t in shared_terms)
    fixes_list = ' / '.join(
        f"『{fx['wrong_ja']}』→『{fx['correct_ja']}』（意味:{fx['zh_meaning']}）"
        for fx in shared_fixes
    ) or '（文脈に応じて判定）'
    self_refs = track_guide.get('album_self_references') or track_guide.get('character_self_reference_ja', '')

    arb_sys = (
        'あなたは日本語の音声作品（ボイスドラマ・ASMR）の音声認識（ASR）校正のスペシャリストです。\n'
        f"【作品設定】{meta.get('synopsis_ja', '')}\n"
        + (
            f"【全作全局背景与核心设定（继承自开篇轨）】\n"
            f"- 全剧核心脉络：{track_guide.get('album_core_summary', '')}\n"
            f"- 全局人物身份与处境约束：{track_guide.get('album_core_scene', '')}\n"
            f"- 角色基调与情感约束：{track_guide.get('album_core_tone', '')}\n"
            if track_guide.get('album_core_summary') else ''
        )
        + f"【本トラックあらすじ】{track_guide['story_summary_zh']}\n"
        f"【シーン・設定と行動領域】{track_guide['scene_constraints_zh']}\n"
        f"【話し方・敬語と本音の切り替え】{track_guide.get('character_tone_zh', '')}\n"
        f'【キャラクター自称・一人称】{self_refs}\n'
        f'【全編共有用語集（必ずこの表記に統一）】{discovered_list}\n'
        f'【本編の文脈から特定された同音異義語・誤変換補正リスト】{fixes_list}\n\n'
        '各音声区間について、性質の異なる2つのASRモデルの出力（A_anime と B_turbo）、およびAB間の差分（hints）が与えられます：\n'
        '- 【A_anime (Anime-Whisper)】：声優の発音（仮名読み）や吐息・小声の聞き取りは非常に正確だが、漢字変換で同音異義語ミスを起こしやすい。\n'
        '- 【B_turbo (Whisper-Turbo)】：数字・小数・標準的な熟語に強いが、無言区間で『ご視聴ありがとうございました』等の幻覚を出すことがある。\n\n'
        '【厳守すべき校正ルール】\n'
        '1. **全作背景・シーン文脈に基づく同音異義語の積極的修正**：\n'
        '   - ASRは一般的な日常語や無関係な漢字（例：招集/召集、痴候、周期、火の中）に誤変換することが多々あります。必ず【全作全局背景】および本トラックのシーン設定（誰が・どのような状況で・何を行っているか）と照合してください。\n'
        '   - 登場人物の境遇や行為に不自然な語がある場合、発音が同一または類似する適切な漢字（例：消臭、恥垢、臭気、胃の中など）へ是正してください。\n'
        '   - 入力に `hints`（AB差異）がある場合、および上記の【誤変換補正リスト】に該当する語がある場合は、作品設定と前後の文脈に合致する表記を `ja_golden` に採用してください。\n'
        '   - AとBのどちらも同音異義語の誤変換を起こしている場合でも、前後の文脈から正しい同音語に補正してください。\n'
        '2. **話者交代・敬語と内心の声の句点（。/！/？）分離**：\n'
        '   - 1つの区間内で複数のキャラクターが掛け合いをしている場合、または表向きの敬語と内心の独白・小声のツッコミが切り替わる場合は、読点『、』で繋げず、必ず文末句点『**。**』『**！**』『**？**』で文を区切ってください。\n'
        '3. **ja_golden（最終日本語テキスト）**：同音異義語・固有名詞・句読点を修正した自然な日本語文を出力してください。過度に連続反復する同一の擬音は自然な長さ（2〜3回＋…程度）に整えてください。'
    )

    batch_sz = 10
    batches = [(k, paired[k:k + batch_sz]) for k in range(0, len(paired), batch_sz)]

    def _arb_sub_batch(b_idx, sub):
        ctx_before = [
            f"[{x['seg_id']}] A:{compress_extreme_repeats(x['ja_anime'])} | B:{compress_extreme_repeats(x['ja_turbo'])}"
            for x in paired[max(0, b_idx - 2):b_idx]
        ]
        ctx_after = [
            f"[{x['seg_id']}] A:{compress_extreme_repeats(x['ja_anime'])} | B:{compress_extreme_repeats(x['ja_turbo'])}"
            for x in paired[b_idx + len(sub):min(len(paired), b_idx + len(sub) + 2)]
        ]
        inp = []
        for x in sub:
            row_in = {
                'seg_id': x['seg_id'],
                'A_anime': compress_extreme_repeats(x['ja_anime']),
                'B_turbo': compress_extreme_repeats(x['ja_turbo']),
            }
            hints = analyze_asr_pair(x['ja_anime'], x['ja_turbo'])
            if hints:
                row_in['hints'] = ' / '.join(hints)
            inp.append(row_in)

        schema = {
            'type': 'object',
            'properties': {
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'seg_id': {'type': 'integer'},
                            'check': {'type': 'string'},
                            'ja_golden': {'type': 'string'},
                        },
                        'required': ['seg_id', 'check', 'ja_golden'],
                        'additionalProperties': False,
                    },
                    'minItems': len(sub),
                    'maxItems': len(sub),
                },
            },
            'required': ['items'],
            'additionalProperties': False,
        }
        ctx_block = ''
        if ctx_before:
            ctx_block += '【前文参考（出力不要）】\n' + '\n'.join(ctx_before) + '\n\n'
        if ctx_after:
            ctx_block += '【後文参考（出力不要）】\n' + '\n'.join(ctx_after) + '\n\n'

        user_msg = (
            ctx_block
            + '以下の対象区間について、まずcheckでhintsおよび同音異義語・主語・数値を検証し、すべての誤変換を修正した正しいja_goldenを出力してください：/no_think\n'
            + json.dumps(inp, ensure_ascii=False)
        )
        last_err = None
        for temp_try in (0.1, 0.3, 0.5):
            try:
                resp = requests.post(
                    f'http://127.0.0.1:{port}/v1/chat/completions',
                    json={
                        'messages': [
                            {'role': 'system', 'content': arb_sys},
                            {'role': 'user', 'content': user_msg},
                        ],
                        'temperature': temp_try,
                        'max_tokens': 4096,
                        'frequency_penalty': 0.18,
                        'repeat_penalty': 1.15,
                        'chat_template_kwargs': {'enable_thinking': False},
                        'response_format': {'type': 'json_schema', 'json_schema': {'name': 'arb_ja', 'strict': True, 'schema': schema}},
                    },
                    timeout=180,
                )
                resp.raise_for_status()
                if resp.json()['choices'][0].get('finish_reason') == 'length':
                    raise ValueError('Arbitration generation truncated')
                txt = resp.json()['choices'][0]['message']['content'].strip()
                if '</think>' in txt:
                    txt = txt.split('</think>', 1)[1].strip()
                arr = json.loads(txt)['items']
                parse_arbitration(arr, sub)
                return arr
            except Exception as e:
                last_err = e
        if len(sub) > 1:
            mid = len(sub) // 2
            print(f'ARBITRATE_BISECT [{stem}] splitting {len(sub)} segs -> {mid}+{len(sub) - mid} ({last_err})', flush=True)
            return _arb_sub_batch(b_idx, sub[:mid]) + _arb_sub_batch(b_idx + mid, sub[mid:])
        fallback_txt = compress_extreme_repeats(sub[0]['ja_anime'] or sub[0]['ja_turbo'])
        print(f"ARBITRATE_SINGLE_FALLBACK [{stem}] seg={sub[0]['seg_id']} ({last_err})", flush=True)
        return [{'seg_id': sub[0]['seg_id'], 'check': 'fallback', 'ja_golden': fallback_txt}]

    def run_arb_batch(item):
        b_idx, sub = item
        arr = _arb_sub_batch(b_idx, sub)
        print(f'ARBITRATE [{stem}] batch @{b_idx} done ({len(arr)} segs)', flush=True)
        return b_idx, arr

    golden_map = {}
    with ThreadPoolExecutor(max_workers=DEFAULT_PARALLEL) as ex:
        for _, arr in ex.map(run_arb_batch, batches):
            for obj in arr:
                cleaned_ja = normalize_ja_names(obj['ja_golden'].strip(), shared_terms, fixes=shared_fixes)
                golden_map[obj['seg_id']] = cleaned_ja

    for c in paired:
        c['ja_golden'] = golden_map[c['seg_id']]
    pass2_sec = round(time.monotonic() - t_p2, 2)
    atomic_write_json(out_dir / 'dual_asr_arbitrated.json', paired)
    print(f'PASS2_ARBITRATION_DONE [{stem}] in {pass2_sec}s', flush=True)


def run_arbitrate_batch(jobs, port=8090):
    if not jobs:
        return
    paths = json.loads((ROOT / 'models.json').read_text(encoding='utf-8'))
    srv = start_llama_server(paths['qwen3-14b'], Path(jobs[0]['out_dir']), port, n_parallel=DEFAULT_PARALLEL, ctx_size=40960, log_name='qwen3-14b.log')
    try:
        # Phase 1: Extract Track Guide for every track in the batch concurrently
        def _run_p1(job):
            return job['stem'], _extract_track_guide(job['stem'], Path(job['out_dir']), job['meta'], port)

        with ThreadPoolExecutor(max_workers=min(DEFAULT_PARALLEL, len(jobs))) as ex:
            track_guides = dict(ex.map(_run_p1, jobs))

        # Broadcast album-wide character names, self-references, and glossary across all tracks
        shared_terms, self_refs = _merge_album_glossary(jobs, track_guides)
        print(f'ALBUM_SHARED_GLOSSARY: {len(shared_terms)} terms (self_refs={self_refs})', flush=True)
        for job in jobs:
            out_dir = Path(job['out_dir'])
            atomic_write_json(out_dir / 'track_guide.json', track_guides[job['stem']])

        # Phase 2: Arbitrate each track using the shared album glossary + phonetic reasoning
        for job in jobs:
            out_dir = Path(job['out_dir'])
            _arbitrate_one_track(job['stem'], out_dir, job['meta'], track_guides[job['stem']], port)
            if job.get('batch_mode'):
                print(f"BATCH_TRACK_STAGE_DONE {job['index']} arbitrate-ja {job['stem']}", flush=True)
    finally:
        stop_server(srv)


def align_chunks_to_cues(model, audio, sr, chunks, text_field, pan_10hz=None):
    norm = lambda s: re.sub(r'[\s、。！？,.!?…～〜ー・♡💛♪「」『』（）()％%]', '', s)
    rows = []
    for chunk in chunks:
        ja_text = chunk[text_field].strip()
        if not ja_text or is_nonverbal_only(ja_text):
            continue

        data = audio[round(chunk['start'] * sr):round(chunk['end'] * sr)]
        try:
            units = model.align(audio=(data, sr), text=ja_text, language='Japanese')[0]
            proj = project_punctuation(ja_text, units)
            words = []
            for w in proj:
                ws = round(max(chunk['start'], min(chunk['end'], w['start'] + chunk['start'])), 3)
                we = round(max(ws + 0.02, min(chunk['end'], w['end'] + chunk['start'])), 3)
                words.append(dict(
                    text=w['text'], start=ws, end=we,
                    pan=compute_span_pan(pan_10hz, ws, we),
                ))
        except Exception as exc:
            print(f'ALIGN_FALLBACK seg={chunk.get("seg_id")} error={type(exc).__name__}: {exc}', flush=True)
            words = []

        if not words:
            c_pan = compute_span_pan(pan_10hz, chunk['start'], chunk['end'])
            rows.append(dict(
                start=chunk['start'], end=chunk['end'], ja=ja_text,
                pan=round(c_pan, 3), pos=pan_to_pos_label(c_pan),
            ))
            continue

        buf = []
        base_idx = len(rows)

        def flush_buf():
            if buf and buf[-1]['end'] > buf[0]['start']:
                text_str = ''.join(w['text'] for w in buf).strip()
                if text_str and not is_nonverbal_only(text_str):
                    b_pan = compute_span_pan(pan_10hz, buf[0]['start'], buf[-1]['end'])
                    rows.append(dict(
                        start=buf[0]['start'], end=buf[-1]['end'], ja=text_str,
                        pan=round(b_pan, 3), pos=pan_to_pos_label(b_pan),
                    ))
                buf.clear()

        for idx_w, w in enumerate(words):
            cur_text = ''.join(x['text'] for x in buf)
            cur_len = len(norm(cur_text))
            if buf:
                gap = w['start'] - buf[-1]['end']
                dur = w['end'] - buf[0]['start']
                buf_pan = compute_span_pan(pan_10hz, buf[0]['start'], buf[-1]['end'])
                pan_shift = abs(w['pan'] - buf_pan) if pan_10hz else 0.0
                next_is_suffix = bool(re.match(r'^(?:た|だ|です|ます|ました|ません|て|で|に|を|の|が|は|と|も|か|ね|よ|な|し|ば|たら|なら|パーセント|%|％)[。！？、…]*$', w['text']))
                prev_is_incomplete = bool(re.search(r'(?:を|に|が|は|と|の|へ|で|まし|てい|され|いたし|ござい)$', norm(cur_text)))
                if not next_is_suffix and not prev_is_incomplete:
                    if (
                        (gap > 0.70 and cur_len >= 6)
                        or (dur > 9.0 and gap > 0.25 and cur_len >= 10)
                        or (gap >= 0.16 and cur_len >= 5 and pan_shift >= 0.13)
                    ):
                        flush_buf()
            buf.append(w)
            cur_text_after = ''.join(x['text'] for x in buf)
            after_len = len(norm(cur_text_after))
            stripped_w = w['text'].rstrip('」』）) ')
            if stripped_w.rstrip('…').endswith(('。', '！', '？', '!', '?')) and after_len >= 5:
                if idx_w + 1 < len(words) and re.match(r'^(?:です|ます|ました|た|ね|よ|か)[。！？…]*$', words[idx_w + 1]['text']):
                    pass
                else:
                    flush_buf()
            elif stripped_w.endswith(('…', '‥')) and after_len >= 5 and idx_w + 1 < len(words):
                nw = words[idx_w + 1]
                ngap = nw['start'] - w['end']
                b_pan = compute_span_pan(pan_10hz, buf[0]['start'], buf[-1]['end'])
                n_shift = abs(nw['pan'] - b_pan) if pan_10hz else 0.0
                n_suffix = bool(re.match(r'^(?:た|だ|です|ます|ました|ません|て|で|に|を|の|が|は|と|も|か|ね|よ|な|し|ば|たら|なら)[。！？、…]*$', nw['text']))
                if not n_suffix and (ngap >= 0.30 or (ngap >= 0.14 and n_shift >= 0.10)):
                    flush_buf()
            elif stripped_w.endswith('、') and after_len >= 11 and idx_w + 1 < len(words):
                nw = words[idx_w + 1]
                ngap = nw['start'] - w['end']
                b_pan = compute_span_pan(pan_10hz, buf[0]['start'], buf[-1]['end'])
                n_shift = abs(nw['pan'] - b_pan) if pan_10hz else 0.0
                if ngap >= 0.42 or (ngap >= 0.18 and n_shift >= 0.08):
                    flush_buf()
        flush_buf()
        if buf:
            rem_text = ''.join(w['text'] for w in buf)
            if not is_nonverbal_only(rem_text):
                if len(rows) > base_idx:
                    rows[-1]['ja'] += rem_text
                    rows[-1]['end'] = max(rows[-1]['end'], buf[-1]['end'])
                else:
                    c_pan = compute_span_pan(pan_10hz, chunk['start'], chunk['end'])
                    rows.append(dict(
                        start=chunk['start'], end=chunk['end'], ja=ja_text,
                        pan=round(c_pan, 3), pos=pan_to_pos_label(c_pan),
                    ))

    merged = []
    for r in rows:
        n_txt = norm(r['ja'])
        starts_with_suffix = bool(re.match(r'^(?:た|だ|です|ます|ました|ません|でございます)[。！？、…]*$', r['ja'].strip()))
        if merged and 0 <= r['start'] - merged[-1]['end'] < 1.5 and r['end'] - merged[-1]['start'] <= 12 and (starts_with_suffix or len(n_txt) < 4):
            merged[-1]['ja'] = merged[-1]['ja'].rstrip('。、 ') + r['ja']
            merged[-1]['end'] = max(merged[-1]['end'], r['end'])
        else:
            merged.append(r)

    final_rows = []
    i = 0
    while i < len(merged):
        r = merged[i]
        n_txt = norm(r['ja'])
        if len(n_txt) <= 3 and i + 1 < len(merged) and 0 <= merged[i + 1]['start'] - r['end'] < 1.5 and merged[i + 1]['end'] - r['start'] <= 12:
            merged[i + 1]['ja'] = r['ja'] + merged[i + 1]['ja']
            merged[i + 1]['start'] = r['start']
            i += 1
            continue
        final_rows.append(r)
        i += 1
    return dedup_boundary_echo(final_rows)


def run_align_batch(jobs):
    if not jobs:
        return
    import torch, soundfile as sf
    from qwen_asr import Qwen3ForcedAligner

    paths = json.loads((ROOT / 'models.json').read_text(encoding='utf-8'))
    model = Qwen3ForcedAligner.from_pretrained(
        paths['Qwen/Qwen3-ForcedAligner-0.6B'], dtype=torch.bfloat16, device_map='cuda:0'
    )
    try:
        for job in jobs:
            out_dir = Path(job['out_dir'])
            stem = job['stem']
            chunks = json.loads((out_dir / 'dual_asr_arbitrated.json').read_text(encoding='utf-8'))
            audio, sr = sf.read(str(job['audio']), dtype='float32')
            pan_10hz = job['meta'].get('stereo_pan_10hz') or []
            t0 = time.monotonic()
            cues = align_chunks_to_cues(model, audio, sr, chunks, 'ja_golden', pan_10hz=pan_10hz)
            if not cues and chunks:
                cues = [dict(start=chunks[0]['start'], end=chunks[-1]['end'], ja=chunks[0]['ja_golden'] or '…')]
            elapsed = round(time.monotonic() - t0, 2)
            atomic_write_json(out_dir / 'aligned_cues.json', cues)
            write_vtt(out_dir / f'{stem}.ja.vtt', cues, 'ja')
            atomic_write_text(out_dir / f'{stem}.ja.txt', '\n'.join(r['ja'] for r in cues) + '\n')
            print(f'ALIGN_DONE [{stem}]: {len(cues)} cues in {elapsed}s', flush=True)
            if job.get('batch_mode'):
                print(f"BATCH_TRACK_STAGE_DONE {job['index']} align {stem}", flush=True)
    finally:
        del model
        torch.cuda.empty_cache()


def build_merged_glossary(meta, track_guide, max_items=30):
    merged = {item['ja']: (item['zh'], item.get('note', '')) for item in meta.get('base_glossary', [])[:max_items]}
    norm_keys = {ja.lower() for ja in merged}
    source_terms = track_guide.get('album_shared_terms') or track_guide.get('discovered_terms', [])
    for item in source_terms:
        ja = item['ja'].strip()
        zh = item['zh'].strip()
        note = item.get('note', '').strip()[:16]
        if ja and ja.lower() not in norm_keys and len(merged) < max_items:
            merged[ja] = (zh, note)
            norm_keys.add(ja.lower())
    return merged


def _anchor_cues_one_track(stem, out_dir, meta, port):
    """Step 4a: Use Qwen3-14B on aligned_cues.json to generate a strict 1-to-1 semantic Chinese anchor per cue."""
    import requests
    out_dir = Path(out_dir)
    cues_in = json.loads((out_dir / 'aligned_cues.json').read_text(encoding='utf-8'))
    if not cues_in:
        return
    if all(r.get('zh_anchor') for r in cues_in):
        return

    track_guide = json.loads((out_dir / 'track_guide.json').read_text(encoding='utf-8'))
    merged_gloss = build_merged_glossary(meta, track_guide, max_items=30)
    gloss_str = ', '.join(f'{ja}->{zh}' for ja, (zh, _) in merged_gloss.items())
    shared_fixes = track_guide.get('album_shared_fixes') or track_guide.get('context_homophone_fixes', [])
    fixes_str = '；'.join(
        f"『{fx['wrong_ja']}/{fx['correct_ja']}』在此场景下意为「{fx['zh_meaning']}」"
        for fx in shared_fixes
    ) or '按场景上下文判断'
    self_refs = track_guide.get('album_self_references') or track_guide.get('character_self_reference_ja', '')
    has_stereo = any(r.get('pos') in ('左声道', '右声道') for r in cues_in)

    album_bg = ''
    if track_guide.get('album_core_summary'):
        album_bg = (
            f"【全作全局背景与核心设定（继承自开篇轨）】\n"
            f"- 全剧核心脉络：{track_guide.get('album_core_summary')}\n"
            f"- 全局人物身份与处境约束：{track_guide.get('album_core_scene')}\n"
            f"- 角色基调与情感约束：{track_guide.get('album_core_tone')}\n"
        )

    anchor_sys = (
        '你是一位精通日语语法与声优台本的资深翻译专家。请将已切分好的日文字幕分句（aligned cues）逐条1对1直译为准确、忠实的简体中文语义锚点（zh_anchor）。\n'
        + album_bg +
        f"【全轨剧情】{track_guide['story_summary_zh']}\n"
        f"【场景身份与核心动作】{track_guide['scene_constraints_zh']}\n"
        f"【角色语气与敬语反差】{track_guide.get('character_tone_zh', '')}（自称：{self_refs}）\n"
        f'【统一术语表】{gloss_str}\n'
        f'【本轨场景同音/多义词消歧约束】{fixes_str}\n\n'
        '【必须严格遵守的通用日语翻译准则】\n'
        '1. **严格1对1对应当前分句**：每个 `id` 的 `zh_anchor` 只能翻译该 `id` 对应的 `ja` 内容！绝对禁止把前一句或后一句合并、提前或重复写进当前 `id` 中！\n'
        '2. **日语零主语与敬语主客体准则（严禁颠倒动作方向）**：\n'
        '   - **尊敬语（动作发出者 = 对方/尊长/您）**：凡是 `お/ご + 動詞 + になる / なさる / なられる` 或尊敬助动词 `～れる/られる`，动作发出者必定是**对方（主人/您）自己**，必须译为「您（自己）…」，严禁译成使役句「让您…」！\n'
        '   - **谦让语与授受关系**：`お/ご + 動詞 + する / いたす / させていただく` 的动作发出者是**说话人自己（我）**；`～てくださる / ～てくれる` 是**对方（您）为我**做某事；`～てあげる / ～てさしあげる` 是**我为对方（您）**做某事。\n'
        '   - **双面语气切换与第一人称补全**：音声角色常在「表面对对方说恭敬敬语（`です/ます/でございます`）」与「背地里小声用常体/俚语独白或吐槽」之间切换。省略主语的句子请严格依据前后文语气推断主语是「我」还是「您」，严格区分陈述句与疑问句。\n'
        '3. **双声道方位（pos）与多人对话换人识别**：\n'
        '   - 若输入包含声道方位 `pos`（`左声道`/`居中`/`右声道`），当相邻分句的声道方位发生切换或对话立场改变时，请按不同角色的对话衔接人称。\n'
        '4. **结合场景消歧多义词与俗语**：请严格参考上方的【统一术语表】与【本轨场景同音/多义词消歧约束】，确保多义词、行业俗语和专有术语符合当前场景，不产生字面误译或偏离原意的改写。'
    )

    batch_sz = 12
    batches = [(k, cues_in[k:k + batch_sz]) for k in range(0, len(cues_in), batch_sz)]

    def _fmt_ctx_line(idx_1based, r):
        pos_str = f"|{r['pos']}" if has_stereo and r.get('pos') else ''
        return f"[{idx_1based}{pos_str}] {r['ja']}"

    def _anchor_sub_batch(b_idx, sub):
        ctx_before = [
            _fmt_ctx_line(b_idx - len(cues_in[max(0, b_idx - 3):b_idx]) + idx + 1, r)
            for idx, r in enumerate(cues_in[max(0, b_idx - 3):b_idx])
        ]
        ctx_after = [
            _fmt_ctx_line(b_idx + len(sub) + idx + 1, r)
            for idx, r in enumerate(cues_in[b_idx + len(sub):min(len(cues_in), b_idx + len(sub) + 3)])
        ]
        inp = []
        for k, r in enumerate(sub):
            item_in = {'id': b_idx + k + 1, 'ja': compress_extreme_repeats(r['ja'])}
            if has_stereo and r.get('pos'):
                item_in['pos'] = r['pos']
            inp.append(item_in)
        schema = {
            'type': 'object',
            'properties': {
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'id': {'type': 'integer'},
                            'zh_anchor': {'type': 'string'},
                        },
                        'required': ['id', 'zh_anchor'],
                        'additionalProperties': False,
                    },
                    'minItems': len(sub),
                    'maxItems': len(sub),
                },
            },
            'required': ['items'],
            'additionalProperties': False,
        }
        ctx_str = ''
        if ctx_before:
            ctx_str += '【前文参考（切勿翻译进当前分句）】\n' + '\n'.join(ctx_before) + '\n\n'
        if ctx_after:
            ctx_str += '【后文参考（切勿提前翻译进当前分句）】\n' + '\n'.join(ctx_after) + '\n\n'
        user_msg = (
            ctx_str
            + '请对以下每个分句 id 严格1对1输出仅对应本句 ja 的中文直译 zh_anchor：/no_think\n'
            + json.dumps(inp, ensure_ascii=False)
        )
        expected_ids = [x['id'] for x in inp]
        last_err = None
        for temp_try in (0.1, 0.25, 0.4):
            try:
                resp = requests.post(
                    f'http://127.0.0.1:{port}/v1/chat/completions',
                    json={
                        'messages': [
                            {'role': 'system', 'content': anchor_sys},
                            {'role': 'user', 'content': user_msg},
                        ],
                        'temperature': temp_try,
                        'max_tokens': 3000,
                        'frequency_penalty': 0.15,
                        'repeat_penalty': 1.12,
                        'chat_template_kwargs': {'enable_thinking': False},
                        'response_format': {'type': 'json_schema', 'json_schema': {'name': 'zh_anc', 'strict': True, 'schema': schema}},
                    },
                    timeout=180,
                )
                resp.raise_for_status()
                if resp.json()['choices'][0].get('finish_reason') == 'length':
                    raise ValueError('Anchor generation truncated')
                txt = resp.json()['choices'][0]['message']['content'].strip()
                if '</think>' in txt:
                    txt = txt.split('</think>', 1)[1].strip()
                arr = json.loads(txt)['items']
                if [x.get('id') for x in arr] != expected_ids or any(not (x.get('zh_anchor') or '').strip() for x in arr):
                    raise ValueError('Anchor IDs mismatch or empty zh_anchor')
                return arr
            except Exception as e:
                last_err = e
        if len(sub) > 1:
            mid = len(sub) // 2
            return _anchor_sub_batch(b_idx, sub[:mid]) + _anchor_sub_batch(b_idx + mid, sub[mid:])
        return [{'id': b_idx + 1, 'zh_anchor': sub[0]['ja']}]

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=DEFAULT_PARALLEL) as ex:
        for _, arr in ex.map(lambda it: (it[0], _anchor_sub_batch(it[0], it[1])), batches):
            for obj in arr:
                cues_in[obj['id'] - 1]['zh_anchor'] = obj['zh_anchor'].strip()
    atomic_write_json(out_dir / 'aligned_cues.json', cues_in)
    print(f'ANCHOR_14B_DONE [{stem}] in {round(time.monotonic() - t0, 2)}s ({len(cues_in)} cues)', flush=True)


def _translate_one_track(stem, out_dir, meta, port):
    import requests
    out_dir = Path(out_dir)
    track_guide = json.loads((out_dir / 'track_guide.json').read_text(encoding='utf-8'))
    merged_gloss = build_merged_glossary(meta, track_guide, max_items=30)

    char_keys = {
        ja for ja, (_, note) in merged_gloss.items()
        if any(k in note for k in ('称呼', '名称', '角色', '自称'))
    }
    terms_json = {
        'cvs': [],
        'characters': [{'ja': ja, 'zh': zh} for ja, (zh, _) in merged_gloss.items() if ja in char_keys],
        'terms': [{'ja': ja, 'zh': zh} for ja, (zh, _) in merged_gloss.items() if ja not in char_keys],
    }

    cues_in = json.loads((out_dir / 'aligned_cues.json').read_text(encoding='utf-8'))
    has_stereo = any(r.get('pos') in ('左声道', '右声道') for r in cues_in)
    batch_sz = 12
    batches = [(k, cues_in[k:k + batch_sz]) for k in range(0, len(cues_in), batch_sz)]

    def _translate_sub_batch(b_idx, sub):
        ctx_before = [
            f"[{b_idx - len(cues_in[max(0, b_idx - 2):b_idx]) + idx + 1}] {r['ja']} -> {r.get('zh_anchor', '')}"
            for idx, r in enumerate(cues_in[max(0, b_idx - 2):b_idx])
        ]
        ctx_after = [
            f"[{b_idx + len(sub) + idx + 1}] {r['ja']} -> {r.get('zh_anchor', '')}"
            for idx, r in enumerate(cues_in[b_idx + len(sub):min(len(cues_in), b_idx + len(sub) + 2)])
        ]
        transcript_arr = []
        for k, r in enumerate(sub):
            row_t = {
                'id': b_idx + k + 1,
                'text': compress_extreme_repeats(r['ja']),
                'ref_zh': r.get('zh_anchor', ''),
                'start': round(r['start'] * 1000),
                'end': round(r['end'] * 1000),
            }
            if has_stereo and r.get('pos'):
                row_t['pos'] = r['pos']
            transcript_arr.append(row_t)

        track_label = meta.get('track_title') or stem
        ctx_str = ''
        if ctx_before:
            ctx_str += '【前文只读上下文（切勿输出到结果中）】\n' + '\n'.join(ctx_before) + '\n\n'
        if ctx_after:
            ctx_str += '【后文只读上下文（切勿输出到结果中）】\n' + '\n'.join(ctx_after) + '\n\n'

        album_bg = ''
        if track_guide.get('album_core_summary'):
            album_bg = (
                f"【全作全局背景与核心设定（继承自开篇轨）】\n"
                f"- 全剧核心脉络：{track_guide.get('album_core_summary')}\n"
                f"- 全局人物身份与处境约束：{track_guide.get('album_core_scene')}\n"
                f"- 角色基调与情感约束：{track_guide.get('album_core_tone')}\n\n"
            )

        prompt = (
            '将以下日语音声逐字稿润色翻译成自然、准确、忠实的简体中文。\n\n'
            f'音轨：{track_label}\n'
            + album_bg +
            f"全轨剧情脉络：{track_guide['story_summary_zh']}\n"
            f"场景与行动约束：{track_guide['scene_constraints_zh']}\n"
            f"角色语气与敬语反差：{track_guide['character_tone_zh']}（自称：{track_guide.get('album_self_references') or track_guide.get('character_self_reference_ja', '')}）\n\n"
            '术语表（请严格统一使用zh栏位的译名，角色名字与称呼全轨必须100%统一）：\n'
            + json.dumps(terms_json, ensure_ascii=False, indent=2) + '\n\n'
            + ctx_str +
            '【核心忠实度与1对1对齐铁律（严禁违反）】\n'
            '1. **严格单句1对1输出**：每个输出项的 `ids` 只能包含**1个**输入 `id`（即 `"ids": [id]`），且该条 `text` 只能翻译该 `id` 对应的 `text`（日文）和 `ref_zh`！严禁合并相邻分句，严禁把前一句或后一句的内容重复或串位到当前 `id`！\n'
            '2. **严格锚定 `ref_zh` 的主客体方向、人称、数字与词义（严禁反向改写）**：\n'
            '   - 严格保持 `ref_zh` 中的动作主客体方向（尊敬语代表对方/尊长的动作，谦让语代表说话人自己的动作，严禁把对方的动作改写成「让您…」或把自己的话改写成第二人称）。\n'
            '   - 严格保持陈述句与疑问句的语气一致，严禁篡改 `ref_zh` 中的数量、倍数或专有名词含义。\n'
            '3. 输出的 text 字段只输出最终中文译文，译文内如需引号请用「」或『』，切勿使用半角双引号。\n\n'
            '输出JSON数组格式（每条仅对应1个id）：\n'
            '[{"ids": [<id>], "input": "<原始日文>", "glossary": {"<日文>": "<简体中文>"}, "text": "<简体中文>", "start": <start_ms>, "end": <end_ms>}]\n\n'
            '待翻译逐字稿：\n'
            + json.dumps(transcript_arr, ensure_ascii=False, indent=2)
            + '\n/no_think'
        )
        last_err = None
        for temp_try in (0.15, 0.3, 0.45):
            try:
                payload = {
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': temp_try,
                    'top_p': 0.85,
                    'max_tokens': 4096,
                    'frequency_penalty': 0.15,
                    'repeat_penalty': 1.12,
                    'chat_template_kwargs': {'enable_thinking': False},
                }
                resp = requests.post(f'http://127.0.0.1:{port}/v1/chat/completions', json=payload, timeout=180)
                resp.raise_for_status()
                if resp.json()['choices'][0].get('finish_reason') == 'length':
                    raise ValueError('Translation generation truncated')
                content = resp.json()['choices'][0]['message']['content'].strip()
                if '</think>' in content:
                    content = content.split('</think>', 1)[1].strip()
                content = re.sub(r'^```(?:json)?\s*|\s*```$', '', content.strip())
                s_i, e_i = content.find('['), content.rfind(']')
                if s_i != -1 and e_i != -1:
                    content = content[s_i:e_i + 1]
                return parse_translation(json.loads(content), sub, b_idx)
            except Exception as e:
                last_err = e
        if len(sub) > 1:
            mid = len(sub) // 2
            print(f'TRANSLATE_BISECT [{stem}] splitting {len(sub)} cues -> {mid}+{len(sub) - mid} ({last_err})', flush=True)
            return _translate_sub_batch(b_idx, sub[:mid]) + _translate_sub_batch(b_idx + mid, sub[mid:])
        fallback_zh = sub[0].get('zh_anchor') or sub[0]['ja']
        print(f'TRANSLATE_SINGLE_FALLBACK [{stem}] cue={b_idx + 1} ({last_err})', flush=True)
        return [dict(ids=[b_idx + 1], start=sub[0]['start'], end=sub[0]['end'], ja=sub[0]['ja'], zh=fallback_zh, glossary={})]

    def proc_batch(item):
        b_idx, sub = item
        return b_idx, _translate_sub_batch(b_idx, sub)

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=DEFAULT_PARALLEL) as ex:
        res = sorted(ex.map(proc_batch, batches), key=lambda x: x[0])
    elapsed = round(time.monotonic() - t0, 2)

    vtt_rows = [row for _, arr in res for row in arr]
    validate_cues(vtt_rows)
    atomic_write_json(out_dir / 'translated_cues.json', vtt_rows)
    write_vtt(out_dir / f'{stem}.vtt', vtt_rows, 'zh')
    atomic_write_text(out_dir / f'{stem}.zh.txt', '\n'.join(r['zh'] for r in vtt_rows) + '\n')
    print(f'TRANSLATE_DONE [{stem}] in {elapsed}s ({len(vtt_rows)} cues)', flush=True)


def run_translate_batch(jobs, port=8095):
    if not jobs:
        return
    paths = json.loads((ROOT / 'models.json').read_text(encoding='utf-8'))
    # Step 4a: Generate 1-to-1 cue-level semantic Chinese anchors (zh_anchor) with Qwen3-14B
    srv_14b = start_llama_server(paths['qwen3-14b'], Path(jobs[0]['out_dir']), port, n_parallel=DEFAULT_PARALLEL, ctx_size=40960, log_name='qwen3-14b-anchor.log')
    try:
        for job in jobs:
            _anchor_cues_one_track(job['stem'], Path(job['out_dir']), job['meta'], port)
    finally:
        stop_server(srv_14b)

    # Step 4b: Polish cue-level translations with ASMR-Qwen3.5-9B-Echo anchored by zh_anchor
    srv_9b = start_llama_server(paths['asmr-qwen35-9b-echo'], Path(jobs[0]['out_dir']), port, n_parallel=DEFAULT_PARALLEL, ctx_size=32768, log_name='asmr-9b.log')
    try:
        for job in jobs:
            out_dir = Path(job['out_dir'])
            _translate_one_track(job['stem'], out_dir, job['meta'], port)
            if job.get('batch_mode'):
                print(f"BATCH_TRACK_STAGE_DONE {job['index']} translate-zh {job['stem']}", flush=True)
    finally:
        stop_server(srv_9b)


def _spawn_stage_subprocess(extra_args, stage_name):
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(ROOT / 'align-packages') + os.pathsep + str(Path(__file__).resolve().parent),
        HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
        USE_TF='0',
        USE_FLAX='0',
    )
    cmd = [sys.executable, '-u', __file__, *extra_args, '--stage', stage_name]
    p_st = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for ln in p_st.stdout:
        print(ln, end='', flush=True)
    if p_st.wait() != 0:
        raise RuntimeError(f'Stage {stage_name} failed')


def run_batch_orchestrator(batch_json_path, stage_only=None, force=False):
    raw_jobs = json.loads(Path(batch_json_path).read_text(encoding='utf-8'))
    jobs = []
    for idx, item in enumerate(raw_jobs):
        audio_path = Path(item['audio'])
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        stem = item.get('stem') or audio_path.name
        out_dir = Path(item['out_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = load_metadata(item.get('metadata_json', ''))
        jobs.append(dict(
            index=item.get('index', idx),
            audio=str(audio_path),
            stem=stem,
            out_dir=str(out_dir),
            metadata_json=item.get('metadata_json', ''),
            meta=meta,
            batch_mode=True,
        ))

    if stage_only == 'dual-asr':
        run_dual_asr_batch(jobs)
        return
    if stage_only == 'arbitrate-ja':
        run_arbitrate_batch(jobs)
        return
    if stage_only == 'align':
        run_align_batch(jobs)
        return
    if stage_only == 'translate-zh':
        run_translate_batch(jobs)
        return

    recomputed = {j['index']: force for j in jobs}
    for st in STAGES:
        todo = []
        for j in jobs:
            if not recomputed[j['index']] and is_stage_done(j['out_dir'], st, j['stem']):
                print(f"=== SKIPPING COMPLETED STAGE: {st} [{j['stem']}] ===", flush=True)
                print(f"BATCH_TRACK_STAGE_DONE {j['index']} {st} {j['stem']}", flush=True)
                continue
            recomputed[j['index']] = True
            todo.append(j)
        if not todo:
            continue
        print(f'=== RUNNING BATCH STAGE: {st} ({len(todo)}/{len(jobs)} tracks) ===', flush=True)
        stage_manifest = Path(batch_json_path).with_name(f'.stage_{st}_manifest.json')
        atomic_write_json(stage_manifest, [
            {k: j[k] for k in ('index', 'audio', 'stem', 'out_dir', 'metadata_json')}
            for j in todo
        ])
        try:
            _spawn_stage_subprocess(['--batch-json', str(stage_manifest)], st)
        finally:
            if stage_manifest.exists():
                stage_manifest.unlink()
    print('ALL_BATCH_STAGES_COMPLETE', len(jobs), flush=True)


def main():
    p = argparse.ArgumentParser(description='Japanese Audio Subtitles 4-Stage Pipeline')
    p.add_argument('--audio', default='', help='Single input 16kHz mono FLAC file')
    p.add_argument('--stem', default='', help='Output filename prefix (e.g. 01.wav)')
    p.add_argument('--out-dir', default='/content/subtitle_out', help='Output working directory')
    p.add_argument('--metadata-json', default='', help='Optional JSON file with work_title, synopsis_ja, base_glossary')
    p.add_argument('--batch-json', default='', help='JSON manifest for multi-track stage-wise batch execution')
    p.add_argument('--stage', choices=STAGES)
    p.add_argument('--force', action='store_true', help='Ignore existing stage checkpoints in --out-dir')
    a = p.parse_args()

    if a.batch_json:
        run_batch_orchestrator(a.batch_json, stage_only=a.stage, force=a.force)
        return

    if not a.audio:
        p.error('Specify either --audio or --batch-json')
    audio_path = Path(a.audio)
    stem = a.stem or audio_path.name
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(a.metadata_json)
    single_job = [dict(index=0, audio=str(audio_path), stem=stem, out_dir=str(out_dir), meta=meta, batch_mode=False)]

    if a.stage == 'dual-asr':
        run_dual_asr_batch(single_job)
        return
    if a.stage == 'arbitrate-ja':
        run_arbitrate_batch(single_job)
        return
    if a.stage == 'align':
        run_align_batch(single_job)
        return
    if a.stage == 'translate-zh':
        run_translate_batch(single_job)
        return

    recompute = a.force
    for st in STAGES:
        if not recompute and is_stage_done(out_dir, st, stem):
            print(f'=== SKIPPING COMPLETED STAGE: {st} [{stem}] ===', flush=True)
            continue
        recompute = True
        print(f'=== RUNNING STAGE: {st} [{stem}] ===', flush=True)
        extra = ['--audio', str(audio_path), '--stem', stem, '--out-dir', str(out_dir)]
        if a.metadata_json:
            extra.extend(['--metadata-json', a.metadata_json])
        _spawn_stage_subprocess(extra, st)
    print(f'ALL_STAGES_COMPLETE -> {out_dir / (stem + ".vtt")}', flush=True)


if __name__ == '__main__':
    main()
