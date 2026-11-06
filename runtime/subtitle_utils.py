"""Pure-Python utilities for WebVTT formatting, punctuation projection, and LLM output parsing."""
import difflib, json, math, os, re, tempfile, unicodedata
from collections import Counter
from pathlib import Path


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
            f.flush()
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path, data):
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def read_nonempty_json(path):
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return None
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
        return data if data else None
    except (ValueError, OSError):
        return None


def sanitize_name(name, fallback='item'):
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', '_', str(name).strip())
    cleaned = cleaned.strip('. ')
    return cleaned[:80] or fallback


def format_timestamp(seconds):
    ms = max(0, round(float(seconds) * 1000))
    return f'{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}'


def write_vtt(path, rows, field):
    blocks = [
        f"{format_timestamp(r['start'])} --> {format_timestamp(r['end'])}\n{' '.join(str(r[field]).split())}"
        for r in rows
    ]
    atomic_write_text(path, 'WEBVTT\n\n' + '\n\n'.join(blocks) + '\n')


def validate_vtt(path):
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        raise ValueError(f'Missing or empty WebVTT file: {p}')
    text = p.read_text(encoding='utf-8-sig')
    if not text.startswith('WEBVTT') or not re.search(r'\d+:\d{2}\.\d{3}\s+-->\s+\d+', text):
        raise ValueError(f'Invalid WebVTT structure: {p}')


def validate_cues(rows):
    if not rows:
        raise ValueError('No subtitle cues produced')
    previous = -1.0
    for r in rows:
        s, e = r['start'], r['end']
        if not all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) for t in (s, e)):
            raise ValueError('Non-finite or non-numeric cue timestamp')
        if s < 0 or e <= s or s < previous - 0.05:
            raise ValueError('Invalid or overlapping cue timestamp')
        previous = e


def is_nonverbal_only(text):
    cleaned = re.sub(r'[\s、。！？…～〜ー・♡💛♪「」『』（）()!?\.,％%]', '', text or '')
    if not cleaned:
        return True
    nonverbal_pat = r'^(?:ん|っ|ぁ|あ|ふ|ぅ|う|ちゅ|じゅ|ぢゅ|れ|ちゃ|く|ぽ|ぷ|ぶ|る|ろ|む|は|ひ|え|ぇ|お|ぉ|ぐ|ぬ)+$'
    if re.fullmatch(nonverbal_pat, cleaned) and len(cleaned) >= 3:
        return True
    return False


def compress_extreme_repeats(ja_text):
    """Collapse runaway 5+ consecutive ASR/onomatopoeia loops while preserving normal 1-4x vocalizations."""
    s = (ja_text or '').strip()
    s = re.sub(r'([^\s、。！？…]{1,12}?)(?:[、。…～〜ー・♡💛♪!?\s]*\1){4,}', r'\1、\1、\1… ', s)
    if len(s) > 240:
        s = s[:240].rstrip('、。 ') + '…'
    return s.strip()


def cap_guide_lines(lines, max_chars=12000):
    joined = '\n'.join(lines)
    if len(joined) <= max_chars or len(lines) <= 90:
        return lines
    head = lines[:40]
    tail = lines[-35:]
    middle_pool = lines[40:-35]
    budget = max(10, min(55, (max_chars - sum(len(x) + 1 for x in head + tail)) // 90))
    step = max(1, len(middle_pool) / budget)
    mid = [middle_pool[min(len(middle_pool) - 1, int(i * step))] for i in range(budget)]
    return head + ['...（中略）...'] + mid + ['...（中略）...'] + tail


def project_punctuation(ja_text, units):
    """Project original punctuation, decimals, %, and quotes onto Qwen3-ForcedAligner word units."""
    if not units:
        return []
    words = []
    pos = 0
    n = len(ja_text)
    norm_text = unicodedata.normalize('NFKC', ja_text)
    use_norm = (len(norm_text) == len(ja_text))
    src_match_str = norm_text.lower() if use_norm else ja_text.lower()

    pending_prefix = ''
    for u in units:
        utxt = str(u.text)
        u_start = float(u.start_time)
        u_end = float(u.end_time)
        u_norm = unicodedata.normalize('NFKC', utxt).lower() if use_norm else utxt.lower()

        while pos < n and ja_text[pos].isspace():
            pos += 1

        idx = src_match_str.find(u_norm, pos, min(n, pos + 24))
        if idx != -1:
            skipped = ja_text[pos:idx]
            orig_token = ja_text[idx:idx + len(utxt)]
            pos = idx + len(utxt)
        else:
            raise ValueError('Aligner token cannot be projected onto source')

        if skipped:
            m_open = re.search(r'[「『（(\[【]+$', skipped)
            if m_open:
                back_part = skipped[:m_open.start()]
                front_part = skipped[m_open.start():]
            else:
                back_part = skipped
                front_part = ''

            if back_part:
                if words:
                    if back_part.strip() in ('.', '．') and words[-1]['text'][-1:].isdigit() and orig_token[:1].isdigit():
                        words[-1]['text'] += back_part.strip() + pending_prefix + orig_token
                        words[-1]['end'] = max(words[-1]['end'], u_end)
                        pending_prefix = front_part
                        continue
                    words[-1]['text'] += back_part
                else:
                    pending_prefix += back_part
            pending_prefix += front_part

        words.append(dict(
            text=pending_prefix + orig_token,
            start=u_start,
            end=u_end,
        ))
        pending_prefix = ''

    tail = ja_text[pos:].strip()
    if tail and words:
        words[-1]['text'] += tail
    if ''.join(''.join(w['text'] for w in words).split()) != ''.join(ja_text.split()):
        raise ValueError('Projection changed source text')
    return words


def compute_span_pan(pan_10hz, start_sec, end_sec):
    """Return mean stereo balance in [-1.0..+1.0] (-1.0=Left, +1.0=Right, 0.0=Center) over [start_sec, end_sec]."""
    if not pan_10hz:
        return 0.0
    i0 = max(0, min(len(pan_10hz) - 1, int(start_sec * 10)))
    i1 = max(i0 + 1, min(len(pan_10hz), int(math.ceil(end_sec * 10))))
    vals = [pan_10hz[i] / 100.0 for i in range(i0, i1) if pan_10hz[i] != 0]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def pan_to_pos_label(pan_val):
    """Map stereo balance [-1.0..+1.0] to a concise spatial speaker label."""
    if pan_val <= -0.11:
        return '左声道'
    if pan_val >= 0.11:
        return '右声道'
    return '居中'


def analyze_asr_pair(ja_anime, ja_turbo):
    """Extract cross-stream diffs between Stream A (Anime-Whisper) and Stream B (Whisper-Turbo)."""
    a = (ja_anime or '').strip()
    b = (ja_turbo or '').strip()
    hints = []

    halluc_pat = r'ご視聴ありがとうございました|チャンネル登録|ご主人様のおかげで、お前は'
    if a and b and not re.search(halluc_pat, b):
        strip_p = lambda s: re.sub(r'[\s、。！？,.!?…～〜ー・♡💛♪「」『』（）()％%]', '', s)
        ca, cb = strip_p(a), strip_p(b)
        if ca and cb and ca != cb:
            sm = difflib.SequenceMatcher(None, ca, cb)
            diffs = []
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == 'replace':
                    sa, sb = ca[i1:i2], cb[j1:j2]
                    if 1 <= len(sa) <= 10 and 1 <= len(sb) <= 10 and sa != sb:
                        diffs.append(f'A「{sa}」/B「{sb}」')
            if diffs:
                hints.append('AB差異: ' + ', '.join(diffs[:5]))
    return hints


def apply_context_homophone_fixes(ja_text, fixes):
    """Apply scene-verified homophone fixes discovered by Pass 1 (only safe 2+ char spans)."""
    out = ja_text
    for item in (fixes or []):
        w = (item.get('wrong_ja') or '').strip()
        c = (item.get('correct_ja') or '').strip()
        if len(w) >= 2 and len(c) >= 1 and w != c and w in out:
            out = out.replace(w, c)
    return out


def normalize_ja_names(ja_text, shared_terms, fixes=None):
    """Unify katakana/hiragana ASR variants of character names and apply scene-verified homophone fixes."""
    out = apply_context_homophone_fixes(ja_text, fixes)
    for t in (shared_terms or []):
        ja = (t.get('ja') or '').strip()
        zh = (t.get('zh') or '').strip()
        if len(ja) >= 3 and ja.endswith('子') and zh.endswith('子'):
            stem = ja[:-1]
            for var in (stem + 'コ', stem + 'ッコ', unicodedata.normalize('NFKC', stem) + 'コ'):
                out = out.replace(var, ja)
            hira_stem = ''.join(chr(ord(c) - 0x60) if 'ァ' <= c <= 'ン' else c for c in stem)
            if hira_stem != stem:
                out = out.replace(hira_stem + '子', ja).replace(hira_stem + 'こ', ja)
    return out


def parse_arbitration(arr, sub):
    expected = [r['seg_id'] for r in sub]
    if not isinstance(arr, list) or any(
        not isinstance(r, dict) or type(r.get('seg_id')) is not int or not isinstance(r.get('ja_golden'), str)
        for r in arr
    ):
        raise ValueError('Invalid arbitration JSON structure')
    if Counter(r['seg_id'] for r in arr) != Counter(expected):
        raise ValueError('Arbitration segment IDs do not match input batch')

    norm = lambda s: re.sub(r'[\s、。！？,.!?…～〜ー・♡💛♪「」『』（）()％%]', '', s or '')
    by_id = {r['seg_id']: r for r in sub}
    for idx, out_row in enumerate(arr):
        sid = out_row['seg_id']
        src = by_id[sid]
        g_norm = norm(out_row['ja_golden'])
        a_norm = norm(src.get('ja_anime', ''))
        b_norm = norm(src.get('ja_turbo', ''))
        if idx > 0 and len(g_norm) >= 6:
            prev_g = norm(arr[idx - 1]['ja_golden'])
            prev_a = norm(by_id[arr[idx - 1]['seg_id']].get('ja_anime', ''))
            if g_norm == prev_g and a_norm != prev_a:
                raise ValueError(f'Arbitration duplicated adjacent segment at seg_id={sid}')
        if len(a_norm) >= 8 and len(g_norm) >= 6 and not is_nonverbal_only(a_norm):
            sim_a = difflib.SequenceMatcher(None, a_norm, g_norm).ratio()
            sim_b = difflib.SequenceMatcher(None, b_norm, g_norm).ratio() if b_norm else 0.0
            if max(sim_a, sim_b) < 0.25:
                raise ValueError(f'Arbitration output drifted from input at seg_id={sid}')
    return arr


def sanitize_cue_timestamps(rows):
    """Ensure cue timestamps are strictly monotonic and non-overlapping (e > s, s >= prev_e)."""
    out = []
    prev_end = 0.0
    for r in rows:
        s = max(float(r['start']), round(prev_end, 3))
        e = float(r['end'])
        if e <= s:
            e = round(s + 0.25, 3)
        r_clean = dict(r, start=round(s, 3), end=round(e, 3))
        out.append(r_clean)
        prev_end = r_clean['end']
    return out


def dedup_boundary_echo(rows):
    """Remove or merge short boundary-echo duplicate clauses caused by chunk splits."""
    if len(rows) < 2:
        return sanitize_cue_timestamps(rows)
    norm = lambda s: re.sub(r'[\s、。！？,.!?…～〜ー・♡💛♪「」『』（）()％%]', '', s or '')
    cleaned = [rows[0]]
    for cur in rows[1:]:
        prev = cleaned[-1]
        gap = cur['start'] - prev['end']
        p_norm = norm(prev.get('ja', ''))
        c_norm = norm(cur.get('ja', ''))
        if gap < 1.5 and len(c_norm) >= 4 and len(p_norm) >= len(c_norm):
            # Case 1: entire current cue is just a repeat of the tail of previous cue
            if p_norm.endswith(c_norm) and (cur['end'] - cur['start']) <= 2.5:
                prev['end'] = max(prev['end'], cur['end'])
                continue
            # Case 2: tail of prev (4..12 chars) is repeated verbatim at the start of cur
            for k in range(min(12, len(p_norm), len(c_norm) - 1), 3, -1):
                overlap = p_norm[-k:]
                if c_norm.startswith(overlap):
                    raw_ja = cur['ja'].lstrip('…、。 ')
                    if len(raw_ja) > k and norm(raw_ja[:k + 2]).startswith(overlap):
                        trimmed = raw_ja[k:].lstrip('…、。 ')
                        if trimmed and not is_nonverbal_only(trimmed):
                            cur = dict(cur, ja=trimmed)
                        break
        cleaned.append(cur)
    return sanitize_cue_timestamps(cleaned)


def parse_translation(arr, sub, offset):
    if not isinstance(arr, list):
        raise ValueError('Expected a JSON array of translated cues')
    source = {offset + k + 1: r for k, r in enumerate(sub)}
    seen = []
    result = []
    for obj in arr:
        ids = obj.get('ids') if isinstance(obj, dict) else None
        if not isinstance(ids, list) or not ids or any(type(i) is not int or i not in source for i in ids):
            raise ValueError('Missing or out-of-range cue IDs in translation')
        if ids != sorted(ids) or ids != list(range(ids[0], ids[-1] + 1)):
            raise ValueError('Translated cue IDs must be contiguous and ordered')
        text = obj.get('text')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Empty translation text')
        seen.extend(ids)
        result.append(dict(
            ids=ids,
            start=source[ids[0]]['start'],
            end=source[ids[-1]]['end'],
            ja=''.join(source[i]['ja'] for i in ids),
            zh=text.strip(),
            glossary=obj.get('glossary', {}),
        ))
    if Counter(seen) != Counter(source.keys()):
        raise ValueError('Duplicate or missing cue IDs in translation batch')
    result.sort(key=lambda r: r['ids'][0])
    validate_cues(result)
    return result
