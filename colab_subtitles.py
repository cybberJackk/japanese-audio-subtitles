"""Automated Japanese Audio Subtitle CLI:
Local audio folder/file -> 16kHz FLAC -> Google Drive -> Headless Colab L4 -> Local sidecar subtitles -> Auto cleanup & Colab stop.

Requires Python 3.11+ standard library and ffmpeg on PATH (zero third-party pip packages on Windows/macOS/Linux host).
"""
import argparse, html, json, os, re, shutil, signal, subprocess, sys, tarfile, tempfile, time, urllib.request, uuid
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = REPO_ROOT / 'runtime'
sys.path.insert(0, str(RUNTIME_DIR))

from subtitle_utils import atomic_write_json, sanitize_name, validate_vtt  # noqa: E402
from drive_store import (  # noqa: E402
    DriveApiStore, default_app_data_dir, default_windows_token_path, run_installed_app_oauth,
)

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.opus', '.wma', '.aiff', '.aif', '.mp4', '.webm'}
SKIP_DIRS = {'.autosub', '.git', 'subtitles', '__pycache__', 'venv', '.venv', 'work', 'japanese-audio-subtitles'}
RUNTIME_FILES = [
    '__init__.py',
    'subtitle_utils.py',
    'drive_store.py',
    'bootstrap.py',
    'pipeline.py',
    'worker.py',
    'telemetry.py',
]


def sidecar_paths(audio_path, vtt_only=False, zh_only=False):
    audio_path = Path(audio_path)
    out = {'zh_vtt': audio_path.with_name(audio_path.name + '.vtt')}
    if not zh_only:
        out['ja_vtt'] = audio_path.with_name(audio_path.name + '.ja.vtt')
    if not vtt_only:
        out['zh_txt'] = audio_path.with_name(audio_path.name + '.zh.txt')
        if not zh_only:
            out['ja_txt'] = audio_path.with_name(audio_path.name + '.ja.txt')
    return out


def discover_audio_files(targets, recursive=True):
    found, seen = [], set()

    def _add(p):
        rp = p.resolve()
        if rp not in seen and rp.is_file() and rp.suffix.lower() in AUDIO_EXTENSIONS:
            seen.add(rp)
            found.append(rp)

    def _walk_dir(d):
        for entry in sorted(d.iterdir(), key=lambda x: x.name.lower()):
            if entry.is_dir():
                if recursive and entry.name not in SKIP_DIRS and not entry.name.startswith('.'):
                    _walk_dir(entry)
            elif entry.is_file():
                _add(entry)

    for raw in targets:
        p = Path(raw).resolve()
        if not p.exists():
            raise FileNotFoundError(f'Target path does not exist: {p}')
        if p.is_file():
            _add(p)
        else:
            _walk_dir(p)
    return found


def _load_glossary_file(path):
    p = Path(path)
    if not p.is_file():
        return []
    try:
        raw_g = json.loads(p.read_text(encoding='utf-8'))
        if isinstance(raw_g, dict):
            return [{'ja': k, 'zh': v, 'note': ''} for k, v in raw_g.items()]
        if isinstance(raw_g, list):
            return [x for x in raw_g if isinstance(x, dict) and x.get('ja') and x.get('zh')]
    except Exception:
        pass
    return []


def _merge_glossary_lists(*lists):
    merged = []
    seen = set()
    for lst in lists:
        for item in (lst or []):
            ja = (item.get('ja') or '').strip()
            zh = (item.get('zh') or '').strip()
            if ja and zh and ja.lower() not in seen:
                merged.append({'ja': ja, 'zh': zh, 'note': item.get('note', '')})
                seen.add(ja.lower())
    return merged


def resolve_track_metadata(audio_path, custom_meta_path='', auto_metadata=True, cache_mem=None):
    if cache_mem is None:
        cache_mem = {}
    audio_path = Path(audio_path)

    local_default_gloss = _merge_glossary_lists(
        _load_glossary_file(REPO_ROOT / 'glossary.local.json'),
        _load_glossary_file(default_app_data_dir() / 'default_glossary.json'),
    )

    if custom_meta_path:
        p_custom = Path(custom_meta_path)
        raw_custom = json.loads(p_custom.read_text(encoding='utf-8'))
        if isinstance(raw_custom, list) or (isinstance(raw_custom, dict) and 'work_title' not in raw_custom and 'base_glossary' not in raw_custom):
            meta = {'track_title': audio_path.stem, 'base_glossary': _merge_glossary_lists(_load_glossary_file(p_custom), local_default_gloss)}
        else:
            meta = dict(raw_custom)
            meta.setdefault('track_title', audio_path.stem)
            meta['base_glossary'] = _merge_glossary_lists(meta.get('base_glossary', []), local_default_gloss)
        return meta

    folder_gloss = []
    meta = {'track_title': audio_path.stem}
    for folder in (audio_path.parent, audio_path.parent.parent):
        cand = folder / 'metadata.json'
        if cand.is_file():
            m_data = dict(json.loads(cand.read_text(encoding='utf-8')))
            m_data.setdefault('track_title', audio_path.stem)
            meta.update(m_data)
            break
        gloss = folder / 'glossary.json'
        if gloss.is_file() and not folder_gloss:
            folder_gloss = _load_glossary_file(gloss)

    merged_g = _merge_glossary_lists(meta.get('base_glossary', []), folder_gloss, local_default_gloss)
    if merged_g:
        meta['base_glossary'] = merged_g

    if not auto_metadata:
        return meta

    hook_file = REPO_ROOT / 'metadata_hook.local.py'
    if hook_file.is_file():
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location('metadata_hook_local', hook_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            info = mod.lookup_work_metadata(audio_path, default_app_data_dir(), cache_mem) or {}
            if info.get('work_title') and not meta.get('work_title'):
                meta['work_title'] = info['work_title']
            if info.get('synopsis_ja') and not meta.get('synopsis_ja'):
                meta['synopsis_ja'] = info['synopsis_ja']
        except Exception:
            pass
    return meta


def assign_project_slugs(pending_items):
    """Assign clean, collision-free Google Drive folder paths (<project>/<track_filename>)."""
    used = set()
    for item in pending_items:
        audio = item['audio']
        proj = sanitize_name(audio.parent.name, 'default')
        track_key = sanitize_name(audio.name, 'track')
        if (proj, track_key) in used and audio.parent.parent != audio.parent:
            proj = sanitize_name(f'{audio.parent.parent.name}_{audio.parent.name}', proj)
        suffix = 2
        base_proj = proj
        while (proj, track_key) in used:
            proj = f'{base_proj}_{suffix}'
            suffix += 1
        used.add((proj, track_key))
        item['project_slug'] = proj
        item['track_slug'] = track_key
        item['drive_input_rel'] = f'input/{proj}/{track_key}.flac'
        item['remote_job_rel'] = f'jobs/{proj}/{track_key}'


def convert_to_16k_flac(audio_path, project_slug, track_slug):
    if not shutil.which('ffmpeg'):
        raise RuntimeError('ffmpeg is required on PATH to convert audio to 16kHz mono FLAC')
    audio_path = Path(audio_path)
    cache_dir = default_app_data_dir() / 'cache' / 'flac' / project_slug
    cache_dir.mkdir(parents=True, exist_ok=True)
    flac_path = cache_dir / f'{track_slug}.flac'
    src_mtime = audio_path.stat().st_mtime
    if not flac_path.is_file() or flac_path.stat().st_size == 0 or flac_path.stat().st_mtime < src_mtime:
        tmp = flac_path.with_suffix('.part')
        subprocess.run(
            ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
             '-i', str(audio_path), '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'flac', '-f', 'flac', str(tmp)],
            check=True,
        )
        os.replace(tmp, flac_path)
    return flac_path


def extract_stereo_pan_10hz(audio_path):
    """Extract 10Hz (100ms step) stereo balance in [-100..+100] (-100=Left, +100=Right, 0=Center/Silence)."""
    import struct
    try:
        raw = subprocess.check_output(
            ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
             '-i', str(audio_path), '-vn', '-ac', '2', '-ar', '1000', '-f', 's16le', '-'],
            timeout=60,
        )
        samples = struct.unpack('<' + 'h' * (len(raw) // 2), raw)
        pans = []
        for i in range(0, len(samples) - 199, 200):
            frame = samples[i:i + 200]
            l_sq = sum(frame[j] * frame[j] for j in range(0, 200, 2))
            r_sq = sum(frame[j] * frame[j] for j in range(1, 200, 2))
            l_rms = (l_sq / 100.0) ** 0.5
            r_rms = (r_sq / 100.0) ** 0.5
            if l_rms + r_rms > 120.0:
                pans.append(int(round((r_rms - l_rms) / (l_rms + r_rms) * 100)))
            else:
                pans.append(0)
        return pans
    except Exception:
        return []


class ColabRunner:
    """Thin wrapper over native Linux/macOS or WSL colab CLI."""

    def __init__(self, wsl_distro='Ubuntu-24.04', colab_bin=''):
        self.is_windows = (os.name == 'nt')
        self.wsl_distro = wsl_distro
        if not self.is_windows:
            self.colab_bin = colab_bin or shutil.which('colab') or str(Path.home() / '.local/bin/colab')
        else:
            self.colab_bin = colab_bin or self._detect_wsl_colab()

    def _detect_wsl_colab(self):
        if os.environ.get('JAS_COLAB_BIN'):
            return os.environ['JAS_COLAB_BIN']
        probe = subprocess.run(
            ['wsl', '-d', self.wsl_distro, '--', 'bash', '-lc', 'command -v colab || ls -1 /home/*/.local/bin/colab 2>/dev/null | head -n1'],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=20,
        )
        found = probe.stdout.strip().splitlines()
        if not found or not found[0].strip():
            raise RuntimeError(f'colab CLI not found in WSL distro {self.wsl_distro}')
        return found[0].strip()

    def to_cli_path(self, local_path):
        p = str(Path(local_path).resolve())
        if not self.is_windows:
            return p
        drive = p[0].lower()
        rest = p[2:].replace('\\', '/')
        return f'/mnt/{drive}{rest}'

    def run(self, *args, timeout=240, check=True, retries=4):
        cmd = ['wsl', '-d', self.wsl_distro, '--', self.colab_bin, *args] if self.is_windows else [self.colab_bin, *args]
        last_proc = None
        for attempt in range(retries):
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                if not check:
                    return subprocess.CompletedProcess(cmd, 124, stdout='', stderr=str(exc))
                if attempt + 1 < retries:
                    time.sleep(2.5 * (attempt + 1))
                    continue
                raise
            last_proc = proc
            if proc.returncode == 0 or not check:
                return proc
            combined = (proc.stdout + '\n' + proc.stderr)
            is_transient = any(k in combined for k in ('502 Bad Gateway', '503 Service', '504 Gateway', 'ProxyError', 'ConnectionResetError', 'RemoteDisconnected', 'Read timed out'))
            if is_transient and attempt + 1 < retries:
                time.sleep(2.5 * (attempt + 1))
                continue
            break
        tail = (last_proc.stdout[-1500:] + '\n' + last_proc.stderr[-1500:]).strip()
        raise RuntimeError(f"colab {' '.join(args[:3])} failed (exit {last_proc.returncode}): {tail}")


SPAWN_REMOTE_WORKER_PY = """
import os, subprocess, sys, tarfile
from pathlib import Path

Path('/content/.jas_drive_token.json').chmod(0o600)
code_dir = Path('/content/jas-runtime')
code_dir.mkdir(parents=True, exist_ok=True)
with tarfile.open('/content/jas_bundle.tar') as tar:
    tar.extractall(code_dir, filter='data')
Path('/content/jas_bundle.tar').unlink()

log_f = open('/content/jas_worker.log', 'w', encoding='utf-8')
proc = subprocess.Popen(
    [sys.executable, '-u', '/content/jas-runtime/worker.py',
     '--batch-spec', '/content/jas-runtime/jas_batch_spec.json',
     '--status-file', '/content/jas_worker_status.json'],
    stdout=log_f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    start_new_session=True, close_fds=True,
)
Path('/content/jas_worker.pid').write_text(str(proc.pid), encoding='utf-8')
print('WORKER_SPAWNED', proc.pid)
"""

POLL_REMOTE_WORKER_PY = """
import json, os
from pathlib import Path

offset = int(os.environ.get('JAS_LOG_OFFSET', '0'))
pid_file = Path('/content/jas_worker.pid')
log_file = Path('/content/jas_worker.log')
status_file = Path('/content/jas_worker_status.json')

pid = int(pid_file.read_text().strip()) if pid_file.is_file() else 0
alive = False
if pid > 0:
    try:
        os.kill(pid, 0)
        alive = True
    except OSError:
        alive = False

chunk = ''
next_offset = offset
if log_file.is_file():
    with log_file.open('rb') as f:
        f.seek(offset)
        data = f.read(262144)
        next_offset = offset + len(data)
    chunk = data.decode('utf-8', 'replace')

status = json.loads(status_file.read_text(encoding='utf-8')) if status_file.is_file() else None
print('JAS_POLL_JSON:' + json.dumps({'alive': alive, 'pid': pid, 'next_offset': next_offset, 'chunk': chunk, 'status': status}, ensure_ascii=False))
"""


def save_track_deliverables_locally(store, item, tr_info, args):
    deliverables_meta = tr_info['deliverables']
    targets = sidecar_paths(item['audio'], vtt_only=args.vtt_only, zh_only=args.zh_only)
    staged = {}
    try:
        for key, dst_path in targets.items():
            tmp_dl = dst_path.with_name(f'.{dst_path.name}.{uuid.uuid4().hex[:8]}.downloading')
            store.download_file(deliverables_meta[key]['rel_path'], tmp_dl)
            if key.endswith('_vtt'):
                validate_vtt(tmp_dl)
            elif tmp_dl.stat().st_size == 0:
                raise ValueError(f'Downloaded text file is empty: {tmp_dl}')
            staged[dst_path] = tmp_dl

        for dst_path, tmp_dl in staged.items():
            os.replace(tmp_dl, dst_path)
    finally:
        for tmp_dl in staged.values():
            if tmp_dl.exists():
                tmp_dl.unlink()

    # Verify final local files on disk before cleaning up cloud files
    for key, dst_path in targets.items():
        if not dst_path.is_file() or dst_path.stat().st_size == 0:
            raise IOError(f'Local subtitle missing or empty after write: {dst_path}')
        if key.endswith('_vtt'):
            validate_vtt(dst_path)

    zh_vtt = targets['zh_vtt']
    others = ', '.join(p.name for k, p in targets.items() if k != 'zh_vtt')
    print(f'SAVED_LOCAL {zh_vtt}' + (f' (+ {others})' if others else ''), flush=True)

    if not args.keep_drive_files:
        remote_job = item['remote_job_rel']
        drive_input = item['drive_input_rel']
        proj_slug = item['project_slug']
        store.delete_path(remote_job)
        store.delete_path(drive_input)
        store.delete_folder_if_empty(f'jobs/{proj_slug}')
        store.delete_folder_if_empty(f'input/{proj_slug}')
        flac_p = item.get('flac_path')
        if flac_p and Path(flac_p).is_file():
            try:
                Path(flac_p).unlink()
                if not any(Path(flac_p).parent.iterdir()):
                    Path(flac_p).parent.rmdir()
            except OSError:
                pass
        print(f"CLEANED_CLOUD_FILES {item['audio'].name} -> deleted {remote_job} & {drive_input}", flush=True)


def main():
    p = argparse.ArgumentParser(description='Automated Local Audio -> Google Drive -> Colab L4 -> Local Sidecar Subtitles')
    p.add_argument('targets', nargs='*', type=Path, help='Local audio folders or audio files')
    p.add_argument('--overwrite', action='store_true', help='Re-generate and overwrite existing .vtt sidecar files')
    p.add_argument('--fresh', action='store_true', help='Ignore any existing stage checkpoints on Google Drive')
    p.add_argument('--metadata', default='', help='Optional path to custom metadata.json or glossary')
    p.add_argument('--no-recursive', action='store_true', help='Only scan top-level of given folders')
    p.add_argument('--no-auto-metadata', action='store_true', help='Disable optional local metadata hook lookup')
    p.add_argument('--vtt-only', action='store_true', help='Save only .vtt/.ja.vtt sidecars (skip .zh.txt/.ja.txt)')
    p.add_argument('--zh-only', action='store_true', help='Save only Chinese sidecars (skip .ja.vtt/.ja.txt)')
    p.add_argument('--keep-drive-files', action='store_true', help='Do not delete cloud audio/job folders after local verification')
    p.add_argument('--dry-run', action='store_true', help='Scan folders and preview pending tracks without uploading or starting GPU')
    p.add_argument('--gpu', default='', help='Colab GPU type (required: A100 or L4, e.g. --gpu A100)')
    p.add_argument('--session', default='', help='Optional custom Colab session name')
    p.add_argument('--keep-alive', action='store_true', help='Keep Colab session alive on exit (debugging only)')
    p.add_argument('--poll-seconds', type=float, default=5.0)
    p.add_argument('--wsl-distro', default='Ubuntu-24.04')
    p.add_argument('--drive-token', default='', help='Path to drive-token.json')
    p.add_argument('--auth-client-secret', default='', help='One-time OAuth setup from Google client_secret.json')
    args = p.parse_args()

    token_path = Path(args.drive_token) if args.drive_token else default_windows_token_path()
    if args.auth_client_secret:
        run_installed_app_oauth(args.auth_client_secret, token_path)
        print('DRIVE_OAUTH_TOKEN_SAVED', token_path, flush=True)
        if not args.targets:
            return

    if not args.targets:
        p.error('Specify at least one local audio folder or audio file')

    # Mandatory manual GPU selection
    valid_gpus = {'A100': 'A100', 'L4': 'L4'}
    chosen_gpu = (args.gpu or '').strip().upper()
    if not chosen_gpu or chosen_gpu not in valid_gpus:
        print('\n' + '=' * 60, file=sys.stderr)
        print('【错误：未指定或输入了无效的 GPU 类型参数】', file=sys.stderr)
        print('为了确保显存和吞吐达到最优压榨参数（避免浪费算力），请显式指定 GPU 类型：', file=sys.stderr)
        print('  --gpu A100   : 使用 A100-SXM4-40GB 极限参数 (41并发14B + 2x24双实例9B + bs192双流ASR)', file=sys.stderr)
        print('  --gpu L4     : 使用 L4-24GB 极限参数 (16并发14B + 48并发单实例9B + bs96双流ASR)', file=sys.stderr)
        print('=' * 60 + '\n', file=sys.stderr)
        p.error('Missing or invalid --gpu parameter. Please rerun with --gpu A100 or --gpu L4.')
    args.gpu = valid_gpus[chosen_gpu]

    discovered = discover_audio_files(args.targets, recursive=not args.no_recursive)
    if not discovered:
        print('NO_AUDIO_FILES_FOUND', flush=True)
        return

    pending = []
    meta_cache = {}
    for audio in discovered:
        zh_vtt = audio.with_name(audio.name + '.vtt')
        if zh_vtt.is_file() and not args.overwrite:
            try:
                validate_vtt(zh_vtt)
                print('SKIP_EXISTING', zh_vtt, flush=True)
                continue
            except ValueError:
                print('REQUEUE_INVALID_VTT', zh_vtt, flush=True)
        meta = resolve_track_metadata(
            audio, custom_meta_path=args.metadata,
            auto_metadata=not args.no_auto_metadata, cache_mem=meta_cache,
        )
        pending.append({'audio': audio, 'metadata': meta})

    if not pending:
        print('ALL_SUBTITLES_UP_TO_DATE', flush=True)
        return

    assign_project_slugs(pending)
    print(f'PENDING_TRACKS: {len(pending)}', flush=True)
    for idx, item in enumerate(pending, 1):
        print(f"  [{idx}/{len(pending)}] {item['audio']} -> Drive:{item['remote_job_rel']}", flush=True)

    if args.dry_run:
        print('DRY_RUN_COMPLETE', flush=True)
        return

    store = DriveApiStore(token_path)
    batch_tracks = []
    for idx, item in enumerate(pending):
        audio = item['audio']
        t0 = time.perf_counter()
        flac_path = convert_to_16k_flac(audio, item['project_slug'], item['track_slug'])
        item['flac_path'] = flac_path
        if 'stereo_pan_10hz' not in item['metadata']:
            item['metadata']['stereo_pan_10hz'] = extract_stereo_pan_10hz(audio)
        drive_rel = item['drive_input_rel']
        print(
            f"UPLOAD_INPUT [{idx + 1}/{len(pending)}] {audio.name} "
            f"({audio.stat().st_size / 1048576:.1f} -> {flac_path.stat().st_size / 1048576:.1f} MiB FLAC)",
            flush=True,
        )
        store.upload_file(flac_path, drive_rel, overwrite=False)
        print(f'INPUT_READY_ON_DRIVE {drive_rel} ({time.perf_counter() - t0:.1f}s)', flush=True)
        batch_tracks.append({
            'track_id': str(idx),
            'stem': audio.name,
            'drive_input_rel': drive_rel,
            'remote_job_rel': item['remote_job_rel'],
            'metadata': item['metadata'],
        })

    colab = ColabRunner(wsl_distro=args.wsl_distro)
    session_name = args.session or f'jas-l4-{uuid.uuid4().hex[:8]}'
    session_allocated = False
    saved_tracks = set()

    with tempfile.TemporaryDirectory(prefix='jas-colab-') as tmp_str:
        tmp_dir = Path(tmp_str)
        spawn_py = tmp_dir / 'spawn_worker.py'
        poll_py = tmp_dir / 'poll_worker.py'
        token_tmp = tmp_dir / 'jas_drive_token.json'
        spec_tmp = tmp_dir / 'jas_batch_spec.json'
        bundle_tar = tmp_dir / 'jas_bundle.tar'

        spawn_py.write_text(SPAWN_REMOTE_WORKER_PY, encoding='utf-8')
        poll_py.write_text(POLL_REMOTE_WORKER_PY, encoding='utf-8')
        atomic_write_json(spec_tmp, {'fresh': args.fresh, 'tracks': batch_tracks})
        with tarfile.open(bundle_tar, 'w') as tar:
            for fname in RUNTIME_FILES:
                tar.add(RUNTIME_DIR / fname, arcname=fname)
            tar.add(spec_tmp, arcname='jas_batch_spec.json')

        def push_fresh_short_token():
            payload = store.get_short_lived_payload(min_ttl_sec=1800)
            atomic_write_json(token_tmp, payload)
            colab.run('upload', '-s', session_name, colab.to_cli_path(token_tmp), '/content/.jas_drive_token.json', timeout=60)
            return payload['expires_at']

        try:
            session_allocated = True
            print(f'ALLOCATING_COLAB_GPU session={session_name} gpu={args.gpu} ...', flush=True)
            colab.run('new', '-s', session_name, '--gpu', args.gpu, timeout=300)
            print(f'COLAB_SESSION_READY {session_name}', flush=True)

            colab.run('upload', '-s', session_name, colab.to_cli_path(bundle_tar), '/content/jas_bundle.tar', timeout=90)
            token_exp = push_fresh_short_token()

            colab.run('exec', '-s', session_name, '-f', colab.to_cli_path(spawn_py), '--timeout', '30', timeout=90)
            print('REMOTE_BATCH_WORKER_STARTED', flush=True)

            log_offset = 0
            poll_failures = 0
            while True:
                if time.time() + 1200 >= token_exp:
                    token_exp = push_fresh_short_token()
                    print('REFRESHED_SHORT_LIVED_COLAB_TOKEN', flush=True)

                try:
                    res = colab.run(
                        'exec', '-s', session_name, '-f', colab.to_cli_path(poll_py),
                        '--env', f'JAS_LOG_OFFSET={log_offset}', '--timeout', '45',
                        timeout=110,
                    )
                    poll_failures = 0
                except Exception as poll_exc:
                    poll_failures += 1
                    print(f'POLL_RETRY ({poll_failures}/12): {poll_exc}', flush=True)
                    if poll_failures >= 12:
                        raise
                    time.sleep(5)
                    continue

                marker_lines = [ln for ln in res.stdout.splitlines() if ln.startswith('JAS_POLL_JSON:')]
                if not marker_lines:
                    time.sleep(args.poll_seconds)
                    continue

                snap = json.loads(marker_lines[-1][len('JAS_POLL_JSON:'):])
                log_offset = snap['next_offset']
                if snap['chunk']:
                    sys.stdout.write(snap['chunk'])
                    if not snap['chunk'].endswith('\n'):
                        sys.stdout.write('\n')
                    sys.stdout.flush()

                w_status = snap.get('status') or {}
                tracks_state = w_status.get('tracks') or {}
                for tid_str, tr_info in tracks_state.items():
                    tid = int(tid_str)
                    if tr_info.get('state') == 'complete' and tid not in saved_tracks:
                        save_track_deliverables_locally(store, pending[tid], tr_info, args)
                        saved_tracks.add(tid)

                phase = w_status.get('phase')
                if phase == 'complete' and len(saved_tracks) == len(pending):
                    print(
                        f"BATCH_PIPELINE_COMPLETE tracks={len(saved_tracks)} "
                        f"peak_gpu_mib={w_status.get('peak_gpu_mib')} elapsed={w_status.get('elapsed_seconds')}s",
                        flush=True,
                    )
                    break

                if phase == 'failed':
                    raise RuntimeError(f"Remote Colab pipeline failed: {w_status.get('error')}")

                if not snap['alive'] and phase != 'complete':
                    raise RuntimeError(f'Remote worker exited unexpectedly in phase={phase}')

                time.sleep(args.poll_seconds)
        finally:
            if session_allocated and not args.keep_alive:
                prev_handler = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                try:
                    colab.run('rm', '-s', session_name, '/content/.jas_drive_token.json', timeout=20, check=False)
                    stop_res = colab.run('stop', '-s', session_name, timeout=90, check=False)
                    print('COLAB_SESSION_STOPPED', session_name, stop_res.stdout.strip(), flush=True)
                finally:
                    signal.signal(signal.SIGINT, prev_handler)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nINTERRUPTED_BY_USER (Colab runtime cleanup completed).', file=sys.stderr, flush=True)
        sys.exit(130)
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
