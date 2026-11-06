"""Colab batch worker: restores models, runs multi-track pipeline, and syncs stage checkpoints to Google Drive."""
import argparse, json, os, re, shutil, signal, subprocess, sys, threading, time
from pathlib import Path
from subtitle_utils import atomic_write_json, validate_vtt
from drive_store import open_drive_store


def deliverable_rel_paths(stem):
    return {
        'zh_vtt': f'outputs/{stem}.vtt',
        'ja_vtt': f'outputs/{stem}.ja.vtt',
        'zh_txt': f'outputs/{stem}.zh.txt',
        'ja_txt': f'outputs/{stem}.ja.txt',
    }


def _verify_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU runtime is required on Colab (select L4 GPU)')


def _ensure_16k_mono_flac(src_audio, dst_flac):
    import soundfile as sf
    src_audio = Path(src_audio)
    dst_flac = Path(dst_flac)
    if dst_flac.is_file() and dst_flac.stat().st_size > 0:
        return
    dst_flac.parent.mkdir(parents=True, exist_ok=True)
    if src_audio.suffix.lower() == '.flac':
        try:
            info = sf.info(str(src_audio))
            if info.samplerate == 16000 and info.channels == 1:
                shutil.copyfile(src_audio, dst_flac)
                return
        except Exception:
            pass
    subprocess.run(
        ['ffmpeg', '-nostdin', '-y', '-i', str(src_audio), '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'flac', str(dst_flac)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )


def run_batch(spec_path, store, sync_seconds=30, status_file=None):
    runtime_dir = Path(__file__).resolve().parent
    spec = json.loads(Path(spec_path).read_text(encoding='utf-8'))
    started = time.time()
    status = {
        'phase': 'starting',
        'pid': os.getpid(),
        'started_at': started,
        'mode': store.mode,
        'tracks': {
            str(t['track_id']): {'state': 'queued', 'stem': t['stem'], 'stage': None}
            for t in spec['tracks']
        },
        'peak_gpu_mib': 0,
        'error': None,
    }

    def emit_status(**updates):
        status.update(updates)
        status['elapsed_seconds'] = round(time.time() - started, 1)
        if status_file:
            atomic_write_json(status_file, status)

    try:
        emit_status(phase='bootstrapping_models')
        boot_cmd = [sys.executable, '-u', str(runtime_dir / 'bootstrap.py')]
        if spec.get('restore_only', False):
            boot_cmd.append('--restore-only')
        p_boot = subprocess.Popen(boot_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for ln in p_boot.stdout:
            print(ln, end='', flush=True)
        if p_boot.wait() != 0:
            raise RuntimeError('Model bootstrap failed')

        _verify_gpu()
        emit_status(phase='preparing_inputs')

        prepared_jobs = []
        batch_manifest = []
        fresh = bool(spec.get('fresh', False))

        for idx, tr in enumerate(spec['tracks']):
            tid = str(tr['track_id'])
            stem = tr['stem']
            remote_job_rel = tr['remote_job_rel']
            local_dir = Path('/content/subtitle-jobs') / remote_job_rel.removeprefix('jobs/')
            out_dir = local_dir / 'outputs'
            local_dir.mkdir(parents=True, exist_ok=True)
            store.resolve_folder(remote_job_rel, create=True)

            if not fresh and not out_dir.exists():
                if store.sync_tree_down(f'{remote_job_rel}/outputs', out_dir):
                    print(f'RESTORED_CHECKPOINT [{stem}] <- {remote_job_rel}/outputs', flush=True)
            out_dir.mkdir(parents=True, exist_ok=True)

            staged_flac = local_dir / 'staged_input.flac'
            normalized = local_dir / 'input.flac'
            if not normalized.is_file():
                store.download_file(tr['drive_input_rel'], staged_flac)
                _ensure_16k_mono_flac(staged_flac, normalized)
                if staged_flac.exists():
                    staged_flac.unlink()

            meta_path = ''
            meta_obj = tr.get('metadata') or {}
            if meta_obj:
                mp = local_dir / 'metadata.json'
                atomic_write_json(mp, meta_obj)
                meta_path = str(mp)

            status['tracks'][tid].update(
                state='prepared',
                remote_job=remote_job_rel,
                drive_input_rel=tr['drive_input_rel'],
            )
            emit_status()

            prepared_jobs.append(dict(
                index=idx,
                track_id=tid,
                stem=stem,
                local_dir=local_dir,
                out_dir=out_dir,
                remote_job_rel=remote_job_rel,
            ))
            batch_manifest.append(dict(
                index=idx,
                audio=str(normalized),
                stem=stem,
                out_dir=str(out_dir),
                metadata_json=meta_path,
            ))

        by_index = {j['index']: j for j in prepared_jobs}
        manifest_path = Path('/content/subtitle-jobs/batch_manifest.json')
        atomic_write_json(manifest_path, batch_manifest)

        stopped = threading.Event()
        sync_lock = threading.Lock()

        def sync_one_job(job):
            with sync_lock:
                store.sync_tree_up(job['out_dir'], f"{job['remote_job_rel']}/outputs")

        def sync_all_jobs():
            for job in prepared_jobs:
                if status['tracks'][job['track_id']].get('state') != 'complete':
                    sync_one_job(job)

        def finalize_track(job):
            deliverables = {}
            for k, rel_sub in deliverable_rel_paths(job['stem']).items():
                local_f = job['local_dir'] / rel_sub
                if not local_f.is_file() or local_f.stat().st_size == 0:
                    raise FileNotFoundError(f'Missing or empty deliverable {local_f}')
                if k.endswith('_vtt'):
                    validate_vtt(local_f)
                deliverables[k] = {'rel_path': f"{job['remote_job_rel']}/{rel_sub}"}
            sync_one_job(job)
            status['tracks'][job['track_id']].update(
                state='complete',
                stage='translate-zh',
                deliverables=deliverables,
            )
            emit_status()
            print(f"TRACK_COMPLETE {job['track_id']} {job['stem']} -> {job['remote_job_rel']}", flush=True)

        def monitor():
            next_sync = time.time() + max(10, sync_seconds)
            while not stopped.is_set():
                try:
                    raw = subprocess.check_output(
                        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                        text=True,
                    ).strip()
                    used_mib = int(raw.splitlines()[0].strip())
                    if used_mib > status['peak_gpu_mib']:
                        status['peak_gpu_mib'] = used_mib
                except Exception:
                    pass
                if time.time() >= next_sync:
                    try:
                        sync_all_jobs()
                    except Exception as e:
                        print('DRIVE_SYNC_RETRY', type(e).__name__, e, flush=True)
                    next_sync = time.time() + max(10, sync_seconds)
                stopped.wait(2)

        mon_thread = threading.Thread(target=monitor, daemon=True)
        mon_thread.start()
        proc = None
        emit_status(phase='running_pipeline')
        try:
            cmd = [sys.executable, '-u', str(runtime_dir / 'pipeline.py'), '--batch-json', str(manifest_path)]
            if fresh:
                cmd.append('--force')
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
            for line in proc.stdout:
                print(line, end='', flush=True)
                m_stage = re.match(r'^=== RUNNING BATCH STAGE:\s*(\S+)', line.strip())
                if m_stage:
                    cur_st = m_stage.group(1)
                    for job in prepared_jobs:
                        tr_st = status['tracks'][job['track_id']]
                        if tr_st['state'] != 'complete':
                            tr_st['state'] = 'running'
                            tr_st['stage'] = cur_st
                    emit_status(current_stage=cur_st)
                m_done = re.match(r'^BATCH_TRACK_STAGE_DONE\s+(\d+)\s+(\S+)\s+(.+)$', line.strip())
                if m_done:
                    j_idx = int(m_done.group(1))
                    st_done = m_done.group(2)
                    job = by_index[j_idx]
                    if st_done == 'translate-zh':
                        finalize_track(job)
                    else:
                        status['tracks'][job['track_id']].update(state='running', stage=st_done + ':done')
                        emit_status()
                        sync_one_job(job)
            if proc.wait() != 0:
                raise RuntimeError('Batch pipeline failed; completed stages have been synced to Google Drive')
            for job in prepared_jobs:
                if status['tracks'][job['track_id']]['state'] != 'complete':
                    finalize_track(job)
            emit_status(phase='complete')
            print('BATCH_RESULTS_READY', len(prepared_jobs), flush=True)
        finally:
            if proc and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            stopped.set()
            mon_thread.join(timeout=60)
            try:
                sync_all_jobs()
            except Exception:
                pass
    except Exception as exc:
        emit_status(phase='failed', error=str(exc))
        raise


def main():
    p = argparse.ArgumentParser(description='Colab batch subtitle worker')
    p.add_argument('--batch-spec', required=True)
    p.add_argument('--status-file', default='/content/jas_worker_status.json')
    p.add_argument('--drive-root', default='/content/drive/MyDrive/japanese-subtitles')
    p.add_argument('--token-file', default='')
    p.add_argument('--sync-seconds', type=int, default=30)
    a = p.parse_args()

    lock_f = open('/tmp/jas-worker.lock', 'w')
    try:
        import fcntl
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (ImportError, BlockingIOError):
        pass

    store = open_drive_store(drive_root=a.drive_root, token_file=a.token_file or None)
    run_batch(a.batch_spec, store, sync_seconds=a.sync_seconds, status_file=a.status_file)


if __name__ == '__main__':
    main()
