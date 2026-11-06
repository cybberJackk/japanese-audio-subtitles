"""Bootstrap and cache speech/translation models between Google Drive and local Colab SSD."""
import argparse, json, os, shutil, subprocess, sys, tarfile, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from subtitle_utils import atomic_write_json
from drive_store import open_drive_store

SNAPSHOTS = [
    'litagin/anime-whisper',
    'openai/whisper-large-v3-turbo',
    'Qwen/Qwen3-ForcedAligner-0.6B',
]
GGUF = {
    'qwen3-14b': ('Qwen/Qwen3-14B-GGUF', 'Qwen3-14B-Q4_K_M.gguf'),
    'asmr-qwen35-9b-echo': (
        'mmis1000/asmr-qwen3.5-9b-zh-cn-echo-gguf-v0.2',
        'asmr-qwen3.5-9b-zh-cn-echo-gguf-v0.2-q6_k.gguf',
    ),
}
LLAMA_TARS = [
    'llama-b12123-bin-ubuntu-cuda-12.8-x64.tar.gz',
    'cudart-llama-b12123-bin-ubuntu-cuda-12.8-x64.tar.gz',
]
PACKAGES = [
    'qwen-asr==0.0.6', 'transformers==4.57.6', 'tokenizers==0.22.2',
    'nagisa==0.2.11', 'DyNet38==2.2', 'soynlp==0.0.493',
    'qwen-omni-utils==0.0.9', 'sox==1.5.0', 'huggingface-hub==0.36.2',
]


def build_archive(src, dest, arcname):
    with tarfile.open(dest, 'w', dereference=True) as tar:
        tar.add(src, arcname=arcname)


def _runtime_ready(root):
    models_file = root / 'models.json'
    ready_file = root / 'align-packages' / 'ready.json'
    if not (models_file.is_file() and ready_file.is_file()):
        return False
    try:
        paths = json.loads(models_file.read_text(encoding='utf-8'))
        if any(not Path(p).exists() for p in paths.values()):
            return False
        return any((root / 'llama').rglob('llama-server'))
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser(description='Restore or initialize persistent models in /content/japanese-subtitles')
    p.add_argument('--drive-root', default='/content/drive/MyDrive/japanese-subtitles')
    p.add_argument('--token-file', default='')
    p.add_argument('--runtime-root', default='/content/japanese-subtitles')
    p.add_argument('--restore-only', action='store_true', help='Require all models to exist on Google Drive')
    a = p.parse_args()

    store = open_drive_store(drive_root=a.drive_root, token_file=a.token_file or None)
    root = Path(a.runtime_root)
    root.mkdir(parents=True, exist_ok=True)
    for folder in ('models', 'jobs', 'input'):
        store.resolve_folder(folder, create=True)

    if _runtime_ready(root):
        print('BOOTSTRAP_ALREADY_READY', flush=True)
        return

    packages = root / 'align-packages'
    ready = packages / 'ready.json'
    if not ready.is_file():
        if packages.exists():
            shutil.rmtree(packages)
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--quiet', '--target', str(packages), '--no-deps', *PACKAGES],
            check=True,
        )
        atomic_write_json(ready, {'packages': PACKAGES})

    state = store.read_json('model_registry.json', default={}) or {}
    os.environ.update(HF_HOME=str(root / 'hf-cache'), HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    sys_path_env = str(packages) + os.pathsep + os.environ.get('PYTHONPATH', '')
    if str(packages) not in sys.path:
        sys.path.insert(0, str(packages))

    state_lock = threading.Lock()
    llama_lock = threading.Lock()
    paths = {}

    def restore(key, destination):
        item = state.get(key)
        if not item:
            return False
        rel_art = item['artifact']
        if item.get('kind') == 'tar':
            local_tar = root / (Path(rel_art).name + '.restore')
            store.download_file(rel_art, local_tar)
            try:
                destination.mkdir(parents=True, exist_ok=True)
                with tarfile.open(local_tar) as tar:
                    tar.extractall(destination, filter='data')
            finally:
                if local_tar.exists():
                    local_tar.unlink()
        else:
            store.download_file(rel_art, destination)
        print('RESTORED_MODEL', key, f'({store.mode})', flush=True)
        return True

    def persist(key, local_file, kind):
        local_file = Path(local_file)
        rel_target = f'models/{local_file.name}'
        store.upload_file(local_file, rel_target, overwrite=True)
        with state_lock:
            state[key] = {'artifact': rel_target, 'bytes': local_file.stat().st_size, 'kind': kind}
            store.write_json('model_registry.json', state)
        print('CACHED_MODEL_TO_DRIVE', key, flush=True)

    def handle_snapshot(repo):
        name = repo.replace('/', '__')
        folder = root / 'snapshots' / name
        folder.parent.mkdir(parents=True, exist_ok=True)
        if folder.is_dir() and any(folder.iterdir()):
            return repo, str(folder)
        if not restore(repo, folder.parent):
            if a.restore_only:
                raise RuntimeError(f'Model {repo} not yet cached on Google Drive; run without --restore-only once')
            from huggingface_hub import snapshot_download
            source = snapshot_download(
                repo, token=False, max_workers=4,
                allow_patterns=['*.json', '*.jsonl', '*.py', '*.safetensors', '*.model', '*.tiktoken', '*.jinja', '*.txt'],
            )
            archive = root / (name + '.tar')
            build_archive(source, archive, name)
            persist(repo, archive, 'tar')
            with tarfile.open(archive) as tar:
                tar.extractall(folder.parent, filter='data')
            archive.unlink()
        return repo, str(folder)

    def handle_gguf(key, repo, filename):
        local = root / filename
        if local.is_file() and local.stat().st_size > 0:
            return key, str(local)
        if not restore(key, local):
            if a.restore_only:
                raise RuntimeError(f'Model {key} not yet cached on Google Drive; run without --restore-only once')
            from huggingface_hub import hf_hub_download
            source = Path(hf_hub_download(repo, filename, token=False))
            persist(key, source, 'file')
            shutil.copyfile(source, local)
        return key, str(local)

    def handle_llama(filename):
        dest = root / 'llama'
        local = root / filename
        if not restore(filename, local):
            if a.restore_only:
                raise RuntimeError(f'Binary {filename} not yet cached on Google Drive; run without --restore-only once')
            import requests
            url = 'https://github.com/ggml-org/llama.cpp/releases/download/b12123/' + filename
            with requests.get(url, stream=True, timeout=(30, 180)) as r:
                r.raise_for_status()
                with local.open('wb') as f:
                    for block in r.iter_content(2 ** 20):
                        f.write(block)
            persist(filename, local, 'file')
        with llama_lock:
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(local) as tar:
                tar.extractall(dest, filter='data')
            for srv_bin in dest.rglob('llama-server'):
                srv_bin.chmod(0o755)
        if local.exists():
            local.unlink()
        return filename, str(dest)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = []
        for repo in SNAPSHOTS:
            futs.append(pool.submit(handle_snapshot, repo))
        for key, (repo, filename) in GGUF.items():
            futs.append(pool.submit(handle_gguf, key, repo, filename))
        for filename in LLAMA_TARS:
            futs.append(pool.submit(handle_llama, filename))
        for fut in as_completed(futs):
            k, val = fut.result()
            if k not in LLAMA_TARS:
                paths[k] = val

    env = dict(os.environ, PYTHONPATH=sys_path_env, USE_TF='0', USE_FLAX='0')
    subprocess.run(
        [sys.executable, '-c', 'from qwen_asr import Qwen3ForcedAligner; from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq; import nagisa'],
        env=env, check=True,
    )
    atomic_write_json(root / 'models.json', paths)
    print('BOOTSTRAP_COMPLETE', f'mode={store.mode}', flush=True)


if __name__ == '__main__':
    main()
