"""Google Drive storage backend (Headless Drive v3 REST API + Colab FUSE mount).

Uses Python standard library only so it runs natively on Windows, Linux, and Colab
without third-party Google API SDK dependencies or expensive SHA-256 re-hashing.
"""
import json, os, shutil, threading, time, urllib.error, urllib.parse, urllib.request, uuid
from pathlib import Path
from subtitle_utils import atomic_write_json

DRIVE_FILES_URL = 'https://www.googleapis.com/drive/v3/files'
DRIVE_UPLOAD_URL = 'https://www.googleapis.com/upload/drive/v3/files'
OAUTH_TOKEN_URL = 'https://oauth2.googleapis.com/token'
FOLDER_MIME = 'application/vnd.google-apps.folder'
DEFAULT_COLAB_TOKEN_PATH = Path('/content/.jas_drive_token.json')
CHUNK_UPLOAD_BYTES = 16 * 1024 * 1024
CHUNK_DOWNLOAD_BYTES = 8 * 1024 * 1024


class _GoogleUploadRedirectHandler(urllib.request.HTTPRedirectHandler):
    def http_error_308(self, req, fp, code, msg, hdrs):
        return fp


def _build_opener():
    return urllib.request.build_opener(_GoogleUploadRedirectHandler())


def default_app_data_dir():
    local_app = os.environ.get('LOCALAPPDATA')
    if local_app:
        return Path(local_app) / 'JapaneseAudioSubtitles'
    return Path.home() / '.config' / 'japanese-audio-subtitles'


def default_windows_token_path():
    env_p = os.environ.get('JAS_DRIVE_TOKEN')
    if env_p:
        return Path(env_p)
    return default_app_data_dir() / 'drive-token.json'


def run_installed_app_oauth(client_secret_path, token_out_path):
    """One-time browser OAuth flow to save refreshable Drive credentials locally."""
    import http.server, webbrowser
    raw = json.loads(Path(client_secret_path).read_text(encoding='utf-8'))
    cfg = raw.get('installed') or raw.get('web')
    if not cfg:
        raise ValueError('Invalid Google OAuth client secret JSON')
    result = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            result['code'] = (qs.get('code') or [None])[0]
            result['error'] = (qs.get('error') or [None])[0]
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write('<h3>Google Drive 授权成功，可关闭此页面返回终端。</h3>'.encode('utf-8'))

        def log_message(self, *args):
            pass

    srv = http.server.HTTPServer(('127.0.0.1', 0), Handler)
    redirect_uri = f'http://127.0.0.1:{srv.server_port}/'
    scopes = ['https://www.googleapis.com/auth/drive']
    auth_url = cfg['auth_uri'] + '?' + urllib.parse.urlencode({
        'client_id': cfg['client_id'],
        'redirect_uri': redirect_uri,
        'response_type': 'code',
        'scope': ' '.join(scopes),
        'access_type': 'offline',
        'prompt': 'consent',
    })
    print('OPENING_BROWSER_FOR_DRIVE_AUTH:', auth_url, flush=True)
    webbrowser.open(auth_url)
    srv.handle_request()
    srv.server_close()
    if result.get('error') or not result.get('code'):
        raise RuntimeError(f"OAuth authorization failed: {result.get('error')}")

    body = urllib.parse.urlencode({
        'code': result['code'],
        'client_id': cfg['client_id'],
        'client_secret': cfg['client_secret'],
        'redirect_uri': redirect_uri,
        'grant_type': 'authorization_code',
    }).encode('utf-8')
    req = urllib.request.Request(cfg.get('token_uri', OAUTH_TOKEN_URL), data=body)
    resp = json.loads(urllib.request.urlopen(req, timeout=30).read().decode('utf-8'))
    token_data = {
        'token': resp['access_token'],
        'refresh_token': resp['refresh_token'],
        'token_uri': cfg.get('token_uri', OAUTH_TOKEN_URL),
        'client_id': cfg['client_id'],
        'client_secret': cfg['client_secret'],
        'scopes': scopes,
        'expires_at': time.time() + int(resp.get('expires_in', 3600)),
    }
    atomic_write_json(token_out_path, token_data)
    return token_data


class DriveApiStore:
    """Google Drive v3 REST API store supporting local refresh tokens and Colab short-lived access tokens."""
    mode = 'api'

    def __init__(self, token_path, project_name='japanese-subtitles', root_id=None):
        self.token_path = Path(token_path)
        self.project_name = project_name
        self._lock = threading.RLock()
        self._token = None
        self._expires_at = 0.0
        self._file_mtime = 0.0
        self._folder_cache = {}
        self._stat_cache = {}
        self._load_or_refresh_token(force=True)
        with self._lock:
            if root_id:
                self.root_id = root_id
            elif self._configured_root_id:
                self.root_id = self._configured_root_id
            else:
                self.root_id = self._find_or_create_folder('root', project_name)
            self._folder_cache[''] = self.root_id

    def _load_or_refresh_token(self, force=False):
        with self._lock:
            if not self.token_path.is_file():
                raise FileNotFoundError(f'Drive token file missing: {self.token_path}')
            st_mtime = self.token_path.stat().st_mtime
            raw = json.loads(self.token_path.read_text(encoding='utf-8'))
            self._configured_root_id = raw.get('root_id')

            if 'access_token' in raw and 'refresh_token' not in raw:
                self._token = raw['access_token']
                self._expires_at = float(raw.get('expires_at', time.time() + 1800))
                self._file_mtime = st_mtime
                return self._token

            now = time.time()
            cached_exp = float(raw.get('expires_at', 0.0))
            if not force and self._token and now < self._expires_at - 300:
                return self._token
            if not force and raw.get('token') and now < cached_exp - 300:
                self._token = raw['token']
                self._expires_at = cached_exp
                return self._token

            body = urllib.parse.urlencode({
                'client_id': raw['client_id'],
                'client_secret': raw['client_secret'],
                'refresh_token': raw['refresh_token'],
                'grant_type': 'refresh_token',
            }).encode('utf-8')
            req = urllib.request.Request(raw.get('token_uri', OAUTH_TOKEN_URL), data=body)
            resp = json.loads(_build_opener().open(req, timeout=30).read().decode('utf-8'))
            self._token = resp['access_token']
            self._expires_at = time.time() + int(resp.get('expires_in', 3600))
            raw['token'] = self._token
            raw['expires_at'] = self._expires_at
            try:
                atomic_write_json(self.token_path, raw)
            except OSError:
                pass
            return self._token

    def get_short_lived_payload(self, min_ttl_sec=1500):
        with self._lock:
            if time.time() + min_ttl_sec >= self._expires_at:
                self._load_or_refresh_token(force=True)
            return {
                'access_token': self._token,
                'expires_at': self._expires_at,
                'root_id': self.root_id,
            }

    def _auth_header(self, force_reload=False):
        with self._lock:
            if force_reload:
                self._load_or_refresh_token(force=True)
            elif self.token_path.is_file():
                st_mtime = self.token_path.stat().st_mtime
                if st_mtime != self._file_mtime or time.time() >= self._expires_at - 300:
                    self._load_or_refresh_token(force=(time.time() >= self._expires_at - 300))
            return {'Authorization': f'Bearer {self._token}'}

    def _request(self, method, url, params=None, data=None, headers=None, timeout=60, raw_resp=False):
        if params:
            sep = '&' if '?' in url else '?'
            url = url + sep + urllib.parse.urlencode(params)
        last_err = None
        for attempt in range(5):
            hdrs = dict(self._auth_header(force_reload=(attempt > 0 and getattr(last_err, 'code', None) == 401)))
            if headers:
                hdrs.update(headers)
            req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
            try:
                resp = _build_opener().open(req, timeout=timeout)
                return resp if raw_resp else resp.read()
            except urllib.error.HTTPError as e:
                if e.code == 308 and raw_resp:
                    return e
                last_err = e
                if e.code in (401, 403, 408, 429, 500, 502, 503, 504) and attempt < 4:
                    time.sleep(min(15, 1.5 * (2 ** attempt)))
                    continue
                detail = ''
                try:
                    detail = e.read().decode('utf-8', 'replace')[:500]
                except Exception:
                    pass
                raise RuntimeError(f'Drive HTTP {e.code} on {method} {url}: {detail}') from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = e
                if attempt < 4:
                    time.sleep(min(15, 1.5 * (2 ** attempt)))
                    continue
                raise
        raise last_err

    def _escape_q(self, name):
        return name.replace('\\', '\\\\').replace("'", "\\'")

    def _list_query(self, q, fields='id,name,mimeType,size'):
        items = []
        page_token = None
        while True:
            params = {'q': q, 'fields': f'nextPageToken,files({fields})', 'pageSize': 1000, 'spaces': 'drive'}
            if page_token:
                params['pageToken'] = page_token
            res = json.loads(self._request('GET', DRIVE_FILES_URL, params=params).decode('utf-8'))
            items.extend(res.get('files', []))
            page_token = res.get('nextPageToken')
            if not page_token:
                break
        return items

    def _find_or_create_folder(self, parent_id, name):
        q = f"'{parent_id}' in parents and name='{self._escape_q(name)}' and mimeType='{FOLDER_MIME}' and trashed=false"
        found = self._list_query(q, fields='id,name')
        if found:
            return found[0]['id']
        payload = json.dumps({'name': name, 'mimeType': FOLDER_MIME, 'parents': [parent_id]}).encode('utf-8')
        created = json.loads(self._request(
            'POST', DRIVE_FILES_URL, params={'fields': 'id'},
            data=payload, headers={'Content-Type': 'application/json'}
        ).decode('utf-8'))
        return created['id']

    def resolve_folder(self, rel_dir, create=True):
        norm = str(rel_dir).strip('/').replace('\\', '/')
        if norm in ('', '.'):
            return self.root_id
        with self._lock:
            if norm in self._folder_cache:
                return self._folder_cache[norm]
            cur_id = self.root_id
            built = ''
            for part in norm.split('/'):
                built = f'{built}/{part}' if built else part
                if built in self._folder_cache:
                    cur_id = self._folder_cache[built]
                    continue
                q = f"'{cur_id}' in parents and name='{self._escape_q(part)}' and mimeType='{FOLDER_MIME}' and trashed=false"
                found = self._list_query(q, fields='id,name')
                if found:
                    cur_id = found[0]['id']
                elif create:
                    cur_id = self._find_or_create_folder(cur_id, part)
                else:
                    return None
                self._folder_cache[built] = cur_id
            return cur_id

    def _split_rel(self, rel_path):
        norm = str(rel_path).strip('/').replace('\\', '/')
        if '/' in norm:
            parent_rel, name = norm.rsplit('/', 1)
        else:
            parent_rel, name = '', norm
        return parent_rel, name

    def stat_file(self, rel_path):
        parent_rel, name = self._split_rel(rel_path)
        parent_id = self.resolve_folder(parent_rel, create=False)
        if not parent_id:
            return None
        q = f"'{parent_id}' in parents and name='{self._escape_q(name)}' and mimeType!='{FOLDER_MIME}' and trashed=false"
        found = self._list_query(q, fields='id,name,size')
        return found[0] if found else None

    def exists(self, rel_path):
        if self.stat_file(rel_path):
            return True
        return self.resolve_folder(rel_path, create=False) is not None

    def delete_path(self, rel_path):
        norm = str(rel_path).strip('/').replace('\\', '/')
        if not norm:
            raise ValueError('Refusing to delete project root folder')
        info = self.stat_file(norm)
        target_id = info['id'] if info else self.resolve_folder(norm, create=False)
        if not target_id:
            return False
        self._request('DELETE', f'{DRIVE_FILES_URL}/{target_id}')
        with self._lock:
            for k in [k for k in self._folder_cache if k == norm or k.startswith(norm + '/')]:
                self._folder_cache.pop(k, None)
            for k in [k for k in self._stat_cache if k[1] == norm or k[1].startswith(norm + '/')]:
                self._stat_cache.pop(k, None)
        return True

    def delete_folder_if_empty(self, rel_dir):
        norm = str(rel_dir).strip('/').replace('\\', '/')
        if not norm:
            return False
        fid = self.resolve_folder(norm, create=False)
        if not fid:
            return False
        children = self._list_query(f"'{fid}' in parents and trashed=false", fields='id')
        if not children:
            return self.delete_path(norm)
        return False

    def move_folder(self, src_rel, dst_rel):
        src_id = self.resolve_folder(src_rel, create=False)
        if not src_id:
            return False
        dst_parent_rel, dst_name = self._split_rel(dst_rel)
        dst_parent_id = self.resolve_folder(dst_parent_rel, create=True)
        meta = json.loads(self._request('GET', f'{DRIVE_FILES_URL}/{src_id}', params={'fields': 'parents'}).decode('utf-8'))
        old_parents = ','.join(meta.get('parents', []))
        params = {'addParents': dst_parent_id, 'fields': 'id,name'}
        if old_parents:
            params['removeParents'] = old_parents
        self._request(
            'PATCH', f'{DRIVE_FILES_URL}/{src_id}',
            params=params,
            data=json.dumps({'name': dst_name}).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
        )
        with self._lock:
            self._folder_cache.clear()
            self._folder_cache[''] = self.root_id
        return True

    def read_json(self, rel_path, default=None):
        info = self.stat_file(rel_path)
        if not info:
            return default
        raw = self._request('GET', f"{DRIVE_FILES_URL}/{info['id']}", params={'alt': 'media'}, timeout=60)
        return json.loads(raw.decode('utf-8'))

    def write_json(self, rel_path, data):
        parent_rel, name = self._split_rel(rel_path)
        parent_id = self.resolve_folder(parent_rel, create=True)
        body = json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8')
        existing = self.stat_file(rel_path)
        if existing:
            self._request(
                'PATCH', f"{DRIVE_UPLOAD_URL}/{existing['id']}",
                params={'uploadType': 'media'},
                data=body, headers={'Content-Type': 'application/json; charset=utf-8'}, timeout=60
            )
        else:
            boundary = 'jas_' + uuid.uuid4().hex
            meta_bytes = json.dumps({'name': name, 'parents': [parent_id]}).encode('utf-8')
            multipart = (
                f'--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n'.encode('utf-8')
                + meta_bytes
                + f'\r\n--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n'.encode('utf-8')
                + body
                + f'\r\n--{boundary}--\r\n'.encode('utf-8')
            )
            self._request(
                'POST', DRIVE_UPLOAD_URL, params={'uploadType': 'multipart'},
                data=multipart, headers={'Content-Type': f'multipart/related; boundary={boundary}'}, timeout=60
            )

    def upload_file(self, local_path, rel_path, overwrite=True):
        local_path = Path(local_path)
        st = local_path.stat()
        size = st.st_size
        existing = self.stat_file(rel_path)
        if existing and not overwrite and int(existing.get('size', -1)) == size:
            return existing

        parent_rel, name = self._split_rel(rel_path)
        parent_id = self.resolve_folder(parent_rel, create=True)

        # Atomic single-request upload for files <= 8 MiB (logs, JSON, WebVTT, small FLACs)
        if size <= 8 * 1024 * 1024:
            raw_bytes = local_path.read_bytes()
            if existing:
                res_bytes = self._request(
                    'PATCH', f"{DRIVE_UPLOAD_URL}/{existing['id']}",
                    params={'uploadType': 'media', 'fields': 'id,name,size'},
                    data=raw_bytes,
                    headers={'Content-Type': 'application/octet-stream', 'Content-Length': str(len(raw_bytes))},
                    timeout=120,
                )
            else:
                boundary = 'jas_' + uuid.uuid4().hex
                meta_part = json.dumps({'name': name, 'parents': [parent_id]}).encode('utf-8')
                multipart = (
                    f'--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n'.encode('utf-8')
                    + meta_part
                    + f'\r\n--{boundary}\r\nContent-Type: application/octet-stream\r\n\r\n'.encode('utf-8')
                    + raw_bytes
                    + f'\r\n--{boundary}--\r\n'.encode('utf-8')
                )
                res_bytes = self._request(
                    'POST', DRIVE_UPLOAD_URL,
                    params={'uploadType': 'multipart', 'fields': 'id,name,size'},
                    data=multipart,
                    headers={'Content-Type': f'multipart/related; boundary={boundary}'},
                    timeout=120,
                )
            return json.loads(res_bytes.decode('utf-8'))

        # Resumable chunked upload for large audio/model files (> 8 MiB)
        if existing:
            init_url = f"{DRIVE_UPLOAD_URL}/{existing['id']}"
            init_method = 'PATCH'
            meta_payload = json.dumps({'name': name}).encode('utf-8')
        else:
            init_url = DRIVE_UPLOAD_URL
            init_method = 'POST'
            meta_payload = json.dumps({'name': name, 'parents': [parent_id]}).encode('utf-8')

        init_resp = self._request(
            init_method, init_url,
            params={'uploadType': 'resumable', 'fields': 'id,name,size'},
            data=meta_payload,
            headers={'Content-Type': 'application/json; charset=UTF-8', 'X-Upload-Content-Length': str(size)},
            timeout=60, raw_resp=True,
        )
        session_uri = init_resp.headers.get('Location')
        init_resp.close()
        if not session_uri:
            raise RuntimeError(f'Missing resumable Location header for {rel_path}')

        offset = 0
        final_meta = None
        with local_path.open('rb') as f:
            while offset < size:
                f.seek(offset)
                chunk = f.read(CHUNK_UPLOAD_BYTES)
                end_byte = offset + len(chunk) - 1
                resp = self._request(
                    'PUT', session_uri, data=chunk,
                    headers={
                        'Content-Length': str(len(chunk)),
                        'Content-Range': f'bytes {offset}-{end_byte}/{size}',
                    },
                    timeout=300, raw_resp=True,
                )
                code = getattr(resp, 'status', None) or resp.getcode()
                if code == 308:
                    rng = resp.headers.get('Range', '')
                    resp.close()
                    offset = int(rng.split('-')[-1]) + 1 if '-' in rng else offset + len(chunk)
                else:
                    body = resp.read()
                    resp.close()
                    final_meta = json.loads(body.decode('utf-8'))
                    offset = size
        return final_meta

    def download_file(self, rel_path, local_path):
        info = self.stat_file(rel_path)
        if not info:
            raise FileNotFoundError(f'Remote Drive file not found: {rel_path}')
        local_path = Path(local_path)
        expected_size = int(info['size']) if info.get('size') is not None else None
        if local_path.is_file() and expected_size is not None and local_path.stat().st_size == expected_size:
            return local_path

        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = local_path.with_name(f'.{local_path.name}.{uuid.uuid4().hex[:8]}.part')
        last_err = None
        for attempt in range(4):
            try:
                resp = self._request(
                    'GET', f"{DRIVE_FILES_URL}/{info['id']}",
                    params={'alt': 'media'}, timeout=300, raw_resp=True
                )
                written = 0
                try:
                    with tmp.open('wb') as out_f:
                        while True:
                            blk = resp.read(CHUNK_DOWNLOAD_BYTES)
                            if not blk:
                                break
                            out_f.write(blk)
                            written += len(blk)
                finally:
                    resp.close()
                if expected_size is not None and written != expected_size:
                    raise IOError(f'Incomplete Drive download for {rel_path}: got {written}/{expected_size} bytes')
                os.replace(tmp, local_path)
                return local_path
            except Exception as e:
                last_err = e
                if tmp.exists():
                    tmp.unlink()
                if attempt < 3:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
        raise last_err

    def sync_tree_up(self, local_dir, rel_dir):
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            return
        for f in sorted(local_dir.rglob('*')):
            if not f.is_file() or f.name.endswith(('.tmp', '.part', '.partial')):
                continue
            sub = f.relative_to(local_dir).as_posix()
            target_rel = f"{rel_dir.strip('/')}/{sub}"
            st = f.stat()
            sig = (st.st_mtime_ns, st.st_size)
            key = (str(f), target_rel)
            if self._stat_cache.get(key) == sig:
                continue
            self.upload_file(f, target_rel, overwrite=True)
            self._stat_cache[key] = sig

    def sync_tree_down(self, rel_dir, local_dir):
        folder_id = self.resolve_folder(rel_dir, create=False)
        if not folder_id:
            return False
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        items = self._list_query(f"'{folder_id}' in parents and trashed=false", fields='id,name,mimeType,size')
        for item in items:
            child_rel = f"{rel_dir.strip('/')}/{item['name']}"
            child_local = local_dir / item['name']
            if item['mimeType'] == FOLDER_MIME:
                self.sync_tree_down(child_rel, child_local)
            else:
                self.download_file(child_rel, child_local)
        return True


class MountedDriveStore:
    """Filesystem store when Google Drive is mounted at /content/drive/MyDrive in Colab Notebook."""
    mode = 'mount'

    def __init__(self, drive_root='/content/drive/MyDrive/japanese-subtitles'):
        self.root = Path(drive_root)
        if not self.root.parent.is_dir():
            raise RuntimeError(f'Google Drive mount not found at {self.root.parent}')
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, rel_path):
        norm = str(rel_path).strip('/').replace('\\', '/')
        return self.root if norm in ('', '.') else self.root / norm

    def resolve_folder(self, rel_dir, create=True):
        p = self._path(rel_dir)
        if create:
            p.mkdir(parents=True, exist_ok=True)
        return str(p) if p.is_dir() else None

    def exists(self, rel_path):
        return self._path(rel_path).exists()

    def delete_path(self, rel_path):
        norm = str(rel_path).strip('/').replace('\\', '/')
        if not norm:
            raise ValueError('Refusing to delete project root folder')
        p = self._path(norm)
        if not p.exists():
            return False
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        return True

    def delete_folder_if_empty(self, rel_dir):
        p = self._path(rel_dir)
        if p.is_dir() and not any(p.iterdir()):
            p.rmdir()
            return True
        return False

    def read_json(self, rel_path, default=None):
        p = self._path(rel_path)
        return json.loads(p.read_text(encoding='utf-8')) if p.is_file() else default

    def write_json(self, rel_path, data):
        atomic_write_json(self._path(rel_path), data)

    def upload_file(self, local_path, rel_path, overwrite=True):
        src = Path(local_path)
        dst = self._path(rel_path)
        if dst.is_file() and not overwrite and dst.stat().st_size == src.stat().st_size:
            return {'size': str(dst.stat().st_size)}
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f'.{dst.name}.part')
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        return {'size': str(dst.stat().st_size)}

    def download_file(self, rel_path, local_path):
        src = self._path(rel_path)
        if not src.is_file():
            raise FileNotFoundError(f'Mounted Drive file missing: {src}')
        dst = Path(local_path)
        if dst.is_file() and dst.stat().st_size == src.stat().st_size:
            return dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f'.{dst.name}.part')
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        return dst

    def sync_tree_up(self, local_dir, rel_dir):
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            return
        for f in sorted(local_dir.rglob('*')):
            if not f.is_file() or f.name.endswith(('.tmp', '.part', '.partial')):
                continue
            sub = f.relative_to(local_dir).as_posix()
            self.upload_file(f, f"{rel_dir.strip('/')}/{sub}", overwrite=True)

    def sync_tree_down(self, rel_dir, local_dir):
        src_dir = self._path(rel_dir)
        if not src_dir.is_dir():
            return False
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(src_dir.rglob('*')):
            if f.is_file() and not f.name.endswith(('.tmp', '.part', '.partial')):
                sub = f.relative_to(src_dir).as_posix()
                self.download_file(f"{rel_dir.strip('/')}/{sub}", local_dir / sub)
        return True


def open_drive_store(drive_root=None, token_file=None, project_name='japanese-subtitles'):
    tok = Path(token_file) if token_file else Path(os.environ.get('JAS_DRIVE_TOKEN', str(DEFAULT_COLAB_TOKEN_PATH)))
    if tok.is_file():
        return DriveApiStore(tok, project_name=project_name)
    win_tok = default_windows_token_path()
    if win_tok.is_file():
        return DriveApiStore(win_tok, project_name=project_name)
    mount_target = Path(drive_root or f'/content/drive/MyDrive/{project_name}')
    if mount_target.parent.is_dir():
        return MountedDriveStore(mount_target)
    raise RuntimeError('No Google Drive credentials or Colab mount available')
