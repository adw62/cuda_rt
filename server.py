import json
import re
import subprocess
import sys
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file, abort

ROOT = Path(__file__).resolve().parent
SCENES_DIR = ROOT / 'scenes'
JOBS_DIR = ROOT / 'jobs'
SCENES_DIR.mkdir(exist_ok=True)
JOBS_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder='editor', static_url_path='')

jobs = {}  # job_id -> {process, dir, total}

NAME_RE = re.compile(r'^[A-Za-z0-9_-]+$')


def _validate_name(name):
    if not NAME_RE.match(name):
        abort(400, 'scene name must match [A-Za-z0-9_-]+')


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/api/scenes', methods=['GET'])
def list_scenes():
    names = sorted(p.stem for p in SCENES_DIR.glob('*.json') if not p.stem.startswith('_'))
    return jsonify(names)


@app.route('/api/scenes/<name>', methods=['GET'])
def get_scene(name):
    _validate_name(name)
    path = SCENES_DIR / f'{name}.json'
    if not path.exists():
        abort(404, 'no such scene')
    return send_file(path, mimetype='application/json')


@app.route('/api/scenes/<name>', methods=['POST'])
def save_scene(name):
    _validate_name(name)
    data = request.get_json(force=True)
    path = SCENES_DIR / f'{name}.json'
    path.write_text(json.dumps(data, indent=2))
    return jsonify({'ok': True})


@app.route('/api/render', methods=['POST'])
def start_render():
    data = request.get_json(force=True)
    total = data.get('frames', 0)

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    (job_dir / 'img').mkdir(parents=True)

    scene_path = job_dir / 'scene.json'
    scene_path.write_text(json.dumps(data))

    log_path = job_dir / 'log.txt'
    log_file = open(log_path, 'w')
    process = subprocess.Popen(
        [sys.executable, str(ROOT / 'main.py'), str(scene_path)],
        cwd=job_dir,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    jobs[job_id] = {'process': process, 'dir': job_dir, 'total': total, 'log': log_file}
    return jsonify({'job_id': job_id})


@app.route('/api/render/<job_id>/status', methods=['GET'])
def render_status(job_id):
    job = jobs.get(job_id)
    if job is None:
        abort(404, 'no such render job')

    done = len(list((job['dir'] / 'img').glob('*.png')))
    returncode = job['process'].poll()
    finished = returncode is not None
    error = None
    if finished and returncode != 0:
        job['log'].flush()
        tail = (job['dir'] / 'log.txt').read_text()[-2000:]
        error = tail or f'process exited with code {returncode}'

    resp = {'done': done, 'total': job['total'], 'finished': finished, 'error': error}
    if finished and error is None:
        resp['video_url'] = f'/api/render/{job_id}/video'
        resp['gif_url'] = f'/api/render/{job_id}/gif'
    return jsonify(resp)


@app.route('/api/render/<job_id>/video', methods=['GET'])
def render_video(job_id):
    job = jobs.get(job_id)
    if job is None:
        abort(404, 'no such render job')
    video_path = job['dir'] / 'video.avi'
    if not video_path.exists():
        abort(404, 'video not ready')
    return send_file(video_path, mimetype='video/x-msvideo')


@app.route('/api/render/<job_id>/gif', methods=['GET'])
def render_gif(job_id):
    job = jobs.get(job_id)
    if job is None:
        abort(404, 'no such render job')
    gif_path = job['dir'] / 'video.gif'
    if not gif_path.exists():
        abort(404, 'gif not ready')
    return send_file(gif_path, mimetype='image/gif')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
