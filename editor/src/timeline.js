const RULER_H = 24;
const ROW_H = 24;
const DIAMOND_R = 5;

function tracksOf(model) {
  const tracks = [
    { id: 'camera', label: 'Camera', keyframes: model.camera.keyframes },
    { id: 'light', label: 'Light', keyframes: model.light.keyframes },
  ];
  for (const obj of model.objects) {
    tracks.push({ id: obj.id, label: obj.id, keyframes: obj.positionKeyframes });
  }
  return tracks;
}

export class Timeline {
  constructor(container, model, callbacks) {
    this.container = container;
    this.model = model;
    this.cb = callbacks; // { onScrub, onMoveKeyframe, onDeleteKeyframe, onAddKeyframe }
    this.currentFrame = 0;
    this.selected = null; // { trackId, frame }
    this.dragging = null; // { type: 'scrub' | 'keyframe', trackId?, frame? }

    container.innerHTML = '';
    container.className = 'timeline';

    this.labelsEl = document.createElement('div');
    this.labelsEl.className = 'timeline-labels';
    const spacer = document.createElement('div');
    spacer.style.height = RULER_H + 'px';
    this.labelsEl.appendChild(spacer);

    this.canvasWrap = document.createElement('div');
    this.canvasWrap.className = 'timeline-canvas-wrap';
    this.canvas = document.createElement('canvas');
    this.canvas.tabIndex = 0;
    this.canvasWrap.appendChild(this.canvas);

    container.appendChild(this.labelsEl);
    container.appendChild(this.canvasWrap);

    this.canvas.addEventListener('mousedown', (e) => this._onMouseDown(e));
    window.addEventListener('mousemove', (e) => this._onMouseMove(e));
    window.addEventListener('mouseup', () => this._onMouseUp());
    this.canvas.addEventListener('keydown', (e) => this._onKeyDown(e));
    window.addEventListener('resize', () => this.render());
  }

  setFrame(frame) {
    this.currentFrame = frame;
  }

  _frameToX(frame) {
    const w = this.canvas.width;
    const span = Math.max(1, this.model.frames - 1);
    return (frame / span) * w;
  }

  _xToFrame(x) {
    const w = this.canvas.width;
    const span = Math.max(1, this.model.frames - 1);
    return (x / w) * span;
  }

  _rowLabelEls() {
    return this.labelsEl.querySelectorAll('.timeline-row-label');
  }

  _rebuildLabels(tracks) {
    // Only rebuild DOM if the set of track ids changed, to avoid losing focus/etc.
    const ids = tracks.map((t) => t.id).join(',');
    if (this._lastLabelIds === ids) return;
    this._lastLabelIds = ids;

    while (this.labelsEl.children.length > 1) this.labelsEl.removeChild(this.labelsEl.lastChild);
    for (const t of tracks) {
      const row = document.createElement('div');
      row.className = 'timeline-row-label';
      row.style.height = ROW_H + 'px';
      const span = document.createElement('span');
      span.textContent = t.label;
      const btn = document.createElement('button');
      btn.textContent = '+';
      btn.title = 'Insert keyframe at playhead';
      btn.onclick = () => this.cb.onAddKeyframe(t.id, this.currentFrame);
      row.appendChild(span);
      row.appendChild(btn);
      this.labelsEl.appendChild(row);
    }
  }

  _hitTestKeyframe(tracks, x, y) {
    const rowIdx = Math.floor((y - RULER_H) / ROW_H);
    if (rowIdx < 0 || rowIdx >= tracks.length) return null;
    const track = tracks[rowIdx];
    const rowCenterY = RULER_H + rowIdx * ROW_H + ROW_H / 2;
    if (Math.abs(y - rowCenterY) > ROW_H / 2) return null;
    for (const kf of track.keyframes) {
      const kx = this._frameToX(kf.frame);
      if (Math.abs(kx - x) <= DIAMOND_R + 3) {
        return { trackId: track.id, frame: kf.frame };
      }
    }
    return { trackId: track.id, frame: null }; // blank area in a valid row
  }

  _onMouseDown(e) {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    this.canvas.focus();

    if (y < RULER_H) {
      this.dragging = { type: 'scrub' };
      this._scrubTo(x);
      return;
    }
    const tracks = tracksOf(this.model);
    const hit = this._hitTestKeyframe(tracks, x, y);
    if (hit && hit.frame !== null) {
      this.selected = hit;
      this.dragging = { type: 'keyframe', trackId: hit.trackId, originalFrame: hit.frame };
      this.render();
    } else {
      this.dragging = { type: 'scrub' };
      this._scrubTo(x);
    }
  }

  _onMouseMove(e) {
    if (!this.dragging) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    if (this.dragging.type === 'scrub') {
      this._scrubTo(x);
    } else if (this.dragging.type === 'keyframe') {
      const frame = Math.max(0, Math.min(this.model.frames - 1, Math.round(this._xToFrame(x))));
      this.dragging.pendingFrame = frame;
      this.render(frame);
    }
  }

  _onMouseUp() {
    if (this.dragging && this.dragging.type === 'keyframe') {
      const { trackId, originalFrame, pendingFrame } = this.dragging;
      if (pendingFrame !== undefined && pendingFrame !== originalFrame) {
        this.cb.onMoveKeyframe(trackId, originalFrame, pendingFrame);
        this.selected = { trackId, frame: pendingFrame };
      }
    }
    this.dragging = null;
    this.render();
  }

  _onKeyDown(e) {
    if ((e.key === 'Delete' || e.key === 'Backspace') && this.selected) {
      this.cb.onDeleteKeyframe(this.selected.trackId, this.selected.frame);
      this.selected = null;
      this.render();
    }
  }

  _scrubTo(x) {
    const frame = Math.max(0, Math.min(this.model.frames - 1, Math.round(this._xToFrame(x))));
    this.currentFrame = frame;
    this.cb.onScrub(frame);
  }

  render() {
    const tracks = tracksOf(this.model);
    this._rebuildLabels(tracks);

    const w = this.canvasWrap.clientWidth;
    const h = RULER_H + tracks.length * ROW_H;
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    const ctx = this.canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);

    // Row backgrounds
    tracks.forEach((t, i) => {
      ctx.fillStyle = i % 2 === 0 ? '#232323' : '#262626';
      ctx.fillRect(0, RULER_H + i * ROW_H, w, ROW_H);
    });

    // Ruler ticks
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, w, RULER_H);
    ctx.strokeStyle = '#555';
    ctx.fillStyle = '#aaa';
    ctx.font = '10px monospace';
    const step = Math.max(1, Math.round((this.model.frames * 34) / Math.max(w, 1)));
    for (let f = 0; f < this.model.frames; f += step) {
      const x = this._frameToX(f);
      ctx.beginPath();
      ctx.moveTo(x, RULER_H - 6);
      ctx.lineTo(x, RULER_H);
      ctx.stroke();
      ctx.fillText(String(f), x + 2, RULER_H - 8);
    }

    // Keyframe diamonds
    tracks.forEach((t, i) => {
      const rowCenterY = RULER_H + i * ROW_H + ROW_H / 2;
      for (const kf of t.keyframes) {
        let frame = kf.frame;
        if (
          this.dragging &&
          this.dragging.type === 'keyframe' &&
          this.dragging.trackId === t.id &&
          this.dragging.originalFrame === kf.frame &&
          this.dragging.pendingFrame !== undefined
        ) {
          frame = this.dragging.pendingFrame;
        }
        const x = this._frameToX(frame);
        const isSelected = this.selected && this.selected.trackId === t.id && this.selected.frame === kf.frame;
        ctx.save();
        ctx.translate(x, rowCenterY);
        ctx.rotate(Math.PI / 4);
        const r = isSelected ? DIAMOND_R + 2 : DIAMOND_R;
        ctx.fillStyle = isSelected ? '#ffcc55' : '#5aa9ff';
        ctx.strokeStyle = '#111';
        ctx.fillRect(-r, -r, r * 2, r * 2);
        ctx.strokeRect(-r, -r, r * 2, r * 2);
        ctx.restore();
      }
    });

    // Playhead
    const px = this._frameToX(this.currentFrame);
    ctx.strokeStyle = '#ff5555';
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, h);
    ctx.stroke();
    ctx.fillStyle = '#ff5555';
    ctx.beginPath();
    ctx.moveTo(px - 5, 0);
    ctx.lineTo(px + 5, 0);
    ctx.lineTo(px, 8);
    ctx.closePath();
    ctx.fill();
  }
}
