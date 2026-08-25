function toHex(rgb01) {
  const c = rgb01.map((v) => Math.round(Math.min(1, Math.max(0, v)) * 255));
  return '#' + c.map((v) => v.toString(16).padStart(2, '0')).join('');
}

function fromHex(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

function field(labelText, inputEl) {
  const wrap = document.createElement('label');
  wrap.className = 'field';
  const span = document.createElement('span');
  span.textContent = labelText;
  wrap.appendChild(span);
  wrap.appendChild(inputEl);
  return wrap;
}

// Static (non-keyframed) per-object material/size properties, matching the
// current renderer scope: only position is animated.
export class Inspector {
  constructor(el, { onChange }) {
    this.el = el;
    this.onChange = onChange;
    this.obj = null;
  }

  render(obj) {
    this.obj = obj;
    this.el.innerHTML = '';
    if (!obj) {
      const empty = document.createElement('div');
      empty.className = 'inspector-empty';
      empty.textContent = 'No object selected';
      this.el.appendChild(empty);
      return;
    }

    const note = document.createElement('div');
    note.className = 'inspector-note';
    note.textContent =
      'Preview color is an approximation of the diffuse color -- the real look ' +
      'comes from the raytraced render.';
    this.el.appendChild(note);

    this._renderGeometryFields(obj);

    const shine = document.createElement('input');
    shine.type = 'range';
    shine.min = '1';
    shine.max = '500';
    shine.step = '1';
    shine.value = obj.shine;
    shine.oninput = () => this._set({ shine: parseFloat(shine.value) });
    this.el.appendChild(field(`Shine (${obj.shine})`, shine));

    const refl = document.createElement('input');
    refl.type = 'range';
    refl.min = '0';
    refl.max = '1';
    refl.step = '0.01';
    refl.value = obj.reflection;
    refl.oninput = () => this._set({ reflection: parseFloat(refl.value) });
    this.el.appendChild(field(`Reflection (${obj.reflection.toFixed(2)})`, refl));

    for (const key of ['ambient', 'diffusion', 'specular']) {
      const color = document.createElement('input');
      color.type = 'color';
      color.value = toHex(obj[key]);
      color.oninput = () => this._set({ [key]: fromHex(color.value) });
      this.el.appendChild(field(key[0].toUpperCase() + key.slice(1), color));
    }

    const physicsNote = document.createElement('div');
    physicsNote.className = 'inspector-note';
    physicsNote.textContent = 'Gravity: falls when physics is baked. Collision: can be hit / can hit other collision objects.';
    this.el.appendChild(physicsNote);

    for (const [key, label] of [['gravity', 'Gravity'], ['collision', 'Collision']]) {
      const wrap = document.createElement('label');
      wrap.className = 'field field-checkbox';
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.checked = !!obj[key];
      checkbox.onchange = () => this._set({ [key]: checkbox.checked });
      const span = document.createElement('span');
      span.textContent = label;
      wrap.appendChild(checkbox);
      wrap.appendChild(span);
      this.el.appendChild(wrap);
    }

    const velocityLabels = ['Velocity X', 'Velocity Y', 'Velocity Z'];
    obj.initialVelocity.forEach((component, i) => {
      const input = document.createElement('input');
      input.type = 'number';
      input.step = '0.1';
      input.value = component;
      input.oninput = () => {
        const next = [...this.obj.initialVelocity];
        next[i] = parseFloat(input.value) || 0;
        this._set({ initialVelocity: next });
      };
      this.el.appendChild(field(velocityLabels[i], input));
    });

    const randomize = document.createElement('input');
    randomize.type = 'number';
    randomize.min = '0';
    randomize.step = '0.1';
    randomize.value = obj.velocityRandomization;
    randomize.oninput = () => this._set({ velocityRandomization: parseFloat(randomize.value) || 0 });
    this.el.appendChild(field('Velocity Randomization', randomize));
  }

  // Static (non-keyframed) geometry: sphere radius, box half-extents, or a
  // plane's normal -- one of these per object, matching its `type`.
  _renderGeometryFields(obj) {
    if (obj.type === 'box') {
      const axisLabels = ['Width (X)', 'Height (Y)', 'Depth (Z)'];
      obj.halfExtents.forEach((half, i) => {
        const input = document.createElement('input');
        input.type = 'range';
        input.min = '0.02';
        input.max = '2';
        input.step = '0.01';
        input.value = half;
        input.oninput = () => {
          const next = [...this.obj.halfExtents];
          next[i] = parseFloat(input.value);
          this._set({ halfExtents: next });
        };
        this.el.appendChild(field(`${axisLabels[i]} (${(half * 2).toFixed(2)})`, input));
      });
    } else if (obj.type === 'plane') {
      const axisLabels = ['Normal X', 'Normal Y', 'Normal Z'];
      obj.normal.forEach((component, i) => {
        const input = document.createElement('input');
        input.type = 'number';
        input.min = '-1';
        input.max = '1';
        input.step = '0.1';
        input.value = component;
        input.oninput = () => {
          const next = [...this.obj.normal];
          next[i] = parseFloat(input.value) || 0;
          this._set({ normal: next });
        };
        this.el.appendChild(field(axisLabels[i], input));
      });
    } else {
      const size = document.createElement('input');
      size.type = 'range';
      size.min = '0.02';
      size.max = '2';
      size.step = '0.01';
      size.value = obj.size;
      size.oninput = () => this._set({ size: parseFloat(size.value) });
      this.el.appendChild(field(`Size (${obj.size.toFixed(2)})`, size));
    }
  }

  _set(patch) {
    Object.assign(this.obj, patch);
    this.onChange(this.obj, patch);
  }
}
