export class Outliner {
  constructor(el, { onSelect, onAdd, onDelete }) {
    this.el = el;
    this.onSelect = onSelect;
    this.onAdd = onAdd;
    this.onDelete = onDelete;
    this.selectedId = null;

    this.listEl = document.createElement('div');
    this.listEl.className = 'outliner-list';

    this.addRow = document.createElement('div');
    this.addRow.className = 'outliner-add-row';
    for (const [type, label] of [['sphere', '+ Sphere'], ['box', '+ Box'], ['plane', '+ Plane']]) {
      const btn = document.createElement('button');
      btn.textContent = label;
      btn.className = 'outliner-add';
      btn.onclick = () => this.onAdd(type);
      this.addRow.appendChild(btn);
    }

    el.appendChild(this.addRow);
    el.appendChild(this.listEl);
  }

  render(objects) {
    this.listEl.innerHTML = '';
    for (const obj of objects) {
      const row = document.createElement('div');
      row.className = 'outliner-row' + (obj.id === this.selectedId ? ' selected' : '');
      const label = document.createElement('span');
      label.textContent = obj.id;
      label.className = 'outliner-label';
      label.onclick = () => {
        this.selectedId = obj.id;
        this.onSelect(obj.id);
      };
      const del = document.createElement('button');
      del.textContent = '×';
      del.className = 'outliner-delete';
      del.onclick = (e) => {
        e.stopPropagation();
        if (this.selectedId === obj.id) this.selectedId = null;
        this.onDelete(obj.id);
      };
      row.appendChild(label);
      row.appendChild(del);
      this.listEl.appendChild(row);
    }
  }

  select(id) {
    this.selectedId = id;
  }
}
