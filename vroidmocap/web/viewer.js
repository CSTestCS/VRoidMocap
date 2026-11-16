import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { FBXLoader } from "three/addons/loaders/FBXLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";

export class Viewer {
  constructor(canvas, pane) {
    this.canvas = canvas;
    this.pane = pane;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.scene = new THREE.Scene();
    this.bg = new THREE.Color(0x14181e);
    this.scene.background = this.bg;
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 1.6));
    const dl = new THREE.DirectionalLight(0xffffff, 1.6);
    dl.position.set(1.5, 3, 2);
    this.scene.add(dl);
    this.grid = new THREE.GridHelper(40, 40, 0x3a4554, 0x262d37);
    this.scene.add(this.grid);
    this.floor = new THREE.Mesh(new THREE.CircleGeometry(20, 64), new THREE.MeshBasicMaterial({ color: 0x1a2028, transparent: true, opacity: 0.6, depthWrite: false }));
    this.floor.rotation.x = -Math.PI / 2;
    this.floor.position.y = -0.002;
    this.scene.add(this.floor);
    // foot contact markers
    this.contactMarkers = [0, 1].map(() => {
      const m = new THREE.Mesh(new THREE.RingGeometry(0.05, 0.075, 24), new THREE.MeshBasicMaterial({ color: 0x4fd1c5, side: THREE.DoubleSide }));
      m.rotation.x = -Math.PI / 2;
      m.visible = false;
      this.scene.add(m);
      return m;
    });
    this.freeCam = new THREE.PerspectiveCamera(40, 1, 0.05, 500);
    this.freeCam.position.set(0, 1.4, 4);
    this.videoCam = new THREE.PerspectiveCamera(40, 16 / 9, 0.05, 500);
    this.controls = new OrbitControls(this.freeCam, canvas);
    this.controls.target.set(0, 1, 0);
    this.controls.enableDamping = true;
    this.mode = "video";
    this.lockMove = false;
    this.yaw = 0;  // the whole animation turned about the vertical (radians), with the video camera
    this.lockHeight = false;
    this.overlay = null;
    this.anim = null;
    this.frame = 0;
    this.lastFrame = -1;
    this.clock = new THREE.Clock();
    this.followTarget = true;
    new ResizeObserver(() => this.resize()).observe(pane);
    this.resize();
    const loop = () => {
      requestAnimationFrame(loop);
      this.render();
    };
    loop();
  }

  _clearAvatar() {
    if (this.vrm) {
      this.scene.remove(this.vrm.scene);
      VRMUtils.deepDispose(this.vrm.scene);
      this.vrm = null;
    }
    if (this.fbx) {
      this.scene.remove(this.fbx.group);
      this.fbx = null;
    }
  }

  /** An FBX character: info = {placement (4x4 row-major: file -> app world), hips, feet} */
  async loadFBX(url, info) {
    const obj = await new FBXLoader().loadAsync(url);
    this._clearAvatar();
    const group = new THREE.Group();
    group.matrixAutoUpdate = false;
    const m = new THREE.Matrix4();
    m.set(...info.placement);
    this.fbxPlacement = m;
    group.matrix.copy(m);
    group.add(obj);
    obj.traverse((o) => {
      o.frustumCulled = false;
      if (o.isMesh) {
        for (const mat of [].concat(o.material)) { if (mat) mat.side = THREE.DoubleSide; }
      }
    });
    // FBX animations in the file itself are not used
    obj.animations = [];
    const byName = new Map();
    obj.traverse((o) => { if (o.name && !byName.has(o.name)) byName.set(o.name, o); });
    const get = (n) => byName.get(THREE.PropertyBinding.sanitizeNodeName(n)) || byName.get(n);
    this.fbx = { group, root: obj, get, hips: get(info.hips), feet: info.feet.map(get), restHipsY: info.rest_hips_y };
    this.scene.add(group);
    this.nodes = null;
    return obj;
  }

  hipsObject() {
    if (this.fbx) return this.fbx.hips;
    return this.vrm?.humanoid.getNormalizedBoneNode("hips");
  }

  async loadVRM(url) {
    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));
    const gltf = await loader.loadAsync(url);
    const vrm = gltf.userData.vrm;
    this._clearAvatar();
    VRMUtils.removeUnnecessaryVertices(gltf.scene);
    VRMUtils.combineSkeletons?.(gltf.scene);
    this.isVrm0 = vrm.meta?.metaVersion === "0";
    if (this.isVrm0) VRMUtils.rotateVRM0(vrm);
    this.baseYaw = vrm.scene.rotation.y;
    vrm.scene.traverse((o) => { o.frustumCulled = false; });
    this.vrm = vrm;
    this.scene.add(vrm.scene);
    this.nodes = null;
    return vrm;
  }

  setAnimation(meta, buf) {
    const L = meta.layout;
    const f32 = (name) => new Float32Array(buf, L[name].offset, L[name].shape.reduce((a, b) => a * b, 1));
    this.anim = {
      meta, frames: meta.frames, fps: meta.fps, bones: meta.bones, B: meta.bones.length,
      quats: f32("quats"), root: f32("root"), camPos: f32("cam_pos"), camQuat: f32("cam_quat"),
      vfov: f32("cam_vfov"), valid: f32("valid"), contacts: f32("contacts"),
      fbxQuats: L.fbx_quats ? f32("fbx_quats") : null, fbxHips: L.fbx_hips ? f32("fbx_hips") : null,
      fbxNames: meta.fbx_names || null,
      lm2d: f32("lm2d"), vis: f32("vis"), h2d: f32("h2d"), hs: f32("hs"), shot: f32("shot_of"), src: f32("src_frame"),
    };
    this.videoCam.aspect = meta.width / meta.height;
    this.nodes = null;
    this.lastFrame = -1;
    this.resize();
  }

  /** Hand-set keys (fingers, expressions): rig from /api/keys, evalFn(t) -> {channel: value} */
  setKeys(rig, evalFn, expressions) {
    this.keyRig = rig;
    this.keyEval = evalFn;
    this.keyExpr = expressions || [];
    this.keyNodes = null;
    this.morphMap = null;
  }

  _applyKeys(f) {
    if (!this.keyRig || !this.keyEval || !this.anim) return;
    const vals = this.keyEval(f / this.anim.fps);
    const fbx = !!this.fbx;
    if (!this.keyNodes) {
      this.keyNodes = this.keyRig.map((e) => (fbx ? (e.node ? this.fbx.get(e.node) : null) : this.vrm?.humanoid.getNormalizedBoneNode(e.bone)));
    }
    const s = !fbx && this.isVrm0 ? -1 : 1;
    const qa = new THREE.Quaternion(), qb = new THREE.Quaternion(), ax = new THREE.Vector3();
    this.keyRig.forEach((e, i) => {
      const n = this.keyNodes[i];
      if (!n) return;
      const base = e.side + e.finger;
      const curl = vals[base + ".curl"], splay = vals[base + ".splay"];
      const ca = fbx ? e.node_curl_axis : e.curl_axis, sa = fbx ? e.node_splay_axis : e.splay_axis;
      qa.setFromAxisAngle(ax.set(ca[0], ca[1], ca[2]), (e.curl_max * Math.PI / 180) * curl / 100);
      qb.setFromAxisAngle(ax.set(sa[0], sa[1], sa[2]), (e.splay_max * Math.PI / 180) * splay / 10);
      qb.multiply(qa);  // splay after curl
      if (fbx) n.quaternion.multiply(qb);  // on top of the file's rest (fingers are not animated)
      else n.quaternion.set(s * qb.x, qb.y, s * qb.z, qb.w);
    });
    if (this.vrm?.expressionManager) {
      for (const nm of this.keyExpr) this.vrm.expressionManager.setValue(nm, (vals["expr." + nm] || 0) / 100);
    } else if (fbx) {
      if (!this.morphMap) {
        this.morphMap = new Map();
        this.fbx.root.traverse((o) => {
          if (!o.isMesh || !o.morphTargetDictionary) return;
          for (const [nm, k] of Object.entries(o.morphTargetDictionary)) {
            if (!this.morphMap.has(nm)) this.morphMap.set(nm, []);
            this.morphMap.get(nm).push([o, k]);
          }
        });
      }
      for (const nm of this.keyExpr) {
        const w = (vals["expr." + nm] || 0) / 100;
        for (const [o, k] of this.morphMap.get(nm) || []) o.morphTargetInfluences[k] = w;
      }
    }
  }

  _bindNodes() {
    if (!this.vrm || !this.anim) return;
    this.nodes = this.anim.bones.map((b) => this.vrm.humanoid.getNormalizedBoneNode(b));
  }

  lockOffset(f) {
    const r = this.anim.root;
    const o = new THREE.Vector3();
    if (this.lockMove) { o.x = r[f * 3]; o.z = r[f * 3 + 2]; }
    if (this.lockMove && this.lockHeight) { o.y = r[f * 3 + 1] - (this.restHipsY ?? r[1]); }
    return o;
  }

  setFrame(f) {
    this.frame = Math.max(0, Math.min(f, (this.anim?.frames ?? 1) - 1));
  }

  _applyFbx(A, f) {
    const F = this.fbx;
    if (!A.fbxQuats) return;
    if (!this.nodes) this.nodes = A.fbxNames.map((n) => F.get(n));
    const M = this.nodes.length;
    for (let i = 0; i < M; i++) {
      const n = this.nodes[i];
      if (!n) continue;
      const k = (f * M + i) * 4;
      n.quaternion.set(A.fbxQuats[k], A.fbxQuats[k + 1], A.fbxQuats[k + 2], A.fbxQuats[k + 3]);
    }
    if (F.hips) F.hips.position.set(A.fbxHips[f * 3], A.fbxHips[f * 3 + 1], A.fbxHips[f * 3 + 2]);
    // "lock movement": move the whole model back instead of its hips
    const off = this.lockOffset(f);
    F.group.matrix.copy(new THREE.Matrix4().makeRotationY(this.yaw).multiply(new THREE.Matrix4().makeTranslation(-off.x, -off.y, -off.z)).multiply(this.fbxPlacement));
    F.group.matrixWorldNeedsUpdate = true;
    F.group.updateMatrixWorld(true);
    if (this._editAt(f)) {
      const M = this._editCharMatrix(F.hips);
      if (M) { F.group.matrix.premultiply(M); F.group.matrixWorldNeedsUpdate = true; F.group.updateMatrixWorld(true); }
      this._editPose();
    }
  }

  applyPose() {
    const A = this.anim;
    if (!A || (!this.vrm && !this.fbx)) return;
    if (this.fbx) {
      this.restHipsY = this.fbx.restHipsY;
      this._applyFbx(A, this.frame);
      this._applyKeys(this.frame);
      this.fbx.group.updateMatrixWorld(true);
      this.lastFrame = this.frame;
      this._applyCommon(A, this.frame, this.fbx.hips, this.fbx.feet);
      return;
    }
    if (!this.nodes) {
      this._bindNodes();
      const hips = this.vrm.humanoid.getNormalizedBoneNode("hips");
      this.restHipsY = hips ? hips.position.y : 1;
    }
    const f = this.frame;
    const s = this.isVrm0 ? -1 : 1;
    this.vrm.scene.position.set(0, 0, 0);
    this.vrm.scene.scale.set(1, 1, 1);
    this.vrm.scene.rotation.set(0, (this.baseYaw ?? 0) + this.yaw, 0);
    const B = A.B;
    for (let i = 0; i < B; i++) {
      const n = this.nodes[i];
      if (!n) continue;
      const k = (f * B + i) * 4;
      n.quaternion.set(s * A.quats[k], A.quats[k + 1], s * A.quats[k + 2], A.quats[k + 3]);
    }
    const off = this.lockOffset(f);
    const hips = this.nodes[A.bones.indexOf("hips")];
    if (hips) {
      const x = A.root[f * 3] - off.x, y = A.root[f * 3 + 1] - off.y, z = A.root[f * 3 + 2] - off.z;
      hips.position.set(s * x, y, s * z);
    }
    this._applyKeys(f);
    if (this._editAt(f)) {
      this.vrm.scene.updateMatrixWorld(true);
      const M = this._editCharMatrix(hips);
      if (M) {
        this.vrm.scene.updateMatrix();
        const m = M.multiply(this.vrm.scene.matrix);
        m.decompose(this.vrm.scene.position, this.vrm.scene.quaternion, this.vrm.scene.scale);
        this.vrm.scene.updateMatrixWorld(true);
      }
      this._editPose();
    }
    const jumped = Math.abs(f - this.lastFrame) > 2;
    const dt = jumped ? 0 : Math.min(0.1, Math.abs(f - this.lastFrame) / A.fps);
    this.vrm.update(dt);
    if (jumped) this.vrm.springBoneManager?.reset?.();
    this.lastFrame = f;
    this._applyCommon(A, f, hips, ["leftFoot", "rightFoot"].map((b) => this.vrm.humanoid.getNormalizedBoneNode(b)));
  }

  _applyCommon(A, f, hips, feet) {
    const off = this.lockOffset(f);
    // contact markers
    for (let side = 0; side < 2; side++) {
      const m = this.contactMarkers[side];
      const on = A.contacts[f * 2 + side] > 0.5;
      const foot = feet[side];
      if (foot && on) {
        const p = new THREE.Vector3();
        foot.getWorldPosition(p);
        m.position.set(p.x, 0.003, p.z);
      }
      m.visible = on && !this.overlay;
    }
    // video camera
    const cp = A.camPos, cq = A.camQuat;
    const qy = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), this.yaw);
    this.videoCam.position.set(cp[f * 3] - off.x, cp[f * 3 + 1] - off.y, cp[f * 3 + 2] - off.z).applyQuaternion(qy);
    this.videoCam.quaternion.set(cq[f * 4], cq[f * 4 + 1], cq[f * 4 + 2], cq[f * 4 + 3]).premultiply(qy);
    this.videoCam.fov = A.vfov[f];
    if (this._editAt(f) && this.edit.camOn) {
      const c = this.edit.cam, cam = this.videoCam, d2r = Math.PI / 180;
      cam.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), c.yaw * d2r));
      cam.quaternion.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), c.pitch * d2r));
      cam.quaternion.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), c.roll * d2r));
      cam.position.add(new THREE.Vector3(0, 0, -1).applyQuaternion(cam.quaternion).multiplyScalar(c.dist || 0));
      if (c.move) cam.position.add(new THREE.Vector3(c.move[0], c.move[1], c.move[2]));
      if (c.fov) cam.fov = c.fov;
    }
    this.videoCam.updateProjectionMatrix();
    if (this.mode === "free" && this.followTarget && hips) {
      const p = new THREE.Vector3();
      hips.getWorldPosition(p);
      const d = p.clone().sub(this.controls.target);
      this.controls.target.add(d);
      this.freeCam.position.add(d);
    }
  }

  // ---- the room scan: the scenery as coloured points (the Edit step's guide)
  setRoom(buf) {
    if (this.room) { this.scene.remove(this.room); this.room.geometry.dispose(); this.room.material.dispose(); this.room = null; }
    if (!buf) return;
    const n = new Uint32Array(buf, 0, 1)[0];
    const pos = new Float32Array(buf, 4, n * 3);
    const col = new Uint8Array(buf, 4 + n * 12, n * 3);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3, true));
    const m = new THREE.PointsMaterial({ size: 3, sizeAttenuation: false, vertexColors: true, transparent: true, opacity: this.roomOpacity ?? 0.6, depthWrite: false });
    this.room = new THREE.Points(g, m);
    this.room.visible = !!this.roomOn;
    this.scene.add(this.room);
  }
  showRoom(on, opacity) {
    this.roomOn = on;
    if (opacity != null) this.roomOpacity = opacity;
    if (this.room) { this.room.visible = on; this.room.material.opacity = this.roomOpacity ?? 0.6; }
  }

  /** The camera of a correction, driven like a free camera over the picture
   * (the view stays the video's frame): drag to turn it, right-drag (or
   * shift-drag) to move it sideways / up and down, wheel to move it in and
   * out (shift-wheel: zoom), W A S D Q E to move it. The scene follows the
   * mouse, so a point dragged onto the picture stays there. `getCam()`
   * returns the correction's camera settings while they can be changed. */
  bindEditCam(getCam, changed) {
    const el = this.canvas;
    let down = null;
    this._dragged = false;
    const depth = () => {
      const h = this.hipsObject();
      const p = new THREE.Vector3(0, 1, 0);
      h?.getWorldPosition(p);
      return Math.max(0.5, p.distanceTo(this.videoCam.position));
    };
    const perPx = () => {
      const r = this.videoRect();
      return (2 * Math.tan((this.videoCam.fov * Math.PI) / 360)) / Math.max(r.h, 1);  // radians (small angles) per pixel
    };
    const axes = () => {
      const q = this.videoCam.quaternion;
      return { right: new THREE.Vector3(1, 0, 0).applyQuaternion(q), up: new THREE.Vector3(0, 1, 0).applyQuaternion(q), fwd: new THREE.Vector3(0, 0, -1).applyQuaternion(q) };
    };
    const move = (c, v) => { c.move = c.move || [0, 0, 0]; c.move[0] += v.x; c.move[1] += v.y; c.move[2] += v.z; };
    el.addEventListener("pointerdown", (ev) => {
      const c = getCam();
      if (!c) return;
      down = { x: ev.clientX, y: ev.clientY, pan: ev.button === 2 || ev.button === 1 || ev.shiftKey, moved: false };
      this._dragged = false;
      this.controls.enabled = false;
      el.setPointerCapture(ev.pointerId);
    });
    el.addEventListener("pointermove", (ev) => {
      const c = getCam();
      if (!down || !c) return;
      const dx = ev.clientX - down.x, dy = ev.clientY - down.y;
      if (!down.moved && Math.hypot(dx, dy) < 3) return;
      down.moved = true;
      this._dragged = true;
      down.x = ev.clientX; down.y = ev.clientY;
      const k = perPx();
      if (down.pan) {
        const { right, up } = axes(), s = k * depth();
        move(c, right.multiplyScalar(-dx * s).add(up.multiplyScalar(dy * s)));
      } else {
        c.yaw = (c.yaw || 0) + (dx * k * 180) / Math.PI;
        c.pitch = Math.max(-89, Math.min(89, (c.pitch || 0) + (dy * k * 180) / Math.PI));
      }
      changed();
    });
    const up = (ev) => { if (down) { down = null; this.controls.enabled = true; try { el.releasePointerCapture(ev.pointerId); } catch (e) { /* */ } } };
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
    el.addEventListener("contextmenu", (ev) => { if (getCam()) ev.preventDefault(); });
    el.addEventListener("wheel", (ev) => {
      const c = getCam();
      if (!c) return;
      ev.preventDefault();
      if (ev.shiftKey || ev.altKey) c.fov = Math.max(5, Math.min(120, (c.fov || this.videoCam.fov) * Math.exp(ev.deltaY * 0.001)));
      else move(c, axes().fwd.multiplyScalar(-ev.deltaY * 0.001 * depth()));
      changed();
    }, { passive: false });
    window.addEventListener("keydown", (ev) => {
      const c = getCam();
      const t = ev.target;
      if (!c || (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT"))) return;
      const key = ev.key.toLowerCase();
      const dir = { w: [0, 0, 1], s: [0, 0, -1], a: [-1, 0, 0], d: [1, 0, 0], q: [0, -1, 0], e: [0, 1, 0] }[key];
      if (!dir) return;
      ev.preventDefault();
      const { right, up, fwd } = axes(), step = depth() * (ev.shiftKey ? 0.1 : 0.02);
      move(c, right.multiplyScalar(dir[0] * step).add(up.multiplyScalar(dir[1] * step)).add(fwd.multiplyScalar(dir[2] * step)));
      changed();
    });
  }
  /** true once after a drag (a click that ends a drag is not a click on the picture) */
  consumeDrag() { const d = this._dragged; this._dragged = false; return d; }

  // ---- corrections (the Edit step): previewed at their frame only
  /** edit = {frame, camOn, cam: {yaw, pitch, roll, fov, dist}, chrOn, chr: {place: [x, z] | null, dx, dz, dy, yaw}} */
  setEdit(edit) { this.edit = edit; this.lastFrame = -1; }
  _editAt(f) { return !!(this.edit && this.edit.frame === f && this.anim); }
  _editCharMatrix(hipsNode) {
    const e = this.edit;
    if (!e || !e.chrOn || !hipsNode) return null;
    const h = new THREE.Vector3();
    hipsNode.getWorldPosition(h);
    // the camera's left/right and forward on the floor (before any camera correction)
    const cq = this.anim.camQuat, f = this.frame;
    const q = new THREE.Quaternion(cq[f * 4], cq[f * 4 + 1], cq[f * 4 + 2], cq[f * 4 + 3]);
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(q); right.y = 0; right.normalize();
    const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(q); fwd.y = 0; fwd.normalize();
    const t = h.clone();
    if (e.chr.place) { t.x = e.chr.place[0]; t.z = e.chr.place[1]; }
    t.addScaledVector(right, e.chr.dx || 0).addScaledVector(fwd, e.chr.dz || 0);
    t.y += e.chr.dy || 0;
    return new THREE.Matrix4().makeTranslation(t.x, t.y, t.z)
      .multiply(new THREE.Matrix4().makeRotationY(((e.chr.yaw || 0) * Math.PI) / 180))
      .multiply(new THREE.Matrix4().makeTranslation(-h.x, -h.y, -h.z));
  }
  boneNode(name) {
    if (this.vrm) return this.vrm.humanoid.getNormalizedBoneNode(name);
    if (this.fbx) { const n = this.anim?.meta?.fbx_bone_nodes?.[name]; return n ? this.fbx.get(n) : null; }
    return null;
  }
  /** The character's own axes now (world): left (her left), up, forward. */
  bodyAxes() {
    const L = new THREE.Vector3(), R = new THREE.Vector3();
    const l = this.boneNode("leftUpperLeg"), r = this.boneNode("rightUpperLeg");
    if (!l || !r) return null;
    l.getWorldPosition(L); r.getWorldPosition(R);
    const left = L.sub(R); left.y = 0; left.normalize();
    const up = new THREE.Vector3(0, 1, 0);
    return { left, up, fwd: new THREE.Vector3().crossVectors(left, up).normalize() };
  }
  static POSE_ORDER = ["hips", "spine", "chest", "upperChest", "neck", "head",
    "leftUpperArm", "leftLowerArm", "leftHand", "rightUpperArm", "rightLowerArm", "rightHand",
    "leftUpperLeg", "leftLowerLeg", "leftFoot", "rightUpperLeg", "rightLowerLeg", "rightFoot"];
  /** edit.pose = {bone: [bend, out, turn] degrees}: each turns the bone (and what hangs from it)
   * about its joint: bend about her left-right axis, out about her forward axis, turn about the vertical. */
  _editPose() {
    const pose = this.edit?.poseOn ? this.edit.pose : null;
    if (!pose) return;
    const ax = this.bodyAxes();
    if (!ax) return;
    const d2r = Math.PI / 180;
    const root = this.vrm ? this.vrm.scene : this.fbx.group;
    for (const b of Viewer.POSE_ORDER) {
      const a = pose[b];
      if (!a || (!a[0] && !a[1] && !a[2])) continue;
      const n = this.boneNode(b);
      if (!n || !n.parent) continue;
      root.updateMatrixWorld(true);
      const side = b.startsWith("right") ? -1 : 1;  // "out" is away from the body on either side
      // (+ bend: forward, as raising a knee or an arm in front; + out: away from the body)
      const dq = new THREE.Quaternion().setFromAxisAngle(ax.up, a[2] * d2r)
        .multiply(new THREE.Quaternion().setFromAxisAngle(ax.fwd, side * a[1] * d2r))
        .multiply(new THREE.Quaternion().setFromAxisAngle(ax.left, -a[0] * d2r));
      const qw = new THREE.Quaternion(), pw = new THREE.Quaternion();
      n.getWorldQuaternion(qw);
      n.parent.getWorldQuaternion(pw);
      n.quaternion.copy(pw.invert().multiply(dq.multiply(qw)));
      n.updateMatrixWorld(true);
    }
    root.updateMatrixWorld(true);
  }
  /** The posed joints (world), in MediaPipe numbering, with face points from the posed head. */
  poseJoints() {
    const P = (name) => { const n = this.boneNode(name); if (!n) return null; const p = new THREE.Vector3(); n.getWorldPosition(p); return p; };
    const map = { 11: "leftUpperArm", 12: "rightUpperArm", 13: "leftLowerArm", 14: "rightLowerArm", 15: "leftHand", 16: "rightHand",
      23: "leftUpperLeg", 24: "rightUpperLeg", 25: "leftLowerLeg", 26: "rightLowerLeg", 27: "leftFoot", 28: "rightFoot", 31: "leftToes", 32: "rightToes" };
    const out = {};
    for (const [k, b] of Object.entries(map)) { const p = P(b); if (p) out[k] = p.toArray(); }
    const head = P("head"), neck = P("neck") || P("upperChest") || P("chest");
    const ax = this.bodyAxes();
    if (head && neck && ax) {
      const up = head.clone().sub(neck).normalize();
      const eL = P("leftEye"), eR = P("rightEye");
      let left = eL && eR ? eL.clone().sub(eR).normalize() : null;
      if (!left && this.vrm) {  // the head's own left axis (normalized bones: +X is her left; VRM 0.x is turned round)
        const q = new THREE.Quaternion();
        this.boneNode("head").getWorldQuaternion(q);
        left = new THREE.Vector3(this.isVrm0 ? -1 : 1, 0, 0).applyQuaternion(q);
      }
      if (!left) left = ax.left.clone();
      const fwd = new THREE.Vector3().crossVectors(left, up).normalize();
      left = new THREE.Vector3().crossVectors(up, fwd).normalize();
      const at = (u, f, l) => head.clone().addScaledVector(up, u).addScaledVector(fwd, f).addScaledVector(left, l).toArray();
      Object.assign(out, { 0: at(0.06, 0.09, 0), 1: at(0.08, 0.07, 0.02), 2: at(0.08, 0.07, 0.03), 3: at(0.08, 0.065, 0.045),
        4: at(0.08, 0.07, -0.02), 5: at(0.08, 0.07, -0.03), 6: at(0.08, 0.065, -0.045), 7: at(0.05, 0, 0.07), 8: at(0.05, 0, -0.07),
        9: at(0.02, 0.08, 0.02), 10: at(0.02, 0.08, -0.02) });
    }
    return out;
  }
  /** The floor point (x, z) under a click on the picture (video camera), or null. */
  floorAt(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect(), r = this.videoRect();
    const x = ((clientX - rect.left - r.x) / r.w) * 2 - 1, y = -(((clientY - rect.top - r.y) / r.h) * 2 - 1);
    if (Math.abs(x) > 1 || Math.abs(y) > 1) return null;
    const ray = new THREE.Raycaster();
    ray.setFromCamera(new THREE.Vector2(x, y), this.videoCam);
    const p = new THREE.Vector3();
    if (!ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), p)) return null;
    return [p.x, p.z];
  }
  _bonePositions() {
    const out = [];
    const v = () => new THREE.Vector3();
    if (this.vrm) {
      for (const b of ["hips", "spine", "chest", "neck", "head", "leftUpperArm", "rightUpperArm", "leftLowerArm", "rightLowerArm",
        "leftHand", "rightHand", "leftUpperLeg", "rightUpperLeg", "leftLowerLeg", "rightLowerLeg", "leftFoot", "rightFoot"]) {
        const n = this.vrm.humanoid.getNormalizedBoneNode(b);
        if (n) { const p = v(); n.getWorldPosition(p); out.push([b, p]); }
      }
      const head = this.vrm.humanoid.getNormalizedBoneNode("head");
      if (head) { const p = v(); head.getWorldPosition(p); p.y += 0.15; out.push(["headTop", p]); }
    } else if (this.fbx) {
      this.fbx.root.traverse((o) => { if (o.isBone) { const p = v(); o.getWorldPosition(p); out.push([o.name, p]); } });
    }
    return out;
  }
  /** What a correction at the current frame says, in the animation's world: the camera the user
   * looked through (and set, if adjusted), and the character's hips, facing and box in the picture. */
  editSnapshot() {
    const cam = this.videoCam, A = this.anim;
    cam.updateMatrixWorld(true);
    const view = { quat: cam.quaternion.toArray(), pos: cam.position.toArray() };
    const out = { view };
    if (this.edit?.camOn) out.camera = { quat: view.quat, pos: view.pos, vfov: cam.fov };
    const posed = this.edit?.poseOn && this.edit.pose && Object.values(this.edit.pose).some((a) => a && (a[0] || a[1] || a[2]));
    // the box around her in the picture (video pixels): where the detector looks again
    const boxOf = (bones) => {
      const W = A.meta.width, H = A.meta.height;
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const [, p] of bones) {
        const q = p.clone().project(cam);
        if (q.z > 1) continue;
        const px = (q.x + 1) / 2 * W, py = (1 - q.y) / 2 * H;
        x0 = Math.min(x0, px); y0 = Math.min(y0, py); x1 = Math.max(x1, px); y1 = Math.max(y1, py);
      }
      const pw = 0.15 * (x1 - x0), ph = 0.1 * (y1 - y0);
      return isFinite(x0) ? [x0 - pw, y0 - ph, x1 + pw, y1 + ph] : null;
    };
    if (this.edit?.chrOn) {
      const bones = this._bonePositions();
      const get = (n) => (bones.find((b) => b[0] === n) || [])[1];
      const hipsN = this.hipsObject();
      const h = new THREE.Vector3();
      hipsN.getWorldPosition(h);
      let L = get("leftUpperLeg"), R = get("rightUpperLeg");
      if ((!L || !R) && this.fbx) { L = new THREE.Vector3(); R = new THREE.Vector3(); this.fbx.feet[0]?.getWorldPosition(L); this.fbx.feet[1]?.getWorldPosition(R); }
      const fwd = new THREE.Vector3().crossVectors(L.clone().sub(R), new THREE.Vector3(0, 1, 0)).normalize();
      out.character = { hips: h.toArray(), fwd: fwd.toArray(), scale: A.meta.scale || 1, box: boxOf(bones) };
    }
    if (posed) {
      out.pose = { joints: this.poseJoints(), scale: A.meta.scale || 1, box: boxOf(this._bonePositions()),
        changed: Object.entries(this.edit.pose).filter(([, a]) => a && (a[0] || a[1] || a[2])).map(([b]) => b) };
    }
    return out;
  }

  setMode(mode) {
    if (mode === "free" && this.mode !== "free" && this.anim) {
      // start the free camera where the video camera is
      this.freeCam.position.copy(this.videoCam.position);
      this.freeCam.quaternion.copy(this.videoCam.quaternion);
      const hips = this.hipsObject();
      const p = new THREE.Vector3(0, 1, 0);
      hips?.getWorldPosition(p);
      this.controls.target.copy(p);
    }
    this.mode = mode;
  }

  setOverlay(img) {
    this.overlay = img;
    this.scene.background = img ? null : this.bg;
    this.resize();
  }

  videoRect() {
    const w = this.pane.clientWidth, h = this.pane.clientHeight;
    const a = this.anim ? this.anim.meta.width / this.anim.meta.height : 16 / 9;
    let vw = w, vh = w / a;
    if (vh > h) { vh = h; vw = h * a; }
    return { x: (w - vw) / 2, y: (h - vh) / 2, w: vw, h: vh };
  }

  resize() {
    const w = this.pane.clientWidth, h = this.pane.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.freeCam.aspect = w / h;
    this.freeCam.updateProjectionMatrix();
    if (this.overlay) {
      const r = this.videoRect();
      Object.assign(this.overlay.style, { left: r.x + "px", top: r.y + "px", width: r.w + "px", height: r.h + "px" });
    }
  }

  render() {
    if (this.mode === "free") this.controls.update();
    this.applyPose();
    const w = this.pane.clientWidth, h = this.pane.clientHeight;
    const useVideo = this.mode === "video" || this.overlay;
    this.grid.visible = this.showFloor !== false && !this.overlay;
    this.floor.visible = this.grid.visible;
    if (useVideo) {
      const r = this.videoRect();
      this.renderer.setScissorTest(true);
      this.renderer.setViewport(0, 0, w, h);
      this.renderer.setScissor(0, 0, w, h);
      this.renderer.setClearColor(0x000000, this.overlay ? 0 : 1);
      this.renderer.clear();
      this.renderer.setViewport(r.x, h - r.y - r.h, r.w, r.h);
      this.renderer.setScissor(r.x, h - r.y - r.h, r.w, r.h);
      this.videoCam.aspect = r.w / r.h;
      this.videoCam.updateProjectionMatrix();
      this.renderer.render(this.scene, this.videoCam);
      this.renderer.setScissorTest(false);
    } else {
      this.renderer.setViewport(0, 0, w, h);
      this.renderer.render(this.scene, this.freeCam);
    }
  }
}

// MediaPipe skeleton connections for the 2D overlay
export const POSE_EDGES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28], [27, 29], [29, 31], [27, 31], [28, 30], [30, 32], [28, 32], [15, 17], [15, 19], [15, 21], [16, 18], [16, 20], [16, 22], [0, 2], [0, 5], [2, 7], [5, 8], [9, 10]];
export const HAND_EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];

export function drawLandmarks(ctx, A, f, sx, sy) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  if (!A || A.valid[f] < 0.5) return;
  const P = (i) => [A.lm2d[(f * 33 + i) * 2] * sx, A.lm2d[(f * 33 + i) * 2 + 1] * sy];
  const V = (i) => A.vis[f * 33 + i];
  ctx.lineCap = "round";
  for (const pass of [0, 1]) {
    for (const [a, b] of POSE_EDGES) {
      if (V(a) < 0.3 || V(b) < 0.3) continue;
      const left = (a % 2 === 1 && a > 10) || (b % 2 === 1 && b > 10);
      ctx.lineWidth = pass ? 2 : 4;
      ctx.strokeStyle = pass ? (left ? "#ff4fd8" : "#ffe14f") : "rgba(0,0,0,.7)";
      const [x1, y1] = P(a), [x2, y2] = P(b);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    }
  }
  for (let side = 0; side < 2; side++) {
    if (A.hs[f * 2 + side] < 0.5) continue;
    const H = (i) => [A.h2d[((f * 2 + side) * 21 + i) * 2] * sx, A.h2d[((f * 2 + side) * 21 + i) * 2 + 1] * sy];
    ctx.strokeStyle = side === 0 ? "#ff9cf0" : "#fff3a0";
    ctx.lineWidth = 1.2;
    for (const [a, b] of HAND_EDGES) {
      const [x1, y1] = H(a), [x2, y2] = H(b);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    }
  }
}
