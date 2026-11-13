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
    this.vrm.scene.rotation.y = (this.baseYaw ?? 0) + this.yaw;
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
    this.videoCam.updateProjectionMatrix();
    if (this.mode === "free" && this.followTarget && hips) {
      const p = new THREE.Vector3();
      hips.getWorldPosition(p);
      const d = p.clone().sub(this.controls.target);
      this.controls.target.add(d);
      this.freeCam.position.add(d);
    }
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
