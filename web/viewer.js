// RenderLoc ROS2 Gaussian map viewer.
//
// - gaussian_map.splat is rendered with @mkkellogg/gaussian-splats-3d.
// - /events streams ROS poses through Server-Sent Events.
// - Incoming poses are visualized as a moving marker, heading arrow, and trail.
// - The map camera remains user-controlled, RViz-style; press F/C for follow modes.
const _bc = (m) => { try { fetch("/clientlog?m=" + encodeURIComponent(m)); } catch (_) {} };
_bc("viewer.js: module-start");

import * as THREE from "three";
import * as GaussianSplats3D from "./lib/gaussian-splats-3d.module.js";

const container = document.getElementById("viewer");
const elConn = document.getElementById("conn");
const elCnt = document.getElementById("cnt");
const elXyz = document.getElementById("xyz");
const elTopic = document.getElementById("topic");
const elDataSub = document.getElementById("dataSub");
const elPoseReady = document.getElementById("poseReady");
const elLocSpeed = document.getElementById("locSpeed");
const elPoseLag = document.getElementById("poseLag");
const elLoad = document.getElementById("load");
const elCam = document.getElementById("camView");
const elCamViews = document.getElementById("camViews");
const elCamPanel = document.getElementById("camPanel");
const elMiniMap = document.getElementById("miniMap");
const elNavPanel = document.getElementById("navPanel");
const elViewMode = document.getElementById("viewmode");
const elHelp = document.getElementById("help");

let lastCameraStampNs = null;

function stampToNs(stamp) {
  if (!stamp) return null;
  if (typeof stamp === "number") {
    return Number.isFinite(stamp) ? stamp * 1e9 : null;
  }
  const sec = Number(stamp.sec ?? 0);
  const nsec = Number(stamp.nanosec ?? stamp.nsec ?? 0);
  if (!Number.isFinite(sec) || !Number.isFinite(nsec)) return null;
  return sec * 1e9 + nsec;
}

function updatePoseLagHud(msg) {
  if (!elPoseLag) return;
  const poseNs = stampToNs(msg.stamp);
  if (poseNs == null) {
    elPoseLag.textContent = "-";
    elPoseLag.className = "warn";
    return;
  }
  const nowNs = Date.now() * 1e6;
  const ageSec = Math.max(0, (nowNs - poseNs) / 1e9);
  const parts = [];
  if (ageSec < 60) {
    parts.push(`age ${ageSec.toFixed(2)}s`);
  } else {
    parts.push("age bag-time");
  }
  let lagSec = ageSec;
  if (lastCameraStampNs != null) {
    const camDeltaSec = (lastCameraStampNs - poseNs) / 1e9;
    lagSec = Math.abs(camDeltaSec);
    parts.push(`cam ${camDeltaSec >= 0 ? "+" : ""}${camDeltaSec.toFixed(2)}s`);
  }
  elPoseLag.textContent = parts.join(" / ");
  elPoseLag.className = lagSec > 1.0 ? "warn" : "ok";
}

function updateViewModeHud() {
  if (!elViewMode) return;
  elViewMode.textContent = followMarker
    ? "follow pose"
    : poseCamera
      ? "look from pose"
      : "free";
}

function setLoad(message, bad = false) {
  if (elLoad) {
    elLoad.textContent = message;
    elLoad.className = bad ? "bad" : "";
  }
  _bc("viewer.js: " + message);
}

setLoad("module loaded");

let viewer;
try {
  viewer = new GaussianSplats3D.Viewer({
    rootElement: container,
    selfDrivenMode: true,
    useBuiltInControls: true,
    cameraUp: [0, 0, 1],
    initialCameraPosition: [-18, -38, 22],
    initialCameraLookAt: [-12, -4, 1.0],
    sharedMemoryForWorkers: false,
  });
  setLoad("viewer created");
} catch (e) {
  setLoad("viewer ctor failed: " + e.message, true);
  throw e;
}

let ready = false;
let lastPose = null;
let followMarker = false;
let poseCamera = false;
let livePoseSeen = false;
let testBagAvailable = false;
let localizerControlAvailable = false;
let gridVisible = true;

const FOLLOW_POSE_BACK_OFFSET_M = 0.45;
const LOOK_FROM_POSE_BACK_OFFSET_M = 1.3;
const POSE_VIEW_LOOKAHEAD_M = 5.0;

const playback = {
  poses: [],
  poseStampNs: [],
  idx: 0,
  playing: false,
  timer: null,
  periodMs: 250,
  autoplay: false,
  syncWithCamera: false,
  syncStartSeq: null,
  imageStride: 1,
};

// Overlay objects live in the normal Three.js scene and render over the splats.
// Declared before viewer.start()/ensureOverlays() so they are not in the TDZ
// when ensureOverlays() runs.
let markerRoot = null;
let navArrow = null;
let trailLine = null;
let grid = null;
const trailPts = [];

viewer.start();
ready = true;
ensureOverlays();
setLoad("viewer started");

// The cache-busting query (?v=...) makes the library's path-based format
// detection fail (it uses endsWith('.splat')), so pass the format explicitly.
// Wrapped in try/catch so a splat problem can never abort the module before
// the SSE/EventSource setup below runs.
const splatUrl = "gaussian_map.splat?v=" + Date.now();
try {
  viewer
    .addSplatScene(splatUrl, {
      format: GaussianSplats3D.SceneFormat.Splat,
      showLoadingUI: true,
      progressiveLoad: true,
      splatAlphaRemovalThreshold: 1,
      onProgress: (pct, label) => {
        const pctText = Number.isFinite(pct) ? `${pct.toFixed(1)}%` : (label || "");
        setLoad(`splat loading ${pctText}`);
      },
    })
    .then(() => {
      ensureOverlays();
      resetView();
      const count = viewer.splatMesh && viewer.splatMesh.getSplatCount
        ? viewer.splatMesh.getSplatCount()
        : "?";
      setLoad(`splat loaded (${count})`);
    })
    .catch((e) => {
      console.error("splat load failed:", e);
      setLoad("splat failed: " + e.message, true);
    });
} catch (e) {
  console.error("splat load failed:", e);
  setLoad("splat failed: " + e.message, true);
}

function ensureOverlays() {
  if (!ready || !viewer.threeScene || markerRoot) return;

  markerRoot = new THREE.Group();
  markerRoot.name = "renderloc_pose_marker";

  // 바닥 평면(XY)에 누운 납작한 네비게이션 화살표. +X 를 진행방향으로 보고
  // markerRoot 의 yaw 회전으로 heading 을 표현한다. 탑다운에서 위에서 보면
  // 네비처럼 보이고, front 뷰에서도 같은 화살표가 함께 나온다.
  const shape = new THREE.Shape();
  shape.moveTo(0.95, 0.0);    // 앞쪽 꼭짓점
  shape.lineTo(-0.55, 0.6);   // 왼쪽 날개
  shape.lineTo(-0.25, 0.0);   // 안쪽 노치
  shape.lineTo(-0.55, -0.6);  // 오른쪽 날개
  shape.closePath();
  const arrowGeo = new THREE.ShapeGeometry(shape);
  const arrowMat = new THREE.MeshBasicMaterial({
    color: 0xff3030,
    depthTest: false,
    transparent: true,
    opacity: 0.95,
    side: THREE.DoubleSide,
  });
  navArrow = new THREE.Mesh(arrowGeo, arrowMat);
  navArrow.renderOrder = 10001;
  // 외곽선으로 어두운 splat 위에서도 잘 보이게.
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(arrowGeo),
    new THREE.LineBasicMaterial({ color: 0xffffff, depthTest: false }),
  );
  edges.renderOrder = 10002;
  navArrow.add(edges);
  markerRoot.add(navArrow);

  grid = new THREE.GridHelper(100, 100, 0x555555, 0x333333);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = 0;
  grid.material.transparent = true;
  grid.material.opacity = 0.18;
  grid.visible = gridVisible;
  viewer.threeScene.add(grid);
  viewer.threeScene.add(markerRoot);

  if (lastPose) updateOverlays(lastPose);
}

function vecFromPose(p) {
  return new THREE.Vector3(p.position.x, p.position.y, p.position.z);
}

function quatFromPose(p) {
  return new THREE.Quaternion(
    p.quaternion.x,
    p.quaternion.y,
    p.quaternion.z,
    p.quaternion.w,
  ).normalize();
}

function quatFromRotationMatrix3(m) {
  const mat = new THREE.Matrix4();
  mat.set(
    m[0][0], m[0][1], m[0][2], 0,
    m[1][0], m[1][1], m[1][2], 0,
    m[2][0], m[2][1], m[2][2], 0,
    0, 0, 0, 1,
  );
  return new THREE.Quaternion().setFromRotationMatrix(mat).normalize();
}

function movementHeadingFromTrail(pos) {
  for (let i = trailPts.length - 1; i >= 0; i--) {
    const d = pos.clone().sub(trailPts[i]);
    d.z = 0;
    if (d.lengthSq() > 0.01) return d.normalize();
  }
  return null;
}

function headingFromPose(p, pos = null) {
  const q = quatFromPose(p);
  const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(q);
  const flatForward = new THREE.Vector3(forward.x, forward.y, 0);
  if (flatForward.lengthSq() > 0.0025) {
    return flatForward.normalize();
  }
  if (pos) {
    const moved = movementHeadingFromTrail(pos);
    if (moved) return moved;
  }
  return new THREE.Vector3(1, 0, 0);
}

function updateTrail(pos) {
  if (!viewer.threeScene) return;
  const last = trailPts[trailPts.length - 1];
  if (!last || last.distanceTo(pos) > 0.03) {
    trailPts.push(pos.clone());
    if (trailPts.length > 5000) trailPts.shift();
  }

  if (!trailLine) {
    const geo = new THREE.BufferGeometry().setFromPoints(trailPts);
    const mat = new THREE.LineBasicMaterial({
      color: 0xff3030,
      depthTest: false,
      transparent: true,
      opacity: 0.95,
    });
    trailLine = new THREE.Line(geo, mat);
    trailLine.renderOrder = 9999;
    viewer.threeScene.add(trailLine);
  } else {
    trailLine.geometry.dispose();
    trailLine.geometry = new THREE.BufferGeometry().setFromPoints(trailPts);
  }
}

function updateViewLocalOrigin(pos) {
  if (!grid) return;
  grid.visible = gridVisible;
  if (followMarker || poseCamera) {
    grid.position.x = pos.x;
    grid.position.y = pos.y;
  } else {
    grid.position.x = 0;
    grid.position.y = 0;
  }
  grid.position.z = 0;
}

function applyPoseView(pos, q) {
  if (!viewer.camera || !viewer.controls) return;
  const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(q).normalize();
  const up = new THREE.Vector3(0, -1, 0).applyQuaternion(q).normalize();
  const backOffset = poseCamera
    ? LOOK_FROM_POSE_BACK_OFFSET_M
    : FOLLOW_POSE_BACK_OFFSET_M;
  const cameraPos = pos.clone().addScaledVector(forward, -backOffset);
  viewer.camera.position.copy(cameraPos);
  viewer.camera.up.copy(up);
  viewer.controls.target.copy(cameraPos.clone().addScaledVector(forward, POSE_VIEW_LOOKAHEAD_M));
  viewer.controls.update();
}

function updateOverlays(p) {
  ensureOverlays();
  if (!markerRoot) return;

  const pos = vecFromPose(p);
  const q = quatFromPose(p);

  // RenderLoc pose is OpenCV optical c2w: local +Z is camera forward.
  // 평면 화살표는 항상 바닥과 평행하게 두고 heading(yaw)만 반영한다.
  const heading = headingFromPose(p, pos);
  const yaw = Math.atan2(heading.y, heading.x);
  markerRoot.position.copy(pos);
  markerRoot.quaternion.setFromAxisAngle(new THREE.Vector3(0, 0, 1), yaw);
  markerRoot.visible = !(followMarker || poseCamera);

  updateTrail(pos);
  updateViewLocalOrigin(pos);

  if (followMarker || poseCamera) {
    applyPoseView(pos, q);
  }
}

function applyPoseMessage(msg, source = "live") {
  lastPose = msg;
  updateOverlays(msg);
  updateMiniMap(msg);
  count += 1;
  elCnt.textContent = String(count);
  elTopic.textContent = source === "trajectory"
    ? `trajectory ${playback.idx + 1}/${playback.poses.length}`
    : (msg.topic || "-");
  elXyz.textContent =
    `${msg.position.x.toFixed(2)}, ${msg.position.y.toFixed(2)}, ${msg.position.z.toFixed(2)}`;
}

function resetView() {
  if (!viewer.camera || !viewer.controls) return;
  viewer.camera.position.set(-18, -38, 22);
  viewer.camera.up.set(0, 0, 1);
  viewer.controls.target.set(-12, -4, 1);
  if (grid) grid.position.set(0, 0, 0);
  if (markerRoot) markerRoot.visible = true;
  viewer.controls.update();
}

function toggleGrid() {
  gridVisible = !gridVisible;
  if (grid) grid.visible = gridVisible;
}

// SSE receiver.
let count = 0;
_bc("viewer.js: creating-EventSource");
const es = new EventSource("/events");
es.onopen = () => {
  _bc("viewer.js: sse-open");
  elConn.textContent = "connected";
  elConn.className = "ok";
};
es.onerror = () => {
  elConn.textContent = "disconnected, retrying";
  elConn.className = "bad";
};
es.onmessage = (ev) => {
  let msg;
  try {
    msg = JSON.parse(ev.data);
  } catch (_) {
    return;
  }
  if (msg.type === "camera_frame") {
    const stampNs = Number(msg.stamp_ns);
    lastCameraStampNs = Number.isFinite(stampNs) ? stampNs : stampToNs(msg.stamp);
    syncTrajectoryToCameraFrame(msg);
    return;
  }
  if (msg.type === "localizer_status") {
    updateLocalizerStatus(msg);
    return;
  }
  if (msg.type !== "pose") return;

  livePoseSeen = true;
  stopTrajectoryPlayback();
  playback.syncWithCamera = false;
  updatePoseLagHud(msg);
  applyPoseMessage(msg, "live");
};

window.addEventListener("keydown", (ev) => {
  if (ev.repeat) return;
  if (ev.code === "KeyF" || ev.key === "f" || ev.key === "F") {
    followMarker = !followMarker;
    if (followMarker) poseCamera = false;
    if (lastPose) updateOverlays(lastPose);
  } else if (ev.code === "KeyC" || ev.key === "c" || ev.key === "C") {
    poseCamera = !poseCamera;
    if (poseCamera) followMarker = false;
    if (lastPose) updateOverlays(lastPose);
  } else if (ev.code === "KeyR" || ev.key === "r" || ev.key === "R") {
    followMarker = false;
    poseCamera = false;
    resetView();
  } else if (ev.code === "KeyG" || ev.key === "g" || ev.key === "G") {
    toggleGrid();
  } else if (ev.code === "KeyP" || ev.key === "p" || ev.key === "P") {
    startTestBagPlayback();
  } else if (ev.code === "BracketLeft" || ev.key === "[") {
    startLocalizer();
  } else if (ev.code === "BracketRight" || ev.key === "]") {
    stopLocalizer();
  } else if (ev.code === "Comma" || ev.key === ",") {
    toggleLocalizationDebug();
  } else {
    return;
  }
  ev.preventDefault();
  ev.stopImmediatePropagation();
  updateViewModeHud();
}, true);
updateViewModeHud();

window.addEventListener("resize", () => {
  if (viewer.camera) {
    viewer.camera.aspect = window.innerWidth / window.innerHeight;
    viewer.camera.updateProjectionMatrix();
  }
});

// ---------------------------------------------------------------------------
// 상단 패널: 카메라(MJPEG) + 2D 네비 미니맵
// ---------------------------------------------------------------------------

const topdownMap = {
  img: null,
  meta: null,
  ready: false,
};

const miniView = {
  centerX: 0,
  centerY: 0,
  mPerPx: 0.1,
  initialized: false,
  poseFollowInitialized: false,
  followPose: true,
  dragging: false,
  dragStartPx: { x: 0, y: 0 },
  dragStartCenter: { x: 0, y: 0 },
};

function syncMiniCanvasSize() {
  if (!elMiniMap) return false;
  const rect = elMiniMap.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));
  if (elMiniMap.width === w && elMiniMap.height === h) return false;
  elMiniMap.width = w;
  elMiniMap.height = h;
  if (miniView.initialized) {
    miniView.mPerPx = clampMiniZoom(miniView.mPerPx);
  }
  return true;
}

function redrawMiniMap() {
  syncMiniCanvasSize();
  if (lastPose) updateMiniMap(lastPose);
  else drawMiniMapIdle();
}

function setupPanelControls(panel) {
  if (!panel) return;
  const toggle = panel.querySelector(".panel-toggle");
  const handle = panel.querySelector(".resize-handle");

  if (toggle) {
    toggle.addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      panel.classList.toggle("collapsed");
      toggle.textContent = panel.classList.contains("collapsed") ? "+" : "−";
      if (panel === elNavPanel && !panel.classList.contains("collapsed")) {
        window.requestAnimationFrame(redrawMiniMap);
      }
    });
  }

  if (!handle) return;
  handle.addEventListener("pointerdown", (ev) => {
    if (panel.classList.contains("collapsed")) return;
    ev.preventDefault();
    ev.stopPropagation();
    const startX = ev.clientX;
    const startY = ev.clientY;
    const startW = panel.offsetWidth;
    const startH = panel.offsetHeight;
    const minW = panel === elNavPanel ? 160 : 220;
    const minH = panel === elNavPanel ? 160 : 150;
    const maxW = Math.max(minW, Math.min(window.innerWidth - 40, 720));
    const maxH = Math.max(minH, Math.min(window.innerHeight - 40, 620));

    const onMove = (moveEv) => {
      const nextW = Math.max(minW, Math.min(maxW, startW - (moveEv.clientX - startX)));
      const nextH = Math.max(minH, Math.min(maxH, startH + (moveEv.clientY - startY)));
      panel.style.width = `${Math.round(nextW)}px`;
      panel.style.height = `${Math.round(nextH)}px`;
      if (panel === elNavPanel) redrawMiniMap();
    };

    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
      if (panel === elNavPanel) redrawMiniMap();
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
  });
}

setupPanelControls(elNavPanel);
setupPanelControls(elCamPanel);
if (window.ResizeObserver && elNavPanel) {
  new ResizeObserver(() => {
    if (!elNavPanel.classList.contains("collapsed")) redrawMiniMap();
  }).observe(elNavPanel);
}

// /config 로 카메라 스트림 사용 가능 여부 확인 후 <img> 연결.
fetch("/config")
  .then((r) => r.json())
  .then((cfg) => {
    const cameras = cfg && Array.isArray(cfg.cameras) && cfg.cameras.length
      ? cfg.cameras
      : (cfg && cfg.camera ? [{ url: cfg.camera, label: "cam" }] : []);
    if (cameras.length && elCamViews && elCamPanel) {
      elCamViews.innerHTML = "";
      elCamViews.style.gridTemplateColumns =
        `repeat(${Math.min(cameras.length, 2)}, minmax(0, 1fr))`;
      for (const [i, cam] of cameras.entries()) {
        const img = document.createElement("img");
        img.className = "camView";
        img.alt = cam.label || `camera view ${i}`;
        img.src = (typeof cam === "string" ? cam : cam.url); // MJPEG → 자동 갱신
        elCamViews.appendChild(img);
      }
      elCamPanel.classList.remove("empty");
    } else if (cfg && cfg.camera && elCam && elCamPanel) {
      elCam.src = cfg.camera;
      elCamPanel.classList.remove("empty");
    }
    if (cfg && cfg.topdown_map_meta) {
      fetch(cfg.topdown_map_meta)
        .then((r) => r.json())
        .then((meta) => {
          const img = new Image();
          img.onload = () => {
            topdownMap.img = img;
            topdownMap.meta = meta;
            topdownMap.ready = true;
            syncMiniCanvasSize();
            fitMiniMapToBounds();
            if (lastPose) updateMiniMap(lastPose);
            else drawMiniMapIdle();
          };
          img.src = (meta.image || cfg.topdown_map) + "?v=" + Date.now();
        })
        .catch(() => {});
    }
    testBagAvailable = !!cfg.test_bag;
    localizerControlAvailable = !!cfg.localizer_control;
    playback.imageStride = Math.max(1, Number(cfg.trajectory_image_stride) || 1);
    if (cfg && cfg.trajectory) {
      playback.autoplay = !!cfg.trajectory_autoplay;
      loadTrajectory(cfg.trajectory);
    }
  })
  .catch(() => {});

// 미니맵: PLY top-down 맵 위에 궤적 + 현재 위치/방향 화살표를 그린다.
const miniPts = [];              // {x, y} 누적 궤적 (월드 좌표, m)
const MINI_MIN_EXTENT = 20;      // 최소 표시 범위(m) — 시작 시 과확대 방지

function mapBounds() {
  return topdownMap.ready && topdownMap.meta && topdownMap.meta.bounds
    ? topdownMap.meta.bounds
    : null;
}

function fitMiniMapToBounds() {
  if (!elMiniMap) return;
  const b = mapBounds();
  if (!b) return;
  miniView.centerX = (b.min_x + b.max_x) / 2;
  miniView.centerY = (b.min_y + b.max_y) / 2;
  const spanX = Math.max(b.max_x - b.min_x, 1e-6);
  const spanY = Math.max(b.max_y - b.min_y, 1e-6);
  miniView.mPerPx = Math.max(spanX / elMiniMap.width, spanY / elMiniMap.height) * 1.08;
  miniView.initialized = true;
}

function ensureMiniView(cur = null) {
  if (!elMiniMap) return;
  if (!miniView.initialized) {
    fitMiniMapToBounds();
    if (!miniView.initialized) {
      miniView.centerX = cur ? cur.x : 0;
      miniView.centerY = cur ? cur.y : 0;
      miniView.mPerPx = MINI_MIN_EXTENT / Math.min(elMiniMap.width, elMiniMap.height);
      miniView.initialized = true;
    }
  }
  if (cur && miniView.followPose && !miniView.dragging) {
    miniView.centerX = cur.x;
    miniView.centerY = cur.y;
    if (!miniView.poseFollowInitialized
        && topdownMap.ready && topdownMap.meta && topdownMap.meta.meters_per_pixel) {
      const mapMpp = Number(topdownMap.meta.meters_per_pixel);
      miniView.mPerPx = clampMiniZoom(
        Math.max(mapMpp * 2.0, 0.08));
      miniView.poseFollowInitialized = true;
    }
  }
}

function currentMiniFrame(W, H) {
  const halfW = W * miniView.mPerPx / 2;
  const halfH = H * miniView.mPerPx / 2;
  return {
    minX: miniView.centerX - halfW,
    maxX: miniView.centerX + halfW,
    minY: miniView.centerY - halfH,
    maxY: miniView.centerY + halfH,
  };
}

function drawMiniMapBase(ctx, W, H, frame) {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0c0f14";
  ctx.fillRect(0, 0, W, H);

  if (topdownMap.ready && topdownMap.img && mapBounds()) {
    drawTopdownMapLayer(ctx, W, H, frame);
    ctx.fillStyle = "rgba(0,0,0,0.18)";
    ctx.fillRect(0, 0, W, H);
  }

  const span = Math.max(frame.maxX - frame.minX, frame.maxY - frame.minY);
  const gridM = niceGridStep(span);
  const toPx = worldToMiniPx(frame, W, H);
  ctx.strokeStyle = topdownMap.ready ? "rgba(255,255,255,0.12)" : "rgba(255,255,255,0.07)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let gx = Math.ceil(frame.minX / gridM) * gridM; gx <= frame.maxX; gx += gridM) {
    const a = toPx(gx, frame.minY);
    const b = toPx(gx, frame.maxY);
    ctx.moveTo(a.px, a.py);
    ctx.lineTo(b.px, b.py);
  }
  for (let gy = Math.ceil(frame.minY / gridM) * gridM; gy <= frame.maxY; gy += gridM) {
    const a = toPx(frame.minX, gy);
    const b = toPx(frame.maxX, gy);
    ctx.moveTo(a.px, a.py);
    ctx.lineTo(b.px, b.py);
  }
  ctx.stroke();
  return gridM;
}

function drawTopdownMapLayer(ctx, W, H, frame) {
  const b = mapBounds();
  const ix0 = Math.max(frame.minX, b.min_x);
  const ix1 = Math.min(frame.maxX, b.max_x);
  const iy0 = Math.max(frame.minY, b.min_y);
  const iy1 = Math.min(frame.maxY, b.max_y);
  if (ix0 >= ix1 || iy0 >= iy1) return;

  const img = topdownMap.img;
  const mapSpanX = Math.max(b.max_x - b.min_x, 1e-6);
  const mapSpanY = Math.max(b.max_y - b.min_y, 1e-6);
  const sx = (ix0 - b.min_x) / mapSpanX * img.width;
  const sy = (b.max_y - iy1) / mapSpanY * img.height;
  const sw = (ix1 - ix0) / mapSpanX * img.width;
  const sh = (iy1 - iy0) / mapSpanY * img.height;
  const toPx = worldToMiniPx(frame, W, H);
  const a = toPx(ix0, iy1);
  const c = toPx(ix1, iy0);
  ctx.drawImage(img, sx, sy, sw, sh, a.px, a.py, c.px - a.px, c.py - a.py);
}

function drawMiniMapIdle() {
  if (!elMiniMap) return;
  const ctx = elMiniMap.getContext("2d");
  if (!ctx) return;
  const W = elMiniMap.width;
  const H = elMiniMap.height;
  ensureMiniView();
  const frame = currentMiniFrame(W, H);
  const gridM = drawMiniMapBase(ctx, W, H, frame);
  drawMiniMapLabels(ctx, W, H, gridM);
}

function worldToMiniPx(frame, W, H) {
  const spanX = Math.max(frame.maxX - frame.minX, 1e-6);
  const spanY = Math.max(frame.maxY - frame.minY, 1e-6);
  return (wx, wy) => ({
    px: (wx - frame.minX) / spanX * W,
    py: (frame.maxY - wy) / spanY * H,
  });
}

function headingFromPreviousPosition(p, cur) {
  if (p.prevPosition) {
    const dx = cur.x - p.prevPosition.x;
    const dy = cur.y - p.prevPosition.y;
    const len = Math.hypot(dx, dy);
    if (len > 1e-4) return { x: dx / len, y: dy / len };
  }

  for (let i = miniPts.length - 1; i >= 0; i--) {
    const dx = cur.x - miniPts[i].x;
    const dy = cur.y - miniPts[i].y;
    const len = Math.hypot(dx, dy);
    if (len > 0.05) return { x: dx / len, y: dy / len };
  }
  return { x: 1, y: 0 };
}

function updateMiniMap(p) {
  if (!elMiniMap) return;
  const ctx = elMiniMap.getContext("2d");
  if (!ctx) return;
  const W = elMiniMap.width;
  const H = elMiniMap.height;

  const cur = { x: p.position.x, y: p.position.y };
  ensureMiniView(cur);
  const last = miniPts[miniPts.length - 1];
  if (!last || Math.hypot(last.x - cur.x, last.y - cur.y) > 0.05) {
    miniPts.push(cur);
    if (miniPts.length > 5000) miniPts.shift();
  }

  // BEV heading은 카메라 orientation 대신 이전 idx -> 현재 idx 이동 방향을 쓴다.
  const heading = headingFromPreviousPosition(p, cur);
  const yaw = Math.atan2(heading.y, heading.x);

  const frame = currentMiniFrame(W, H);
  const toPx = worldToMiniPx(frame, W, H);
  const gridM = drawMiniMapBase(ctx, W, H, frame);

  // 궤적
  if (miniPts.length > 1) {
    ctx.strokeStyle = "#ff7a7a";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < miniPts.length; i++) {
      const s = toPx(miniPts[i].x, miniPts[i].y);
      if (i === 0) ctx.moveTo(s.px, s.py);
      else ctx.lineTo(s.px, s.py);
    }
    ctx.stroke();

    // 시작점
    const st = toPx(miniPts[0].x, miniPts[0].y);
    ctx.fillStyle = "#4ade80";
    ctx.beginPath();
    ctx.arc(st.px, st.py, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  // 현재 위치/방향 화살표
  const c = toPx(cur.x, cur.y);
  const markerOnScreen = c.px >= -20 && c.px <= W + 20 && c.py >= -20 && c.py <= H + 20;
  ctx.save();
  ctx.translate(
    markerOnScreen ? c.px : Math.max(12, Math.min(W - 12, c.px)),
    markerOnScreen ? c.py : Math.max(12, Math.min(H - 12, c.py)));
  ctx.rotate(-yaw);  // 캔버스 y 반전 → 회전 부호 반전
  ctx.beginPath();
  ctx.moveTo(11, 0);
  ctx.lineTo(-7, 7);
  ctx.lineTo(-3, 0);
  ctx.lineTo(-7, -7);
  ctx.closePath();
  ctx.fillStyle = "#ff3030";
  ctx.globalAlpha = markerOnScreen ? 1.0 : 0.65;
  ctx.fill();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.restore();

  // 방위(N) 표시
  drawMiniMapLabels(ctx, W, H, gridM);
}

function drawMiniMapLabels(ctx, W, H, gridM) {
  ctx.fillStyle = "rgba(255,255,255,0.7)";
  ctx.font = "11px ui-monospace, monospace";
  ctx.fillText("N↑", 6, 14);
  ctx.fillText(`${gridM} m`, 6, H - 8);
}

function miniCanvasPoint(ev) {
  const r = elMiniMap.getBoundingClientRect();
  return {
    x: (ev.clientX - r.left) * elMiniMap.width / Math.max(r.width, 1),
    y: (ev.clientY - r.top) * elMiniMap.height / Math.max(r.height, 1),
  };
}

function miniPxToWorld(px, py) {
  return {
    x: miniView.centerX + (px - elMiniMap.width / 2) * miniView.mPerPx,
    y: miniView.centerY - (py - elMiniMap.height / 2) * miniView.mPerPx,
  };
}

function clampMiniZoom(mPerPx) {
  const b = mapBounds();
  if (!b) return Math.max(0.02, Math.min(8.0, mPerPx));
  const span = Math.max(b.max_x - b.min_x, b.max_y - b.min_y, MINI_MIN_EXTENT);
  const minZoom = Math.max((topdownMap.meta.meters_per_pixel || 0.03) * 0.6, 0.02);
  const maxZoom = Math.max(span / Math.min(elMiniMap.width, elMiniMap.height) * 3.0, minZoom);
  return Math.max(minZoom, Math.min(maxZoom, mPerPx));
}

if (elMiniMap) {
  elMiniMap.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    ensureMiniView(lastPose ? { x: lastPose.position.x, y: lastPose.position.y } : null);
    const p = miniCanvasPoint(ev);
    const before = miniPxToWorld(p.x, p.y);
    const zoom = Math.exp(Math.sign(ev.deltaY) * 0.22);
    miniView.mPerPx = clampMiniZoom(miniView.mPerPx * zoom);
    miniView.centerX = before.x - (p.x - elMiniMap.width / 2) * miniView.mPerPx;
    miniView.centerY = before.y + (p.y - elMiniMap.height / 2) * miniView.mPerPx;
    miniView.followPose = false;
    if (lastPose) updateMiniMap(lastPose);
    else drawMiniMapIdle();
  }, { passive: false });

  elMiniMap.addEventListener("pointerdown", (ev) => {
    ensureMiniView(lastPose ? { x: lastPose.position.x, y: lastPose.position.y } : null);
    miniView.dragging = true;
    miniView.followPose = false;
    miniView.dragStartPx = miniCanvasPoint(ev);
    miniView.dragStartCenter = { x: miniView.centerX, y: miniView.centerY };
    elMiniMap.classList.add("dragging");
    elMiniMap.setPointerCapture(ev.pointerId);
  });

  elMiniMap.addEventListener("pointermove", (ev) => {
    if (!miniView.dragging) return;
    const p = miniCanvasPoint(ev);
    const dx = p.x - miniView.dragStartPx.x;
    const dy = p.y - miniView.dragStartPx.y;
    miniView.centerX = miniView.dragStartCenter.x - dx * miniView.mPerPx;
    miniView.centerY = miniView.dragStartCenter.y + dy * miniView.mPerPx;
    if (lastPose) updateMiniMap(lastPose);
    else drawMiniMapIdle();
  });

  const endMiniDrag = (ev) => {
    if (!miniView.dragging) return;
    miniView.dragging = false;
    elMiniMap.classList.remove("dragging");
    try { elMiniMap.releasePointerCapture(ev.pointerId); } catch (_) {}
  };
  elMiniMap.addEventListener("pointerup", endMiniDrag);
  elMiniMap.addEventListener("pointercancel", endMiniDrag);
  elMiniMap.addEventListener("dblclick", () => {
    miniView.followPose = true;
    if (lastPose) updateMiniMap(lastPose);
    else {
      fitMiniMapToBounds();
      drawMiniMapIdle();
    }
  });
}

function loadTrajectory(url) {
  fetch(url + "?v=" + Date.now())
    .then((r) => r.json())
    .then((data) => {
      playback.poses = parseTrajectoryPoses(data);
      playback.poseStampNs = playback.poses.map((p) => p.stampNs);
      playback.idx = 0;
      if (playback.poses.length === 0) return;
      appendTrajectoryHelp();
      applyTrajectoryIndex(0);
      if (playback.autoplay && !livePoseSeen) startTrajectoryPlayback();
    })
    .catch((e) => {
      console.warn("trajectory load failed:", e);
    });
}

function parseTrajectoryPoses(data) {
  const items = Array.isArray(data)
    ? data.map((m, i) => [String(i).padStart(6, "0"), m])
    : Object.entries(data);
  items.sort((a, b) => a[0].localeCompare(b[0], undefined, { numeric: true }));
  const poses = [];
  for (const [name, m] of items) {
    if (!Array.isArray(m) || m.length < 3) continue;
    const q = quatFromRotationMatrix3(m);
    poses.push({
      type: "pose",
      source: "trajectory",
      name,
      position: {
        x: Number(m[0][3]),
        y: Number(m[1][3]),
        z: Number(m[2][3]),
      },
      quaternion: { x: q.x, y: q.y, z: q.z, w: q.w },
      stamp: 0,
      stampNs: stampNsFromTrajectoryName(name),
      frame_id: "map",
      topic: "trajectory",
    });
  }
  const valid = poses.filter((p) =>
    Number.isFinite(p.position.x)
    && Number.isFinite(p.position.y)
    && Number.isFinite(p.position.z));
  for (let i = 1; i < valid.length; i++) {
    valid[i].prevPosition = { ...valid[i - 1].position };
  }
  return valid;
}

function applyTrajectoryIndex(idx) {
  if (playback.poses.length === 0) return;
  playback.idx = ((idx % playback.poses.length) + playback.poses.length) % playback.poses.length;
  const msg = playback.poses[playback.idx];
  applyPoseMessage(msg, "trajectory");
}

function startTrajectoryPlayback() {
  if (playback.timer || playback.poses.length === 0) return;
  playback.syncWithCamera = false;
  playback.playing = true;
  playback.timer = window.setInterval(() => {
    applyTrajectoryIndex(playback.idx + 1);
  }, playback.periodMs);
}

function stopTrajectoryPlayback() {
  playback.playing = false;
  if (playback.timer) {
    window.clearInterval(playback.timer);
    playback.timer = null;
  }
}

function toggleTrajectoryPlayback() {
  if (playback.poses.length === 0) return;
  if (playback.timer) stopTrajectoryPlayback();
  else startTrajectoryPlayback();
}

function resetTrajectoryVisuals() {
  trailPts.length = 0;
  miniPts.length = 0;
  if (trailLine) {
    trailLine.geometry.dispose();
    trailLine.geometry = new THREE.BufferGeometry().setFromPoints([]);
  }
}

function stampNsFromTrajectoryName(name) {
  const base = String(name || "").split(/[\\/]/).pop().replace(/\.[^.]+$/, "");
  const match = base.match(/\d{12,}/);
  if (!match) return null;
  let stampNs = Number(match[0]);
  if (match[0].length <= 13) {
    stampNs *= 1000000; // milliseconds -> nanoseconds
  } else if (match[0].length <= 16) {
    stampNs *= 1000; // microseconds -> nanoseconds
  }
  return Number.isFinite(stampNs) ? stampNs : null;
}

function nearestTrajectoryIndexByStamp(stampNs) {
  const stamps = playback.poseStampNs;
  if (!stamps.length || !Number.isFinite(stampNs) || stampNs <= 0) return null;
  if (!Number.isFinite(stamps[0])) return null;
  let lo = 0;
  let hi = stamps.length - 1;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (stamps[mid] < stampNs) lo = mid + 1;
    else hi = mid;
  }
  if (lo <= 0) return 0;
  const prev = lo - 1;
  return Math.abs(stamps[lo] - stampNs) < Math.abs(stampNs - stamps[prev])
    ? lo
    : prev;
}

function syncTrajectoryToCameraFrame(msg) {
  if (!playback.syncWithCamera || playback.poses.length === 0) return;

  let idx = nearestTrajectoryIndexByStamp(Number(msg.stamp_ns));
  if (idx === null) {
    const seq = Number(msg.seq);
    if (!Number.isFinite(seq)) return;
    if (playback.syncStartSeq === null) playback.syncStartSeq = seq;
    const rel = Math.max(0, seq - playback.syncStartSeq);
    idx = Math.floor(rel / playback.imageStride);
  }

  idx = Math.max(0, Math.min(playback.poses.length - 1, idx));
  if (idx !== playback.idx || !lastPose) applyTrajectoryIndex(idx);
}

async function startTestBagPlayback() {
  if (!testBagAvailable) {
    console.warn("test bag is not configured");
    return;
  }
  stopTrajectoryPlayback();
  try {
    setLoad("starting bag playback");
    const r = await fetch("/test_bag/play", { method: "POST" });
    const body = await r.json().catch(() => ({}));
    if (!r.ok || !body.ok) {
      console.warn("test bag play failed:", body.error || r.statusText);
      const detail = body.detail ? `: ${String(body.detail).split("\n").slice(-2).join(" ")}` : "";
      setLoad("bag play failed" + detail, true);
      return;
    }
    setLoad(body.already_running
      ? `bag already running (${body.pid || "-"})`
      : `bag playback started (${body.pid || "-"})`);
    resetTrajectoryVisuals();
    if (playback.poses.length > 0) {
      playback.syncWithCamera = true;
      playback.syncStartSeq = null;
      applyTrajectoryIndex(0);
    }
  } catch (e) {
    console.warn("test bag play failed:", e);
    setLoad("bag play failed: " + e.message, true);
  }
}

async function setLocalizerEnabled(enabled) {
  if (!localizerControlAvailable) {
    console.warn("localizer control is not configured");
    return;
  }
  try {
    const r = await fetch(enabled ? "/localizer/start" : "/localizer/stop", {
      method: "POST",
    });
    const body = await r.json().catch(() => ({}));
    if (!r.ok || !body.ok) {
      console.warn("localizer control failed:", body.error || r.statusText);
      return;
    }
    setLoad(enabled ? "localizer enabled" : "localizer stopped");
  } catch (e) {
    console.warn("localizer control failed:", e);
  }
}

function startLocalizer() {
  setLocalizerEnabled(true);
}

function stopLocalizer() {
  setLocalizerEnabled(false);
}

async function toggleLocalizationDebug() {
  try {
    const r = await fetch("/localizer/debug/toggle", { method: "POST" });
    const body = await r.json().catch(() => ({}));
    if (!r.ok || !body.ok) {
      console.warn("localizer debug toggle failed:", body.error || r.statusText);
      return;
    }
    setLoad(body.enabled ? "timing debug started" : "timing debug stopped");
  } catch (e) {
    console.warn("localizer debug toggle failed:", e);
  }
}

function updateLocalizerStatus(msg) {
  if (elDataSub) {
    const cams = msg.cams || {};
    const names = Object.keys(cams).sort();
    const camText = names.map((name) =>
      `${name}:${cams[name].recent ? "ok" : "wait"}`);
    if (msg.lidar && msg.lidar.required) {
      camText.push(`lidar:${msg.lidar.recent ? "ok" : "wait"}`);
    }
    elDataSub.textContent = camText.length ? camText.join(" ") : "-";
    elDataSub.className = msg.data_ready ? "ok" : (camText.length ? "warn" : "bad");
  }
  if (elPoseReady) {
    if (msg.pose_ready && msg.enabled) {
      elPoseReady.textContent = `running (${msg.processed || 0})`;
      elPoseReady.className = "ok";
    } else if (msg.pose_ready) {
      elPoseReady.textContent = "ready, press [";
      elPoseReady.className = "ok";
    } else {
      elPoseReady.textContent = msg.undistort_ready ? "waiting data" : "waiting calib";
      elPoseReady.className = "bad";
    }
  }
  if (elLocSpeed) {
    const loc = msg.localization || {};
    const sec = Number(loc.last_sec);
    const lim = Number(loc.rate_limit_hz);
    const pnp = loc.pnp || {};
    const parts = [];
    if (Number.isFinite(sec)) parts.push(`${sec.toFixed(2)}s`);
    if (Number.isFinite(lim) && lim > 0) parts.push(`limit ${lim.toFixed(1)}hz`);
    if (loc.debug_enabled) parts.push(`debug ${loc.debug_samples || 0}`);
    if (pnp.best_cam) {
      const used = Array.isArray(pnp.cams_used) && pnp.cams_used.length
        ? ` used=${pnp.cams_used.join("+")}`
        : "";
      const inliers = Number.isFinite(Number(pnp.inliers))
        ? ` inl=${Number(pnp.inliers)}`
        : "";
      const view = pnp.view_cam
        ? ` view=${pnp.view_cam}${pnp.view_source ? `:${pnp.view_source}` : ""}`
        : "";
      parts.push(`pnp ${pnp.best_cam}${used}${inliers}${view}`);
    }
    if (loc.last_ok === false) parts.push("last fail");
    elLocSpeed.textContent = parts.join(" / ");
    elLocSpeed.className = msg.enabled ? "ok" : "";
  }
}

function appendTrajectoryHelp() {
  if (!elHelp || elHelp.dataset.trajectoryHelp === "1") return;
  elHelp.dataset.trajectoryHelp = "1";
}

// span(표시 범위)에 맞춰 보기 좋은 격자 간격(1/2/5×10ⁿ m) 선택.
function niceGridStep(span) {
  const target = span / 6;
  const pow = Math.pow(10, Math.floor(Math.log10(target)));
  const f = target / pow;
  const step = f >= 5 ? 5 : f >= 2 ? 2 : 1;
  return step * pow;
}
