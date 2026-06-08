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
const elLoad = document.getElementById("load");
const elCam = document.getElementById("camView");
const elCamPanel = document.getElementById("camPanel");
const elMiniMap = document.getElementById("miniMap");
const elViewMode = document.getElementById("viewmode");

function updateViewModeHud() {
  if (!elViewMode) return;
  elViewMode.textContent = followMarker
    ? "follow marker"
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

function updateOverlays(p) {
  ensureOverlays();
  if (!markerRoot) return;

  const pos = vecFromPose(p);
  const q = quatFromPose(p);

  // RenderLoc pose is OpenCV optical c2w: local +Z is camera forward.
  // 평면 화살표는 항상 바닥과 평행하게 두고 heading(yaw)만 반영한다.
  const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(q).normalize();
  const yaw = Math.atan2(forward.y, forward.x);
  markerRoot.position.copy(pos);
  markerRoot.quaternion.setFromAxisAngle(new THREE.Vector3(0, 0, 1), yaw);

  updateTrail(pos);

  if (followMarker && viewer.controls) {
    viewer.controls.target.copy(pos);
  }
  if (poseCamera && viewer.camera && viewer.controls) {
    const up = new THREE.Vector3(0, -1, 0).applyQuaternion(q).normalize();
    const eye = pos.clone().addScaledVector(forward, -7.0).addScaledVector(up, 2.5);
    viewer.camera.position.copy(eye);
    viewer.camera.up.copy(up);
    viewer.controls.target.copy(pos.clone().addScaledVector(forward, 3.0));
    viewer.controls.update();
  }
}

function resetView() {
  if (!viewer.camera || !viewer.controls) return;
  viewer.camera.position.set(-18, -38, 22);
  viewer.camera.up.set(0, 0, 1);
  viewer.controls.target.set(-12, -4, 1);
  viewer.controls.update();
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
  if (msg.type !== "pose") return;

  lastPose = msg;
  updateOverlays(msg);
  updateMiniMap(msg);
  count += 1;
  elCnt.textContent = String(count);
  elTopic.textContent = msg.topic || "-";
  elXyz.textContent =
    `${msg.position.x.toFixed(2)}, ${msg.position.y.toFixed(2)}, ${msg.position.z.toFixed(2)}`;
};

window.addEventListener("keydown", (ev) => {
  if (ev.key === "f" || ev.key === "F") {
    followMarker = !followMarker;
    if (followMarker) poseCamera = false;
  } else if (ev.key === "c" || ev.key === "C") {
    poseCamera = !poseCamera;
    if (poseCamera) followMarker = false;
  } else if (ev.key === "r" || ev.key === "R") {
    followMarker = false;
    poseCamera = false;
    resetView();
  } else {
    return;
  }
  updateViewModeHud();
});
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

// /config 로 카메라 스트림 사용 가능 여부 확인 후 <img> 연결.
fetch("/config")
  .then((r) => r.json())
  .then((cfg) => {
    if (cfg && cfg.camera && elCam && elCamPanel) {
      elCam.src = cfg.camera;            // MJPEG → 자동 갱신
      elCamPanel.classList.remove("empty");
    }
  })
  .catch(() => {});

// 미니맵: 바닥(XY) 평면을 위에서 내려다본 2D 네비. 궤적 + 현재 위치/방향 화살표.
const miniPts = [];              // {x, y} 누적 궤적 (월드 좌표, m)
const MINI_MIN_EXTENT = 20;      // 최소 표시 범위(m) — 시작 시 과확대 방지

function updateMiniMap(p) {
  if (!elMiniMap) return;
  const ctx = elMiniMap.getContext("2d");
  if (!ctx) return;
  const W = elMiniMap.width;
  const H = elMiniMap.height;

  const cur = { x: p.position.x, y: p.position.y };
  const last = miniPts[miniPts.length - 1];
  if (!last || Math.hypot(last.x - cur.x, last.y - cur.y) > 0.05) {
    miniPts.push(cur);
    if (miniPts.length > 5000) miniPts.shift();
  }

  // heading(yaw): pose 는 OpenCV optical c2w → forward = local +Z.
  const q = quatFromPose(p);
  const fwd = new THREE.Vector3(0, 0, 1).applyQuaternion(q).normalize();
  const yaw = Math.atan2(fwd.y, fwd.x);

  // 모든 점 + 현재 위치를 담는 바운드 계산(여백 포함, 최소 범위 보장).
  let minX = cur.x, maxX = cur.x, minY = cur.y, maxY = cur.y;
  for (const pt of miniPts) {
    if (pt.x < minX) minX = pt.x;
    if (pt.x > maxX) maxX = pt.x;
    if (pt.y < minY) minY = pt.y;
    if (pt.y > maxY) maxY = pt.y;
  }
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  let span = Math.max(maxX - minX, maxY - minY, MINI_MIN_EXTENT) * 1.2;
  const scale = Math.min(W, H) / span;    // px per meter

  // 월드(x:동, y:북) → 캔버스(우:+x, 위:+y 이므로 y 반전). 북쪽이 위.
  const toPx = (wx, wy) => ({
    px: W / 2 + (wx - cx) * scale,
    py: H / 2 - (wy - cy) * scale,
  });

  // 배경 + 격자
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0c0f14";
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = "rgba(255,255,255,0.07)";
  ctx.lineWidth = 1;
  const gridM = niceGridStep(span);
  ctx.beginPath();
  for (let gx = Math.ceil(minX / gridM) * gridM; gx <= maxX + gridM; gx += gridM) {
    const a = toPx(gx, cy - span); const b = toPx(gx, cy + span);
    ctx.moveTo(a.px, a.py); ctx.lineTo(b.px, b.py);
  }
  for (let gy = Math.ceil(minY / gridM) * gridM; gy <= maxY + gridM; gy += gridM) {
    const a = toPx(cx - span, gy); const b = toPx(cx + span, gy);
    ctx.moveTo(a.px, a.py); ctx.lineTo(b.px, b.py);
  }
  ctx.stroke();

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
  ctx.save();
  ctx.translate(c.px, c.py);
  ctx.rotate(-yaw);  // 캔버스 y 반전 → 회전 부호 반전
  ctx.beginPath();
  ctx.moveTo(11, 0);
  ctx.lineTo(-7, 7);
  ctx.lineTo(-3, 0);
  ctx.lineTo(-7, -7);
  ctx.closePath();
  ctx.fillStyle = "#ff3030";
  ctx.fill();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.restore();

  // 방위(N) 표시
  ctx.fillStyle = "rgba(255,255,255,0.7)";
  ctx.font = "11px ui-monospace, monospace";
  ctx.fillText("N↑", 6, 14);
  ctx.fillText(`${gridM} m`, 6, H - 8);
}

// span(표시 범위)에 맞춰 보기 좋은 격자 간격(1/2/5×10ⁿ m) 선택.
function niceGridStep(span) {
  const target = span / 6;
  const pow = Math.pow(10, Math.floor(Math.log10(target)));
  const f = target / pow;
  const step = f >= 5 ? 5 : f >= 2 ? 2 : 1;
  return step * pow;
}
