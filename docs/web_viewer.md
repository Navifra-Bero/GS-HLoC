# RenderLoc 웹 라이브 뷰어 구조

ROS 2 가 추정한 pose 와 카메라 영상을, **별도 의존성(rosbridge/Foxglove) 없이**
표준 라이브러리 HTTP 서버 하나로 브라우저에 실시간 전달하는 경량 뷰어다.
Gaussian splat 맵 위에 현재 위치/이동 궤적을 그리고, 상단에 2D 네비 미니맵과
실제 카메라 영상을 함께 보여준다.

관련 다이어그램: [web_viewer_architecture.drawio](web_viewer_architecture.drawio)

---

## 1. 전체 흐름 한눈에

```
ROS 2 토픽 ──(DDS)──▶ web_pose_bridge.py ──(HTTP/SSE/MJPEG)──▶ 브라우저(web/)
  pose/이미지            ROS node + HTTP 서버              three.js + gaussian-splats-3d
```

- **ROS_DOMAIN_ID** 가 필요한 구간은 `토픽 발행 노드 ↔ bridge` 뿐이다.
- **브라우저**는 순수 HTTP 클라이언트라 ROS 를 전혀 모른다. 같은 네트워크면
  다른 PC 에서 `http://<서버IP>:<port>` 로 접속 가능(WebGL2 + 하드웨어 가속 필요).

---

## 2. 구성 요소

### 2.1 오프라인 준비 (launch 시 1회)
- [`scripts/ros/ply_to_splat.py`](../scripts/ros/ply_to_splat.py)
  - `aligned_map.ply`(3DGS PLY) → `gaussian_map.splat`(antimatter15/mkkellogg 포맷)
  - `--z-min/--z-max` 로 Z 범위 슬라이스 가능 (바닥 = z=0 으로 정렬돼 있음, step0 기준)
- [`launch/gaussian_web_viewer.launch.py`](../launch/gaussian_web_viewer.launch.py)
  - 전체 splat 과 (선택) top-down 슬라이스 splat `*_topdown.splat` 을 없으면 생성
  - bridge 노드 실행

### 2.2 브리지 — [`scripts/ros/web_pose_bridge.py`](../scripts/ros/web_pose_bridge.py)
하나의 프로세스 안에서 **ROS 2 노드 + `ThreadingHTTPServer`** 를 함께 돌린다.

- **구독**
  - pose: `PoseStamped` 또는 `Odometry` (기본 `/vps/current_pose`) → JSON 으로 변환
  - 카메라: `CompressedImage`/`Image` (기본 `/cam0/image_raw/compressed`) → JPEG 프레임
- **브로드캐스터(`_Broadcaster`)**: 클라이언트별 큐 + 최신값 캐시.
  - pose 용 1개(SSE), 카메라 프레임용 1개(MJPEG)
  - 느린 클라이언트는 오래된 항목을 버리고 최신만 유지
- **HTTP 정적 서빙**: `web/` 디렉터리 (index.html, viewer.js, lib/*.js)
  - 개발 편의상 모든 응답에 `Cache-Control: no-cache` (브라우저 캐시로 옛 코드가
    뜨는 문제 방지)

### 2.3 브라우저 — [`web/`](../web)
- [`index.html`](../web/index.html): HUD(좌상단), 상단 패널 2개(미니맵 + 카메라), importmap
- [`viewer.js`](../web/viewer.js): 메인 로직
  - `gaussian-splats-3d` 로 splat 렌더 (RViz 식 자유 카메라)
  - `EventSource('/events')` 로 pose 수신 → 마커/방향 화살표/궤적 갱신
  - `<img src="/camera.mjpg">` 로 카메라 영상 표시
  - 2D 캔버스 미니맵에 궤적 + 현재 위치/방향 화살표 그리기
  - `fetch('/config')` 로 카메라 사용 가능 여부 확인
- `lib/three.module.js` + `lib/three.core.js` + `lib/gaussian-splats-3d.module.js`
  (importmap 으로 `three` → 로컬 빌드 매핑)

---

## 3. HTTP 엔드포인트

| 경로 | 형식 | 설명 |
|------|------|------|
| `GET /` , `/viewer.js`, `/lib/*` | HTTP 정적 | 뷰어 페이지/스크립트/three·gsplat 라이브러리 |
| `GET /gaussian_map.splat` | octet-stream | 전체 Gaussian splat 맵 |
| `GET /gaussian_map_topdown.splat` | octet-stream | 바닥 슬라이스 splat (있을 때) |
| `GET /events` | **SSE** | pose 스트림. `data: {type,position,quaternion,stamp,frame_id,topic}` |
| `GET /camera.mjpg` | **MJPEG** (multipart/x-mixed-replace) | 카메라 JPEG 프레임 연속 스트림 |
| `GET /config` | JSON | 사용 가능 스트림 알림 `{topdown, camera}` |
| `GET /clientlog?m=…` | 204 | 브라우저 → 서버 디버그 로그(터미널 출력) |

---

## 4. 런타임 시퀀스

1. 페이지 로드 → `viewer.js` 가 `gaussian_map.splat` 다운로드/렌더 시작
2. `fetch('/config')` → 카메라 가능하면 카메라 패널 활성, `<img>` 를 `/camera.mjpg` 에 연결
3. `EventSource('/events')` 연결 → 좌상단 HUD `연결: connected`
4. ROS 가 pose 발행 → bridge 가 JSON 으로 SSE push → 브라우저가:
   - 3D 씬의 위치 마커/방향 화살표/궤적 갱신
   - 2D 미니맵에 점 추가 + 현재 위치/방향 화살표 갱신
   - HUD 의 `pose# / topic / xyz` 갱신
5. 카메라가 이미지 발행 → bridge 가 JPEG 프레임을 MJPEG 로 push → 카메라 패널 자동 갱신

### 카메라 프레임 변환 규칙
- `CompressedImage` 이고 format 이 jpeg/jpg → **그대로 패스스루**(변환 0)
- 그 외(png 등) 또는 raw `Image` → `cv2`/`cv_bridge` 로 디코드 후 JPEG(품질 80) 재인코딩

---

## 5. 화면 구성 / 조작

- **좌상단 HUD**: 연결 상태, 로드 상태, pose 수, 토픽, xyz, `view:` 상태
- **상단 우측 패널 2개**
  - `navigation`: 2D 미니맵 (궤적 + 현재 위치/방향 화살표 + 격자 + N 북쪽, 자동 스케일)
  - `camera view`: 실제 카메라 MJPEG (토픽 없으면 숨김)
- **메인 3D 뷰**: Gaussian splat 맵 + 바닥 평면 네비 화살표 + 빨간 궤적
- **키 조작**
  - `F` follow marker (마커 추적) — HUD `view: follow marker`
  - `C` look from pose (1인칭 시점) — HUD `view: look from pose`
  - `R` reset view (자유 시점) — HUD `view: free`
  - 마우스: orbit / pan / zoom

---

## 6. 실행 / 반영

```bash
ros2 launch render_loc gaussian_web_viewer.launch.py \
  repo_root:=/home/park/loc_ws/src/render_loc \
  aligned_ply:=.../output/gs_sdf_omni_2/aligned_map.ply \
  splat_path:=.../output/gs_sdf_omni_2/gaussian_map.splat \
  pose_topic:=/vps/current_pose pose_topic_type:=pose \
  image_topic:=/cam0/image_raw/compressed \
  port:=8081
```

- `web/` 파일(viewer.js/index.html)은 소스에서 직접 서빙 → **수정 시 브라우저
  새로고침(Ctrl+Shift+R)만** 하면 반영.
- `web_pose_bridge.py`/launch 를 고쳤다면 ROS 설치본을 쓰므로 **재빌드 필요**:
  `colcon build --packages-select render_loc --symlink-install`

### 주요 launch 인자(기본값)
| 인자 | 기본값 | 설명 |
|------|--------|------|
| `port` | 8080 | HTTP 포트 |
| `pose_topic` | `/vps/current_pose` | pose 토픽 |
| `pose_topic_type` | auto | auto/pose/odom |
| `image_topic` | `/cam0/image_raw/compressed` | 카메라 토픽(빈 값이면 패널 비활성) |
| `image_topic_type` | auto | auto/raw/compressed |
| `topdown_z_min` / `topdown_z_max` | -1.0 / 3.0 | top-down 슬라이스 Z 범위 |
