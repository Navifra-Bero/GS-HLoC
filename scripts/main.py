#!/usr/bin/env python3
"""
RenderLoc Pipeline — Modular Entry Point
=========================================
Offline (DB 구축):
  python3 scripts/main.py --ply_map /path/to/map.ply --step all
  python3 scripts/main.py --ply_map /path/to/map.ply --step 2_render --render_mode gs

Online (로컬라이제이션):
  python3 scripts/main.py --ply_map /path/to/map.ply --step online --query_image /path/to/query.png

Batch test:
  python3 scripts/main.py --ply_map /path/to/map.ply --step test --test_dir /path/to/imgs
"""
import argparse, os, pickle, glob
import numpy as np
import re

from pipeline import (
    load_config, default_config, load_pkl,
    step0_align, step1_viewpoints, step2_render, step2_render_2dgs, step2_render_sgs,
    step2_scaffold_render,
    step3_global_desc, step4_build_db,
    step5_retrieval, step6_match, step6a_match_viz,
    step6_match_dedode, step6a_match_viz_dedode,
    step7_pnp,
    run_test_batch,
    load_multi_cam_config, parse_kapture_records, find_sister_images,
    OFFLINE_STEPS, ONLINE_STEPS, STEPS,
)


def main():
    parser = argparse.ArgumentParser(description="RenderLoc Pipeline")
    parser.add_argument("--ply_map",    required=True)
    parser.add_argument("--config",     default="config/render_loc.yaml")
    parser.add_argument("--output_dir", default="output/gs_test")
    parser.add_argument("--step",       default="all",
                        choices=["all", "offline", "online", "test"] + STEPS)
    parser.add_argument("--query_image", default=None,
                        help="Query 이미지 경로 (없으면 DB 내 self-test)")
    parser.add_argument("--query_dir",  default=None,
                        help="step 6a_match_viz: 배치 쿼리 이미지 폴더")
    parser.add_argument("--test_dir",    default=None,
                        help="배치 테스트용 이미지 디렉토리 또는 단일 파일")
    parser.add_argument("--gt_poses",   default=None,
                        help='GT poses JSON. 형식: {"filename": [[4x4]] or [x,y,z]}')
    parser.add_argument("--save", action="store_true", default=False,
                        help="--step test 시 각 프레임 중간 이미지 저장 (step5/6/7 png, pkl). "
                             "미지정 시 trajectory 파일만 저장.")

    # Render mode
    parser.add_argument("--render_mode", default="gs",
                        choices=["pointcloud", "gs", "2dgs", "sgs", "scaffold_gs"],
                        help="렌더링 방식: pointcloud (O3D), gs (3DGS), 2dgs (2DGS), sgs (Scaffold-GS), "
                             "scaffold_gs (사전 학습된 Scaffold-GS, --sgs_model_path 필요)")

    # scaffold_gs-specific options
    parser.add_argument("--sgs_model_path", default=None,
                        help="scaffold_gs 렌더링 모드용 사전 학습된 모델 폴더 경로")
    parser.add_argument("--sgs_iteration", type=int, default=-1,
                        help="scaffold_gs 렌더링 모드용 모델 iteration (-1 = 최신)")
    parser.add_argument("--sgs_use_train_cameras", action="store_true", default=False,
                        help="cameras.json 학습 카메라를 직접 사용 (train.py 방식). "
                             "step1 viewpoints 대신 사용.")
    parser.add_argument("--sgs_no_ply_compare", action="store_true", default=False,
                        help="scaffold_gs 렌더 시 PLY 비교 렌더를 저장하지 않음 (기본: 저장).")
    parser.add_argument("--sgs_no_ply_depth", action="store_true", default=False,
                        help="scaffold_gs 렌더 시 GS depth가 없을 때 aligned PLY depth fallback을 저장하지 않음 (기본: 저장).")
    parser.add_argument("--sgs_ckpt", default=None,
                        help="scaffold_gs .pth 체크포인트 경로 직접 지정 (예: --sgs_ckpt path/to/chkpnt_best.pth). "
                             "지정 시 --sgs_iteration 무시. 생략 시 자동 결정 (chkpnt_best.json → max iteration).")

    # GS-specific options
    parser.add_argument("--kapture_dir", default="kapture/sensors",
                        help="GS 렌더링용 kapture_mapping/sensors 디렉토리")
    parser.add_argument("--gs_epochs",  type=int, default=150,
                        help="3DGS 학습 epoch 수 (1 epoch = 전체 이미지 1회)")
    parser.add_argument("--gs_subsample", type=int, default=1,
                        help="매핑 이미지 서브샘플 간격 (기본 1=전체)")
    parser.add_argument("--gs_voxel_size", type=float, default=0.03,
                        help="GS 학습 전 voxel downsample 크기(m). 예: 0.05")
    parser.add_argument("--gs_train_size", type=int, default=1920,
                        help="GS 학습 시 이미지 리사이즈 (긴 쪽 기준, 비율 유지)")
    parser.add_argument("--gs_accum", type=int, default=8,
                        help="Gradient accumulation steps (effective batch size)")
    parser.add_argument("--ppisp", action="store_true", default=False,
                        help="PPISP 사용: 카메라간 노출/화이트밸런스 차이 보정 (3DGS 전용)")

    args = parser.parse_args()

    config = load_config(args.config) if os.path.exists(args.config) else default_config()
    os.makedirs(args.output_dir, exist_ok=True)
    run_offline = args.step in ("all", "offline")
    run_online  = args.step in ("all", "online")
    run_test    = args.step == "test"

    # ── Offline ───────────────────────────────────────────────────────
    s0 = None
    if run_offline or args.step == "0_align":
        s0 = step0_align(args.ply_map, config, args.output_dir)
    else:
        s0 = load_pkl(args.output_dir, "step0_data.pkl")

    if run_offline or args.step == "1_viewpoints":
        vps = step1_viewpoints(args.ply_map, config, args.output_dir, s0)
    else:
        d = load_pkl(args.output_dir, "step1_data.pkl")
        vps = d["viewpoints"] if d else []

    if run_offline or args.step == "2_render":
        if args.render_mode == "scaffold_gs":
            if not args.sgs_model_path:
                print("ERROR: --sgs_model_path 를 지정하세요.")
                return
            rendered = step2_scaffold_render(
                args.ply_map, vps, config, args.output_dir,
                step0_data=s0,
                sgs_model_path=args.sgs_model_path,
                sgs_iteration=args.sgs_iteration,
                sgs_ckpt_path=args.sgs_ckpt,
                use_train_cameras=args.sgs_use_train_cameras,
                save_ply_compare=not args.sgs_no_ply_compare,
                save_ply_depth=not args.sgs_no_ply_depth,
            )
        elif args.render_mode in ("gs", "2dgs", "sgs"):
            if not args.kapture_dir:
                print("ERROR: --kapture_dir 을 지정하세요.")
                return
            gs_kwargs = dict(
                kapture_dir=args.kapture_dir,
                gs_epochs=args.gs_epochs,
                subsample=args.gs_subsample,
                voxel_size=args.gs_voxel_size,
                train_img_size=args.gs_train_size,
                accum_steps=args.gs_accum,
                use_ppisp=args.ppisp,
            )
            if args.render_mode == "2dgs":
                rendered = step2_render_2dgs(
                    args.ply_map, vps, config, args.output_dir,
                    step0_data=s0, **gs_kwargs)
            elif args.render_mode == "sgs":
                rendered = step2_render_sgs(
                    args.ply_map, vps, config, args.output_dir,
                    step0_data=s0, **gs_kwargs)
            else:
                rendered = step2_render(
                    args.ply_map, vps, config, args.output_dir,
                    step0_data=s0, mode="gs", **gs_kwargs)
        else:
            rendered = step2_render(
                args.ply_map, vps, config, args.output_dir,
                step0_data=s0, mode="pointcloud",
            )
    else:
        rendered = load_pkl(args.output_dir, "step2_data.pkl") or []
        if not rendered:
            rgb_dir   = os.path.join(args.output_dir, "rendered", "rgb")
            depth_dir = os.path.join(args.output_dir, "rendered", "depth")
            if os.path.isdir(rgb_dir):
                rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")) +
                                   glob.glob(os.path.join(rgb_dir, "*.png")))
                rendered = []
                for i, rgb_path in enumerate(rgb_files):
                    stem = os.path.splitext(os.path.basename(rgb_path))[0]
                    depth_path = os.path.join(depth_dir, stem + ".depth")
                    if not os.path.exists(depth_path):
                        depth_path = os.path.join(depth_dir, stem + ".png")
                    rendered.append({
                        "id":         i,
                        "rgb_path":   rgb_path,
                        "depth_path": depth_path if os.path.exists(depth_path) else "",
                        "pose":       np.eye(4),
                        "floor":      0,
                        "yaw":        0.0,
                    })
                print(f"  [step2 fallback] {len(rendered)} real images loaded from {rgb_dir}")

    need_gd = run_offline or args.step in ("3_global_desc", "4_build_db")
    if run_offline or args.step == "3_global_desc":
        rendered, _ = step3_global_desc(rendered, config, args.output_dir)
    elif need_gd:
        _s3 = load_pkl(args.output_dir, "step3_data.pkl")
        if isinstance(_s3, dict):
            rendered = _s3.get("rendered") or rendered
        elif _s3:
            rendered = _s3

    db = None
    if run_offline or args.step == "4_build_db":
        db = step4_build_db(rendered, args.output_dir)

    # ── DB 로드 (online 단독 실행 시) ────────────────────────────────
    need_db = run_online or run_test or args.step in ONLINE_STEPS
    if need_db and db is None:
        db = load_pkl(args.output_dir, "step4_database.pkl")
        if db is None:
            print("ERROR: step4_database.pkl not found. Run --step 4_build_db first.")
            return

    # ── Batch test ────────────────────────────────────────────────────
    if run_test:
        if not args.test_dir:
            print("ERROR: --step test 사용 시 --test_dir 을 지정하세요.")
            return
        
        _cam = re.search(r'cam_\d+', os.path.abspath(args.test_dir))
        _cam_sub = _cam.group(0) if _cam else "cam_unknown"
        run_test_batch(args.test_dir, db, config, args.output_dir, args.gt_poses,
                       save_images=args.save)
        print(f"\n=== Done === Results in: {args.output_dir}/test_results/{_cam_sub}/")
        return

    # ── Online single query ───────────────────────────────────────────
    # multi-cam: kapture records 로드 (enabled 시)
    _mc_enabled, _mc_cam_ids, _mc_kapture_dir, _mc_primary = load_multi_cam_config(config)
    _mc_records = None
    if _mc_enabled:
        if not os.path.isabs(_mc_kapture_dir):
            _mc_kapture_dir = os.path.join(os.getcwd(), _mc_kapture_dir)
        if os.path.exists(_mc_kapture_dir):
            _mc_records = parse_kapture_records(_mc_kapture_dir)
            print(f"  Multi-cam: cams={_mc_cam_ids}  primary={_mc_primary}  "
                  f"records={len(_mc_records)} timestamps")
        else:
            print(f"  WARNING: multi_cam enabled but kapture_dir not found: {_mc_kapture_dir}")

    def _build_query_images(query_path):
        """query_path에 대한 multi-cam {cam_id: path} 딕셔너리 반환."""
        if not _mc_enabled or _mc_records is None or not query_path:
            return None
        sisters = find_sister_images(query_path, _mc_records, _mc_cam_ids)
        return sisters if len(sisters) > 1 else None

    s5 = None
    if run_online or args.step == "5_retrieval":
        _qimgs = _build_query_images(args.query_image)
        s5 = step5_retrieval(args.query_image, db, config, args.output_dir,
                             query_images=_qimgs)
    elif args.step in ("6_match", "6_match_dedode", "7_pnp"):
        s5 = load_pkl(args.output_dir, "step5_data.pkl")
        if s5 is None:
            print("ERROR: step5_data.pkl 없음. 5_retrieval 먼저 실행.")
            return

    _matcher_name = config.get("features", {}).get("matcher_name", "eloftr").lower()
    _use_dedode   = _matcher_name == "dedode_lightglue"

    if args.step == "6a_match_viz":
        if not args.query_dir:
            print("ERROR: --query_dir 를 지정하세요."); return
        if _use_dedode:
            step6a_match_viz_dedode(args.query_dir, db, config, args.output_dir)
        else:
            step6a_match_viz(args.query_dir, db, config, args.output_dir)
        return

    s6 = None
    if run_online or args.step == "6_match":
        if _use_dedode:
            s6 = step6_match_dedode(s5, config, args.output_dir)
        else:
            s6 = step6_match(s5, config, args.output_dir)
    elif args.step == "6_match_dedode":
        s6 = step6_match_dedode(s5, config, args.output_dir)
    elif args.step == "7_pnp":
        s6 = load_pkl(args.output_dir, "step6_data.pkl")
        if s6 is None:
            print("ERROR: step6_data.pkl 없음. 6_match 먼저 실행.")
            return

    if run_online or args.step == "7_pnp":
        step7_pnp(s6, s5, config, args.output_dir)

    print(f"\n=== Done === Results in: {args.output_dir}/")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith(".png"):
            print(f"  {f}")


if __name__ == "__main__":
    main()
