from plyfile import PlyData, PlyElement
import numpy as np

ply_in = "/home/park/loc_ws/src/render_loc/output/colmap_sgs_test/aligned_map.ply"          # 실제 파일명 수정
ply_out = "/home/park/loc_ws/src/render_loc/output/colmap_sgs_test/splat_filtered.ply"

alpha_thresh = 0.05       # 0.01, 0.03, 0.05 순서로 테스트
scale_percentile = 99.5   # 너무 큰 gaussian 제거

ply = PlyData.read(ply_in)
v = ply["vertex"].data

mask = np.ones(len(v), dtype=bool)

if "opacity" in v.dtype.names:
    opacity = np.asarray(v["opacity"])
    alpha = 1.0 / (1.0 + np.exp(-opacity))
    mask &= alpha > alpha_thresh
    print("after alpha filter:", mask.sum(), "/", len(mask))

scale_names = [n for n in ["scale_0", "scale_1", "scale_2"] if n in v.dtype.names]
if len(scale_names) == 3:
    # 3DGS scale은 log-scale로 저장되는 경우가 많으므로 exp 적용
    scales = np.stack([np.exp(np.asarray(v[n])) for n in scale_names], axis=1)
    max_scale = scales.max(axis=1)
    th = np.percentile(max_scale, scale_percentile)
    mask &= max_scale < th
    print("scale threshold:", th)
    print("after scale filter:", mask.sum(), "/", len(mask))

filtered_v = v[mask]

new_ply = PlyData(
    [PlyElement.describe(filtered_v, "vertex")],
    text=ply.text
)
new_ply.write(ply_out)

print("saved:", ply_out)