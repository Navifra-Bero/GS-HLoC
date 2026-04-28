from .config import load_config, default_config, load_pkl
from .step.step0_align import step0_align
from .step.step1_viewpoints import step1_viewpoints
from .step.step2_render import step2_render
from .step.step2_render_2d import step2_render_2dgs
from .step.step2_render_sgs import step2_render_sgs
from .step.step2_scaffold_render import step2_scaffold_render
from .step.step3_global_desc import step3_global_desc
from .step.step4_build_db import step4_build_db
from .step.step5_retrieval import step5_retrieval
from .step.step6_match import step6_match, step6a_match_viz
from .step.step6_match_dedode import step6_match_dedode, step6a_match_viz_dedode
from .step.step7_pnp import step7_pnp
from .step.multi_cam import (parse_kapture_records, parse_kapture_sensors,
                              parse_kapture_rigs,
                              find_sister_images, load_multi_cam_config)
from .batch_test import localize_single, run_test_batch

OFFLINE_STEPS = ["0_align", "1_viewpoints", "2_render", "3_global_desc", "4_build_db"]
ONLINE_STEPS  = ["5_retrieval", "6_match", "6_match_dedode", "6a_match_viz", "7_pnp"]
STEPS = OFFLINE_STEPS + ONLINE_STEPS
