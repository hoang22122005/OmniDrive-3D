import io
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from omnishield.data.acquire import read_catalog
from omnishield.data.dataset import OccupancyDataset
from omnishield.data.labels import add_camera_free_proxy
from omnishield.data.manifest import load_manifest
from omnishield.data.prepare import audit, synthetic
from omnishield.engine.checkpoints import save_checkpoint
from omnishield.engine.losses import loss_parts
from omnishield.eval.demo import RECT_HW, DemoRuntime
from omnishield.eval.occupancy import OccupancyMetrics
from omnishield.geometry.calibration import Grid, project
from omnishield.io import load_json, write_json
from omnishield.models.edge import OmniShieldEdge


ROOT = Path(__file__).resolve().parents[1]


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.raw, cls.prepared = Path(cls.tmp.name) / "raw", Path(cls.tmp.name) / "prepared"
        cls.config = load_json(ROOT / "configs/model/smoke.json")
        synthetic(cls.raw, cls.prepared, cls.config)
        cls.manifest_path = cls.prepared / "manifest.json"
        cls.manifest = load_manifest(cls.manifest_path, cls.config, allow_synthetic=True)

    def test_fixture_audit(self):
        report = audit(self.raw, self.prepared, self.manifest, self.config, write=False)
        self.assertEqual(report, load_json(self.prepared / "audit.json"))
        for split in ("train", "val"):
            self.assertTrue(all(count > 0 for count in report["occupancy_classes"][split]))
            self.assertGreater(report["coverage"][split]["depth"], 0)

    def test_mvp_catalog_uses_two_complete_and_disjoint_clips(self):
        index = ROOT / "configs/data/phenikaa_mvp/index.json"
        _, splits, selected, files = read_catalog(index)
        self.assertEqual(splits, {"train": ["COMPLEX CITY"], "val": ["COMPLEX URBAN"], "test": []})
        self.assertEqual([len(row) for row in selected.values()], [200, 200])
        self.assertEqual(len(files), 2802)
        self.assertFalse(set(selected["COMPLEX CITY"]) & set(selected["COMPLEX URBAN"]))
        _, _, sample, selected_files = read_catalog(index, max_frames=4)
        self.assertEqual(len(selected_files), 58)
        for name, frames in sample.items():
            self.assertEqual(frames[0], selected[name][0])
            self.assertEqual(frames[-1], selected[name][-1])

    def test_free_proxy_requires_positive_evidence_and_keeps_occupied(self):
        config = self.config
        grid = Grid.from_config(config["grid"])
        rig = self.manifest["rigs"]["fixture"]
        occ = np.full(grid.shape, -1, np.int8)
        sem = np.full(grid.shape, -1, np.int16)
        cells, valid = grid.indices(np.asarray([[3., 0., .7]]))
        self.assertTrue(valid[0])
        hit = tuple(cells[0])
        occ[hit], sem[hit] = 1, 3
        maps = np.full((4, config["image_hw"][0] // 4, config["image_hw"][1] // 4), 8.)
        add_camera_free_proxy(occ, sem, maps, config, grid, rig, [None] * 4)
        self.assertFalse((occ == 0).any())
        add_camera_free_proxy(occ, sem, maps, config, grid, rig, ["present", None, None, None])
        self.assertTrue((occ == 0).any())
        self.assertTrue((occ == -1).any())
        self.assertEqual((occ[hit], sem[hit]), (1, 3))

    def test_rejects_synthetic_for_real_runs_and_split_overlap(self):
        with self.assertRaisesRegex(ValueError, "smoke"):
            load_manifest(self.manifest_path, self.config)
        overlap = load_json(self.manifest_path)
        overlap["frames"][1]["split"] = "val"
        invalid = self.prepared / "invalid.json"
        write_json(invalid, overlap)
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            load_manifest(invalid, self.config, allow_synthetic=True)

    def test_projection_behind_optical_plane_in_fisheye(self):
        camera = {
            "slot": "front", "model": "kb4", "image_hw": [100, 150],
            "K": [[20, 0, 75], [0, 20, 50], [0, 0, 1]], "max_theta_deg": 95,
            "T_cam_from_ego": np.eye(4), "distortion": [0, 0, 0, 0],
        }
        uv, _, valid = project(np.array([[1, 0, -0.04], [1, 0, -0.2]]), camera, (100, 150))
        self.assertEqual(valid.tolist(), [True, False])
        self.assertAlmostEqual(uv[0, 0], 75 + 20 * np.arctan2(1, -0.04))

    @torch.no_grad()
    def test_rgb_forward_and_all_camera_dropout(self):
        dataset = OccupancyDataset(self.manifest, self.config, self.raw, self.prepared, "train")
        inputs, targets = next(iter(DataLoader(dataset, batch_size=2)))
        model = OmniShieldEdge(self.config).eval()
        result = model(**inputs)
        self.assertEqual(result["semantic_prob"].shape, (2, 5, 4, 24, 24))
        torch.testing.assert_close(result["mass"].sum(dim=1), torch.ones_like(result["p_occ"]))
        weights = {"occupancy": 1, "semantics": 1, "depth": 0.8, "velocity": 0.3}
        self.assertTrue(all(torch.isfinite(value) for value in loss_parts(result, targets, inputs, weights).values()))
        inputs["sensor_valid"].zero_()
        result = model(**inputs)
        self.assertFalse(result["valid"].any())
        torch.testing.assert_close(result["mass"][:, 2], torch.ones_like(result["p_occ"]))
        torch.testing.assert_close(result["p_occ"], torch.full_like(result["p_occ"], 0.5))

    def test_smoke_model_learns_positive_occupancy_iou(self):
        torch.manual_seed(42)
        dataset = OccupancyDataset(self.manifest, self.config, self.raw, self.prepared, "train")
        inputs, targets = next(iter(DataLoader(dataset, batch_size=2)))
        model = OmniShieldEdge(self.config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
        weights = {"occupancy": 1.0, "semantics": 1.0, "depth": 0.8}
        for _ in range(20):
            optimizer.zero_grad()
            loss = sum(loss_parts(model(**inputs), targets, inputs, weights).values())
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            metrics = OccupancyMetrics(len(self.config["classes"]), torch.device("cpu"))
            metrics.update(model(**inputs), targets)
        self.assertGreater(metrics.result()["occupancy_iou"], 0.01)

    def test_foreground_range_encodes_nearest_measured_surface(self):
        from omnishield.eval.scene3d import pack_foreground_ranges

        camera = {
            "slot": "front", "model": "pinhole", "image_hw": [64, 64], "max_theta_deg": 85,
            "K": [[32, 0, 31.5], [0, 32, 31.5], [0, 0, 1]], "T_cam_from_ego": np.eye(4).tolist(),
        }
        rig = {"cameras": [camera], "ego_xy_bounds_m": [-0.1, 0.1, -0.1, 0.1]}
        # Two returns on the same ray should report the near hit in centimeters.
        point = np.array([0.4, 0.2, 2.0])
        payload = pack_foreground_ranges(np.stack([point * 1.5, point]), rig, height=32, width=32)
        np.testing.assert_array_equal(np.frombuffer(payload, dtype="<u4", count=2), [32, 32])
        ranges = np.frombuffer(payload, dtype="<u2", offset=8, count=32 * 32)
        self.assertGreater(np.count_nonzero(ranges), 0)
        np.testing.assert_array_equal(np.unique(ranges[ranges > 0]), [round(100 * np.linalg.norm(point))])

    def test_camera_only_replay_without_lidar_targets(self):
        args = SimpleNamespace(raw=self.raw, prepared=self.prepared, model=ROOT / "configs/model/smoke.json",
                               no_ai=True, split="val", device="cpu", allow_synthetic=True,
                               threshold=0.5, max_uncertainty=0.85)
        with patch("omnishield.eval.demo.load_targets", side_effect=AssertionError("target read")), \
                patch("omnishield.eval.demo.load_lidar", side_effect=AssertionError("scan read")):
            runtime = DemoRuntime(args)
            mask = [True] * 4
            self.assertFalse(any(row["lidar_present"] for row in runtime.frame_meta))
            for i in range(len(runtime.dataset)):
                self.assertIsNone(runtime.foreground_binary(i))
                self.assertFalse(runtime.infer(i, mask, 0.5, 0.85)["adaptive_bowl"]["seam_zones"])
                self.assertTrue(runtime.image_bytes(i, 0)[0])
            front = next(iter(runtime.surround.values())).rect_cameras[0]
            self.assertEqual(front["horizontal_fov_deg"], 120)
            self.assertAlmostEqual(np.asarray(front["T_cam_from_ego"])[2, 2], -np.sin(np.deg2rad(17.5)))
            self.assertTrue(runtime.rect_jpeg(0, 0))
            self.assertTrue(runtime.underbody_jpeg(0, mask))

    def test_demo_runtime_serves_metadata_images_and_dropout_inference(self):
        ckpt_path = self.prepared / "demo_smoke.pt"
        model = OmniShieldEdge(self.config).eval()
        save_checkpoint(ckpt_path, {
            "schema": 1, "model_config": self.config, "dino_revision": None,
            "model": model.state_dict(), "supervised_heads": ["occupancy", "semantics"],
        })
        args = type("Args", (), {
            "raw": self.raw, "prepared": self.prepared, "checkpoint": ckpt_path,
            "dino_source": None, "split": "val", "device": "cpu",
            "allow_synthetic": True, "threshold": 0.5, "max_uncertainty": 0.85,
        })()
        runtime = DemoRuntime(args)
        meta = runtime.metadata()
        self.assertEqual(len(meta["frames"]), 3)
        img_bytes, mime = runtime.image_bytes(0, 0)
        self.assertEqual(mime, "image/png")
        self.assertGreater(len(img_bytes), 0)
        res_on = runtime.infer(0, [True, True, True, True], 0.5, 0.85)
        self.assertEqual(res_on["sensor_valid"], [True, True, True, True])
        self.assertTrue(runtime.bev_jpeg(0, [True, True, True, True]).startswith(b"\xff\xd8"))
        surround = runtime.surround[meta["source"]]
        rh, rw = RECT_HW
        for cam_idx, (rect, source) in enumerate(zip(surround.rect_cameras, surround.cameras, strict=True)):
            with Image.open(io.BytesIO(runtime.rect_jpeg(0, cam_idx))) as image:
                self.assertEqual(image.size, (rw, rh))
            virtual = np.asarray(rect["T_cam_from_ego"])
            self.assertAlmostEqual(virtual[0, 2], 0.0)
            k = np.asarray(rect["K"])
            v, u = rh // 2, rw // 2
            ray = np.linalg.solve(k, [u, v, 1])
            point = 3 * ray @ virtual[:3, :3] - virtual[:3, 3] @ virtual[:3, :3]
            uv, _, valid = project(point[None], source, source["image_hw"])
            self.assertTrue(valid[0])
            sh, sw = source["image_hw"]
            np.testing.assert_allclose(surround.rect_grids[cam_idx, v, u], 2 * (uv[0] + 0.5) / [sw, sh] - 1, atol=1e-6)
            np.testing.assert_allclose(surround._project_rect_line(cam_idx, point[None]), [[(u + 0.5) / rw, (v + 0.5) / rh]], atol=1e-6)
        self.assertTrue(runtime.underbody_jpeg(0, [True, True, True, True]).startswith(b"\xff\xd8"))
        self.assertIn("odom_from_prev", meta["frames"][0])
        self.assertIn("steer_rad", meta["frames"][0])
        self.assertGreater(len(runtime.bowl_binary()), 1024)
        self.assertIn("camera_calib", meta["surround"])
        self.assertEqual(len(meta["surround"]["camera_calib"]), 4)
        seams = meta["surround"]["bowl_seams"]
        self.assertEqual([pair["cameras"] for pair in seams["pairs"]], [[0, 2], [2, 1], [1, 3], [3, 0]])
        slots = seams["rings"] * len(seams["offsets"])
        self.assertTrue(all(len(mapping) == slots for pair in seams["pairs"] for mapping in pair["uv"]))
        zero = seams["displacements"].index(0.0)
        for pair in seams["pairs"]:
            self.assertEqual(pair["uv"], [samples[zero] for samples in pair["refine_uv"]])
            self.assertTrue(all(len(samples) == len(seams["displacements"]) for samples in pair["refine_uv"]))
        overlap_uv = meta["surround"]["overlap_uv"]
        self.assertEqual(len(overlap_uv), 4)
        self.assertEqual(len({len(points) for points in overlap_uv}), 1)
        self.assertTrue(all(0 <= value < 1 for points in overlap_uv
                            for uv in points if uv is not None for value in uv))
        for c_cal in meta["surround"]["camera_calib"]:
            self.assertEqual(len(c_cal["ds"]), 6)
            self.assertLess(c_cal["ds_rmse_px"], 1.5)
        self.assertEqual(len(res_on["adaptive_bowl"]["inflect_radii"]), 120)
        self.assertEqual(len(res_on["adaptive_bowl"]["cam_gains"]), 4)
        self.assertIn("objects", res_on["ground_truth"])
        self.assertIn("road_edges", res_on["ground_truth"])
        self.assertIn("corridor", res_on["ground_truth"])
        self.assertIn("seam_zones", res_on["ground_truth"])
        self.assertEqual(len(res_on["ground_truth"]["pdc_sectors"]), 8)
        gt_mesh = runtime.mesh_binary(0, [True, True, True, True], 0.5, 0.85, source="gt")
        self.assertGreaterEqual(len(gt_mesh), 8)
        for src in ("pred", "gt"):
            splat_blob = runtime.splats_binary(0, [True, True, True, True], 0.5, 0.85, source=src)
            n_splats = int(np.frombuffer(splat_blob[:4], dtype="<u4")[0])
            self.assertGreater(n_splats, 100)
            self.assertEqual(len(splat_blob), 4 + n_splats * 40)
        res_off = runtime.infer(0, [False, False, False, False], 0.5, 0.85)
        self.assertEqual(res_off["prediction"]["counts"]["occupied"], 0)
        self.assertEqual(res_off["p_occ_max"], 0.5)

    def test_double_sphere_exact_closed_form_roundtrip_and_kb4_fit(self):
        from omnishield.geometry.double_sphere import (
            ds_project,
            ds_unproject,
            fit_double_sphere_from_camera,
        )

        # 1. Exact analytical inverse round-trip: pi(pi^-1(u, v)) == (u, v) to < 1e-4 px
        ds_params = np.array([455.0, 456.0, 941.0, 534.0, -0.25, 0.598], dtype=np.float64)
        uu, vv = np.meshgrid(np.linspace(150, 1770, 20), np.linspace(100, 980, 14))
        uv_in = np.stack([uu.ravel(), vv.ravel()], axis=-1)
        rays, valid_u = ds_unproject(uv_in, ds_params)
        self.assertTrue(valid_u.all())
        np.testing.assert_allclose(np.linalg.norm(rays, axis=-1), 1.0, atol=1e-6)
        uv_rt, valid_p = ds_project(rays * 5.0, ds_params)
        self.assertTrue(valid_p.all())
        err_px = np.linalg.norm(uv_rt - uv_in, axis=-1)
        self.assertLess(float(err_px.max()), 1e-4)

        # 2. Fitting DS to a 190-deg KB4 fisheye camera achieves sub-pixel RMSE
        kb4_cam = {
            "slot": "front", "model": "kb4", "image_hw": [1080, 1920],
            "K": [[570.0, 0.0, 960.0], [0.0, 570.0, 540.0], [0.0, 0.0, 1.0]],
            "distortion": [-0.04, 0.005, -0.001, 0.0], "max_theta_deg": 92.0,
            "T_cam_from_ego": np.eye(4),
        }
        fit = fit_double_sphere_from_camera(kb4_cam)
        self.assertEqual(len(fit["params"]), 6)
        self.assertLess(fit["rmse_px"], 1.0)

    def test_scene3d_adaptive_bowl_smooth_mesh_and_object_yaw(self):
        import numpy as np
        from omnishield.eval.scene3d import (
            bowl_base_radii,
            build_smooth_occupancy_mesh_binary,
            compute_adaptive_bowl_radii,
            compute_corridor_profile,
            compute_pdc_sectors,
            extract_dynamic_objects,
            extract_seam_shift_zones,
        )
        from omnishield.geometry.calibration import Grid

        grid = Grid.from_config(self.config["grid"])
        rig = next(iter(OccupancyDataset(
            load_manifest(self.prepared / "manifest.json", self.config, allow_synthetic=True),
            self.config, self.raw, self.prepared, "val",
        ).rigs.values()))
        occ = np.zeros(grid.shape, dtype=np.int8)
        sem = np.zeros(grid.shape, dtype=np.int8)
        # Place a close Vehicle cluster on the left side (y ~ 2.4m, x ~ 0..3.6m) -> yaw ~ 0
        centers = grid.centers()
        veh_box = (centers[..., 0] >= 0.2) & (centers[..., 0] <= 3.4) & (centers[..., 1] >= 1.9) & (centers[..., 1] <= 3.1) & (centers[..., 2] >= 0.4)
        occ[veh_box] = 1
        sem[veh_box] = 1
        # Place a static bridge wall on the right side (y <= -3.5m)
        wall_box = (centers[..., 1] <= -3.5) & (centers[..., 2] >= 0.4)
        occ[wall_box] = 1

        classes = ["free", "vehicle", "rider", "pedestrian"]
        objs, dyn_mask = extract_dynamic_objects(occ, sem, grid, rig, classes)
        self.assertEqual(len(objs), 1)
        self.assertEqual(objs[0]["cls"], "vehicle")
        self.assertLess(abs(objs[0]["yaw"]), 0.25)
        self.assertLess(objs[0]["dist_m"], 2.0)

        pdc = compute_pdc_sectors(occ, sem, grid, rig)
        self.assertEqual(len(pdc), 8)
        self.assertLess(min(pdc), 1.8)

        corr = compute_corridor_profile(occ, dyn_mask, grid, rig)
        self.assertLess(corr["right_poly"][2], -2.5)  # Right wall at y <= -3.5m
        zones = extract_seam_shift_zones(occ, grid, rig, corr, objs)
        self.assertGreaterEqual(len(zones), 1)
        self.assertEqual(zones[0][2], 2.0)  # Left vehicle crossing FL seam prefers Left cam (2)

        adapt = compute_adaptive_bowl_radii(occ, sem, grid, rig)
        base = bowl_base_radii(np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False))
        self.assertLess(adapt[30], base[30])  # Left sector (~90 deg) pulled inward by close vehicle

        static_field = (occ & (~dyn_mask)).astype(np.float32)
        blob = build_smooth_occupancy_mesh_binary(static_field, grid, level=0.40)
        n_v, n_i = np.frombuffer(blob[:8], dtype="<u4")
        self.assertGreater(n_v, 20)
        self.assertGreater(n_i, 30)
        verts = np.frombuffer(blob[8:8 + int(n_v) * 28], dtype="<f4").reshape(int(n_v), 7)
        faces = np.frombuffer(blob[8 + int(n_v) * 28:], dtype="<u4").reshape(-1, 3)
        self.assertTrue((verts[:, 6] >= 0.34).all() and (verts[:, 6] <= 1.0).all())
        tri_n = np.cross(verts[faces[:, 1], :3] - verts[faces[:, 0], :3], verts[faces[:, 2], :3] - verts[faces[:, 0], :3])
        dots = (tri_n * verts[faces[:, 0], 3:6]).sum(axis=1)
        self.assertGreater(float((dots > 0).mean()), 0.85)

    def test_standard_rig_calibration_targetless_and_profile_loading(self):
        from omnishield.eval.scene3d import (
            bowl_seam_calibration,
            build_standard_rig_calibration,
            export_camera_calib,
            load_calibrated_rig_profile,
        )

        rig = next(iter(self.manifest["rigs"].values()))
        kb4_rig = {
            "rig_id": "kb4-customer-test",
            "ground_height_m": 0.0,
            "ego_xy_bounds_m": rig["ego_xy_bounds_m"],
            "cameras": [
                {
                    **cam,
                    "model": "kb4",
                    "distortion": [0.0, 0.0, 0.0, 0.0],
                    "max_theta_deg": 90.0,
                }
                for cam in rig["cameras"]
            ],
        }
        rows = [r for r in self.manifest["frames"] if r["split"] == "val"] * 2
        profile = build_standard_rig_calibration(
            kb4_rig, rows, self.raw, mode="camera-only", device=torch.device("cpu"),
        )
        self.assertEqual(profile["schema_version"], 1)
        self.assertEqual(profile["calibration_mode"], "targetless_camera_only")
        self.assertEqual(len(profile["rig"]["cameras"]), 4)
        self.assertEqual(len(profile["quality"]["rot_deg"]), 4)
        self.assertEqual(len(profile["quality"]["calib_dz_cm"]), 4)

        out_path = self.prepared / "customer_rig_calibrated.json"
        write_json(out_path, profile)
        loaded = load_calibrated_rig_profile(out_path)
        self.assertEqual(loaded["rig_id"], "kb4-customer-test")

        seams = bowl_seam_calibration(kb4_rig, export_camera_calib(kb4_rig), 0.0)
        self.assertAlmostEqual(seams["max_offset"], 0.26)
        self.assertEqual(len(seams["offsets"]), 25)

        args = SimpleNamespace(
            raw=self.raw, prepared=self.prepared, model=ROOT / "configs/model/smoke.json",
            calibrated_rig=out_path, no_ai=True, split="val", device="cpu",
            allow_synthetic=True, threshold=0.5, max_uncertainty=0.85,
        )
        runtime = DemoRuntime(args)
        self.assertEqual(runtime.calibrated_rigs["val"]["rig_id"], "kb4-customer-test")

    def test_bowl_mesh_integrity_near_coverage_and_depth_discontinuity_surface(self):
        from omnishield.eval.scene3d import (
            FeedForward3DGSDecoder,
            export_camera_calib,
            measure_near_field_coverage,
            verify_bowl_mesh_integrity,
        )

        # 1. Verify 0 inverted/degenerate triangles & radial monotonicity even under extreme curb deformation
        walls = np.full(128, 3.10, dtype=np.float32)
        walls[16:48] = 2.45
        polar = np.zeros((128, 48), dtype=np.float32)
        polar[:, 12:] = 0.28
        integrity = verify_bowl_mesh_integrity(
            ground_plane=(0.006, 0.002, 0.02),
            wall_ranges=walls,
            polar_height=polar,
            left_poly=(0.0, 0.0, 2.35),
            right_poly=(0.0, 0.0, -2.15),
        )
        self.assertTrue(integrity["radial_monotonic"])
        self.assertEqual(integrity["inverted_triangles"], 0)
        self.assertEqual(integrity["degenerate_triangles"], 0)

        # 2. Verify near-field coverage audit distinguishes optical vs shader visibility
        rig = next(iter(self.manifest["rigs"].values()))
        cov_syn = measure_near_field_coverage(rig, (0.0, 0.0, 0.0), band_m=1.25)
        self.assertGreater(cov_syn["total_points"], 50)
        self.assertGreater(cov_syn["shader_covered"], 0)
        def make_cam_t(origin_xyz, fwd_dir):
            z = np.asarray(fwd_dir, dtype=np.float64)
            z /= np.linalg.norm(z)
            x = np.cross(z, [0.0, 0.0, 1.0])
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            r_ec = np.column_stack([x, y, z])
            t = np.eye(4)
            t[:3, :3] = r_ec.T
            t[:3, 3] = -r_ec.T @ np.asarray(origin_xyz, dtype=np.float64)
            return t.tolist()

        kb4_near_rig = {
            "rig_id": "kb4-near-test",
            "ground_height_m": 0.0,
            "ego_xy_bounds_m": [-2.6, 2.6, -1.15, 1.15],
            "cameras": [
                {"slot": "front", "model": "kb4", "image_hw": [1080, 1920], "max_theta_deg": 95,
                 "K": [[600, 0, 960], [0, 600, 540], [0, 0, 1]], "distortion": [0, 0, 0, 0],
                 "T_cam_from_ego": make_cam_t([2.41, 0.0, 0.64], [0.9165, 0.0, -0.40])},
                {"slot": "rear", "model": "kb4", "image_hw": [1080, 1920], "max_theta_deg": 95,
                 "K": [[600, 0, 960], [0, 600, 540], [0, 0, 1]], "distortion": [0, 0, 0, 0],
                 "T_cam_from_ego": make_cam_t([-2.41, 0.0, 0.92], [-0.9165, 0.0, -0.40])},
                {"slot": "left", "model": "kb4", "image_hw": [1080, 1920], "max_theta_deg": 95,
                 "K": [[600, 0, 960], [0, 600, 540], [0, 0, 1]], "distortion": [0, 0, 0, 0],
                 "T_cam_from_ego": make_cam_t([0.99, 0.97, 1.03], [0.0, 0.75, -0.66])},
                {"slot": "right", "model": "kb4", "image_hw": [1080, 1920], "max_theta_deg": 95,
                 "K": [[600, 0, 960], [0, 600, 540], [0, 0, 1]], "distortion": [0, 0, 0, 0],
                 "T_cam_from_ego": make_cam_t([0.99, -0.97, 1.03], [0.0, -0.75, -0.66])},
            ],
        }
        cov_kb4 = measure_near_field_coverage(kb4_near_rig, (0.0, 0.0, 0.0), band_m=0.25)
        self.assertGreater(cov_kb4["shader_covered"], 150)

        # 3. Verify depth-discontinuity surface cuts across sharp depth jumps (no stretched rubber-sheet triangles)
        calibs = export_camera_calib(rig)
        dec = FeedForward3DGSDecoder(rig, calibs, tuple(self.config["image_hw"]), feature_stride=4)
        depth_m = np.full((4, dec.dh, dec.dw), 2.5, dtype=np.float32)
        depth_m[0, :, dec.dw // 2:] = 8.0  # Sharp 5.5m jump down the middle of front camera
        gt_valid = np.ones((4, dec.dh, dec.dw), dtype=bool)
        blob = dec.decode_depth_surface_binary(depth_m, [True, True, True, True], gt_valid_mask=gt_valid)
        n_v, n_i = np.frombuffer(blob[:8], dtype="<u4")
        self.assertGreater(n_v, 10)
        self.assertGreater(n_i, 12)
        verts = np.frombuffer(blob[8:8 + int(n_v) * 24], dtype="<f4").reshape(int(n_v), 6)
        faces = np.frombuffer(blob[8 + int(n_v) * 24:], dtype="<u4").reshape(-1, 3)
        p0, p1, p2 = verts[faces[:, 0], :3], verts[faces[:, 1], :3], verts[faces[:, 2], :3]
        max_edge = max(
            float(np.linalg.norm(p1 - p0, axis=1).max()),
            float(np.linalg.norm(p2 - p1, axis=1).max()),
            float(np.linalg.norm(p0 - p2, axis=1).max()),
        )
        self.assertLessEqual(max_edge, 0.66)

    def test_fisheye_kb4_4point_pnp_recovers_6dof_pose(self):
        import copy
        from omnishield.eval.scene3d import solve_fisheye_pnp_camera
        from omnishield.geometry.calibration import project, rigid_matrix

        rig = next(iter(self.manifest["rigs"].values()))
        cam_gt = copy.deepcopy(rig["cameras"][0])
        cam_gt["model"] = "kb4"
        cam_gt["distortion"] = [0.05, -0.02, 0.004, -0.001]
        cam_gt["max_theta_deg"] = 95.0
        sh, sw = cam_gt["image_hw"]

        # 4 non-collinear calibration mat corners on z = 0 in front of vehicle
        world_pts = np.array([
            [3.5,  1.4, 0.0],
            [3.5, -1.4, 0.0],
            [5.5, -1.4, 0.0],
            [5.5,  1.4, 0.0],
        ], dtype=np.float64)
        uv_gt, _, valid = project(world_pts, cam_gt, (sh, sw))
        self.assertTrue(bool(valid.all()))

        # Perturb initial extrinsics significantly (+6 deg rot, +12 cm shift) and verify PnP recovers exact pose!
        t_gt = rigid_matrix(cam_gt["T_cam_from_ego"], cam_gt["slot"])
        ang = np.deg2rad(6.0)
        r_pert = np.array([[np.cos(ang), 0.0, np.sin(ang)], [0.0, 1.0, 0.0], [-np.sin(ang), 0.0, np.cos(ang)]])
        t_pert = t_gt.copy()
        t_pert[:3, :3] = r_pert @ t_gt[:3, :3]
        t_pert[:3, 3] = t_gt[:3, 3] + np.array([0.08, -0.05, 0.12])
        cam_init = copy.deepcopy(cam_gt)
        cam_init["T_cam_from_ego"] = t_pert.tolist()

        res = solve_fisheye_pnp_camera(cam_init, uv_gt, world_pts)
        self.assertLess(res["reproj_rmse_px"], 0.1)
        t_rec = np.asarray(res["T_cam_from_ego"], dtype=np.float64)
        self.assertLess(float(np.linalg.norm(t_rec[:3, 3] - t_gt[:3, 3])), 0.01)


if __name__ == "__main__":
    unittest.main()
