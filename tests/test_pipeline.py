import io
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
