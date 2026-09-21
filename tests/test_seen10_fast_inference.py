from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from csgo_seen10.fast_inference import (
    COMPILE_MODE,
    AsyncJpegWriter,
    compile_mode_label,
    compile_mode_uses_cudagraphs,
    euler_time_grid,
    fixed_position_ids,
    fixed_sequence_cu_seqlens,
    make_latent_batch,
    pad_collated_batch,
    stable_seed,
    torch_compile_kwargs,
)
from infer_seen10 import (
    _benchmark_report,
    _marker_compatible,
    _provenance_compatible,
    _rank0_exception_payload,
    _run_rank0_stage,
    _validate_benchmark_options,
)


class FixedGeometryTests(unittest.TestCase):
    def test_position_axes_match_native_448_geometry(self):
        position_ids = fixed_position_ids()
        self.assertEqual(tuple(position_ids.shape), (1, 807, 3))
        self.assertTrue(torch.equal(position_ids[0, :23, 0], torch.arange(23, dtype=torch.int32)))
        self.assertTrue(torch.all(position_ids[0, 23:, 0] == 23))
        self.assertTrue(torch.equal(position_ids[0, 23:, 1].view(28, 28)[:, 0], torch.arange(28, dtype=torch.int32)))
        self.assertTrue(torch.equal(position_ids[0, 23:, 2].view(28, 28)[0], torch.arange(28, dtype=torch.int32)))

    def test_varlen_boundaries_are_one_fixed_segment_per_sample(self):
        self.assertTrue(torch.equal(fixed_sequence_cu_seqlens(3, 23), torch.tensor([0, 23, 46, 69], dtype=torch.int32)))
        self.assertTrue(torch.equal(fixed_sequence_cu_seqlens(3, 784), torch.tensor([0, 784, 1568, 2352], dtype=torch.int32)))
        self.assertTrue(torch.equal(fixed_sequence_cu_seqlens(3, 807), torch.tensor([0, 807, 1614, 2421], dtype=torch.int32)))

    def test_euler_grid_matches_transport_shift_formula(self):
        reference = torch.linspace(0.0, 1.0, 28, dtype=torch.float32)
        reference = reference / (reference + 6.0 - 6.0 * reference)
        self.assertTrue(torch.equal(euler_time_grid(28, 6.0), reference))
        with self.assertRaises(ValueError):
            euler_time_grid(1, 6.0)


class BatchAndSeedTests(unittest.TestCase):
    def test_latents_are_independent_and_batching_preserves_sample_seed(self):
        sample_ids = ["sample-a", "sample-b", "sample-a"]
        batched = make_latent_batch(sample_ids, seed=7, task="discrete", dtype=torch.float32, device="cpu")
        single = make_latent_batch(["sample-a"], seed=7, task="discrete", dtype=torch.float32, device="cpu")
        self.assertEqual(tuple(batched.shape), (3, 16, 56, 56))
        self.assertTrue(torch.equal(batched[0], single[0]))
        self.assertTrue(torch.equal(batched[0], batched[2]))
        self.assertFalse(torch.equal(batched[0], batched[1]))
        self.assertNotEqual(stable_seed(7, "discrete", "sample-a"), stable_seed(7, "continuous", "sample-a"))

    def test_tail_batch_padding_repeats_final_record_and_tensor(self):
        batch = {
            "sample_id": ["a", "b"],
            "radar": torch.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4),
            "pose": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        }
        padded, real_size = pad_collated_batch(batch, 4)
        self.assertEqual(real_size, 2)
        self.assertEqual(padded["sample_id"], ["a", "b", "b", "b"])
        self.assertTrue(torch.equal(padded["radar"][1], padded["radar"][3]))
        self.assertTrue(torch.equal(padded["pose"][1], padded["pose"][2]))


class AsyncJpegTests(unittest.TestCase):
    def test_atomic_writer_emits_valid_rgb_jpeg(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "map" / "frame.jpg"
            pixels = np.full((16, 16, 3), 127, dtype=np.uint8)
            with AsyncJpegWriter(workers=1, max_pending=1, quality=95) as writer:
                writer.submit(output, pixels, size=16)
            with Image.open(output) as image:
                self.assertEqual((image.format, image.mode, image.size), ("JPEG", "RGB", (16, 16)))

    def test_async_writer_propagates_validation_error(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "wrong-size.jpg"
            pixels = np.full((8, 8, 3), 127, dtype=np.uint8)
            writer = AsyncJpegWriter(workers=1, max_pending=1, quality=95)
            writer.submit(output, pixels, size=16)
            with self.assertRaises(RuntimeError):
                writer.close()
            self.assertFalse(output.exists())


class ExistingSmokeCompatibilityTests(unittest.TestCase):
    def test_existing_smoke_metadata_remains_accepted_for_default_resume(self):
        project_root = Path(__file__).resolve().parents[1]
        smoke_root = project_root / "outputs/csgo_benchmark_v2_seen10/Lumina-Image-2.0/seed_0/smoke"
        discrete = smoke_root / "discrete"
        marker_path = discrete / "inference_run.json"
        provenance_path = discrete / "provenance.jsonl"
        if not marker_path.is_file() or not provenance_path.is_file():
            self.skipTest("Existing seed-0 smoke metadata is not present in this checkout")

        old_marker = json.loads(marker_path.read_text(encoding="utf-8"))
        new_marker = dict(old_marker)
        new_marker.update(
            {
                "engine": "eager",
                "batch_size": 1,
                "vae_batch_size": 1,
                "compile_mode": "none",
                "sampler": "euler",
                "time_shifting_factor": 6.0,
                "seed_policy": "sha256(seed\\0task\\0sample_id) mod (2^63-1)",
                "output_root": str(project_root / "outputs/csgo_benchmark_v2_seen10/Lumina-Image-2.0"),
                "jpeg_quality": 95,
            }
        )
        self.assertTrue(_marker_compatible(old_marker, new_marker, accept_legacy=True))

        old_rows = [
            json.loads(line)
            for line in provenance_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        current_metadata = {
            key: new_marker[key]
            for key in (
                "engine",
                "batch_size",
                "vae_batch_size",
                "compile_mode",
                "sampler",
                "sampling_steps",
                "time_shifting_factor",
                "seed_policy",
                "jpeg_quality",
            )
        }
        expected_rows = [dict(row, **current_metadata) for row in old_rows]
        self.assertTrue(
            _provenance_compatible(
                old_rows,
                expected_rows,
                inference_metadata=current_metadata,
                accept_legacy=True,
            )
        )
        self.assertFalse(
            _provenance_compatible(
                old_rows,
                expected_rows,
                inference_metadata=current_metadata,
                accept_legacy=False,
            )
        )


class CompileModeTests(unittest.TestCase):
    def test_compile_configuration_keeps_graph_modes_distinct(self):
        reduce_kwargs = torch_compile_kwargs("reduce-overhead")
        default_kwargs = torch_compile_kwargs("default")
        self.assertEqual(reduce_kwargs, {"mode": "reduce-overhead", "fullgraph": True})
        self.assertEqual(
            default_kwargs,
            {"fullgraph": True, "options": {"triton.cudagraphs": False}},
        )
        self.assertEqual(compile_mode_label("reduce-overhead"), COMPILE_MODE)
        self.assertEqual(compile_mode_label("default"), "default/fullgraph")
        self.assertTrue(compile_mode_uses_cudagraphs("reduce-overhead"))
        self.assertFalse(compile_mode_uses_cudagraphs("default"))
        with self.assertRaises(ValueError):
            torch_compile_kwargs("unknown")

    def test_compile_mode_markers_cannot_be_mixed(self):
        reduce_marker = {"engine": "compiled", "compile_mode": COMPILE_MODE, "cudagraphs": True}
        default_marker = {
            "engine": "compiled",
            "compile_mode": "default/fullgraph",
            "cudagraphs": False,
        }
        self.assertFalse(_marker_compatible(reduce_marker, default_marker, accept_legacy=False))
        self.assertFalse(_marker_compatible(default_marker, reduce_marker, accept_legacy=False))

        legacy_reduce_marker = {key: value for key, value in reduce_marker.items() if key != "cudagraphs"}
        self.assertTrue(_marker_compatible(legacy_reduce_marker, reduce_marker, accept_legacy=False))
        self.assertFalse(_marker_compatible(legacy_reduce_marker, default_marker, accept_legacy=False))

        reduce_metadata = {"compile_mode": COMPILE_MODE, "cudagraphs": True}
        default_metadata = {"compile_mode": "default/fullgraph", "cudagraphs": False}
        previous = [
            {
                "map_name": "map-a",
                "file_frame": "frame_1",
                "checkpoint_sha256": "checkpoint-hash",
                **reduce_metadata,
            }
        ]
        expected = [
            {
                "map_name": "map-a",
                "file_frame": "frame_1",
                "checkpoint_sha256": "checkpoint-hash",
                **default_metadata,
            }
        ]
        self.assertFalse(
            _provenance_compatible(
                previous,
                expected,
                inference_metadata=default_metadata,
                accept_legacy=False,
            )
        )

        full_reduce_metadata = {"engine": "compiled", **reduce_metadata}
        previous_without_cudagraphs = [
            {
                "map_name": "map-a",
                "file_frame": "frame_1",
                "checkpoint_sha256": "checkpoint-hash",
                "engine": "compiled",
                "compile_mode": COMPILE_MODE,
            }
        ]
        expected_reduce = [dict(previous_without_cudagraphs[0], cudagraphs=True)]
        self.assertTrue(
            _provenance_compatible(
                previous_without_cudagraphs,
                expected_reduce,
                inference_metadata=full_reduce_metadata,
                accept_legacy=False,
            )
        )


class BenchmarkHelperTests(unittest.TestCase):
    def test_benchmark_requires_isolated_single_task_full_step_run(self):
        valid = {
            "batches": 3,
            "output_root": "/tmp/lumina-bench",
            "configured_root": "/tmp/lumina-formal",
            "task": "discrete",
            "smoke": False,
            "world_size": 1,
            "sampling_steps": 28,
        }
        _validate_benchmark_options(**valid)
        for override in (
            {"output_root": None},
            {"task": "all"},
            {"smoke": True},
            {"world_size": 2},
            {"sampling_steps": 2},
            {"batches": 0},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                _validate_benchmark_options(**(valid | override))

    def test_report_calculates_cold_and_steady_batch_and_total_rates(self):
        report = _benchmark_report(
            task="discrete",
            engine="compiled",
            compile_mode="default/fullgraph",
            cudagraphs=False,
            batch_size=16,
            vae_batch_size=4,
            sampling_steps=28,
            batch_seconds=[8.0, 4.0, 2.0],
            total_seconds=15.0,
            model_initialization_seconds=7.5,
            peak_allocated_bytes=1024,
            peak_reserved_bytes=2048,
        )
        self.assertEqual(report["rows"], 48)
        self.assertEqual(report["batches"], 3)
        self.assertEqual(report["batch_timings"][0]["phase"], "cold")
        self.assertTrue(report["batch_timings"][0]["includes_lazy_compile"])
        self.assertEqual(report["batch_timings"][0]["images_per_second"], 2.0)
        self.assertEqual(report["batch_timings"][1]["phase"], "steady")
        self.assertFalse(report["batch_timings"][1]["includes_lazy_compile"])
        self.assertEqual(report["total_images_per_second"], 3.2)
        self.assertEqual(report["model_initialization_seconds"], 7.5)
        self.assertEqual(report["cuda_peak_allocated_bytes"], 1024)
        self.assertEqual(report["cuda_peak_reserved_bytes"], 2048)


class Rank0StageErrorTests(unittest.TestCase):
    def test_single_process_preserves_original_stage_exception(self):
        expected = FileExistsError("marker belongs to another run")

        def fail():
            raise expected

        with self.assertRaises(FileExistsError) as raised:
            _run_rank0_stage("marker check", fail)
        self.assertIs(raised.exception, expected)

    def test_rank0_error_payload_is_broadcast_safe_and_informative(self):
        payload = _rank0_exception_payload(
            "provenance write",
            OSError("permission denied"),
        )
        self.assertEqual(
            payload,
            {
                "stage": "provenance write",
                "exception_type": "builtins.OSError",
                "message": "permission denied",
            },
        )


if __name__ == "__main__":
    unittest.main()
