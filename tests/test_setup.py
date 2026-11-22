"""Setup contract tests. All synthetic images/labels live in temporary folders."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd

from src.config import NUM_CLASSES
from src.data_utils import (
    DatasetValidationError, load_image, load_metadata, make_train_validation_split,
    resolve_image_path, split_fingerprint, validate_dataset,
)
from src.degradations import (
    add_noise, apply_condition, apply_motion_blur, equalize_histogram,
    gaussian_filter, reduce_brightness,
)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        # Deliberately tiny test-only fixtures, not fabricated GTSRB records.
        for split, per_class in (("Train", 5), ("Test", 1)):
            rows = []
            for class_id in range(NUM_CLASSES):
                for index in range(per_class):
                    relative = f"{split}/{class_id}/{index}.png"
                    path = self.root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    bgr = np.full((6, 8, 3), (10, 20, 200), dtype=np.uint8)
                    self.assertTrue(cv2.imwrite(str(path), bgr))
                    rows.append({"Path": relative, "ClassId": class_id})
            pd.DataFrame(rows).to_csv(self.root / f"{split}.csv", index=False)

    def validate(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return validate_dataset(self.root, **kwargs)

    def test_load_rgb_and_complete_fixture(self):
        train, test = self.validate(require_full_counts=False, check_images="all")
        self.assertEqual((len(train), len(test)), (215, 43))
        image = load_image(self.root, train.iloc[0]["Path"])
        self.assertEqual(image.shape, (32, 32, 3))
        np.testing.assert_array_equal(image[0, 0], [200, 20, 10])
        self.assertEqual(load_image(self.root, train.iloc[0]["Path"], None).shape, (6, 8, 3))

    def test_partial_upload_and_missing_labels_fail(self):
        with self.assertRaisesRegex(DatasetValidationError, "expected 39,209"):
            self.validate()
        (self.root / "Train.csv").unlink()
        (self.root / "Test.csv").unlink()
        with self.assertRaises(DatasetValidationError) as caught:
            self.validate()
        self.assertIn("Train.csv", str(caught.exception))
        self.assertIn("Test.csv", str(caught.exception))

    def test_missing_and_unreadable_images_fail(self):
        target = self.root / "Test/0/0.png"
        target.unlink()
        with self.assertRaisesRegex(DatasetValidationError, "referenced images missing"):
            self.validate(require_full_counts=False)
        target.write_bytes(b"not an image")
        with self.assertRaisesRegex(DatasetValidationError, "unreadable images"):
            self.validate(require_full_counts=False, check_images="all")

    def test_labels_paths_and_classes_are_checked(self):
        frame = load_metadata(self.root)
        for bad_label in (-1, 43, 1.5, "unknown", None):
            with self.subTest(label=bad_label):
                altered = frame.astype({"ClassId": object})
                altered.loc[0, "ClassId"] = bad_label
                altered.to_csv(self.root / "Train.csv", index=False)
                with self.assertRaises(DatasetValidationError):
                    load_metadata(self.root)
        frame[frame.ClassId != 42].to_csv(self.root / "Train.csv", index=False)
        with self.assertRaisesRegex(DatasetValidationError, "missing class IDs: \\[42\\]"):
            self.validate(require_full_counts=False)
        for bad_path in ("../elsewhere.png", "/absolute.png", "C:/absolute.png"):
            with self.assertRaises(DatasetValidationError):
                resolve_image_path(self.root, bad_path)

    def test_split_is_stable_disjoint_stratified_and_training_only(self):
        frame = load_metadata(self.root)
        train, validation = make_train_validation_split(frame)
        shuffled = make_train_validation_split(frame.sample(frac=1, random_state=7))
        self.assertEqual(split_fingerprint(train, validation), split_fingerprint(*shuffled))
        self.assertEqual(len(train), 172)
        self.assertEqual(len(validation), 43)
        self.assertFalse(set(train.Path) & set(validation.Path))
        self.assertEqual(set(train.Path) | set(validation.Path), set(frame.Path))
        for part in (train, validation):
            self.assertEqual(part.ClassId.nunique(), NUM_CLASSES)
        with self.assertRaisesRegex(DatasetValidationError, "Train.csv only"):
            make_train_validation_split(load_metadata(self.root, "Test"))
        with self.assertRaisesRegex(DatasetValidationError, "Duplicate"):
            make_train_validation_split(pd.concat([frame, frame.iloc[:1]]))


class DegradationTests(unittest.TestCase):
    def test_types_ranges_shapes_and_nonmutation(self):
        rng = np.random.default_rng(42)
        for shape in ((20, 30), (20, 30, 3)):
            for dtype in (np.uint8, np.float32, np.float64):
                image = rng.random(shape)
                image = (image * 255).astype(dtype) if dtype == np.uint8 else image.astype(dtype)
                original = image.copy()
                for transform in (reduce_brightness, add_noise, apply_motion_blur,
                                  equalize_histogram, gaussian_filter):
                    with self.subTest(shape=shape, dtype=dtype, transform=transform.__name__):
                        result = transform(image)
                        self.assertEqual(result.dtype, image.dtype)
                        self.assertEqual(result.shape, image.shape)
                        self.assertGreaterEqual(result.min(), 0)
                        self.assertLessEqual(result.max(), 255 if dtype == np.uint8 else 1)
                        self.assertFalse(np.shares_memory(image, result))
                        np.testing.assert_array_equal(image, original)

    def test_meaningful_transform_properties(self):
        image = np.full((32, 32, 3), 100, dtype=np.uint8)
        np.testing.assert_array_equal(reduce_brightness(image), np.full_like(image, 50))
        np.testing.assert_array_equal(add_noise(image, std=0, seed=42), image)
        np.testing.assert_array_equal(apply_motion_blur(image), image)
        impulse = np.zeros((17, 17), dtype=np.float32)
        impulse[8, 8] = 1
        blurred = apply_motion_blur(impulse, kernel_size=9)
        expected = np.zeros_like(impulse)
        expected[8, 4:13] = 1 / 9
        # OpenCV may use a DFT internally, leaving tiny numerical residuals.
        np.testing.assert_allclose(blurred, expected, atol=1e-7)
        self.assertAlmostEqual(float(blurred.sum()), 1)

    def test_noise_reproducibility_and_global_rng_isolation(self):
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        first = apply_condition(image, "noise", "Test/00001.png")
        second = apply_condition(image, "noise", "Test/00002.png")
        np.testing.assert_array_equal(first, apply_condition(image, "noise", "Test/00001.png"))
        self.assertFalse(np.array_equal(first, second))
        np.testing.assert_array_equal(add_noise(image, seed=7), add_noise(image, seed=7))
        np.random.seed(3)
        expected = np.random.random(4)
        np.random.seed(3)
        add_noise(image, seed=7)
        np.testing.assert_array_equal(np.random.random(4), expected)
        clean = apply_condition(image, "clean", "Test/00001.png")
        np.testing.assert_array_equal(clean, image)
        self.assertFalse(np.shares_memory(clean, image))

    def test_invalid_ranges_and_parameters_fail(self):
        image = np.zeros((8, 8), dtype=np.uint8)
        for function, kwargs in (
            (reduce_brightness, {"factor": -0.1}), (add_noise, {"std": float("nan")}),
            (apply_motion_blur, {"kernel_size": 2}), (gaussian_filter, {"sigma": 0}),
        ):
            with self.assertRaises(ValueError):
                function(image, **kwargs)
        with self.assertRaises(ValueError):
            add_noise(np.full((3, 3), 255.0))
        with self.assertRaises(TypeError):
            add_noise(image.astype(np.int16))


if __name__ == "__main__":
    unittest.main()
