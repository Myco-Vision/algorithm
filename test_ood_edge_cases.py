"""
test_ood_edge_cases.py — Out-of-Distribution (OOD) & Non-Mushroom Edge Case Test
==================================================================================
Tests the confidence threshold check in predict.py against:
  1. Low-confidence non-mushroom edge cases from the dataset
  2. Synthesized out-of-distribution inputs (random noise, textures, blank scenes)
  3. Real verified mushrooms to ensure zero false rejections
"""

import os
import sys
import tempfile
from pathlib import Path

# Ensure algorithm directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("KERAS_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from PIL import Image
from predict import predict_mushroom


def create_ood_synthetic_images(temp_dir: Path) -> dict:
    """Create distinct out-of-distribution test images."""
    paths = {}

    # 1. Uniform Random Noise (Pure static OOD)
    noise_arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    noise_path = temp_dir / "ood_random_noise.jpg"
    Image.fromarray(noise_arr).save(noise_path)
    paths["OOD Synthetic: Uniform Noise"] = noise_path

    # 2. Smooth Color Gradient (Artificial non-nature)
    x = np.linspace(0, 255, 224, dtype=np.uint8)
    gradient = np.tile(x, (224, 1))
    grad_arr = np.stack([gradient, gradient.T, 255 - gradient], axis=-1)
    grad_path = temp_dir / "ood_gradient.jpg"
    Image.fromarray(grad_arr).save(grad_path)
    paths["OOD Synthetic: Color Gradient"] = grad_path

    # 3. High-frequency Grid / Checkerboard pattern
    check = np.indices((224, 224)).sum(axis=0) % 16 < 8
    check_arr = (check * 255).astype(np.uint8)
    check_arr = np.stack([check_arr, check_arr, check_arr], axis=-1)
    check_path = temp_dir / "ood_checkerboard.jpg"
    Image.fromarray(check_arr).save(check_path)
    paths["OOD Synthetic: Geometric Grid"] = check_path

    return paths


def main():
    print("=" * 75)
    print("TESTING OUT-OF-DISTRIBUTION (OOD) & NON-MUSHROOM EDGE CASES")
    print("=" * 75)

    test_cases = [
        # Real Mushroom Positives
        ("Real Poisonous: obf1.jpg", BASE_DIR / "mushroom_dataset" / "poisonous" / "obf1.jpg", True),
        ("Real Edible: o1.jpeg", BASE_DIR / "mushroom_dataset" / "edible" / "o1.jpeg", True),
        ("Real Edible: e_mushroom.png", BASE_DIR / "e_mushroom.png", True),

        # Dataset Non-Mushroom Hard Edge Cases (Previously split logits)
        ("Dataset Hard Edge Case 1", BASE_DIR / "mushroom_dataset" / "not_mushroom" / "commons_103166292.jpg", False),
        ("Dataset Hard Edge Case 2", BASE_DIR / "mushroom_dataset" / "not_mushroom" / "inat_192237353.jpg", False),

        # Confident Normal Non-Mushrooms
        ("Confident Non-Mushroom 1", BASE_DIR / "mushroom_dataset" / "not_mushroom" / "commons_10014798.jpg", False),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        synth_cases = create_ood_synthetic_images(tmp_path)
        for label, path in synth_cases.items():
            test_cases.append((label, path, False))

        results = []
        print(f"{'Sample Description':<32} | {'Expected':<10} | {'Detected':<12} | {'Max Conf':<8} | {'Status':<18}")
        print("-" * 90)

        for name, path, expected_mushroom in test_cases:
            if not path.exists():
                print(f"{name:<32} | SKIP (File missing)")
                continue

            res = predict_mushroom(str(path))
            is_m = res["is_mushroom"]
            is_unc = res.get("is_uncertain", False)
            raw_max = res.get("raw_max_prob", 0.0)
            result_name = res.get("result_name", res.get("label"))

            # Determine pass/fail
            if expected_mushroom:
                passed = is_m
                status_str = "✅ PASS (Mushroom)" if passed else "❌ FAIL (False Neg)"
            else:
                passed = not is_m
                if passed:
                    status_str = "✅ PASS (Uncertain/OOD)" if is_unc else "✅ PASS (Not Mushroom)"
                else:
                    status_str = "❌ FAIL (False Pos)"

            print(f"{name:<32} | {('Mushroom' if expected_mushroom else 'Non-Mush'):<10} | {result_name:<12} | {raw_max*100:>6.1f}%  | {status_str:<18}")
            results.append((name, expected_mushroom, is_m, is_unc, raw_max, result_name, passed))

        print("-" * 90)
        total = len(results)
        passed_count = sum(1 for r in results if r[-1])
        print(f"\nSummary: {passed_count}/{total} edge cases handled correctly ({passed_count/total*100:.1f}% pass rate).")
        print("=" * 75 + "\n")

    return 0 if passed_count == total else 1


if __name__ == "__main__":
    sys.exit(main())

