"""Integration tests for dynamic calibration."""
from hearing_agent.agent import get_calibration_factors
from hearing_agent.retrieval import LOCAL_DATABASE


def run_tests():
  print("==================================================")
  print("RUNNING CALIBRATION AGENT INTEGRATION TESTS")
  print("==================================================")

  # Test case 1: Standard headphone calibration
  print("\n--- Test Case 1: Standard Headphone Calibration ---")
  user_model = "Sony WH-1000XM4"
  baseline = "Google Pixel Buds Pro"
  res1 = get_calibration_factors(user_model, baseline)

  print(f"User Headphone: {res1["user_headphone"]}")
  print(f"Baseline: {res1["baseline_headphone"]}")
  print(f"Status: {res1["status"]}")
  print(f"Frequencies: {res1["frequencies"]}")
  print(f"Correction Factors (dB): {res1["correction_factors_db"]}")
  print(f"Raw Correction Factors (dB): {res1["raw_correction_factors_db"]}")
  print(f"Bone Conduction Warning: {res1["bone_conduction_warning"]}")
  print(f"Clipping Warning: {res1["clipping_warning"]}")
  print(f"Vetting Warning: {res1["vetting_warning"]}")

  assert (
      res1["status"] == "success"
  ), "Test Case 1 failed: status should be success"
  assert (
      len(res1["correction_factors_db"]) == 8
  ), "Should have 8 correction factors"
  # Sony WH-1000XM4 mock response:
  # [1.5, 0.8, -0.2, -1.0, -2.0, -3.5, -2.5, -1.5]
  # Google Pixel Buds Pro mock response:
  # [-0.2, 0.2, 0.5, 1.5, 0.8, -1.0, -2.0, -3.0]
  # Raw correction = Baseline - User
  # 250Hz: -1.7, 500Hz: -0.6, 1000Hz: 0.7, 2000Hz: 2.5, 3000Hz: 2.8,
  # 4000Hz: 2.5, 6000Hz: 0.5, 8000Hz: -1.5
  expected_raw = [-1.7, -0.6, 0.7, 2.5, 2.8, 2.5, 0.5, -1.5]
  assert (
      res1["raw_correction_factors_db"] == expected_raw
  ), (
      f"Correction values mismatch. Got {res1["raw_correction_factors_db"]},"
      f" expected {expected_raw}"
  )
  print("✅ Test Case 1 Passed!")

  # Test case 2: Bone conduction detection
  print("\n--- Test Case 2: Bone Conduction Device Detection ---")
  bc_model = "Shokz OpenRun"
  res2 = get_calibration_factors(bc_model, baseline)
  print(f"User Headphone: {res2["user_headphone"]}")
  print(f"Status: {res2["status"]}")
  print(f"Bone Conduction Warning: {res2["bone_conduction_warning"]}")

  assert res2["status"] == "success"
  assert (
      res2["bone_conduction_warning"] is True
  ), "Test Case 2 failed: should detect bone conduction"
  print("✅ Test Case 2 Passed!")

  # Test case 3: Clipping limits (Safety check)
  print("\n--- Test Case 3: Safety Calibration Clipping ---")
  # We will simulate a very sensitive user headphone by setting up an imaginary
  # model which has extremely high response, forcing clipping.
  # We will temporarily modify the mock database to test this.

  LOCAL_DATABASE["ultra sensitive phones"] = {
      "source": "oratory1990",
      "form_factor": "over-ear",
      "frequencies": [250, 500, 1000, 2000, 4000, 8000],
      "smoothed": [25.0, 30.0, 28.0, 32.0, 25.0, 20.0],  # high db
      "raw_url": (
          "https://raw.githubusercontent.com/jaakkopasanen/AutoEq/master/"
          "results/oratory1990/over-ear/UltraSens/UltraSens.csv"
      ),
  }

  res3 = get_calibration_factors("ultra sensitive phones", baseline)
  print(f"Raw Correction: {res3["raw_correction_factors_db"]}")
  print(f"Clipped Correction: {res3["correction_factors_db"]}")
  print(f"Clipping Warning: {res3["clipping_warning"]}")

  assert res3["status"] == "success"
  assert any(
      val == -15.0 for val in res3["correction_factors_db"]
  ), "Correction factors should be clipped to MIN_CORRECTION_DB (-15.0)"
  assert (
      res3["clipping_warning"] is not None
  ), "Should generate a clipping warning"
  print("✅ Test Case 3 Passed!")

  print("\n==================================================")
  print("ALL TESTS PASSED SUCCESSFULLY!")
  print("==================================================")


if __name__ == "__main__":
  run_tests()
