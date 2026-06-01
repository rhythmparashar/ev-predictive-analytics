# Battery Intelligence Engine — Module 2.1 Completeness Report
> Generated: 2026-05-19 10:34 UTC
> Vehicle: EV01

---

## Executive Summary

| Check | Value |
|-------|-------|
| Expected cells | **180** (M1–M5, C1–C36) |
| Cells in weak_cell_profile | 173 |
| Cells missing from weak_cell_profile | **7** |
| Cells in cell_delta_profile | 180 |
| Cells missing from cell_delta_profile | **0** |
| Expected sensors | **90** (M1–M5, T1–T18) |
| Sensors in hot_sensor_profile | 89 |
| Sensors missing from hot_sensor_profile | **1** |
| All cells in schema | ✅ Yes |
| All cells in silver | ✅ Yes |
| All cells in gold | ✅ Yes |
| All sensors in schema | ✅ Yes |
| All sensors in silver | ✅ Yes |
| Module 2 rerun needed | ✅ No |

---

## Root Cause

> **All missing cells/sensors are present in schema, silver, and gold. They are absent only from weak_cell_profile / hot_sensor_profile because they were NEVER the extreme (lowest voltage or hottest) in their module across 1,643,119 timestamps. This is correct Module 2 logic — the profile only records cells/sensors that appeared as the module extremum at least once.**

The `weak_cell_profile` is built from `module_*_lowest_cell` — which records the single
lowest-voltage cell in each module at every timestamp. A cell that is consistently at or
above the module mean voltage will **never** appear as the module minimum, so it is
correctly absent from the profile. It is still fully present in silver and gold.

The `hot_sensor_profile` follows the same logic using `module_*_hottest_sensor`.

---

## Missing from weak_cell_profile
> 7 cells never appeared as their module's lowest-voltage cell.

| cell_id | module_id | present_in_schema | present_in_silver_cells | present_in_gold_cell_features | present_in_cell_delta_profile | silver_valid_count | silver_null_pct | reason_if_missing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1_C2 | 1 | True | True | True | True | 2283812 | 1.225 | NEVER LOWEST IN MODULE — cell voltage consistently at or above module mean; a positive or near-zero mean_delta confirms this is a healthy cell. This is correct behavior — not a pipeline bug. |
| M1_C3 | 1 | True | True | True | True | 2283812 | 1.225 | NEVER LOWEST IN MODULE — cell voltage consistently at or above module mean; a positive or near-zero mean_delta confirms this is a healthy cell. This is correct behavior — not a pipeline bug. |
| M1_C4 | 1 | True | True | True | True | 2283812 | 1.225 | NEVER LOWEST IN MODULE — cell voltage consistently at or above module mean; a positive or near-zero mean_delta confirms this is a healthy cell. This is correct behavior — not a pipeline bug. |
| M1_C27 | 1 | True | True | True | True | 2283745 | 1.228 | NEVER LOWEST IN MODULE — cell voltage consistently at or above module mean; a positive or near-zero mean_delta confirms this is a healthy cell. This is correct behavior — not a pipeline bug. |
| M3_C11 | 3 | True | True | True | True | 2283753 | 1.228 | NEVER LOWEST IN MODULE — cell voltage consistently at or above module mean; a positive or near-zero mean_delta confirms this is a healthy cell. This is correct behavior — not a pipeline bug. |
| M4_C27 | 4 | True | True | True | True | 2283720 | 1.229 | NEVER LOWEST IN MODULE — cell voltage consistently at or above module mean; a positive or near-zero mean_delta confirms this is a healthy cell. This is correct behavior — not a pipeline bug. |
| M5_C4 | 5 | True | True | True | True | 2283837 | 1.224 | NEVER LOWEST IN MODULE — cell voltage consistently at or above module mean; a positive or near-zero mean_delta confirms this is a healthy cell. This is correct behavior — not a pipeline bug. |

---

## Missing from hot_sensor_profile
> 1 sensor(s) never appeared as their module's hottest sensor.

| sensor_id | module_id | present_in_schema | present_in_silver_cells | silver_valid_count | silver_null_pct | reason_if_missing |
| --- | --- | --- | --- | --- | --- | --- |
| M4_T18 | 4 | True | True | 2307816 | 0.187 | NEVER HOTTEST IN MODULE — sensor temperature consistently below other sensors; sensor is present and valid in silver but never wins the hottest-sensor race. This is correct behavior — not a pipeline bug. |

---

## Missing from cell_delta_profile
> 0 cells missing from cell_delta_profile.

_None — all 180 cells present in cell_delta_profile._ ✅

---

## Recommended Fix

> No fix required. Module 2 logic is correct. If a full 180-cell inventory is desired in weak_cell_profile, the function could be extended to include all cells with pct_lowest_in_module=0 and weakness_status='Normal'. This would be an enhancement, not a bug fix.

---

## Full Cell Completeness Table

| cell_id | module_id | present_in_schema | present_in_silver_cells | present_in_gold_cell_features | present_in_weak_cell_profile | present_in_cell_delta_profile | silver_valid_count | silver_null_count | silver_null_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1_C1 | 1 | True | True | True | True | True | 2283812 | 28327 | 1.225 |
| M1_C2 | 1 | True | True | True | False | True | 2283812 | 28327 | 1.225 |
| M1_C3 | 1 | True | True | True | False | True | 2283812 | 28327 | 1.225 |
| M1_C4 | 1 | True | True | True | False | True | 2283812 | 28327 | 1.225 |
| M1_C5 | 1 | True | True | True | True | True | 2283747 | 28392 | 1.228 |
| M1_C6 | 1 | True | True | True | True | True | 2283747 | 28392 | 1.228 |
| M1_C7 | 1 | True | True | True | True | True | 2283747 | 28392 | 1.228 |
| M1_C8 | 1 | True | True | True | True | True | 2283747 | 28392 | 1.228 |
| M1_C9 | 1 | True | True | True | True | True | 2283781 | 28358 | 1.226 |
| M1_C10 | 1 | True | True | True | True | True | 2283781 | 28358 | 1.226 |
| M1_C11 | 1 | True | True | True | True | True | 2283781 | 28358 | 1.226 |
| M1_C12 | 1 | True | True | True | True | True | 2283781 | 28358 | 1.226 |
| M1_C13 | 1 | True | True | True | True | True | 2283820 | 28319 | 1.225 |
| M1_C14 | 1 | True | True | True | True | True | 2283820 | 28319 | 1.225 |
| M1_C15 | 1 | True | True | True | True | True | 2283820 | 28319 | 1.225 |
| M1_C16 | 1 | True | True | True | True | True | 2283820 | 28319 | 1.225 |
| M1_C17 | 1 | True | True | True | True | True | 2283804 | 28335 | 1.225 |
| M1_C18 | 1 | True | True | True | True | True | 2283804 | 28335 | 1.225 |
| M1_C19 | 1 | True | True | True | True | True | 2283804 | 28335 | 1.225 |
| M1_C20 | 1 | True | True | True | True | True | 2283804 | 28335 | 1.225 |
| M1_C21 | 1 | True | True | True | True | True | 2283720 | 28419 | 1.229 |
| M1_C22 | 1 | True | True | True | True | True | 2283720 | 28419 | 1.229 |
| M1_C23 | 1 | True | True | True | True | True | 2283720 | 28419 | 1.229 |
| M1_C24 | 1 | True | True | True | True | True | 2283720 | 28419 | 1.229 |
| M1_C25 | 1 | True | True | True | True | True | 2283745 | 28394 | 1.228 |
| M1_C26 | 1 | True | True | True | True | True | 2283745 | 28394 | 1.228 |
| M1_C27 | 1 | True | True | True | False | True | 2283745 | 28394 | 1.228 |
| M1_C28 | 1 | True | True | True | True | True | 2283745 | 28394 | 1.228 |
| M1_C29 | 1 | True | True | True | True | True | 2283831 | 28308 | 1.224 |
| M1_C30 | 1 | True | True | True | True | True | 2283831 | 28308 | 1.224 |
| M1_C31 | 1 | True | True | True | True | True | 2283831 | 28308 | 1.224 |
| M1_C32 | 1 | True | True | True | True | True | 2283831 | 28308 | 1.224 |
| M1_C33 | 1 | True | True | True | True | True | 2283759 | 28380 | 1.227 |
| M1_C34 | 1 | True | True | True | True | True | 2283759 | 28380 | 1.227 |
| M1_C35 | 1 | True | True | True | True | True | 2283759 | 28380 | 1.227 |
| M1_C36 | 1 | True | True | True | True | True | 2283759 | 28380 | 1.227 |
| M2_C1 | 2 | True | True | True | True | True | 2282802 | 29337 | 1.269 |
| M2_C2 | 2 | True | True | True | True | True | 2282802 | 29337 | 1.269 |
| M2_C3 | 2 | True | True | True | True | True | 2282802 | 29337 | 1.269 |
| M2_C4 | 2 | True | True | True | True | True | 2282802 | 29337 | 1.269 |
| M2_C5 | 2 | True | True | True | True | True | 2283842 | 28297 | 1.224 |
| M2_C6 | 2 | True | True | True | True | True | 2283842 | 28297 | 1.224 |
| M2_C7 | 2 | True | True | True | True | True | 2283842 | 28297 | 1.224 |
| M2_C8 | 2 | True | True | True | True | True | 2283842 | 28297 | 1.224 |
| M2_C9 | 2 | True | True | True | True | True | 2283847 | 28292 | 1.224 |
| M2_C10 | 2 | True | True | True | True | True | 2283847 | 28292 | 1.224 |
| M2_C11 | 2 | True | True | True | True | True | 2283847 | 28292 | 1.224 |
| M2_C12 | 2 | True | True | True | True | True | 2283847 | 28292 | 1.224 |
| M2_C13 | 2 | True | True | True | True | True | 2283087 | 29052 | 1.256 |
| M2_C14 | 2 | True | True | True | True | True | 2283087 | 29052 | 1.256 |
| M2_C15 | 2 | True | True | True | True | True | 2283087 | 29052 | 1.256 |
| M2_C16 | 2 | True | True | True | True | True | 2283087 | 29052 | 1.256 |
| M2_C17 | 2 | True | True | True | True | True | 2283759 | 28380 | 1.227 |
| M2_C18 | 2 | True | True | True | True | True | 2283759 | 28380 | 1.227 |
| M2_C19 | 2 | True | True | True | True | True | 2283759 | 28380 | 1.227 |
| M2_C20 | 2 | True | True | True | True | True | 2283759 | 28380 | 1.227 |
| M2_C21 | 2 | True | True | True | True | True | 2283840 | 28299 | 1.224 |
| M2_C22 | 2 | True | True | True | True | True | 2283840 | 28299 | 1.224 |
| M2_C23 | 2 | True | True | True | True | True | 2283840 | 28299 | 1.224 |
| M2_C24 | 2 | True | True | True | True | True | 2283840 | 28299 | 1.224 |
| M2_C25 | 2 | True | True | True | True | True | 2283731 | 28408 | 1.229 |
| M2_C26 | 2 | True | True | True | True | True | 2283731 | 28408 | 1.229 |
| M2_C27 | 2 | True | True | True | True | True | 2283731 | 28408 | 1.229 |
| M2_C28 | 2 | True | True | True | True | True | 2283731 | 28408 | 1.229 |
| M2_C29 | 2 | True | True | True | True | True | 2283748 | 28391 | 1.228 |
| M2_C30 | 2 | True | True | True | True | True | 2283748 | 28391 | 1.228 |
| M2_C31 | 2 | True | True | True | True | True | 2283748 | 28391 | 1.228 |
| M2_C32 | 2 | True | True | True | True | True | 2283748 | 28391 | 1.228 |
| M2_C33 | 2 | True | True | True | True | True | 2283725 | 28414 | 1.229 |
| M2_C34 | 2 | True | True | True | True | True | 2283725 | 28414 | 1.229 |
| M2_C35 | 2 | True | True | True | True | True | 2283725 | 28414 | 1.229 |
| M2_C36 | 2 | True | True | True | True | True | 2283725 | 28414 | 1.229 |
| M3_C1 | 3 | True | True | True | True | True | 2283826 | 28313 | 1.225 |
| M3_C2 | 3 | True | True | True | True | True | 2283826 | 28313 | 1.225 |
| M3_C3 | 3 | True | True | True | True | True | 2283826 | 28313 | 1.225 |
| M3_C4 | 3 | True | True | True | True | True | 2283826 | 28313 | 1.225 |
| M3_C5 | 3 | True | True | True | True | True | 2283661 | 28478 | 1.232 |
| M3_C6 | 3 | True | True | True | True | True | 2283661 | 28478 | 1.232 |
| M3_C7 | 3 | True | True | True | True | True | 2283661 | 28478 | 1.232 |
| M3_C8 | 3 | True | True | True | True | True | 2283661 | 28478 | 1.232 |
| M3_C9 | 3 | True | True | True | True | True | 2283753 | 28386 | 1.228 |
| M3_C10 | 3 | True | True | True | True | True | 2283753 | 28386 | 1.228 |
| M3_C11 | 3 | True | True | True | False | True | 2283753 | 28386 | 1.228 |
| M3_C12 | 3 | True | True | True | True | True | 2283753 | 28386 | 1.228 |
| M3_C13 | 3 | True | True | True | True | True | 2283755 | 28384 | 1.228 |
| M3_C14 | 3 | True | True | True | True | True | 2283755 | 28384 | 1.228 |
| M3_C15 | 3 | True | True | True | True | True | 2283755 | 28384 | 1.228 |
| M3_C16 | 3 | True | True | True | True | True | 2283755 | 28384 | 1.228 |
| M3_C17 | 3 | True | True | True | True | True | 2283256 | 28883 | 1.249 |
| M3_C18 | 3 | True | True | True | True | True | 2283256 | 28883 | 1.249 |
| M3_C19 | 3 | True | True | True | True | True | 2283256 | 28883 | 1.249 |
| M3_C20 | 3 | True | True | True | True | True | 2283256 | 28883 | 1.249 |
| M3_C21 | 3 | True | True | True | True | True | 2283810 | 28329 | 1.225 |
| M3_C22 | 3 | True | True | True | True | True | 2283810 | 28329 | 1.225 |
| M3_C23 | 3 | True | True | True | True | True | 2283810 | 28329 | 1.225 |
| M3_C24 | 3 | True | True | True | True | True | 2283810 | 28329 | 1.225 |
| M3_C25 | 3 | True | True | True | True | True | 2283872 | 28267 | 1.223 |
| M3_C26 | 3 | True | True | True | True | True | 2283872 | 28267 | 1.223 |
| M3_C27 | 3 | True | True | True | True | True | 2283872 | 28267 | 1.223 |
| M3_C28 | 3 | True | True | True | True | True | 2283872 | 28267 | 1.223 |
| M3_C29 | 3 | True | True | True | True | True | 2282893 | 29246 | 1.265 |
| M3_C30 | 3 | True | True | True | True | True | 2282893 | 29246 | 1.265 |
| M3_C31 | 3 | True | True | True | True | True | 2282893 | 29246 | 1.265 |
| M3_C32 | 3 | True | True | True | True | True | 2282893 | 29246 | 1.265 |
| M3_C33 | 3 | True | True | True | True | True | 2283737 | 28402 | 1.228 |
| M3_C34 | 3 | True | True | True | True | True | 2283737 | 28402 | 1.228 |
| M3_C35 | 3 | True | True | True | True | True | 2283737 | 28402 | 1.228 |
| M3_C36 | 3 | True | True | True | True | True | 2283737 | 28402 | 1.228 |
| M4_C1 | 4 | True | True | True | True | True | 2283730 | 28409 | 1.229 |
| M4_C2 | 4 | True | True | True | True | True | 2283730 | 28409 | 1.229 |
| M4_C3 | 4 | True | True | True | True | True | 2283730 | 28409 | 1.229 |
| M4_C4 | 4 | True | True | True | True | True | 2283730 | 28409 | 1.229 |
| M4_C5 | 4 | True | True | True | True | True | 2283775 | 28364 | 1.227 |
| M4_C6 | 4 | True | True | True | True | True | 2283775 | 28364 | 1.227 |
| M4_C7 | 4 | True | True | True | True | True | 2283775 | 28364 | 1.227 |
| M4_C8 | 4 | True | True | True | True | True | 2283775 | 28364 | 1.227 |
| M4_C9 | 4 | True | True | True | True | True | 2283719 | 28420 | 1.229 |
| M4_C10 | 4 | True | True | True | True | True | 2283719 | 28420 | 1.229 |
| M4_C11 | 4 | True | True | True | True | True | 2283719 | 28420 | 1.229 |
| M4_C12 | 4 | True | True | True | True | True | 2283719 | 28420 | 1.229 |
| M4_C13 | 4 | True | True | True | True | True | 2283558 | 28581 | 1.236 |
| M4_C14 | 4 | True | True | True | True | True | 2283558 | 28581 | 1.236 |
| M4_C15 | 4 | True | True | True | True | True | 2283558 | 28581 | 1.236 |
| M4_C16 | 4 | True | True | True | True | True | 2283558 | 28581 | 1.236 |
| M4_C17 | 4 | True | True | True | True | True | 2283806 | 28333 | 1.225 |
| M4_C18 | 4 | True | True | True | True | True | 2283806 | 28333 | 1.225 |
| M4_C19 | 4 | True | True | True | True | True | 2283806 | 28333 | 1.225 |
| M4_C20 | 4 | True | True | True | True | True | 2283806 | 28333 | 1.225 |
| M4_C21 | 4 | True | True | True | True | True | 2283730 | 28409 | 1.229 |
| M4_C22 | 4 | True | True | True | True | True | 2283730 | 28409 | 1.229 |
| M4_C23 | 4 | True | True | True | True | True | 2283730 | 28409 | 1.229 |
| M4_C24 | 4 | True | True | True | True | True | 2283730 | 28409 | 1.229 |
| M4_C25 | 4 | True | True | True | True | True | 2283720 | 28419 | 1.229 |
| M4_C26 | 4 | True | True | True | True | True | 2283720 | 28419 | 1.229 |
| M4_C27 | 4 | True | True | True | False | True | 2283720 | 28419 | 1.229 |
| M4_C28 | 4 | True | True | True | True | True | 2283720 | 28419 | 1.229 |
| M4_C29 | 4 | True | True | True | True | True | 2283481 | 28658 | 1.239 |
| M4_C30 | 4 | True | True | True | True | True | 2283481 | 28658 | 1.239 |
| M4_C31 | 4 | True | True | True | True | True | 2283481 | 28658 | 1.239 |
| M4_C32 | 4 | True | True | True | True | True | 2283481 | 28658 | 1.239 |
| M4_C33 | 4 | True | True | True | True | True | 2283733 | 28406 | 1.229 |
| M4_C34 | 4 | True | True | True | True | True | 2283733 | 28406 | 1.229 |
| M4_C35 | 4 | True | True | True | True | True | 2283733 | 28406 | 1.229 |
| M4_C36 | 4 | True | True | True | True | True | 2283733 | 28406 | 1.229 |
| M5_C1 | 5 | True | True | True | True | True | 2283837 | 28302 | 1.224 |
| M5_C2 | 5 | True | True | True | True | True | 2283837 | 28302 | 1.224 |
| M5_C3 | 5 | True | True | True | True | True | 2283837 | 28302 | 1.224 |
| M5_C4 | 5 | True | True | True | False | True | 2283837 | 28302 | 1.224 |
| M5_C5 | 5 | True | True | True | True | True | 2283873 | 28266 | 1.223 |
| M5_C6 | 5 | True | True | True | True | True | 2283873 | 28266 | 1.223 |
| M5_C7 | 5 | True | True | True | True | True | 2283873 | 28266 | 1.223 |
| M5_C8 | 5 | True | True | True | True | True | 2283873 | 28266 | 1.223 |
| M5_C9 | 5 | True | True | True | True | True | 2282955 | 29184 | 1.262 |
| M5_C10 | 5 | True | True | True | True | True | 2282955 | 29184 | 1.262 |
| M5_C11 | 5 | True | True | True | True | True | 2282955 | 29184 | 1.262 |
| M5_C12 | 5 | True | True | True | True | True | 2282955 | 29184 | 1.262 |
| M5_C13 | 5 | True | True | True | True | True | 2283807 | 28332 | 1.225 |
| M5_C14 | 5 | True | True | True | True | True | 2283807 | 28332 | 1.225 |
| M5_C15 | 5 | True | True | True | True | True | 2283807 | 28332 | 1.225 |
| M5_C16 | 5 | True | True | True | True | True | 2283807 | 28332 | 1.225 |
| M5_C17 | 5 | True | True | True | True | True | 2283837 | 28302 | 1.224 |
| M5_C18 | 5 | True | True | True | True | True | 2283837 | 28302 | 1.224 |
| M5_C19 | 5 | True | True | True | True | True | 2283837 | 28302 | 1.224 |
| M5_C20 | 5 | True | True | True | True | True | 2283837 | 28302 | 1.224 |
| M5_C21 | 5 | True | True | True | True | True | 2283764 | 28375 | 1.227 |
| M5_C22 | 5 | True | True | True | True | True | 2283764 | 28375 | 1.227 |
| M5_C23 | 5 | True | True | True | True | True | 2283764 | 28375 | 1.227 |
| M5_C24 | 5 | True | True | True | True | True | 2283764 | 28375 | 1.227 |
| M5_C25 | 5 | True | True | True | True | True | 2283768 | 28371 | 1.227 |
| M5_C26 | 5 | True | True | True | True | True | 2283768 | 28371 | 1.227 |
| M5_C27 | 5 | True | True | True | True | True | 2283768 | 28371 | 1.227 |
| M5_C28 | 5 | True | True | True | True | True | 2283768 | 28371 | 1.227 |
| M5_C29 | 5 | True | True | True | True | True | 2283748 | 28391 | 1.228 |
| M5_C30 | 5 | True | True | True | True | True | 2283748 | 28391 | 1.228 |
| M5_C31 | 5 | True | True | True | True | True | 2283748 | 28391 | 1.228 |
| M5_C32 | 5 | True | True | True | True | True | 2283748 | 28391 | 1.228 |
| M5_C33 | 5 | True | True | True | True | True | 2283823 | 28316 | 1.225 |
| M5_C34 | 5 | True | True | True | True | True | 2283823 | 28316 | 1.225 |
| M5_C35 | 5 | True | True | True | True | True | 2283823 | 28316 | 1.225 |
| M5_C36 | 5 | True | True | True | True | True | 2283823 | 28316 | 1.225 |

---

## Full Sensor Completeness Table

| sensor_id | module_id | present_in_schema | present_in_silver_cells | present_in_hot_sensor_profile | silver_valid_count | silver_null_count | silver_null_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1_T1 | 1 | True | True | True | 2308751 | 3388 | 0.147 |
| M1_T2 | 1 | True | True | True | 2308751 | 3388 | 0.147 |
| M1_T3 | 1 | True | True | True | 2308751 | 3388 | 0.147 |
| M1_T4 | 1 | True | True | True | 2308751 | 3388 | 0.147 |
| M1_T5 | 1 | True | True | True | 2308751 | 3388 | 0.147 |
| M1_T6 | 1 | True | True | True | 2308751 | 3388 | 0.147 |
| M1_T7 | 1 | True | True | True | 2308751 | 3388 | 0.147 |
| M1_T8 | 1 | True | True | True | 2308751 | 3388 | 0.147 |
| M1_T9 | 1 | True | True | True | 2308740 | 3399 | 0.147 |
| M1_T10 | 1 | True | True | True | 2308740 | 3399 | 0.147 |
| M1_T11 | 1 | True | True | True | 2308740 | 3399 | 0.147 |
| M1_T12 | 1 | True | True | True | 2308740 | 3399 | 0.147 |
| M1_T13 | 1 | True | True | True | 2308736 | 3403 | 0.147 |
| M1_T14 | 1 | True | True | True | 2308736 | 3403 | 0.147 |
| M1_T15 | 1 | True | True | True | 2308736 | 3403 | 0.147 |
| M1_T16 | 1 | True | True | True | 2308736 | 3403 | 0.147 |
| M1_T17 | 1 | True | True | True | 2308655 | 3484 | 0.151 |
| M1_T18 | 1 | True | True | True | 2308655 | 3484 | 0.151 |
| M2_T1 | 2 | True | True | True | 2308655 | 3484 | 0.151 |
| M2_T2 | 2 | True | True | True | 2308655 | 3484 | 0.151 |
| M2_T3 | 2 | True | True | True | 2308655 | 3484 | 0.151 |
| M2_T4 | 2 | True | True | True | 2308655 | 3484 | 0.151 |
| M2_T5 | 2 | True | True | True | 2308655 | 3484 | 0.151 |
| M2_T6 | 2 | True | True | True | 2308655 | 3484 | 0.151 |
| M2_T7 | 2 | True | True | True | 2308679 | 3460 | 0.15 |
| M2_T8 | 2 | True | True | True | 2308679 | 3460 | 0.15 |
| M2_T9 | 2 | True | True | True | 2308679 | 3460 | 0.15 |
| M2_T10 | 2 | True | True | True | 2308679 | 3460 | 0.15 |
| M2_T11 | 2 | True | True | True | 2308676 | 3463 | 0.15 |
| M2_T12 | 2 | True | True | True | 2308676 | 3463 | 0.15 |
| M2_T13 | 2 | True | True | True | 2308676 | 3463 | 0.15 |
| M2_T14 | 2 | True | True | True | 2308676 | 3463 | 0.15 |
| M2_T15 | 2 | True | True | True | 2308664 | 3475 | 0.15 |
| M2_T16 | 2 | True | True | True | 2308664 | 3475 | 0.15 |
| M2_T17 | 2 | True | True | True | 2308664 | 3475 | 0.15 |
| M2_T18 | 2 | True | True | True | 2308664 | 3475 | 0.15 |
| M3_T1 | 3 | True | True | True | 2308662 | 3477 | 0.15 |
| M3_T2 | 3 | True | True | True | 2308662 | 3477 | 0.15 |
| M3_T3 | 3 | True | True | True | 2308661 | 3478 | 0.15 |
| M3_T4 | 3 | True | True | True | 2308661 | 3478 | 0.15 |
| M3_T5 | 3 | True | True | True | 2308632 | 3507 | 0.152 |
| M3_T6 | 3 | True | True | True | 2308632 | 3507 | 0.152 |
| M3_T7 | 3 | True | True | True | 2308632 | 3507 | 0.152 |
| M3_T8 | 3 | True | True | True | 2308632 | 3507 | 0.152 |
| M3_T9 | 3 | True | True | True | 2308630 | 3509 | 0.152 |
| M3_T10 | 3 | True | True | True | 2308630 | 3509 | 0.152 |
| M3_T11 | 3 | True | True | True | 2308629 | 3510 | 0.152 |
| M3_T12 | 3 | True | True | True | 2308629 | 3510 | 0.152 |
| M3_T13 | 3 | True | True | True | 2308691 | 3448 | 0.149 |
| M3_T14 | 3 | True | True | True | 2308691 | 3448 | 0.149 |
| M3_T15 | 3 | True | True | True | 2308691 | 3448 | 0.149 |
| M3_T16 | 3 | True | True | True | 2308691 | 3448 | 0.149 |
| M3_T17 | 3 | True | True | True | 2308687 | 3452 | 0.149 |
| M3_T18 | 3 | True | True | True | 2308687 | 3452 | 0.149 |
| M4_T1 | 4 | True | True | True | 2308686 | 3453 | 0.149 |
| M4_T2 | 4 | True | True | True | 2308686 | 3453 | 0.149 |
| M4_T3 | 4 | True | True | True | 2308665 | 3474 | 0.15 |
| M4_T4 | 4 | True | True | True | 2308665 | 3474 | 0.15 |
| M4_T5 | 4 | True | True | True | 2308663 | 3476 | 0.15 |
| M4_T6 | 4 | True | True | True | 2308663 | 3476 | 0.15 |
| M4_T7 | 4 | True | True | True | 2308662 | 3477 | 0.15 |
| M4_T8 | 4 | True | True | True | 2308662 | 3477 | 0.15 |
| M4_T9 | 4 | True | True | True | 2308660 | 3479 | 0.15 |
| M4_T10 | 4 | True | True | True | 2308660 | 3479 | 0.15 |
| M4_T11 | 4 | True | True | True | 2307830 | 4309 | 0.186 |
| M4_T12 | 4 | True | True | True | 2307830 | 4309 | 0.186 |
| M4_T13 | 4 | True | True | True | 2307817 | 4322 | 0.187 |
| M4_T14 | 4 | True | True | True | 2307817 | 4322 | 0.187 |
| M4_T15 | 4 | True | True | True | 2307816 | 4323 | 0.187 |
| M4_T16 | 4 | True | True | True | 2307816 | 4323 | 0.187 |
| M4_T17 | 4 | True | True | True | 2307816 | 4323 | 0.187 |
| M4_T18 | 4 | True | True | False | 2307816 | 4323 | 0.187 |
| M5_T1 | 5 | True | True | True | 2308356 | 3783 | 0.164 |
| M5_T2 | 5 | True | True | True | 2308356 | 3783 | 0.164 |
| M5_T3 | 5 | True | True | True | 2308343 | 3796 | 0.164 |
| M5_T4 | 5 | True | True | True | 2308343 | 3796 | 0.164 |
| M5_T5 | 5 | True | True | True | 2308343 | 3796 | 0.164 |
| M5_T6 | 5 | True | True | True | 2308343 | 3796 | 0.164 |
| M5_T7 | 5 | True | True | True | 2308329 | 3810 | 0.165 |
| M5_T8 | 5 | True | True | True | 2308329 | 3810 | 0.165 |
| M5_T9 | 5 | True | True | True | 2308372 | 3767 | 0.163 |
| M5_T10 | 5 | True | True | True | 2308372 | 3767 | 0.163 |
| M5_T11 | 5 | True | True | True | 2308328 | 3811 | 0.165 |
| M5_T12 | 5 | True | True | True | 2308328 | 3811 | 0.165 |
| M5_T13 | 5 | True | True | True | 2308328 | 3811 | 0.165 |
| M5_T14 | 5 | True | True | True | 2308328 | 3811 | 0.165 |
| M5_T15 | 5 | True | True | True | 2308328 | 3811 | 0.165 |
| M5_T16 | 5 | True | True | True | 2308328 | 3811 | 0.165 |
| M5_T17 | 5 | True | True | True | 2307806 | 4333 | 0.187 |
| M5_T18 | 5 | True | True | True | 2307806 | 4333 | 0.187 |
