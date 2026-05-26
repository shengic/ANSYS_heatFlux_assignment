# TDD_TEST_PLAN.md — AL-1605-0740 SPECTRA → ANSYS Heat Flux Assignment
## 測試驅動開發計畫
**Version 2.0 · Albert Sheng · 2025**
**依據 12-Rule Template Rule 9：測試 WHY，不只是 WHAT**

---

## 總覽

| 測試檔案 | 對應模組 | 測試數量 | 最關鍵測試 |
|---|---|---|---|
| `test_geometry.py` | `math_core/geometry.py` | 8 | `test_right_handed_coordinate_system` |
| `test_bilinear_interpolation.py` | `math_core/bilinear_interpolation.py` | 10 | 4 個邊界條件測試 |
| `test_coordinate_transform.py` | `math_core/coordinate_transform.py` | 9 | `test_batch_matches_individual_results` |
| `test_grazing_angle.py` | `math_core/grazing_angle.py` | 7 | `test_result_is_positive_without_abs` |
| `test_unit_conversion.py` | `math_core/unit_conversion.py` | 7 | `test_formula_equivalence_with_vba` |
| `test_spatial_search.py` | `math_core/spatial_search.py` | 14 | `test_right_boundary_returns_last_column`、`test_batch_matches_individual_results` |
| `test_ansys_reader.py` | `io/ansys_reader.py` | 11 | `test_heatflux_corner_nodes_are_indices_5_to_8` |
| `test_ansys_node_store.py` | `model/ansys_node_store.py` | 6 | `test_batch_matches_individual`、`test_memory_smaller_than_dict` |
| `test_spectra_reader.py` | `io/spectra_reader.py` | 9 | `test_elements_built_with_correct_node_layout` |
| `test_output_writer.py` | `io/output_writer.py` | 11 | `test_output_has_exactly_five_columns`、`test_batch_output_matches_sequential` |
| `test_mapping_pipeline.py` | `pipeline/mapping_pipeline.py` | 5 | `test_vectorized_matches_sequential` |
| **合計** | | **97** | |

---

## 實作順序原則

測試必須在對應模組實作**之前**撰寫（TDD Red-Green-Refactor）。

```
第一批：math_core 層（無 I/O 依賴）
  test_geometry, test_bilinear, test_coordinate_transform,
  test_grazing_angle, test_unit_conversion, test_spatial_search

第二批：model 層
  test_ansys_node_store

第三批：io 層
  test_ansys_reader, test_spectra_reader, test_output_writer

第四批：pipeline 層
  test_mapping_pipeline
```

---

## `tests/test_geometry.py`

**對應模組：** `math_core/geometry.py` — `SourceGeometry.from_points()`

| # | 測試函式名稱 | 測試類型 | 驗證內容 | WHY（業務意圖）|
|---|---|---|---|---|
| 1 | `test_unit_vectors_have_length_one` | 數值正確性 | `‖ê_X‖=‖ê_Y‖=‖ê_Z‖=1.0`（誤差<1e-12）| 非單位向量讓 arcsin 產生錯誤 mrad，導致熱通量映射系統性偏移 |
| 2 | `test_basis_vectors_are_orthogonal` | 數值正確性 | `ê_X·ê_Y=ê_Y·ê_Z=ê_X·ê_Z=0`（誤差<1e-12）| 非正交基底造成 xmrad 和 ymrad 互相污染 |
| 3 | `test_right_handed_coordinate_system` | 幾何正確性 | `det([ê_X,ê_Y,ê_Z])>0` | 左手系讓 mrad 符號反轉，熱通量映射到錯誤方向 |
| 4 | `test_beam_axis_aligns_with_ez` | 幾何正確性 | `(target−source)/‖…‖≈ê_Z`（誤差<1e-12）| ê_Z 必須與束軸對齊，否則掠射角基準錯誤 |
| 5 | `test_side_point_in_ex_plane` | 幾何正確性 | `(side−source)·ê_Y≈0`（誤差<1e-12）| side 點定義水平面，ê_Y 有分量代表基底建構錯誤 |
| 6 | `test_degenerate_source_target_same_point` | 邊界條件 | `source==target` → 拋出 `ValueError` | 零向量無法正規化，應明確報錯而非產生 NaN 靜默傳播 |
| 7 | `test_degenerate_side_colinear_with_axis` | 邊界條件 | side 在束軸上 → 拋出 `ValueError` | 叉積為零，ê_Y 無法定義 |
| 8 | `test_known_axis_aligned_case` | 回歸測試 | `source=(0,0,0),target=(0,0,1),side=(1,0,0)` → `ê_X=(1,0,0),ê_Y=(0,1,0),ê_Z=(0,0,1)` | 標準座標系精確結果，防止重構時靜默改變建構順序 |

---

## `tests/test_bilinear_interpolation.py`

**對應模組：** `math_core/bilinear_interpolation.py` — `bilinear_coefficients()`

Node layout：Node1(x1,y2)、Node2(x1,y1)、Node3(x2,y2)、Node4(x2,y1)

| # | 測試函式名稱 | 測試類型 | 驗證內容 | WHY（業務意圖）|
|---|---|---|---|---|
| 1 | `test_boundary_condition_node2_bottom_left` | 數學正確性 | `φ(x1,y1)=f2` | 邊界條件是充要條件，4 個必須全通過 |
| 2 | `test_boundary_condition_node1_top_left` | 數學正確性 | `φ(x1,y2)=f1` | 同上 |
| 3 | `test_boundary_condition_node4_bottom_right` | 數學正確性 | `φ(x2,y1)=f4` | 同上 |
| 4 | `test_boundary_condition_node3_top_right` | 數學正確性 | `φ(x2,y2)=f3` | 同上 |
| 5 | `test_center_point_is_average_of_four_corners` | 數學性質 | `φ(mid,mid)=(f1+f2+f3+f4)/4` | 驗證 a₃ 項對稱性 |
| 6 | `test_uniform_field_returns_constant` | 退化案例 | `f1=f2=f3=f4=C` → `φ=C` | 均勻場插值仍為常數，`a₁=a₂=a₃=0` |
| 7 | `test_linear_x_variation_no_y_dependence` | 退化案例 | `f1=f2=0,f3=f4=1` → `φ=(x−x1)/(x2−x1)` | 確認 `a₂=a₃=0` |
| 8 | `test_linear_y_variation_no_x_dependence` | 退化案例 | `f1=f3=1,f2=f4=0` → `φ=(y−y1)/(y2−y1)` | 確認 `a₁=a₃=0` |
| 9 | `test_coefficient_delta_zero_raises` | 邊界條件 | `x1==x2` 或 `y1==y2` → 拋出 `ValueError` | Δ=0 除以零應明確報錯 |
| 10 | `test_known_numerical_values` | 回歸測試 | 手算係數比對至 1e-10 | 防止重構時靜默改變公式 |

---

## `tests/test_coordinate_transform.py`

**對應模組：** `math_core/coordinate_transform.py`
**涵蓋：** `map_to_mrad()`（單點）+ `map_to_mrad_batch()`（批次）

| # | 測試函式名稱 | 測試類型 | 驗證內容 | WHY（業務意圖）|
|---|---|---|---|---|
| 1 | `test_point_on_beam_axis_returns_zero_mrad` | 幾何正確性 | 形心在束軸上 → `xmrad=0,ymrad=0` | 束軸上的點角度必須為零 |
| 2 | `test_pure_x_displacement_gives_nonzero_xmrad_only` | 幾何正確性 | 只在 ê_X 方向偏移 → `xmrad≠0,ymrad=0` | 兩角度分量相互獨立 |
| 3 | `test_pure_y_displacement_gives_nonzero_ymrad_only` | 幾何正確性 | 只在 ê_Y 方向偏移 → `xmrad=0,ymrad≠0` | 同上，Y 方向 |
| 4 | `test_distance_is_euclidean_norm` | 數值正確性 | `distance_mm=‖centroid−source‖` | 距離用於單位換算，錯誤影響所有 W/mm² |
| 5 | `test_small_angle_arcsin_matches_atan2` | 精度驗證 | 5mrad 以內差異 < 0.004% | 確認小角近似在波動器範圍內有效 |
| 6 | `test_known_geometry_numerical_result` | 回歸測試 | 已知配置 → 預期 mrad（誤差<1e-6）| 防止重構時靜默改變公式 |
| 7 | `test_source_at_centroid_raises` | 邊界條件 | `centroid==source` → 拋出 `ValueError` | R=0 除以零應明確報錯 |
| 8 | `test_batch_matches_individual_results` | **批次一致性** | N 個點：`map_to_mrad_batch` 與 N 次 `map_to_mrad` 結果完全一致（誤差<1e-12）| 向量化路徑的正確性保證 |
| 9 | `test_batch_performance` | 效能 | 10,000 點批次計算 < 50ms | 確認 numpy 向量化效益實際存在 |

---

## `tests/test_grazing_angle.py`

**對應模組：** `math_core/grazing_angle.py` — `get_grazing_angle_rad()`

> ⚠️ 此測試檔最重要任務：驗證 **VBA Bug #2 已修正**（`v1×v2`，不依賴 `abs()`）

| # | 測試函式名稱 | 測試類型 | 驗證內容 | WHY（業務意圖）|
|---|---|---|---|---|
| 1 | `test_normal_incidence_returns_zero` | 幾何正確性 | 束方向與法線平行（90°入射）→ `θ=0` | 正向入射 `sin(0)=0`，投影功率密度為零 |
| 2 | `test_grazing_incidence_45_degrees` | 幾何正確性 | 45° 掠射 → `θ=π/4`（誤差<1e-10）| 已知角度精確驗證 |
| 3 | `test_result_is_positive_without_abs` | **Bug #2 修正驗證** | 所有案例回傳值 > 0，函式內無 `abs()` | 正確 `v1×v2` 產生外法線，`n̂·v̂>0`，自然為正 |
| 4 | `test_outward_normal_points_toward_source` | 幾何正確性 | `n̂·beam_vec>0` | 確認外法線（v1×v2）而非內法線（v2×v1）|
| 5 | `test_flat_xy_plane_element_with_z_beam` | 基本案例 | 水平面，束從 +z 入射 → `θ=0` | 最直觀案例，易於手算 |
| 6 | `test_tilted_surface_known_angle` | 回歸測試 | 已知傾斜面（手算）→ 比對數值（誤差<1e-10）| 防止重構時靜默改變叉積方向 |
| 7 | `test_degenerate_zero_area_element_raises` | 邊界條件 | 4 角點共線 → 拋出 `ValueError` | 法向量為零無法正規化 |

---

## `tests/test_unit_conversion.py`

**對應模組：** `math_core/unit_conversion.py`

公式：`flux [W/mm²] = pd [kW/mrad²] × 10⁹ / R_mm²`

| # | 測試函式名稱 | 測試類型 | 驗證內容 | WHY（業務意圖）|
|---|---|---|---|---|
| 1 | `test_known_value_at_1000mm` | 數值正確性 | `1kW/mrad²` at `R=1000mm` → `1000 W/mm²` | 反平方律基準驗證點 |
| 2 | `test_inverse_square_law_doubles_distance` | 物理性質 | R×2 → 通量÷4 | 確認 R² 反比，非線性關係 |
| 3 | `test_inverse_square_law_halves_distance` | 物理性質 | R÷2 → 通量×4 | 反向驗證 |
| 4 | `test_linearity_in_power_density` | 物理性質 | pd×2 → flux×2 | 確認線性比例 |
| 5 | `test_formula_equivalence_with_vba` | 等價驗證 | `pd*1e9/R²` == `pd*1e3/(R/1e3)²`（VBA 公式）| Python 與 VBA 數值完全等價 |
| 6 | `test_zero_distance_raises` | 邊界條件 | `distance_mm=0` → 拋出 `ValueError` | 除以零應明確報錯 |
| 7 | `test_projected_flux_sin_theta` | 數值正確性 | θ=30° → `projected=0.5×normal` | `sin(π/6)=0.5` 直接驗證 |

---

## `tests/test_spatial_search.py`

**對應模組：** `math_core/spatial_search.py`
**涵蓋：** `find_element()`（單點）+ `find_elements_batch()`（批次）

> ⚠️ searchsorted 右邊界處理是最容易出錯的 edge case，必須明確測試。

| # | 測試函式名稱 | 測試類型 | 驗證內容 | WHY（業務意圖）|
|---|---|---|---|---|
| 1 | `test_interior_point_returns_correct_index` | 功能正確性 | Grid 內部點 → 正確 1-based index | 索引錯誤取到鄰近元素係數，造成系統性插值誤差 |
| 2 | `test_left_boundary_returns_first_column` | 邊界條件 | `xmrad=x_min` → 第 1 欄元素（非 None）| 邊界點必須落在 grid 內 |
| 3 | `test_right_boundary_returns_last_column` | **關鍵邊界** | `xmrad=x_max` → 最後一欄（非 None）| searchsorted 右邊界問題，最容易出錯 |
| 4 | `test_bottom_boundary_returns_first_row` | 邊界條件 | `ymrad=y_min` → 第 1 列（非 None）| Y 方向同上 |
| 5 | `test_top_boundary_returns_last_row` | **關鍵邊界** | `ymrad=y_max` → 最後一列（非 None）| Y 方向右邊界 |
| 6 | `test_outside_right_returns_none` | 邊界條件 | `xmrad>x_max` → None | Grid 外節點熱通量為 0，不能取到錯誤元素 |
| 7 | `test_outside_left_returns_none` | 邊界條件 | `xmrad<x_min` → None | 同上 |
| 8 | `test_outside_top_returns_none` | 邊界條件 | `ymrad>y_max` → None | 同上 |
| 9 | `test_outside_bottom_returns_none` | 邊界條件 | `ymrad<y_min` → None | 同上 |
| 10 | `test_index_is_one_based` | 介面契約 | 左上角元素回傳 `1`，不是 `0` | 與 VBA 語意一致 |
| 11 | `test_row_major_ordering` | 介面契約 | 3×3 grid index 順序：左→右、上→下（1,2,3,4…）| pipeline 用此 index 查找 spectra_elements |
| 12 | `test_batch_matches_individual_results` | **批次一致性** | N 個點：`find_elements_batch` 與 N 次 `find_element` 結果完全一致 | 向量化路徑的正確性保證 |
| 13 | `test_batch_handles_out_of_bounds` | 批次邊界 | 批次輸入含 grid 外的點 → 對應位置回傳 -1 | batch 路徑的邊界處理正確 |
| 14 | `test_performance_large_grid` | 效能 | 1000×1000 grid，1000 次批次查詢 < 10ms | searchsorted O(log N) 效益實際存在 |

---

## `tests/test_ansys_reader.py`

**對應模組：** `io/ansys_reader.py`

| # | 測試函式名稱 | 測試類型 | 驗證內容 | WHY（業務意圖）|
|---|---|---|---|---|
| 1 | `test_parse_node_line_extracts_id_and_xyz` | 解析正確性 | `"  12345  1.23456E+02  0.0  5.678E+01"` → `AnsysNode(12345,123.456,0,56.78)` | 節點座標是整個計算鏈起點 |
| 2 | `test_parse_node_with_leading_spaces` | 格式穩健性 | 多個前置空格仍正確解析 | ANSYS 不同版本縮排不同 |
| 3 | `test_node_section_terminates_on_minus_one` | 狀態機正確性 | `-1` 後停止解析節點 | 未終止會解析到 element section |
| 4 | `test_element_section_counts_only` | 記憶體策略 | element section 後 node store 不新增資料，只有計數增加 | 大型模型記憶體保護策略 |
| 5 | `test_heatflux_line_extracts_element_and_8_nodes` | 解析正確性 | heat flux 行 → 正確 element_id 和 8 個 node 引用 | node 索引錯誤導致面積和掠射角全錯 |
| 6 | `test_heatflux_corner_nodes_are_indices_5_to_8` | **Token offset** | `corner_nodes[0..3]` 對應 token[5..8] | 偏移 1 會靜默產生錯誤結果 |
| 7 | `test_heatflux_midside_nodes_are_indices_9_to_12` | **Token offset** | `midside_nodes[0..3]` 對應 token[9..12] | 同上 |
| 8 | `test_heatflux_surface_area_is_computed_on_load` | 初始化行為 | 解析後 `surface_area_mm2>0` | 面積在讀檔時 eager 計算 |
| 9 | `test_section_transition_node_to_element` | 狀態機正確性 | Elements 觸發後不再解析節點 | 狀態機轉換錯誤是最常見 parser bug |
| 10 | `test_section_transition_element_to_heatflux` | 狀態機正確性 | Heat Flux 觸發後開始解析 hf | 同上 |
| 11 | `test_full_file_minimal_fixture` | 整合測試 | 最小合法 .dat → 正確節點數、元素計數、flux 元素數 | 端對端驗證三個 section 協作 |

---

## `tests/test_ansys_node_store.py`（新增）

**對應模組：** `model/ansys_node_store.py`
**存在理由：** AnsysNodeStore 是效能改進的核心資料結構，
必須獨立驗證其正確性，尤其是批次查詢與逐點查詢的等價性。

| # | 測試函式名稱 | 測試類型 | 驗證內容 | WHY（業務意圖）|
|---|---|---|---|---|
| 1 | `test_get_xyz_returns_correct_coordinates` | 功能正確性 | `get_xyz(node_id)` 回傳正確 (x,y,z) | 座標查詢是所有幾何計算的基礎 |
| 2 | `test_get_xyz_batch_matches_individual` | **批次一致性** | `get_xyz_batch([id1,id2,...])` 與逐次 `get_xyz()` 完全一致 | 向量化形心計算的正確性保證 |
| 3 | `test_unknown_node_id_raises` | 邊界條件 | 查詢不存在的 node_id → 拋出 `KeyError` | 靜默回傳錯誤座標比報錯更危險 |
| 4 | `test_memory_smaller_than_dict` | **效能/記憶體** | 100,000 節點：`AnsysNodeStore` 記憶體用量 < `dict[int,AnsysNode]` 的 20% | 記憶體改進是引入此結構的核心理由 |
| 5 | `test_build_from_numpy_arrays` | 建構正確性 | 從 `(N,)` id array 和 `(N,3)` xyz array 建構後查詢正確 | 向量化讀取後的建構路徑 |
| 6 | `test_batch_query_performance` | 效能 | 8 個節點批次查詢（模擬 heat flux 元素）× 10,000 元素 < 100ms | 相較逐點查詢應有顯著提升 |

---

## `tests/test_spectra_reader.py`

**對應模組：** `io/spectra_reader.py`

| # | 測試函式名稱 | 測試類型 | 驗證內容 | WHY（業務意圖）|
|---|---|---|---|---|
| 1 | `test_two_header_lines_are_skipped` | 格式正確性 | 前 2 行不解析為資料點 | Header 解析為節點導致 grid 維度全錯 |
| 2 | `test_grid_dimensions_auto_detected` | **核心功能** | 3×4 資料 → `n_row=3,n_col=4` | Grid 維度決定所有元素 node index 計算 |
| 3 | `test_total_nodes_equals_rows_times_cols` | 一致性 | `len(nodes)==n_row×n_col` | 維度不一致代表漏行或重複 |
| 4 | `test_node_xmrad_ymrad_power_correctly_parsed` | 解析正確性 | 第一節點 xmrad,ymrad,power_density 正確 | x/y 對調會使功率密度映射到錯誤方向 |
| 5 | `test_peak_power_density_detected` | 統計正確性 | `peak=max(node.power_density)` | GUI 顯示，使用者用來確認資料正確 |
| 6 | `test_total_power_accumulated` | 統計正確性 | `total≈Σ(pd×area_mrad2)` | GUI 顯示 |
| 7 | `test_empty_line_terminates_reading` | 格式穩健性 | 空行 → 停止解析，不拋例外 | SPECTRA 輸出有時有尾端空行 |
| 8 | `test_elements_built_with_correct_node_layout` | **元素建構** | `element[0].nodes[0]` 是 top-left(x1,y2)；`nodes[1]` 是 bottom-left(x1,y1) | Node layout 錯誤讓邊界條件失敗，插值結果系統性偏移 |
| 9 | `test_bilinear_coefficients_satisfy_boundary_conditions` | 整合驗證 | 每個 SpectraElement 滿足 4 個邊界條件 | spectra_reader 與 bilinear_interpolation 的整合驗證 |

---

## `tests/test_output_writer.py`

**對應模組：** `io/output_writer.py`

> ⚠️ 最重要任務：驗證 **VBA Bug #1 已修正**（剛好 5 欄）及批次輸出正確性。

| # | 測試函式名稱 | 測試類型 | 驗證內容 | WHY（業務意圖）|
|---|---|---|---|---|
| 1 | `test_output_has_exactly_five_columns` | **Bug #1 修正驗證** | 每行 `split(',')` 後剛好 5 個元素 | 直接驗證 Bug #1 修正 |
| 2 | `test_no_trailing_zero_columns` | **Bug #1 修正驗證** | 無第 6 欄以後的 `"0.0000E+00"` | 反面確認無多餘欄位 |
| 3 | `test_column_order_x_y_z_scaled_unscaled` | 格式正確性 | 欄位順序：x,y,z,pd×ratio,pd | ANSYS External Data 按欄位位置匹配 |
| 4 | `test_scientific_notation_format` | 格式正確性 | 符合 `{:.4E}` 格式 | ANSYS 對數值格式有要求 |
| 5 | `test_total_power_ratio_applied_to_column4_only` | 功能正確性 | `ratio=2.0` → 第4欄=2×第5欄 | ratio 只縮放輸出欄，不改 dataclass 原始值 |
| 6 | `test_ratio_1_gives_identical_col4_and_col5` | 邊界條件 | `ratio=1.0` → 第4欄==第5欄 | 預設值驗證 |
| 7 | `test_zero_power_density_element_written_as_zeros` | 邊界條件 | grid 外元素（pd=0.0）→ 第4、5欄均為 `0.0000E+00` | 這些元素仍須寫入（保持行數）|
| 8 | `test_element_count_matches_input` | 完整性 | 輸入 N 個 → 輸出 N 行 | 不能遺漏元素，ANSYS 依行數對應節點 |
| 9 | `test_xyz_values_are_centroids` | 數值正確性 | 輸出 x,y,z 與 `element.x/y/z` 一致 | 輸出形心座標，不是節點座標 |
| 10 | `test_batch_output_matches_sequential` | **批次一致性** | 批次 buffer 輸出與逐行輸出結果完全相同（byte-for-byte）| 效能改進不改變輸出內容 |
| 11 | `test_batch_output_performance` | 效能 | 50,000 元素批次輸出 < 1 秒 | 確認批次 buffer 效益實際存在 |

---

## `tests/test_mapping_pipeline.py`（新增）

**對應模組：** `pipeline/mapping_pipeline.py`
**存在理由：** pipeline 整合多個模組，必須獨立驗證向量化路徑與逐元素路徑的等價性，
這是整個效能改進的正確性保證。

| # | 測試函式名稱 | 測試類型 | 驗證內容 | WHY（業務意圖）|
|---|---|---|---|---|
| 1 | `test_vectorized_matches_sequential` | **核心等價性** | 同一輸入：`vectorized=True` 與 `vectorized=False` 所有輸出欄位誤差 < 1e-10 | 向量化路徑的最終正確性保證；任何數值差異都代表實作錯誤 |
| 2 | `test_out_of_bounds_elements_get_zero_power` | 功能正確性 | 形心落在 SPECTRA grid 外的元素 → `projected_power_density_w_mm2==0.0` | 靜默歸零而非報錯；使用者需靠 GUI 統計知道有多少元素在外 |
| 3 | `test_progress_callback_called_for_each_chunk` | UI 整合 | `progress_cb` 被呼叫；最後一次呼叫時 `current==total` | GUI progress bar 需要正確的進度回報 |
| 4 | `test_none_progress_callback_silent` | 介面契約 | `progress_cb=None` 時不拋例外，靜默完成 | CLI/batch 模式不需要 GUI 回呼 |
| 5 | `test_total_power_sum_is_consistent` | 物理一致性 | `Σ(element.total_power_w)` 與手算值誤差 < 0.1% | 總功率是使用者驗證結果正確性的主要指標 |

---

## 測試 Fixture 設計

### 最小合法 ANSYS .dat fixture

```
/com,*** Nodes for the whole assembly
NBLOCK,6,SOLID,,
(3i9,6e21.13e3)
        1        0        0  1.00000000000E+002  0.00000000000E+000  5.00000000000E+001
        2        0        0  2.00000000000E+002  0.00000000000E+000  5.00000000000E+001
        3        0        0  1.50000000000E+002  0.00000000000E+000  6.00000000000E+001
        4        0        0  1.50000000000E+002  0.00000000000E+000  4.00000000000E+001
-1
/com,*** Elements for the solver
EBLOCK,19,SOLID,,1
(19i9)
        1        1        1        1        0        0        0        0        8        0        1        2        3        4        5        6        7        8
-1
/com,*** Create "Heat Flux" on surface
CMBLOCK,HeatFlux,ELEM,1
(8i10)
         1         0         0         0         0        1        2        3        4        5        6        7        8
-1
```

### 最小合法 SPECTRA .dta fixture（3×3 grid，4 個元素）

```
# Power density distribution
# Units: kW/mrad^2
 -1.0  -1.0   5.00000E+00
  0.0  -1.0   8.00000E+00
  1.0  -1.0   5.00000E+00
 -1.0   0.0   8.00000E+00
  0.0   0.0  1.20000E+01
  1.0   0.0   8.00000E+00
 -1.0   1.0   5.00000E+00
  0.0   1.0   8.00000E+00
  1.0   1.0   5.00000E+00
```

→ `n_row=3, n_col=3`，4 個 SpectraElement

---

## Claude Code 指令範本

每次一個測試檔，格式如下：

```
實作 tests/test_ansys_node_store.py。

成功條件：
1. pytest tests/test_ansys_node_store.py 全部 6 個測試通過
2. test_batch_matches_individual：batch 與逐點查詢誤差為 0（完全相同）
3. test_memory_smaller_than_dict：記憶體用量 < dict 版本的 20%
4. 不要動 model/ansys_node_store.py 以外的檔案

參考文件：
- CONTEXT.md §4.1（AnsysNodeStore 設計）
- AL1605_project_structure.md model/ansys_node_store.py 職責說明
```

---

## 效能測試基準

測試機器：一般工程工作站（Intel i7 / AMD Ryzen，16GB RAM）

| 測試 | 基準輸入規模 | 通過門檻 |
|---|---|---|
| `test_batch_performance`（座標轉換）| 10,000 點 | < 50ms |
| `test_performance_large_grid`（搜尋）| 1000×1000 grid，1000次查詢 | < 10ms |
| `test_batch_query_performance`（節點查詢）| 8節點×10,000元素 | < 100ms |
| `test_batch_output_performance`（輸出）| 50,000 元素 | < 1s |

---

*此文件 v2.0 新增：test_ansys_node_store.py（6 個測試）、test_mapping_pipeline.py（5 個測試）、*
*各批次一致性測試、效能測試基準。總計 97 個測試。*
*與 CLAUDE.md、CONTEXT.md、AL1605_project_structure.md 配合使用。*
