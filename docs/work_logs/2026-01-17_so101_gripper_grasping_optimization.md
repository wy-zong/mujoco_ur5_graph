# [2026-01-17] SO101 夾爪抓取穩定性優化

## 問題描述

SO101 機械手臂在模擬環境中夾取物體（圓柱體/方塊）時，物體容易滑落。問題在於 SO101 使用**旋轉式夾爪結構**（hinge joint），與 UR5e 的 2F85 平行夾爪不同，接觸面會隨角度變化而傾斜。

---

## 解決方案歷程

### 1. 補齊碰撞墊物理參數（部分有效）

**邏輯**：比較 UR5e 2F85 夾爪設定，發現 SO101 碰撞墊缺少 `friction`、`solimp`、`solref`、`priority` 等參數。

**實作**：
```xml
<geom name="so101_fixed_pad" type="box" 
      friction="0.7" solimp="0.95 0.99 0.001" solref="0.004 1"
      priority="1" condim="6" .../>
```

**結果**：夾取有改善，但**角度不對時仍然滑脫**。

---

### 2. 多段式碰撞墊（無效）

**邏輯**：旋轉夾爪的接觸面是弧形，單一 box 無法精確模擬，所以用 3 段 box 碰撞墊貼合弧度。

**實作**：將單一碰撞塊替換為 `_tip`、`_mid`、`_base` 三段設計。

**結果**：**仍然滑脫**，因為多個小 box 並沒有改變根本問題。

---

### 3. 球形/膠囊碰撞墊（部分有效但難調整）

**邏輯**：Sphere/Capsule 的圓形接觸面在任何角度都能產生接觸。

**實作**：
```xml
<geom name="so101_fixed_pad" type="capsule" size="0.008 0.04" .../>
<geom name="so101_moving_pad" type="sphere" size="0.012" .../>
```

**結果**：
- ⚠️ 位置難以調整，需要反覆微調 `pos`、`quat`
- ⚠️ Sphere 表面與物體接觸面積小，抓握力不足

---

### 4. Flexcomp Grid 軟墊（失敗）

**邏輯**：MuJoCo 3.0 的 `flexcomp` 可以模擬可變形軟墊，軟墊能貼合物體表面，增加接觸面積。

**實作**：
```xml
<flexcomp type="grid" count="2 3 5" spacing="0.015 0.02 0.025"
          pos="-0.02 -0.005 -0.06" dim="3" mass="0.05" radius="0.006" ...>
  <edge equality="true" damping="0.1"/>
  <pin id="0 1 2 3 4 5"/>
</flexcomp>
```

**結果**：
- ❌ 軟墊與夾爪結構重疊
- ❌ 尾部脫落（`<pin>` 頂點 ID 設定不正確）
- ❌ 模擬極慢（physics 1234ms）
- ❌ 產生 NaN 不穩定警告

**失敗原因**：`flexcomp grid` 自動生成網格，無法精確貼合夾爪內側。需要製作專用 mesh 檔案才能正確使用。

---

### 5. 啟用 multiccd + nativeccd（有效）

**邏輯**：根據 MuJoCo 文檔 "Preventing slip" 建議，啟用這兩個選項可以在平面接觸時產生更多接觸點。

**實作**：
```xml
<option timestep="0.002" integrator="implicitfast" solver="PGS" cone="elliptic" impratio="10">
    <flag multiccd="enable" nativeccd="enable"/>
</option>
```

**結果**：✅ **有效改善抓取穩定性**

---

### 6. 大幅提高摩擦係數（有效）

**邏輯**：直接增加滑動摩擦係數，讓物體在接觸面上更難滑動。

**實作**：`friction` 從 `0.7` 提高到 `5.0`

**結果**：✅ **有效改善抓取穩定性**

---

### 7. 使用原生 Mesh 碰撞替代 Box 碰撞墊（當前方案）

**邏輯**：既然高摩擦參數有效，將這些參數直接應用到夾爪原生 mesh 碰撞上，而非額外的 box 碰撞墊。這樣更接近真實夾爪形狀。

**實作**：
```xml
<!-- 固定爪 mesh 碰撞 -->
<geom type="mesh" class="so101_collision" 
      mesh="so101_wrist_roll_follower_so101_v1"
      friction="5.0" solimp="0.95 0.99 0.001" solref="0.004 1"
      priority="1" condim="6"/>

<!-- 活動爪 mesh 碰撞 -->
<geom type="mesh" class="so101_collision" 
      mesh="so101_moving_jaw_so101_v1"
      friction="5.0" solimp="0.95 0.99 0.001" solref="0.004 1"
      priority="1" condim="6"/>
```

**結果**：✅ **穩定運行，夾取功能正常**

---

## 最終配置

| 檔案 | 變更 |
|------|------|
| `so101_pick_box_scene.xml` | 啟用 `multiccd` + `nativeccd` |
| `so101_new_calib.xml` | 夾爪 mesh 碰撞設定 `friction="5.0"`, `condim="6"`, `priority="1"` |

---

## 關鍵學習

1. **SO101 旋轉夾爪 vs 2F85 平行夾爪**：結構差異導致簡單的碰撞墊設計不能直接複製。

2. **Flexcomp 需要精確設計**：`type="grid"` 不適合直接使用，需要用 `type="mesh"` 配合專用 mesh 檔案。

3. **multiccd 是關鍵**：在平面接觸時產生多個接觸點，大幅改善穩定性。

4. **高摩擦係數有效但需謹慎**：`friction=5.0` 遠超真實物理（橡膠約 0.7-1.0），但在模擬中可接受。

---

## 後續可改進方向

- [ ] 製作專用軟墊 mesh 檔案（flexcomp type="mesh"）
- [ ] 測試不同物體形狀（球體、不規則形狀）
- [ ] 調整夾爪閉合力（actuator forcerange）
