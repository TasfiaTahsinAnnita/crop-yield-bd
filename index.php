<?php
declare(strict_types=1);
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    set_time_limit(360);
}
require_once __DIR__ . '/includes/api_client.php';

$metaResult = api_request('/metadata');
$meta = $metaResult['data'] ?? null;
$metricsResult = api_request('/metrics');
$metrics = $metricsResult['data'] ?? null;
$prediction = null;
$formError = null;
$values = $_POST;

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $required = ['region','year','crop','season','rainfall_mm_season','avg_temperature_c','relative_humidity_pct','soil_ph','soil_moisture_m3_m3','solar_radiation_mj_m2_day','ndvi','fertilizer_kg_ha','irrigation_used_0_no_1_yes','previous_season_yield_tonnes_ha'];
    $payload = [];
    foreach ($required as $field) {
        if (!isset($_POST[$field]) || trim((string)$_POST[$field]) === '') { $formError = 'Please complete every field.'; break; }
        $payload[$field] = $_POST[$field];
    }
    if (!$formError) {
        foreach (['year','irrigation_used_0_no_1_yes'] as $field) {
            if (filter_var($payload[$field], FILTER_VALIDATE_INT) === false) { $formError = 'Year and irrigation must be valid whole numbers.'; break; }
            $payload[$field] = (int)$payload[$field];
        }
        foreach (['rainfall_mm_season','avg_temperature_c','relative_humidity_pct','soil_ph','soil_moisture_m3_m3','solar_radiation_mj_m2_day','ndvi','fertilizer_kg_ha','previous_season_yield_tonnes_ha'] as $field) {
            if (!is_numeric($payload[$field]) || !is_finite((float)$payload[$field])) { $formError = 'Please enter valid finite numbers for all measurements.'; break; }
            $payload[$field] = (float)$payload[$field];
        }
    }
    if (!$formError) {
        $api = api_request('/predict', $payload);
        if ($api['ok']) $prediction = $api['data'];
        else $formError = $api['error'];
    }
}

$categories = $meta['categories'] ?? [];
$fields = [
    ['region','Region','select','Crop location'],
    ['crop','Crop','select','Crop type'],
    ['season','Season','select','Growing season'],
    ['year','Year','select','Year represented in the dataset'],
    ['rainfall_mm_season','Seasonal rainfall (mm)','number','e.g. 850'],
    ['avg_temperature_c','Average temperature (°C)','number','e.g. 26.5'],
    ['relative_humidity_pct','Relative humidity (%)','number','e.g. 75'],
    ['solar_radiation_mj_m2_day','Solar radiation (MJ/m²/day)','number','e.g. 18.2'],
    ['ndvi','NDVI','number','e.g. 0.68'],
    ['soil_ph','Soil pH','number','e.g. 6.4'],
    ['soil_moisture_m3_m3','Soil moisture (m³/m³)','number','e.g. 0.32'],
    ['fertilizer_kg_ha','Fertilizer (kg/ha)','number','e.g. 100'],
    ['irrigation_used_0_no_1_yes','Irrigation used?','select','0 = No · 1 = Yes'],
    ['previous_season_yield_tonnes_ha','Previous-season yield (tonnes/ha)','number','e.g. 4.2'],
];
function esc($value): string { return htmlspecialchars((string)$value, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); }
?>
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Bangladesh Crop Yield Prediction</title>
  <link rel="stylesheet" href="assets/style.css?v=2">
</head>
<body>
<header class="topbar"><a class="brand" href="#top"><span class="brand-mark">✳</span> Fieldnotes <span class="brand-tag">RESEARCH</span></a><span class="top-note">TabFM <i>×</i> TabPFN</span></header>
<main id="top" class="page">
  <section class="hero card">
    <div class="hero-copy"><p class="eyebrow">BANGLADESH · AGRICULTURE · MACHINE LEARNING</p><h1>Bangladesh Crop Yield Prediction</h1><p class="lede">Explore two tabular foundation models with a shared set of crop, soil and seasonal inputs.</p></div>
    <div class="hero-art" aria-hidden="true"><span class="sun"></span><span class="field field-one"></span><span class="field field-two"></span><span class="field field-three"></span><span class="sprout">✳</span></div>
  </section>
  <aside class="notice"><span class="notice-icon">i</span><div><strong>Synthetic prototype data</strong><p>This dataset is labeled SYNTHETIC_FOR_PROTOTYPE. Predictions and scores are software demonstrations, not validated estimates for real Bangladesh farms.</p></div></aside>

  <?php if (!$meta): ?><div class="alert error">Dataset metadata is unavailable. Start the Python API and confirm the CSV is in the project folder.</div><?php endif; ?>
  <?php if ($formError): ?><div class="alert error" role="alert"><?=esc($formError)?></div><?php endif; ?>

  <form method="post" class="form" id="prediction-form">
    <?php $groups = ['Crop & location'=>['region','crop','season','year'],'Weather & environment'=>['rainfall_mm_season','avg_temperature_c','relative_humidity_pct','solar_radiation_mj_m2_day','ndvi'],'Soil & farm practices'=>['soil_ph','soil_moisture_m3_m3','fertilizer_kg_ha','irrigation_used_0_no_1_yes','previous_season_yield_tonnes_ha']];
    foreach ($groups as $group => $keys): ?>
      <section class="card form-section"><div class="section-heading"><span class="section-dot"></span><div><p class="eyebrow">INPUT DETAILS</p><h2><?=esc($group)?></h2></div></div><div class="fields">
      <?php foreach ($fields as [$key,$label,$type,$placeholder]): if (!in_array($key,$keys,true)) continue; ?>
        <label class="field" for="<?=esc($key)?>"><span><?=esc($label)?></span>
          <?php if ($type === 'select'): ?><select id="<?=esc($key)?>" name="<?=esc($key)?>" required><option value="">Choose…</option><?php
            $options = $key === 'irrigation_used_0_no_1_yes' ? ['0'=>'No','1'=>'Yes'] : [];
            if ($key !== 'irrigation_used_0_no_1_yes') foreach (($categories[$key] ?? []) as $category) $options[(string)$category] = (string)$category;
            foreach ($options as $optionValue=>$optionLabel): $optionValue = (string)$optionValue; ?>
              <option value="<?=esc($optionValue)?>" <?=((string)($values[$key] ?? '') === (string)$optionValue ? 'selected' : '')?>><?=esc($optionLabel)?></option>
            <?php endforeach; ?></select>
          <?php else: ?><input id="<?=esc($key)?>" name="<?=esc($key)?>" type="number" step="any" required placeholder="<?=esc($placeholder)?>" value="<?=esc($values[$key] ?? '')?>" <?php if ($key === 'relative_humidity_pct'): ?>min="0" max="100"<?php elseif (in_array($key,['rainfall_mm_season','soil_moisture_m3_m3','fertilizer_kg_ha','previous_season_yield_tonnes_ha'],true)): ?>min="0"<?php elseif ($key === 'ndvi'): ?>min="-1" max="1"<?php elseif ($key === 'soil_ph'): ?>min="0" max="14"<?php endif; ?>><?php endif; ?>
          <?php if ($key === 'previous_season_yield_tonnes_ha'): ?><small>Use only if that prior-season value would be known when making this prediction.</small><?php endif; ?>
        </label>
      <?php endforeach; ?></div></section>
    <?php endforeach; ?>
    <div class="form-actions"><p>Evaluation uses a fixed 80/20 held-out split (random seed 42).</p><button class="button" type="submit" id="submit-button"><span>Run prediction</span><span aria-hidden="true">↗</span></button></div>
  </form>

  <?php if ($prediction): ?><section class="card results" id="results"><p class="eyebrow">YOUR RESULTS</p><h2>Predicted crop yield</h2><p class="result-caption">Actual inference · tonnes per hectare</p><div class="prediction-grid">
    <?php foreach (($prediction['results'] ?? []) as $name=>$result): ?><article class="prediction-card"><div class="model-name"><?=esc($name)?><span class="status-dot"></span></div><?php if (($result['status'] ?? '') === 'ok'): ?><strong><?=number_format((float)$result['prediction_tonnes_ha'],2)?></strong><small>tonnes/ha</small><?php else: ?><p class="model-error"><?=esc($result['message'] ?? 'This model did not return a prediction.')?></p><?php endif; ?></article><?php endforeach; ?>
  </div><p class="result-footnote">This estimate reflects a synthetic-data demonstration and should not guide farm decisions.</p></section><?php endif; ?>

  <section class="card metrics" id="metrics"><div class="section-heading"><span class="section-dot"></span><div><p class="eyebrow">HELD-OUT EVALUATION</p><h2>Model performance</h2></div></div><p class="metrics-intro">Metrics are calculated at backend startup from actual predictions on the same 20% test split. RMSE and MAE are in tonnes/ha; R² can be negative.</p>
    <?php if (!$metrics): ?><div class="alert">Metrics will appear when the backend has initialized. A model that fails to load will be shown as unavailable.</div><?php else: ?><div class="table-wrap"><table><thead><tr><th>Model</th><th>RMSE ↓</th><th>MAE ↓</th><th>R²</th><th>Status</th></tr></thead><tbody>
      <?php foreach (($metrics['models'] ?? []) as $name=>$entry): ?><tr><th scope="row"><?=esc($name)?></th><?php if (!empty($entry['metrics'])): ?><td><?=number_format($entry['metrics']['RMSE'],3)?></td><td><?=number_format($entry['metrics']['MAE'],3)?></td><td><?=number_format($entry['metrics']['R2'],4)?></td><td><span class="ready">Ready</span></td><?php else: ?><td colspan="3" class="muted">— no held-out metrics</td><td><span class="unready">Unavailable</span></td><?php endif; ?></tr><?php endforeach; ?></tbody></table></div><?php endif; ?>
    <div class="metric-notes"><span><b>RMSE</b> penalizes larger errors more heavily.</span><span><b>MAE</b> is the average absolute error.</span><span><b>R²</b> compares the model with a test-set baseline.</span></div>
    <p class="result-footnote">All current scores use synthetic prototype data; they are not evidence of real-world generalization.</p>
  </section>
  <footer>Research prototype <span>·</span> Bangladesh crop-yield exploration</footer>
</main>
<script src="assets/script.js" defer></script>
</body></html>
