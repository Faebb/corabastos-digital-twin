#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pruebas de sanidad del modelo. Uso:  python3 -I tests.py [output_folder]
Sin pytest: cada prueba es una funcion test_*; al final se ejecutan todas e informa.
Si se da la carpeta con las salidas (csv/json), se hacen ademas chequeos de consistencia de archivos."""
import sys, os, copy, json, tempfile, importlib.util, traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else None
_tmp = tempfile.mkdtemp()
sys.argv = [sys.argv[0], _tmp]                  # el modulo lee argv[1] al importarse (efecto lateral)
_spec = importlib.util.spec_from_file_location("sim", os.path.join(HERE, "digital_twin_simulation.py"))
simulation = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(simulation)
from scipy import stats

BASELINE_PARAMETERS = simulation.BASE_PARAMETERS
FIXED_DECAY_RATES = np.array([0.0025, 0.0028])                # lambdas fijos (no hace falta calibrar para estas pruebas)
NUM_LOTS_FOR_TESTS = 60000
RANDOM_DRAWS = simulation.generate_random_draws(np.random.default_rng(2024), NUM_LOTS_FOR_TESTS)
SCENARIOS = simulation.SCENARIOS


def total_loss(loss_array):
    return float(np.sum(loss_array))


def run_scenario(scenario_name, parameters=BASELINE_PARAMETERS, decay_rates=FIXED_DECAY_RATES, random_draws=RANDOM_DRAWS):
    return simulation.simulate_scenario(parameters, decay_rates, random_draws, SCENARIOS[scenario_name])


# ---------------------------------------------------------------- casos limite
def test_p_detect_zero_S3_equals_S0():
    parameters = copy.deepcopy(BASELINE_PARAMETERS); parameters["detection_probability"] = 0.0
    baseline_loss, baseline_true_alerts, baseline_false_alerts = simulation.simulate_scenario(parameters, FIXED_DECAY_RATES, RANDOM_DRAWS, {})
    s3_loss, s3_true_alerts, s3_false_alerts = simulation.simulate_scenario(parameters, FIXED_DECAY_RATES, RANDOM_DRAWS, SCENARIOS["S3 Sensores y alertas"])
    assert np.array_equal(baseline_loss, s3_loss), "S3 con p_detect=0 debe ser identico a S0"
    assert s3_true_alerts == 0 and s3_false_alerts > 0     # no hay alertas verdaderas; las falsas existen pero no actuan
    # con respond_to_false=True ya no debe coincidir (control de que la prueba discrimina)
    parameters["respond_to_false"] = True
    s3_loss_responding, _, _ = simulation.simulate_scenario(parameters, FIXED_DECAY_RATES, RANDOM_DRAWS, SCENARIOS["S3 Sensores y alertas"])
    assert total_loss(s3_loss_responding) < total_loss(baseline_loss)


def test_response_factor_one_S3_equals_S0():
    parameters = copy.deepcopy(BASELINE_PARAMETERS); parameters["response_factor"] = 1.0
    assert np.array_equal(simulation.simulate_scenario(parameters, FIXED_DECAY_RATES, RANDOM_DRAWS, {})[0],
                           simulation.simulate_scenario(parameters, FIXED_DECAY_RATES, RANDOM_DRAWS, SCENARIOS["S3 Sensores y alertas"])[0])


def test_central_factor_one_S2_equals_S0():
    parameters = copy.deepcopy(BASELINE_PARAMETERS); parameters["central_time_factor"] = 1.0
    assert np.array_equal(simulation.simulate_scenario(parameters, FIXED_DECAY_RATES, RANDOM_DRAWS, {})[0],
                           simulation.simulate_scenario(parameters, FIXED_DECAY_RATES, RANDOM_DRAWS, SCENARIOS["S2 Menos tiempo en central (-30 %)"])[0])


def test_reefer_equals_transport_T_S1_equals_S0():
    parameters = copy.deepcopy(BASELINE_PARAMETERS)
    parameters["reefer_temperature_celsius"] = parameters["mean_temperature_celsius"][2]
    parameters["reefer_temperature_sd_celsius"] = parameters["temperature_sd_celsius"][2]
    assert np.allclose(simulation.simulate_scenario(parameters, FIXED_DECAY_RATES, RANDOM_DRAWS, {})[0],
                        simulation.simulate_scenario(parameters, FIXED_DECAY_RATES, RANDOM_DRAWS, SCENARIOS["S1 Cadena de frio en transporte"])[0], rtol=0, atol=1e-15)


def test_lambda_zero_loss_zero():
    for scenario_name in SCENARIOS:
        loss, _, _ = simulation.simulate_scenario(BASELINE_PARAMETERS, np.array([0.0, 0.0]), RANDOM_DRAWS, SCENARIOS[scenario_name])
        assert np.all(loss == 0.0), scenario_name


def test_time_zero_loss_zero():
    parameters = copy.deepcopy(BASELINE_PARAMETERS); parameters["mean_residence_time_hours"] = np.zeros(5)
    loss, _, _ = simulation.simulate_scenario(parameters, FIXED_DECAY_RATES, RANDOM_DRAWS, {})
    assert np.all(loss == 0.0)


def test_lambda_huge_loses_everything():
    loss, _, _ = simulation.simulate_scenario(BASELINE_PARAMETERS, np.array([1e6, 1e6]), RANDOM_DRAWS, {})
    assert abs(loss.sum() - 1.0) < 1e-12
    assert loss[0] > 0.999                       # casi todo se pierde ya en la primera etapa


# ---------------------------------------------------------------- conservacion de masa
def slow_reference_implementation(parameters, decay_rates, random_draws, scenario_options, num_lots=1500):
    """Reimplementacion independiente, lote por lote y con bucles, para los primeros num_lots lotes."""
    stage_loss_total = np.zeros(5); true_alert_count = false_alert_count = 0
    for lot_index in range(num_lots):
        residence_time = [parameters["mean_residence_time_hours"][stage_index] * np.exp(
            parameters["residence_time_log_sigma"] * random_draws["time_z_scores"][lot_index, stage_index] - parameters["residence_time_log_sigma"] ** 2 / 2)
            for stage_index in range(5)]
        temperature = [parameters["mean_temperature_celsius"][stage_index] + parameters["temperature_sd_celsius"][stage_index] * random_draws["temperature_z_scores"][lot_index, stage_index]
            for stage_index in range(5)]
        if scenario_options.get("reefer"):
            temperature[2] = parameters["reefer_temperature_celsius"] + parameters["reefer_temperature_sd_celsius"] * random_draws["temperature_z_scores"][lot_index, 2]
        if scenario_options.get("fast_central"):
            residence_time[3] *= parameters["central_time_factor"]
        if scenario_options.get("sensors"):
            exceeds_threshold = (temperature[1] > parameters["storage_alert_threshold_celsius"]) or (temperature[2] > parameters["transport_alert_threshold_celsius"])
            true_alert = exceeds_threshold and random_draws["detection_uniform"][lot_index] < parameters["detection_probability"]
            false_alert = (not exceeds_threshold) and random_draws["false_alarm_uniform"][lot_index] < parameters["false_alarm_rate"]
            true_alert_count += true_alert; false_alert_count += false_alert
            if true_alert or (false_alert and parameters["respond_to_false"]):
                residence_time[3] *= parameters["response_factor"]; residence_time[4] *= parameters["response_factor"]
        remaining_mass = 1.0
        for stage_index in range(5):
            rate = decay_rates[0 if stage_index < 2 else 1] * parameters["Q10"] ** ((temperature[stage_index] - 20) / 10) * residence_time[stage_index]
            stage_loss = remaining_mass * (1 - np.exp(-rate))
            stage_loss_total[stage_index] += stage_loss; remaining_mass -= stage_loss
        assert 0.0 <= remaining_mass <= 1.0
    return stage_loss_total / num_lots, true_alert_count, false_alert_count


def test_matches_independent_reference():
    num_lots = 1500
    truncated_draws = {key: value[:num_lots] for key, value in RANDOM_DRAWS.items()}
    for scenario_name, scenario_options in SCENARIOS.items():
        for respond_to_false in (False, True):
            parameters = copy.deepcopy(BASELINE_PARAMETERS); parameters["respond_to_false"] = respond_to_false
            vectorized_loss, vectorized_true_alerts, vectorized_false_alerts = simulation.simulate_scenario(parameters, FIXED_DECAY_RATES, truncated_draws, scenario_options)
            reference_loss, reference_true_alerts, reference_false_alerts = slow_reference_implementation(parameters, FIXED_DECAY_RATES, truncated_draws, scenario_options, num_lots)
            assert np.allclose(vectorized_loss, reference_loss, rtol=0, atol=1e-12), (scenario_name, respond_to_false)
            assert (vectorized_true_alerts, vectorized_false_alerts) == (reference_true_alerts, reference_false_alerts), (scenario_name, respond_to_false)


def test_mass_conservation():
    """perdida acumulada + masa final = 1 por lote; con los mismos lotes, masa final no negativa y perdidas en [0,1]."""
    for scenario_name, scenario_options in SCENARIOS.items():
        loss, _, _ = simulation.simulate_scenario(BASELINE_PARAMETERS, FIXED_DECAY_RATES, RANDOM_DRAWS, scenario_options)
        assert np.all(loss >= 0) and loss.sum() < 1.0
    # reconstruccion de la masa final por producto de supervivencias, lote a lote
    residence_time = BASELINE_PARAMETERS["mean_residence_time_hours"] * np.exp(
        BASELINE_PARAMETERS["residence_time_log_sigma"] * RANDOM_DRAWS["time_z_scores"] - BASELINE_PARAMETERS["residence_time_log_sigma"] ** 2 / 2)
    temperature = BASELINE_PARAMETERS["mean_temperature_celsius"] + BASELINE_PARAMETERS["temperature_sd_celsius"] * RANDOM_DRAWS["temperature_z_scores"]
    deterioration_multiplier = BASELINE_PARAMETERS["Q10"] ** ((temperature - 20) / 10)
    stage_decay_rates = np.where(simulation.STAGE_GROUP == 1, FIXED_DECAY_RATES[0], FIXED_DECAY_RATES[1])
    survival = np.exp(-stage_decay_rates * deterioration_multiplier * residence_time)   # supervivencia por etapa
    final_mass = survival.prod(axis=1).mean()
    loss, _, _ = simulation.simulate_scenario(BASELINE_PARAMETERS, FIXED_DECAY_RATES, RANDOM_DRAWS, {})
    assert abs(loss.sum() + final_mass - 1.0) < 1e-12


# ---------------------------------------------------------------- alertas analiticas
def _analytical_rates(parameters, reefer):
    storage_excursion_prob = 1 - stats.norm.cdf((parameters["storage_alert_threshold_celsius"] - parameters["mean_temperature_celsius"][1]) / parameters["temperature_sd_celsius"][1])
    if reefer:
        transport_excursion_prob = 1 - stats.norm.cdf((parameters["transport_alert_threshold_celsius"] - parameters["reefer_temperature_celsius"]) / parameters["reefer_temperature_sd_celsius"])
    else:
        transport_excursion_prob = 1 - stats.norm.cdf((parameters["transport_alert_threshold_celsius"] - parameters["mean_temperature_celsius"][2]) / parameters["temperature_sd_celsius"][2])
    exceeds_prob = 1 - (1 - storage_excursion_prob) * (1 - transport_excursion_prob)       # T de almacenamiento y de transporte son independientes
    return parameters["detection_probability"] * exceeds_prob, parameters["false_alarm_rate"] * (1 - exceeds_prob)


def test_analytical_alerts():
    for reefer, scenario_name in ((False, "S3 Sensores y alertas"), (True, "S4 Combinado (S1+S2+S3)")):
        expected_true_rate, expected_false_rate = _analytical_rates(BASELINE_PARAMETERS, reefer)
        _, true_alert_count, false_alert_count = run_scenario(scenario_name)
        for observed, expected in ((true_alert_count / NUM_LOTS_FOR_TESTS, expected_true_rate), (false_alert_count / NUM_LOTS_FOR_TESTS, expected_false_rate)):
            standard_error = np.sqrt(expected * (1 - expected) / NUM_LOTS_FOR_TESTS)
            assert abs(observed - expected) < 5 * standard_error, (scenario_name, observed, expected)
    # valores citados en el texto (simulados): 17,2 / 7,8 (S3) y 5,4 / 9,3 (S4) por 100 lotes; los analiticos son
    # 17,19 / 7,85 y 5,34 / 9,33, es decir coinciden dentro de 0,1 (el error MC con 1,2 M lotes es ~0,02)
    for reefer, (expected_true_pct, expected_false_pct) in ((False, (17.2, 7.8)), (True, (5.4, 9.3))):
        true_rate, false_rate = _analytical_rates(BASELINE_PARAMETERS, reefer)
        assert abs(true_rate * 100 - expected_true_pct) < 0.1 and abs(false_rate * 100 - expected_false_pct) < 0.1, (reefer, true_rate * 100, false_rate * 100)


def test_no_sensors_no_alerts():
    for scenario_name, scenario_options in SCENARIOS.items():
        _, true_alert_count, false_alert_count = simulation.simulate_scenario(BASELINE_PARAMETERS, FIXED_DECAY_RATES, RANDOM_DRAWS, scenario_options)
        if not scenario_options.get("sensors"):
            assert true_alert_count == 0 and false_alert_count == 0


# ---------------------------------------------------------------- distribuciones
def test_lognormal_times_have_correct_mean():
    residence_time = BASELINE_PARAMETERS["mean_residence_time_hours"] * np.exp(
        BASELINE_PARAMETERS["residence_time_log_sigma"] * RANDOM_DRAWS["time_z_scores"] - BASELINE_PARAMETERS["residence_time_log_sigma"] ** 2 / 2)
    assert np.allclose(residence_time.mean(axis=0) / BASELINE_PARAMETERS["mean_residence_time_hours"], 1.0, atol=0.012)


# ---------------------------------------------------------------- escenarios
def test_scenarios_act_where_they_should():
    baseline_loss = run_scenario("S0 Linea base")[0]
    reefer_loss = run_scenario("S1 Cadena de frio en transporte")[0]
    fast_central_loss = run_scenario("S2 Menos tiempo en central (-30 %)")[0]
    sensors_loss = run_scenario("S3 Sensores y alertas")[0]
    combined_loss = run_scenario("S4 Combinado (S1+S2+S3)")[0]
    assert np.array_equal(baseline_loss[:2], reefer_loss[:2]) and reefer_loss[2] < baseline_loss[2]              # S1: baja transporte, no toca etapas previas
    assert np.array_equal(baseline_loss[:3], fast_central_loss[:3]) and fast_central_loss[3] < baseline_loss[3]  # S2: no toca etapas previas a la central
    assert np.array_equal(baseline_loss[:3], sensors_loss[:3])                                                   # S3: solo actua aguas abajo
    assert total_loss(combined_loss) < min(total_loss(reefer_loss), total_loss(fast_central_loss), total_loss(sensors_loss)) < total_loss(baseline_loss)


def test_same_seed_same_output():
    decay_rates = simulation.calibrate_decay_rates(BASELINE_PARAMETERS, num_lots=20000)
    assert np.array_equal(decay_rates, simulation.calibrate_decay_rates(BASELINE_PARAMETERS, num_lots=20000))
    first_run = simulation.run_replicates(BASELINE_PARAMETERS, decay_rates, SCENARIOS["S4 Combinado (S1+S2+S3)"], num_replicates=2, num_lots=5000)
    second_run = simulation.run_replicates(BASELINE_PARAMETERS, decay_rates, SCENARIOS["S4 Combinado (S1+S2+S3)"], num_replicates=2, num_lots=5000)
    assert np.array_equal(first_run, second_run)


# ---------------------------------------------------------------- calibracion
def test_calibration_meets_targets():
    for target_total_loss in (0.2, 0.3, 0.4):
        parameters = copy.deepcopy(BASELINE_PARAMETERS); parameters["TOTAL_LOSS_TARGET"] = target_total_loss
        decay_rates = simulation.calibrate_decay_rates(parameters, num_lots=60000, seed=12345)
        calibration_draws = simulation.generate_random_draws(np.random.default_rng(12345), 60000)       # mismos lotes de la calibracion: ajuste exacto
        loss, _, _ = simulation.simulate_scenario(parameters, decay_rates, calibration_draws, {})
        assert abs(loss.sum() - target_total_loss) < 1e-6
        assert abs(loss[:2].sum() / loss.sum() - parameters["GROUP1_SHARE_TARGET"]) < 1e-6
        fresh_draws = simulation.generate_random_draws(np.random.default_rng(7), 60000)           # lotes nuevos: error de muestreo pequeno
        fresh_loss, _, _ = simulation.simulate_scenario(parameters, decay_rates, fresh_draws, {})
        assert abs(fresh_loss.sum() - target_total_loss) < 0.003 and abs(fresh_loss[:2].sum() / fresh_loss.sum() - parameters["GROUP1_SHARE_TARGET"]) < 0.005


# ---------------------------------------------------------------- apply_parameter_variation
SPECIAL_VARIATION_KEYS = {"transport_temperature_shift", "transport_time_multiplier", "central_time_multiplier",
                           "retail_time_multiplier", "transport_temperature_sd", "postharvest_temperature"}


def test_apply_var_each_variation():
    parameters = copy.deepcopy(BASELINE_PARAMETERS)
    original_parameters = copy.deepcopy(BASELINE_PARAMETERS)
    expected_changes = {                                                    # (campo, indice, nuevo valor)
        "transport_temperature_shift": ("mean_temperature_celsius", 2, BASELINE_PARAMETERS["mean_temperature_celsius"][2] + 2.0),
        "transport_time_multiplier": ("mean_residence_time_hours", 2, BASELINE_PARAMETERS["mean_residence_time_hours"][2] * 1.3),
        "central_time_multiplier": ("mean_residence_time_hours", 3, BASELINE_PARAMETERS["mean_residence_time_hours"][3] * 1.3),
        "retail_time_multiplier": ("mean_residence_time_hours", 4, BASELINE_PARAMETERS["mean_residence_time_hours"][4] * 1.3),
        "transport_temperature_sd": ("temperature_sd_celsius", 2, 6.0),
        "postharvest_temperature": ("mean_temperature_celsius", 0, 30.0)}
    variation_values = {"transport_temperature_shift": 2.0, "transport_time_multiplier": 1.3, "central_time_multiplier": 1.3,
                         "retail_time_multiplier": 1.3, "transport_temperature_sd": 6.0, "postharvest_temperature": 30.0}
    for variation_key, (field, stage_index, expected_value) in expected_changes.items():
        varied_parameters = simulation.apply_parameter_variation(parameters, variation_key, variation_values[variation_key])
        assert np.isclose(varied_parameters[field][stage_index], expected_value), variation_key
        other_stage_indices = [index for index in range(5) if index != stage_index]
        assert np.array_equal(varied_parameters[field][other_stage_indices], BASELINE_PARAMETERS[field][other_stage_indices]), variation_key  # solo cambia la etapa indicada
        for field_name in ("mean_residence_time_hours", "mean_temperature_celsius", "temperature_sd_celsius"):    # el original no se muta
            assert np.array_equal(parameters[field_name], original_parameters[field_name]), (variation_key, field_name)
        for field_name in ("mean_residence_time_hours", "mean_temperature_celsius", "temperature_sd_celsius"):
            if field_name != field:
                assert np.array_equal(varied_parameters[field_name], BASELINE_PARAMETERS[field_name]), (variation_key, field_name)
    for direct_key in ("Q10", "TOTAL_LOSS_TARGET", "reefer_temperature_celsius", "detection_probability", "response_factor", "central_time_factor"):
        assert direct_key in BASELINE_PARAMETERS, direct_key                                                   # claves directas deben existir en BASE_PARAMETERS
        assert simulation.apply_parameter_variation(parameters, direct_key, 0.123)[direct_key] == 0.123 and parameters[direct_key] == original_parameters[direct_key]


def test_variations_use_valid_keys():
    source_code = open(os.path.join(HERE, "digital_twin_simulation.py"), encoding="utf-8").read()
    import re
    variation_keys = set(re.findall(r'\("[^"]+",\s*"(\w+)",', source_code)) | set(re.findall(r'"(\w+)":\s*\(-?[\d.]+,\s*-?[\d.]+\)', source_code))
    assert len(variation_keys) >= 12, variation_keys
    for key in variation_keys:
        assert key in SPECIAL_VARIATION_KEYS or key in BASELINE_PARAMETERS, f"clave desconocida (apply_parameter_variation la crearia en silencio): {key}"


def test_variations_only_affect_their_scenarios():
    """reefer_temperature_celsius solo cambia S1/S4; central_time_factor solo S2/S4; detection_probability y response_factor solo S3/S4."""
    cases = {"reefer_temperature_celsius": (12.0, {"S1", "S4"}), "central_time_factor": (0.9, {"S2", "S4"}),
             "detection_probability": (0.6, {"S3", "S4"}), "response_factor": (0.7, {"S3", "S4"})}
    for variation_key, (value, affected_scenarios) in cases.items():
        varied_parameters = simulation.apply_parameter_variation(BASELINE_PARAMETERS, variation_key, value)
        for scenario_name in SCENARIOS:
            baseline_result = total_loss(simulation.simulate_scenario(BASELINE_PARAMETERS, FIXED_DECAY_RATES, RANDOM_DRAWS, SCENARIOS[scenario_name])[0])
            varied_result = total_loss(simulation.simulate_scenario(varied_parameters, FIXED_DECAY_RATES, RANDOM_DRAWS, SCENARIOS[scenario_name])[0])
            if scenario_name[:2] in affected_scenarios:
                assert baseline_result != varied_result, (variation_key, scenario_name)
            else:
                assert baseline_result == varied_result, (variation_key, scenario_name)


def test_baseline_variations_affect_calibration():
    """Cambios de tiempo/temperatura de las etapas si cambian la linea base (y por tanto la recalibracion)."""
    for variation_key, value in (("transport_temperature_shift", 3.0), ("transport_time_multiplier", 1.3), ("central_time_multiplier", 1.3),
                                  ("retail_time_multiplier", 1.3), ("transport_temperature_sd", 6.0), ("postharvest_temperature", 30.0)):
        varied_parameters = simulation.apply_parameter_variation(BASELINE_PARAMETERS, variation_key, value)
        assert total_loss(simulation.simulate_scenario(varied_parameters, FIXED_DECAY_RATES, RANDOM_DRAWS, {})[0]) != total_loss(
            simulation.simulate_scenario(BASELINE_PARAMETERS, FIXED_DECAY_RATES, RANDOM_DRAWS, {})[0]), variation_key


# ---------------------------------------------------------------- consistencia de salidas (opcional)
def test_outputs_are_consistent():
    if not OUTPUT_DIR:
        return
    import pandas as pd
    scenario_results_df = pd.read_csv(os.path.join(OUTPUT_DIR, "scenario_results.csv"))
    calibration_json = json.load(open(os.path.join(OUTPUT_DIR, "calibration_and_assumptions.json"), encoding="utf-8"))
    national_inscope_Mt = calibration_json["nacional_en_alcance_Mt"]
    assert abs(national_inscope_Mt - 9.76 * 0.404 * 0.62) < 1e-9
    for _, row in scenario_results_df.iterrows():
        assert abs(row["toneladas_evitadas_nacional_Mt"] - row["reduccion_relativa_pct"] / 100 * national_inscope_Mt) < 0.0001
    # kg salvados por tonelada = (perdida base - perdida escenario) en puntos x 10
    baseline_pct = scenario_results_df.loc[0, "perdida_total_pct"]
    assert abs((baseline_pct - scenario_results_df.loc[1, "perdida_total_pct"]) * 10 - 25.1) < 0.2
    s4_sensitivity_df = pd.read_csv(os.path.join(OUTPUT_DIR, "sensitivity_S4.csv"))
    by_intervention_df = pd.read_csv(os.path.join(OUTPUT_DIR, "sensitivity_by_intervention.csv"))
    for _, row in s4_sensitivity_df.iterrows():                                      # bajo/alto de cada tabla coinciden entre archivos
        matching_rows = by_intervention_df[by_intervention_df["parametro"] == row["parametro"]]
        assert np.isclose(matching_rows.iloc[0]["valor"], row["valor_bajo"]) and np.isclose(matching_rows.iloc[1]["valor"], row["valor_alto"])
        assert np.isclose(matching_rows.iloc[0]["S4"], row["reduccion_con_bajo_pct"]) and np.isclose(matching_rows.iloc[1]["S4"], row["reduccion_con_alto_pct"])
        assert np.isclose(matching_rows.iloc[0]["S1"], row["S1_bajo"]) and np.isclose(matching_rows.iloc[1]["S3"], row["S3_alto"])
    joint_df = pd.read_csv(os.path.join(OUTPUT_DIR, "joint_sensitivity.csv"))
    for scenario_key in ("S1", "S2", "S3", "S4"):
        assert np.isclose(calibration_json["sensibilidad_conjunta"][scenario_key]["mediana"], joint_df[scenario_key].median())
        assert np.isclose(calibration_json["sensibilidad_conjunta"][scenario_key]["p5"], np.percentile(joint_df[scenario_key], 5))
        assert np.isclose(calibration_json["sensibilidad_conjunta"][scenario_key]["p95"], np.percentile(joint_df[scenario_key], 95))
    assert np.isclose(calibration_json["sensibilidad_conjunta"]["fraccion_S1_mayor"], (joint_df["S1"] > joint_df[["S2", "S3"]].max(axis=1)).mean())
    assert np.array_equal(joint_df["S1_es_la_mayor"].values, (joint_df["S1"] > joint_df[["S2", "S3"]].max(axis=1)).values)


if __name__ == "__main__":
    test_functions = [(name, func) for name, func in sorted(globals().items()) if name.startswith("test_") and callable(func)]
    failure_count = 0
    for name, func in test_functions:
        try:
            func(); print("OK    ", name)
        except Exception:
            failure_count += 1; print("FALLA ", name); traceback.print_exc()
    print(f"\n{len(test_functions) - failure_count}/{len(test_functions)} pruebas pasan")
    sys.exit(1 if failure_count else 0)
