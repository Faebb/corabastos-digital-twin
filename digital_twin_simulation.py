#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemelo digital simulado de la cadena de frutas y verduras hacia Corabastos (Bogota).

Modelo Monte Carlo por lotes: cada lote pasa por cinco etapas (poscosecha, almacenamiento,
transporte, central de abastos, comercio minorista). En cada etapa la fraccion perdida es
    f = 1 - exp(-lambda_g * Q10**((T-20)/10) * t)
con t = tiempo de residencia (h) y T = temperatura (C). lambda_g se calibra para cada grupo
de etapas del DNP: G1 = poscosecha y almacenamiento; G2 = distribucion y comercio.

Calibracion (dos objetivos, dos parametros):
  - perdida total en el alcance = TOTAL_LOSS_TARGET (supuesto, con sensibilidad)
  - proporcion G1 = 19,8/(19,8+20,6) del total en el alcance (DNP, 2016)

TODOS los parametros de tiempo, temperatura, Q10 y respuesta a alertas son SUPUESTOS del
modelo (no medidos). Se reportan con analisis de sensibilidad.

Uso:  python3 digital_twin_simulation.py [output_folder]
"""
import sys, os, json, copy
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUTPUT_DIR, exist_ok=True)

STAGE_NAMES = ["Poscosecha", "Almacenamiento", "Transporte", "Central de abastos", "Comercio minorista"]
STAGE_GROUP = np.array([1, 1, 2, 2, 2])

BASE_PARAMETERS = dict(
    mean_residence_time_hours=np.array([10.0, 36.0, 10.0, 20.0, 48.0]),   # horas de residencia (supuesto)
    residence_time_log_sigma=0.5,                                        # dispersion lognormal (supuesto)
    mean_temperature_celsius=np.array([28.0, 22.0, 26.0, 15.0, 18.0]),    # temperatura media C (supuesto)
    temperature_sd_celsius=np.array([3.0, 4.0, 4.0, 3.0, 3.0]),
    Q10=2.2,                                             # supuesto; rango tipico 2-3
    TOTAL_LOSS_TARGET=0.30,                              # perdida total en el alcance (supuesto)
    GROUP1_SHARE_TARGET=19.8 / (19.8 + 20.6),            # DNP (2024)
    # intervenciones
    reefer_temperature_celsius=8.0, reefer_temperature_sd_celsius=1.5,    # transporte refrigerado
    central_time_factor=0.7,                             # -30 % tiempo en central
    detection_probability=0.80,                          # probabilidad de detectar excursion
    response_factor=0.5,                                 # tiempo aguas abajo del lote alertado
    storage_alert_threshold_celsius=28.0, transport_alert_threshold_celsius=30.0,   # umbrales de alerta (C)
    false_alarm_rate=0.10,                                # tasa de falsas alarmas
    respond_to_false=False,                               # True: el operador tambien prioriza las falsas alarmas
    random_marking_rate=0.172,                            # contrafactual: marca al azar esta fraccion de lotes (≈ tasa de alertas verdaderas de S3)
    fruit_veg_share=0.62, national_total_loss_Mt=9.76, bogota_share=0.3809,  # extrapolacion (DNP, 2024; fruit_veg_share es supuesto)
)

SCENARIOS = {
    "S0 Linea base": dict(),
    "S1 Cadena de frio en transporte": dict(reefer=True),
    "S2 Menos tiempo en central (-30 %)": dict(fast_central=True),
    "S3 Sensores y alertas": dict(sensors=True),
    "S4 Combinado (S1+S2+S3)": dict(reefer=True, fast_central=True, sensors=True),
}

NUM_LOTS = 40000
NUM_REPLICATES = 30


def apply_parameter_variation(parameters, variation_key, value):
    """Devuelve una copia de 'parameters' con una variacion aplicada (para el analisis de sensibilidad)."""
    varied = copy.deepcopy(parameters)
    if variation_key == "transport_temperature_shift":
        varied["mean_temperature_celsius"] = varied["mean_temperature_celsius"].copy()
        varied["mean_temperature_celsius"][2] += value
    elif variation_key == "transport_time_multiplier":
        varied["mean_residence_time_hours"] = varied["mean_residence_time_hours"].copy()
        varied["mean_residence_time_hours"][2] *= value
    elif variation_key == "central_time_multiplier":
        varied["mean_residence_time_hours"] = varied["mean_residence_time_hours"].copy()
        varied["mean_residence_time_hours"][3] *= value
    elif variation_key == "retail_time_multiplier":
        varied["mean_residence_time_hours"] = varied["mean_residence_time_hours"].copy()
        varied["mean_residence_time_hours"][4] *= value
    elif variation_key == "transport_temperature_sd":
        varied["temperature_sd_celsius"] = varied["temperature_sd_celsius"].copy()
        varied["temperature_sd_celsius"][2] = value
    elif variation_key == "postharvest_temperature":
        varied["mean_temperature_celsius"] = varied["mean_temperature_celsius"].copy()
        varied["mean_temperature_celsius"][0] = value
    else:
        varied[variation_key] = value
    return varied


def generate_random_draws(rng, num_lots):
    return dict(time_z_scores=rng.standard_normal((num_lots, 5)), temperature_z_scores=rng.standard_normal((num_lots, 5)),
                detection_uniform=rng.random(num_lots), false_alarm_uniform=rng.random(num_lots))


def simulate_scenario(parameters, decay_rates, random_draws, scenario_options):
    """Devuelve perdida por etapa (fraccion de la masa inicial), n_alertas_verdaderas, n_falsas."""
    num_lots = random_draws["time_z_scores"].shape[0]
    residence_time = parameters["mean_residence_time_hours"] * np.exp(
        parameters["residence_time_log_sigma"] * random_draws["time_z_scores"] - parameters["residence_time_log_sigma"] ** 2 / 2)
    temperature = parameters["mean_temperature_celsius"] + parameters["temperature_sd_celsius"] * random_draws["temperature_z_scores"]
    if scenario_options.get("reefer"):
        temperature[:, 2] = parameters["reefer_temperature_celsius"] + parameters["reefer_temperature_sd_celsius"] * random_draws["temperature_z_scores"][:, 2]
    if scenario_options.get("fast_central"):
        residence_time[:, 3] = residence_time[:, 3] * parameters["central_time_factor"]
    true_alert = np.zeros(num_lots, dtype=bool)
    false_alert = np.zeros(num_lots, dtype=bool)
    if scenario_options.get("sensors"):
        excursion = (temperature[:, 1] > parameters["storage_alert_threshold_celsius"]) | (temperature[:, 2] > parameters["transport_alert_threshold_celsius"])
        true_alert = excursion & (random_draws["detection_uniform"] < parameters["detection_probability"])
        false_alert = (~excursion) & (random_draws["false_alarm_uniform"] < parameters["false_alarm_rate"])
        # el lote alertado se vende con prioridad: se acortan los tiempos aguas abajo
        prioritized = true_alert | false_alert if parameters["respond_to_false"] else true_alert
        if scenario_options.get("random_mark"):
            prioritized = random_draws["detection_uniform"] < parameters["random_marking_rate"]   # contrafactual: el sensor no aporta informacion
        residence_time[prioritized, 3] *= parameters["response_factor"]
        residence_time[prioritized, 4] *= parameters["response_factor"]
    deterioration_multiplier = parameters["Q10"] ** ((temperature - 20.0) / 10.0)
    remaining_mass = np.ones(num_lots)
    stage_loss = np.zeros(5)
    for stage_index in range(5):
        loss_fraction = 1.0 - np.exp(-decay_rates[STAGE_GROUP[stage_index] - 1] * deterioration_multiplier[:, stage_index] * residence_time[:, stage_index])
        stage_loss[stage_index] = np.mean(remaining_mass * loss_fraction)
        remaining_mass = remaining_mass * (1.0 - loss_fraction)
    return stage_loss, int(true_alert.sum()), int(false_alert.sum())


def calibrate_decay_rates(parameters, num_lots=200000, seed=12345):
    random_draws = generate_random_draws(np.random.default_rng(seed), num_lots)

    def calibration_equations(log_decay_rates):
        decay_rates = np.exp(log_decay_rates)
        stage_loss, _, _ = simulate_scenario(parameters, decay_rates, random_draws, {})
        total_loss = stage_loss.sum()
        group1_share = stage_loss[STAGE_GROUP == 1].sum() / total_loss
        return [total_loss - parameters["TOTAL_LOSS_TARGET"], group1_share - parameters["GROUP1_SHARE_TARGET"]]

    solution, info, convergence_flag, message = fsolve(calibration_equations, x0=np.log([0.005, 0.005]), full_output=True)
    assert convergence_flag == 1, message
    return np.exp(solution)


def run_replicates(parameters, decay_rates, scenario_options, num_replicates=NUM_REPLICATES, num_lots=NUM_LOTS, seed_start=1000):
    replicate_rows = []
    for replicate_index in range(num_replicates):
        random_draws = generate_random_draws(np.random.default_rng(seed_start + replicate_index), num_lots)
        stage_loss, true_alert_count, false_alert_count = simulate_scenario(parameters, decay_rates, random_draws, scenario_options)
        replicate_rows.append(np.concatenate([stage_loss, [true_alert_count / num_lots, false_alert_count / num_lots]]))
    return np.array(replicate_rows)


def main():
    parameters = copy.deepcopy(BASE_PARAMETERS)
    decay_rates = calibrate_decay_rates(parameters)
    print("lambda calibrado (G1, G2) [1/h a 20 C]:", decay_rates)

    # ---------- Calibracion: comparacion modelo vs DNP, con replicas ----------
    baseline_replicates = run_replicates(parameters, decay_rates, {})
    total_loss = baseline_replicates[:, :5].sum(axis=1)
    group1_share = baseline_replicates[:, :5][:, :2].sum(axis=1) / total_loss
    calibration_summary = dict(
        lambda_G1=float(decay_rates[0]), lambda_G2=float(decay_rates[1]),
        total_model_mean=float(total_loss.mean()), total_model_sd=float(total_loss.std(ddof=1)),
        share_G1_model_mean=float(group1_share.mean()), share_G1_model_sd=float(group1_share.std(ddof=1)),
        share_G1_dnp=float(parameters["GROUP1_SHARE_TARGET"]),
        share_G2_model_mean=float(1 - group1_share.mean()), share_G2_dnp=float(1 - parameters["GROUP1_SHARE_TARGET"]),
        error_pp_G1=float((group1_share.mean() - parameters["GROUP1_SHARE_TARGET"]) * 100),
        error_pp_G2=float(((1 - group1_share.mean()) - (1 - parameters["GROUP1_SHARE_TARGET"])) * 100),
        max_abs_error_pp_across_reps=float(np.max(np.abs(group1_share - parameters["GROUP1_SHARE_TARGET"])) * 100),
    )
    baseline_stage_loss = baseline_replicates[:, :5].mean(axis=0)
    # perdida en la central como % de la masa que llega a la central (comparacion con 2 % de 1998-2000)
    # masa que llega a la central = 1 - perdidas de las 3 primeras etapas
    mass_reaching_central = 1 - baseline_stage_loss[:3].sum()
    calibration_summary["central_loss_pct_of_inflow"] = float(baseline_stage_loss[3] / mass_reaching_central * 100)
    calibration_summary["stage_loss_base_pct_of_initial"] = {stage: float(value * 100) for stage, value in zip(STAGE_NAMES, baseline_stage_loss)}

    # ---------- Escenarios ----------
    scenario_rows = []
    scenario_replicates = {}
    for scenario_name, scenario_options in SCENARIOS.items():
        scenario_replicates[scenario_name] = run_replicates(parameters, decay_rates, scenario_options)
    baseline_total_loss = scenario_replicates["S0 Linea base"][:, :5].sum(axis=1)
    national_inscope_Mt = parameters["national_total_loss_Mt"] * (0.198 + 0.206) * parameters["fruit_veg_share"]   # supuesto: F&V = 62 % en cada etapa
    bogota_share = parameters["bogota_share"]
    t_critical = stats.t.ppf(0.975, NUM_REPLICATES - 1)          # IC con t de Student (29 gl) en vez de 1,96
    for scenario_name, scenario_rep in scenario_replicates.items():
        scenario_total_loss = scenario_rep[:, :5].sum(axis=1)
        relative_reduction = (baseline_total_loss - scenario_total_loss) / baseline_total_loss    # reduccion relativa por replica (mismos lotes)
        row = dict(
            escenario=scenario_name,
            perdida_total_pct=scenario_total_loss.mean() * 100,
            perdida_total_ic95_lo=(scenario_total_loss.mean() - t_critical * scenario_total_loss.std(ddof=1) / np.sqrt(len(scenario_total_loss))) * 100,
            perdida_total_ic95_hi=(scenario_total_loss.mean() + t_critical * scenario_total_loss.std(ddof=1) / np.sqrt(len(scenario_total_loss))) * 100,
            reduccion_relativa_pct=relative_reduction.mean() * 100,
            reduccion_ic95_lo=(relative_reduction.mean() - t_critical * relative_reduction.std(ddof=1) / np.sqrt(len(relative_reduction))) * 100,
            reduccion_ic95_hi=(relative_reduction.mean() + t_critical * relative_reduction.std(ddof=1) / np.sqrt(len(relative_reduction))) * 100,
            alertas_verdaderas_por_100_lotes=scenario_rep[:, 5].mean() * 100,
            falsas_alertas_por_100_lotes=scenario_rep[:, 6].mean() * 100,
        )
        for stage_index, stage_name in enumerate(STAGE_NAMES):
            row["perdida_" + stage_name.replace(" ", "_") + "_pct"] = scenario_rep[:, stage_index].mean() * 100
        row["toneladas_evitadas_nacional_Mt"] = relative_reduction.mean() * national_inscope_Mt
        row["toneladas_evitadas_Bogota_Mt_ilustrativo"] = relative_reduction.mean() * national_inscope_Mt * bogota_share
        scenario_rows.append(row)
    scenario_results = pd.DataFrame(scenario_rows)
    scenario_results.to_csv(os.path.join(OUTPUT_DIR, "scenario_results.csv"), index=False, encoding="utf-8-sig")

    # ---------- Sensibilidad (se recalibra el modelo en cada variacion para seguir igualando al DNP) ----------
    def reduction_pct_for_all_scenarios(varied_parameters, num_replicates=12, num_lots=30000, calibration_lots=100000):
        """Reduccion relativa (%) de S1..S4 con una sola calibracion y los mismos lotes."""
        varied_decay_rates = calibrate_decay_rates(varied_parameters, num_lots=calibration_lots)
        baseline_total = run_replicates(varied_parameters, varied_decay_rates, {}, num_replicates=num_replicates, num_lots=num_lots)[:, :5].sum(axis=1)
        reduction_by_scenario = {}
        for scenario_name, scenario_options in SCENARIOS.items():
            if scenario_name.startswith("S0"):
                continue
            scenario_total = run_replicates(varied_parameters, varied_decay_rates, scenario_options, num_replicates=num_replicates, num_lots=num_lots)[:, :5].sum(axis=1)
            reduction_by_scenario[scenario_name.split()[0]] = float(((baseline_total - scenario_total) / baseline_total).mean() * 100)
        return reduction_by_scenario

    sensitivity_variations = [
        ("Q10 (1,8 / 2,6)", "Q10", 1.8, 2.6),
        ("Pérdida total de calibración (20 % / 40 %)", "TOTAL_LOSS_TARGET", 0.20, 0.40),
        ("Temperatura refrigerada (4 / 12 °C)", "reefer_temperature_celsius", 4.0, 12.0),
        ("Probabilidad de detección (0,6 / 0,95)", "detection_probability", 0.60, 0.95),
        ("Respuesta a la alerta (×0,3 / ×0,7 del tiempo)", "response_factor", 0.30, 0.70),
        ("Recorte del tiempo en la central en S2 (−50 % / −10 %)", "central_time_factor", 0.5, 0.9),
        ("Temperatura media del transporte (−3 / +3 °C)", "transport_temperature_shift", -3.0, 3.0),
        ("Tiempo en el transporte (×0,7 / ×1,3)", "transport_time_multiplier", 0.7, 1.3),
        ("Tiempo en la central (×0,7 / ×1,3)", "central_time_multiplier", 0.7, 1.3),
        ("Tiempo en el comercio minorista (×0,7 / ×1,3)", "retail_time_multiplier", 0.7, 1.3),
        ("Variación de la temperatura en el transporte (σ 2 / 6 °C)", "transport_temperature_sd", 2.0, 6.0),
        ("Temperatura media en poscosecha (22 / 30 °C)", "postharvest_temperature", 22.0, 30.0),
    ]
    baseline_reduction_all = reduction_pct_for_all_scenarios(parameters)
    baseline_S4_reduction = baseline_reduction_all["S4"]
    sensitivity_summary, sensitivity_by_intervention = [], []
    for label, variation_key, low_value, high_value in sensitivity_variations:
        outcomes = [reduction_pct_for_all_scenarios(apply_parameter_variation(parameters, variation_key, candidate_value)) for candidate_value in (low_value, high_value)]
        for candidate_value, outcome in zip((low_value, high_value), outcomes):
            sensitivity_by_intervention.append(dict(parametro=label, valor=candidate_value, **outcome))
        sensitivity_summary.append(dict(parametro=label, valor_bajo=low_value, valor_alto=high_value,
                         reduccion_con_bajo_pct=outcomes[0]["S4"], reduccion_con_alto_pct=outcomes[1]["S4"],
                         S1_bajo=outcomes[0]["S1"], S1_alto=outcomes[1]["S1"], S2_bajo=outcomes[0]["S2"], S2_alto=outcomes[1]["S2"],
                         S3_bajo=outcomes[0]["S3"], S3_alto=outcomes[1]["S3"]))
    sensitivity_by_intervention_df = pd.DataFrame(sensitivity_by_intervention)
    sensitivity_by_intervention_df["S1_es_la_mayor"] = sensitivity_by_intervention_df["S1"] > sensitivity_by_intervention_df[["S2", "S3"]].max(axis=1)
    sensitivity_by_intervention_df.to_csv(os.path.join(OUTPUT_DIR, "sensitivity_by_intervention.csv"), index=False, encoding="utf-8-sig")
    sensitivity_summary_df = pd.DataFrame(sensitivity_summary)
    sensitivity_summary_df["base_pct"] = baseline_S4_reduction
    sensitivity_summary_df["rango_pp"] = (sensitivity_summary_df[["reduccion_con_bajo_pct", "reduccion_con_alto_pct"]].max(axis=1)
                        - sensitivity_summary_df[["reduccion_con_bajo_pct", "reduccion_con_alto_pct"]].min(axis=1))
    sensitivity_summary_df.sort_values("rango_pp", ascending=False).to_csv(
        os.path.join(OUTPUT_DIR, "sensitivity_S4.csv"), index=False, encoding="utf-8-sig")

    # ---------- Variacion conjunta: 400 combinaciones al azar de los supuestos inciertos ----------
    joint_rng = np.random.default_rng(777)
    joint_ranges = {"Q10": (1.8, 2.6), "TOTAL_LOSS_TARGET": (0.20, 0.40), "transport_temperature_shift": (-3.0, 3.0), "transport_time_multiplier": (0.7, 1.3),
              "central_time_multiplier": (0.7, 1.3), "retail_time_multiplier": (0.7, 1.3), "transport_temperature_sd": (2.0, 6.0), "postharvest_temperature": (22.0, 30.0),
              "detection_probability": (0.6, 0.95), "response_factor": (0.3, 0.7), "reefer_temperature_celsius": (4.0, 12.0),
              "central_time_factor": (0.5, 0.9)}
    joint_outcomes = []
    for _ in range(400):
        sampled_parameters = copy.deepcopy(parameters)
        for variation_key, (low_value, high_value) in joint_ranges.items():
            sampled_parameters = apply_parameter_variation(sampled_parameters, variation_key, float(joint_rng.uniform(low_value, high_value)))
        joint_outcomes.append(reduction_pct_for_all_scenarios(sampled_parameters, num_replicates=4, num_lots=20000, calibration_lots=60000))
    joint_sensitivity_df = pd.DataFrame(joint_outcomes)
    joint_sensitivity_df["S1_es_la_mayor"] = joint_sensitivity_df["S1"] > joint_sensitivity_df[["S2", "S3"]].max(axis=1)
    joint_sensitivity_df.to_csv(os.path.join(OUTPUT_DIR, "joint_sensitivity.csv"), index=False, encoding="utf-8-sig")
    joint_summary = {scenario: dict(p5=float(joint_sensitivity_df[scenario].quantile(0.05)), mediana=float(joint_sensitivity_df[scenario].median()), p95=float(joint_sensitivity_df[scenario].quantile(0.95)))
                     for scenario in ["S1", "S2", "S3", "S4"]}
    joint_summary["fraccion_S1_mayor"] = float(joint_sensitivity_df["S1_es_la_mayor"].mean())

    # ---------- Contrafactuales ----------
    respond_to_false_parameters = copy.deepcopy(parameters); respond_to_false_parameters["respond_to_false"] = True
    counterfactual_respond_to_false = reduction_pct_for_all_scenarios(respond_to_false_parameters)   # el operador tambien prioriza las falsas alarmas
    SCENARIOS["S3r Marcado al azar"] = dict(sensors=True, random_mark=True)
    SCENARIOS["S4r Combinado con marcado al azar"] = dict(reefer=True, fast_central=True, sensors=True, random_mark=True)
    counterfactual_random_mark = reduction_pct_for_all_scenarios(parameters)      # marcado al azar con la misma tasa de alertas
    counterfactual_random_mark = {scenario: value for scenario, value in counterfactual_random_mark.items() if scenario == "S3r"}
    del SCENARIOS["S3r Marcado al azar"], SCENARIOS["S4r Combinado con marcado al azar"]
    # casos extremos: transporte corto y/o ya frio (probar donde S1 deja de ser la mayor)
    extreme_cases = {}
    for label, modifications in {"transporte x0,4 (4 h)": [("transport_time_multiplier", 0.4)],
                      "transporte a 18 C": [("transport_temperature_shift", -8.0)],
                      "transporte x0,4 y a 18 C": [("transport_time_multiplier", 0.4), ("transport_temperature_shift", -8.0)],
                      "refrigerado a 12 C": [("reefer_temperature_celsius", 12.0)]}.items():
        modified_parameters = copy.deepcopy(parameters)
        for variation_key, value in modifications:
            modified_parameters = apply_parameter_variation(modified_parameters, variation_key, value)
        extreme_cases[label] = reduction_pct_for_all_scenarios(modified_parameters)
    # kilogramos salvados por tonelada que entra a la cadena (puntos de perdida x 10)
    baseline_loss_pct = scenario_results.loc[0, "perdida_total_pct"]
    scenario_results["kg_salvados_por_tonelada"] = (baseline_loss_pct - scenario_results["perdida_total_pct"]) * 10
    scenario_results.to_csv(os.path.join(OUTPUT_DIR, "scenario_results.csv"), index=False, encoding="utf-8-sig")

    json.dump(dict(calibracion=calibration_summary, supuestos={key: (value.tolist() if hasattr(value, "tolist") else value) for key, value in BASE_PARAMETERS.items()},
                   n_lotes=NUM_LOTS, n_replicas=NUM_REPLICATES, nacional_en_alcance_Mt=national_inscope_Mt,
                   reduccion_S4_base_sens=baseline_S4_reduction, sensibilidad_conjunta=joint_summary, contrafactual_priorizar_falsas=counterfactual_respond_to_false,
                   contrafactual_marcado_al_azar=counterfactual_random_mark, casos_extremos=extreme_cases),
              open(os.path.join(OUTPUT_DIR, "calibration_and_assumptions.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---------- Figuras ----------
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    COLOR_GROUP1, COLOR_GROUP2 = "#1f5fa8", "#d9822b"

    def format_decimal_comma(value, decimals=1):
        return f"{value:.{decimals}f}".replace(".", ",")

    # Fig 1: calibracion
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    row_labels = ["DNP (2016)", "Modelo (media de 30 réplicas)"]
    group1_pct = [parameters["GROUP1_SHARE_TARGET"] * 100, calibration_summary["share_G1_model_mean"] * 100]
    group2_pct = [100 - group1_pct[0], 100 - group1_pct[1]]
    ax.barh(row_labels, group1_pct, color=COLOR_GROUP1, label="Poscosecha y almacenamiento")
    ax.barh(row_labels, group2_pct, left=group1_pct, color=COLOR_GROUP2, label="Distribución y comercio")
    for row_index in range(2):
        ax.text(group1_pct[row_index] / 2, row_index, f"{format_decimal_comma(group1_pct[row_index])} %", ha="center", va="center", color="white", fontweight="bold")
        ax.text(group1_pct[row_index] + group2_pct[row_index] / 2, row_index, f"{format_decimal_comma(group2_pct[row_index])} %", ha="center", va="center", color="white", fontweight="bold")
    ax.set_xlim(0, 100); ax.set_xlabel("Participación en la pérdida dentro del alcance (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.30), ncol=2, frameon=False, fontsize=8.5)
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "fig1_calibration.png"), dpi=200); plt.close(fig)

    # Fig 2: perdida por etapa y escenario
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    scenario_names = list(scenario_results["escenario"])
    scenario_short_labels = ["S0\nBase", "S1\nFrío en\ntransporte", "S2\nMenos tiempo\nen central", "S3\nSensores\ny alertas", "S4\nCombinado"]
    stage_colors = ["#1f5fa8", "#4f86c6", "#d9822b", "#e8b06b", "#8a8f98"]
    stacked_bottom = np.zeros(len(scenario_names))
    for stage_index, stage_name in enumerate(STAGE_NAMES):
        stage_values = scenario_results["perdida_" + stage_name.replace(" ", "_") + "_pct"].values
        ax.bar(range(len(scenario_names)), stage_values, bottom=stacked_bottom, color=stage_colors[stage_index], label=stage_name, width=0.62)
        stacked_bottom += stage_values
    for scenario_index, total_value in enumerate(stacked_bottom):
        ax.text(scenario_index, total_value + 0.5, f"{format_decimal_comma(total_value)} %", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(scenario_names))); ax.set_xticklabels(scenario_short_labels, fontsize=8.5)
    ax.set_ylabel("Pérdida (% de la masa que entra)"); ax.set_ylim(0, stacked_bottom.max() * 1.12)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3)
    fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "fig2_scenarios.png"), dpi=200); plt.close(fig)

    # Fig 3: tornado (reduccion de S4); azul = valor bajo del parametro, naranja = valor alto
    sorted_sensitivity = sensitivity_summary_df.sort_values("rango_pp")
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for row_index, (_, row) in enumerate(sorted_sensitivity.iterrows()):
        low_reduction, high_reduction = row["reduccion_con_bajo_pct"], row["reduccion_con_alto_pct"]
        ax.barh(row_index, low_reduction - baseline_S4_reduction, left=baseline_S4_reduction, color=COLOR_GROUP1, height=0.55, label="Valor bajo" if row_index == 0 else None)
        ax.barh(row_index, high_reduction - baseline_S4_reduction, left=baseline_S4_reduction, color=COLOR_GROUP2, height=0.55, label="Valor alto" if row_index == 0 else None)
    ax.axvline(baseline_S4_reduction, color="black", lw=1)
    ax.set_yticks(range(len(sorted_sensitivity))); ax.set_yticklabels(sorted_sensitivity["parametro"], fontsize=7.5)
    ax.set_xlabel("Reducción relativa de la pérdida en S4 (%)")
    ax.text(baseline_S4_reduction, len(sorted_sensitivity) - 0.3, f" base {format_decimal_comma(baseline_S4_reduction)} %", fontsize=8, va="bottom")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: format_decimal_comma(value, 0)))
    fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "fig3_sensitivity.png"), dpi=200); plt.close(fig)

    print(json.dumps(calibration_summary, indent=2, ensure_ascii=False))
    print(scenario_results.round(2).T.to_string())
    print(sensitivity_summary_df.round(2).to_string())


if __name__ == "__main__":
    main()
