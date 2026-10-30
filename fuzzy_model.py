
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def build_control_system():
    # Input 1: Soil moisture, dataset-relative scale
    soil_moisture = ctrl.Antecedent(
        np.arange(0, 81, 1), "soil_moisture"
    )

    soil_moisture["very_dry"] = fuzz.trapmf(
        soil_moisture.universe, [0, 0, 20, 25]
    )
    soil_moisture["dry"] = fuzz.trapmf(
        soil_moisture.universe, [15, 20, 30, 40]
    )
    soil_moisture["moderate"] = fuzz.trimf(
        soil_moisture.universe, [30, 40, 60]
    )
    soil_moisture["wet"] = fuzz.trapmf(
        soil_moisture.universe, [45, 55, 65, 75]
    )
    soil_moisture["very_wet"] = fuzz.trapmf(
        soil_moisture.universe, [60, 65, 80, 80]
    )

    # Input 2: Air temperature in degrees Celsius
    temperature = ctrl.Antecedent(
        np.arange(0, 51, 1), "temperature"
    )
    temperature["low"] = fuzz.trapmf(
        temperature.universe, [0, 0, 15, 22]
    )
    temperature["medium"] = fuzz.trimf(
        temperature.universe, [15, 25, 35]
    )
    temperature["high"] = fuzz.trapmf(
        temperature.universe, [28, 35, 50, 50]
    )

    # Input 3: Relative humidity, percent
    humidity = ctrl.Antecedent(
        np.arange(0, 101, 1), "humidity"
    )
    humidity["low"] = fuzz.trapmf(
        humidity.universe, [0, 0, 25, 45]
    )
    humidity["medium"] = fuzz.trimf(
        humidity.universe, [30, 55, 80]
    )
    humidity["high"] = fuzz.trapmf(
        humidity.universe, [65, 80, 100, 100]
    )

    # Output: Fuzzy irrigation-demand score from 0 to 100
    demand = ctrl.Consequent(
        np.arange(0, 101, 1), "demand"
    )
    demand["low"] = fuzz.trimf(
        demand.universe, [0, 15, 30]
    )
    demand["medium"] = fuzz.trimf(
        demand.universe, [25, 50, 75]
    )
    demand["high"] = fuzz.trimf(
        demand.universe, [70, 85, 100]
    )

    # Priority rules and broad fallback rules
    rules = [
        ctrl.Rule(soil_moisture["very_dry"], demand["high"]),
        ctrl.Rule(
            soil_moisture["dry"]
            & (temperature["high"] | humidity["low"]),
            demand["high"]
        ),
        ctrl.Rule(
            soil_moisture["dry"]
            & temperature["medium"]
            & humidity["medium"],
            demand["medium"]
        ),
        ctrl.Rule(
            soil_moisture["dry"]
            & temperature["low"]
            & humidity["high"],
            demand["medium"]
        ),
        ctrl.Rule(
            soil_moisture["moderate"]
            & temperature["high"]
            & humidity["low"],
            demand["medium"]
        ),
        ctrl.Rule(
            soil_moisture["moderate"]
            & temperature["high"]
            & humidity["medium"],
            demand["medium"]
        ),
        ctrl.Rule(
            soil_moisture["moderate"] & temperature["medium"],
            demand["medium"]
        ),
        ctrl.Rule(
            soil_moisture["moderate"] & temperature["low"],
            demand["low"]
        ),
        ctrl.Rule(soil_moisture["wet"], demand["low"]),
        ctrl.Rule(soil_moisture["very_wet"], demand["low"]),
        ctrl.Rule(soil_moisture["dry"], demand["medium"]),
        ctrl.Rule(soil_moisture["moderate"], demand["medium"]),
    ]

    return ctrl.ControlSystem(rules)


def predict_demand(soil_moisture, temperature, humidity):
    # Validate inputs before running inference
    if not 0 <= soil_moisture <= 80:
        raise ValueError("Soil moisture must be between 0 and 80.")

    if not 0 <= temperature <= 50:
        raise ValueError("Temperature must be between 0 and 50 °C.")

    if not 0 <= humidity <= 100:
        raise ValueError("Humidity must be between 0 and 100%.")

    control_system = build_control_system()
    simulation = ctrl.ControlSystemSimulation(control_system)

    simulation.input["soil_moisture"] = float(soil_moisture)
    simulation.input["temperature"] = float(temperature)
    simulation.input["humidity"] = float(humidity)
    simulation.compute()

    return float(simulation.output["demand"])
