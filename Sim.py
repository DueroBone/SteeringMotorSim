### Vibe coded via GhatGPT codex ###


import math
import itertools
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

# ============================================================
# USER INPUTS
# ============================================================

TORQUE_CONSTANT_MNM_PER_A = 93.808  # motor torque constant [mN*m / A]
KV_RPM_PER_V = 102.0  # motor speed constant [RPM / V]
RESISTANCE_OHM = 0.05  # motor winding resistance [ohm]

MOTOR_INERTIA_KGM2 = 0.001  # motor rotor inertia [kg*m^2]
ANGLE_DEG = 140.0  # output angle to travel [degrees]

MAX_VOLTAGES = [24.0, 40.0, 60.0, 80.0, 95.0]
MAX_AMPS = [30.0, 40.0, 50.0, 60.0, 70.0]
GEAR_RATIOS = [2.0, 2.5, 3.0, 5.0, 7.5, 10.0, 15.0]
RESISTANCE_TORQUES_NM = [10.0, 20.0, 30.0]  # opposing torques to check [N*m]5

# Integration timestep.
# Reduce this for higher accuracy.
DT = 0.0001

# Fractional parameter change used to calculate local move-time derivatives.
# For example, 0.01 perturbs each parameter by 1% above and below its value.
DERIVATIVE_RELATIVE_STEP = 0.01


# ============================================================
# MOTOR / SERVO MODEL
# ============================================================


def simulate_move(
    torque_constant_mNm_A,
    kv_rpm_V,
    resistance_ohm,
    motor_inertia_kgm2,
    angle_deg,
    resistance_torque_Nm,
    max_voltage,
    max_current,
    gear_ratio,
    dt=0.0001,
    max_sim_time=2.0,
):
    """
    Simulate a servo moving through angle_deg using maximum acceleration
    followed by maximum braking.

    gear_ratio:
        motor speed / output speed

        Example:
            10.0 = 10:1 reduction
            motor turns 10 revolutions for one output revolution.

    motor_inertia_kgm2:
        Motor rotor inertia. No output/load inertia is included.

    resistance_torque_Nm:
        Constant torque opposing output shaft motion.
    """

    # Convert motor torque constant:
    # mN*m/A -> N*m/A
    kt = torque_constant_mNm_A / 1000.0

    # Convert Kv RPM/V to motor back-EMF constant.
    #
    # omega_motor [rad/s] = Kv_rad_per_sec_per_V * voltage
    #
    kv_rad_s_V = kv_rpm_V * 2.0 * math.pi / 60.0

    # Ke in V / (rad/s)
    ke = 1.0 / kv_rad_s_V

    target = math.radians(angle_deg)

    # Current state
    theta = 0.0  # output angle [rad]
    omega = 0.0  # output speed [rad/s]
    time = 0.0

    max_output_speed = 0.0
    max_seen_current = 0.0
    switch_time = None
    switched_to_braking = False

    # --------------------------------------------------------
    # Function determining output acceleration for a requested
    # motor voltage.
    # --------------------------------------------------------
    def acceleration(output_speed, commanded_voltage):
        # Motor speed caused by gearbox
        motor_speed = output_speed * gear_ratio

        # Back EMF
        back_emf = ke * motor_speed

        # Motor current from DC motor equation:
        #
        # V = I*R + Ke*omega
        #
        current = (commanded_voltage - back_emf) / resistance_ohm

        # Apply controller/current-limit
        current = max(-max_current, min(max_current, current))

        # Motor torque
        motor_torque = kt * current

        # Ideal gearbox output torque
        output_motor_torque = motor_torque * gear_ratio

        # Friction/load always opposes the current direction of motion.
        # During startup, assume resistance opposes positive motion.
        if output_speed > 1e-12:
            load_torque = resistance_torque_Nm
        elif output_speed < -1e-12:
            load_torque = -resistance_torque_Nm
        else:
            # At zero velocity, determine based on intended torque.
            if output_motor_torque > 0:
                load_torque = resistance_torque_Nm
            elif output_motor_torque < 0:
                load_torque = -resistance_torque_Nm
            else:
                load_torque = 0.0

        net_torque = output_motor_torque - load_torque

        # Reflect the motor rotor inertia through the ideal gearbox to the
        # output shaft. No separate output/load inertia is modeled.
        effective_output_inertia = motor_inertia_kgm2 * gear_ratio**2
        accel = net_torque / effective_output_inertia

        return accel, current, output_motor_torque

    # --------------------------------------------------------
    # Estimate braking distance from current speed.
    #
    # This numerically simulates a braking maneuver starting
    # at the current velocity.
    # --------------------------------------------------------
    def braking_distance(initial_speed):
        if initial_speed <= 0:
            return 0.0

        speed = initial_speed
        distance = 0.0

        # Separate integration step for stopping-distance estimate.
        brake_dt = dt

        for _ in range(1_000_000):
            accel, _, _ = acceleration(speed, -max_voltage)

            # If even reverse voltage can't decelerate, stopping isn't possible.
            if accel >= 0:
                return float("inf")

            new_speed = speed + accel * brake_dt

            if new_speed <= 0:
                # Approximate the fractional timestep required to reach zero.
                fraction = speed / (speed - new_speed)

                avg_speed = speed / 2.0
                distance += avg_speed * brake_dt * fraction
                return distance

            distance += 0.5 * (speed + new_speed) * brake_dt
            speed = new_speed

        return float("inf")

    # --------------------------------------------------------
    # Main simulation
    # --------------------------------------------------------

    reached_target = False

    while time < max_sim_time:

        distance_remaining = target - theta

        if distance_remaining <= 0 and omega <= 1e-3:
            reached_target = True
            break

        # Figure out how much distance is needed to stop from
        # the current velocity.
        stop_distance = braking_distance(omega)

        # Bang-bang controller:
        #
        # Full positive voltage until stopping distance becomes
        # equal to remaining distance, then full negative voltage.
        if stop_distance >= distance_remaining and omega > 0:
            command_voltage = -max_voltage

            if not switched_to_braking:
                switched_to_braking = True
                switch_time = time
        else:
            command_voltage = max_voltage

        accel, current, output_torque = acceleration(omega, command_voltage)

        max_seen_current = max(max_seen_current, abs(current))
        max_output_speed = max(max_output_speed, abs(omega))

        # Integrate position and velocity.
        #
        # Constant acceleration approximation within timestep.
        theta += omega * dt + 0.5 * accel * dt**2
        new_omega = omega + accel * dt

        # Don't permit reverse motion once braking reaches zero.
        if switched_to_braking and new_omega < 0:
            # Approximate time inside this integration interval
            # where velocity reaches zero.
            if accel != 0:
                stop_fraction = -omega / (accel * dt)
                stop_fraction = max(0.0, min(1.0, stop_fraction))
            else:
                stop_fraction = 1.0

            time += dt * stop_fraction
            omega = 0.0

            # If we're sufficiently near the destination, call move complete.
            if abs(target - theta) < math.radians(0.05):
                reached_target = True
                break

            # Numerical errors may leave us slightly short.
            switched_to_braking = False
            continue

        omega = new_omega
        time += dt

    return {
        "time_s": time if reached_target else float("nan"),
        "max_output_speed_rad_s": max_output_speed,
        "max_output_speed_deg_s": math.degrees(max_output_speed),
        "max_motor_rpm": (max_output_speed * gear_ratio * 60.0 / (2.0 * math.pi)),
        "peak_current_A": max_seen_current,
        "switch_time_s": switch_time,
        "completed": reached_target,
    }


# Parameters for which move-time sensitivity is reported.  The label includes
# the units of d(move time) / d(parameter).
DERIVATIVE_PARAMETERS = [
    ("max_voltage", "dTime/dVoltage (s/V)"),
    ("max_current", "dTime/dCurrent Limit (s/A)"),
    ("gear_ratio", "dTime/dGear Ratio (s/ratio)"),
    ("resistance_torque_Nm", "dTime/dResistance Torque (s/(N*m))"),
    # ("torque_constant_mNm_A", "dTime/dKt (s/(mN*m/A))"),
    # ("kv_rpm_V", "dTime/dKv (s/(RPM/V))"),
    # ("resistance_ohm", "dTime/dWinding Resistance (s/ohm)"),
    ("motor_inertia_kgm2", "dTime/dMotor Inertia (s/(kg*m^2))"),
    ("angle_deg", "dTime/dAngle (s/deg)"),
]


def move_time_derivatives(parameters, relative_step=0.01):
    """Return local numerical derivatives of move time for one setting.

    A centered finite difference is used for positive parameter values.  A
    forward difference is used at zero so that quantities such as opposing
    torque are not perturbed to an invalid negative value.
    """
    base_time = simulate_move(**parameters)["time_s"]
    derivatives = {}

    for parameter_name, output_label in DERIVATIVE_PARAMETERS:
        value = parameters[parameter_name]
        step = max(abs(value) * relative_step, 1e-6)

        upper_parameters = parameters.copy()
        upper_parameters[parameter_name] = value + step
        upper_time = simulate_move(**upper_parameters)["time_s"]

        if value - step > 0:
            lower_parameters = parameters.copy()
            lower_parameters[parameter_name] = value - step
            lower_time = simulate_move(**lower_parameters)["time_s"]
            time_difference = upper_time - lower_time
            parameter_difference = 2.0 * step
        else:
            lower_time = base_time
            time_difference = upper_time - base_time
            parameter_difference = step

        if math.isfinite(upper_time) and math.isfinite(lower_time):
            derivatives[output_label] = time_difference / parameter_difference
        else:
            derivatives[output_label] = float("nan")

    return derivatives


# ============================================================
# RUN ALL COMBINATIONS
# ============================================================


def run_combination(combination):
    voltage, amps, gearing, resistance_torque = combination
    parameters = dict(
        torque_constant_mNm_A=TORQUE_CONSTANT_MNM_PER_A,
        kv_rpm_V=KV_RPM_PER_V,
        resistance_ohm=RESISTANCE_OHM,
        motor_inertia_kgm2=MOTOR_INERTIA_KGM2,
        angle_deg=ANGLE_DEG,
        resistance_torque_Nm=resistance_torque,
        max_voltage=voltage,
        max_current=amps,
        gear_ratio=gearing,
        dt=DT,
    )
    result = simulate_move(**parameters)
    derivatives = move_time_derivatives(
        parameters,
        relative_step=DERIVATIVE_RELATIVE_STEP,
    )

    return {
        "Voltage (V)": voltage,
        "Current Limit (A)": amps,
        "Gear Ratio": gearing,
        "Resistance Torque (N*m)": resistance_torque,
        "Move Time (s)": result["time_s"],
        **derivatives,
        # "Peak Output Speed (deg/s)": result["max_output_speed_deg_s"],
        # "Peak Motor Speed (RPM)": result["max_motor_rpm"],
        # "Peak Current (A)": result["peak_current_A"],
        # "Accel→Brake Time (s)": result["switch_time_s"],
        # "Completed": result["completed"],
    }


combinations = itertools.product(
    MAX_VOLTAGES,
    MAX_AMPS,
    GEAR_RATIOS,
    RESISTANCE_TORQUES_NM,
)
with ProcessPoolExecutor() as executor:
    results = list(executor.map(run_combination, combinations))

# results = [r for r in results if math.isfinite(r["Move Time (s)"])]
results = [r for r in results if (r["dTime/dVoltage (s/V)"] != 0.0 or r["Voltage (V)"] == max(MAX_VOLTAGES))]
results = [r for r in results if not math.isnan(r["Move Time (s)"])]

# ============================================================
# OUTPUT TABLE
# ============================================================

df = pd.DataFrame(results)

# Sort fastest first
df = df.sort_values(by="Move Time (s)", na_position="last").reset_index(drop=True)

# Make terminal output easier to read
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 400)
pd.set_option("display.max_columns", None)

print(
    "Derivative sign: positive means increasing the parameter increases "
    "move time; negative means it decreases move time."
)
print(
    df.round(
        {
            "Move Time (s)": 4,
            **{output_label: 6 for _, output_label in DERIVATIVE_PARAMETERS},
            "Peak Output Speed (deg/s)": 1,
            "Peak Motor Speed (RPM)": 0,
            "Peak Current (A)": 2,
            "Accel→Brake Time (s)": 4,
        }
    )
)

# Optional:
df.to_csv("servo_move_results.csv", index=False)
