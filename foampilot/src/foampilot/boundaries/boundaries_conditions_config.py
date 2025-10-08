
# Configuration des conditions aux limites pour foampilot

from foampilot.utilities.manageunits import Quantity

# Constantes physiques et de modélisation
CONSTANTS = {
    "Cmu": 0.09,
    "kappa": 0.41,
    "E": 9.8,
}

# Définition des fonctions de paroi
WALL_FUNCTIONS = {
    "kEpsilon": {
        "k": {
            "noSlip": {"type": "kqRWallFunction", "value": "uniform 0"},
            "fixedValue": {"type": "kqRWallFunction", "value": "uniform {value}"},
        },
        "epsilon": {
            "default": {"type": "epsilonWallFunction", "value": "uniform 0", **CONSTANTS},
        },
        "nut": {
            "default": {"type": "nutkWallFunction", "value": "uniform 0", **CONSTANTS},
        },
    },
    "kOmegaSST": {
        "k": {
            "noSlip": {"type": "kWallFunction", "value": "uniform 0"},
            "fixedValue": {"type": "kWallFunction", "value": "uniform {value}"},
        },
        "omega": {
            "default": {"type": "omegaWallFunction", "value": "uniform 0"},
        },
        "nut": {
            "default": {"type": "nutkWallFunction", "value": "uniform 0", **CONSTANTS},
        },
    },
}

# Définition des conditions aux limites par type et par champ
BOUNDARY_CONDITIONS_CONFIG = {
    "kEpsilon": {
        "velocityInlet": {
            "U": {"type": "fixedValue", "value": "uniform ({u_ms} {v_ms} {w_ms})"},
            "p": {"type": "zeroGradient"},
            "nut": {"type": "calculated", "value": "uniform 0"},
            "k": {
                "withTurbulence": {"type": "fixedValue", "value": "uniform {k_value}"},
                "default": {"type": "zeroGradient"},
            },
            "epsilon": {
                "withTurbulence": {"type": "fixedValue", "value": "uniform {epsilon_value}"},
                "default": {"type": "zeroGradient"},
            },
        },
        "pressureInlet": {
            "U": {"type": "zeroGradient"},
            "p": {"type": "fixedValue", "value": "uniform {p_pa}"},
            "nut": {"type": "calculated", "value": "uniform 0"},
            "k": {
                "withTurbulence": {"type": "fixedValue", "value": "uniform {k_value}"},
                "default": {"type": "zeroGradient"},
            },
            "epsilon": {
                "withTurbulence": {"type": "fixedValue", "value": "uniform {epsilon_value}"},
                "default": {"type": "zeroGradient"},
            },
        },
        "pressureOutlet": {
            "U": {"type": "pressureInletOutletVelocity", "value": "uniform ({u_ms} {v_ms} {w_ms})"},
            "p": {"type": "fixedValue", "value": "uniform 0"},
            "k": {"type": "zeroGradient"},
            "epsilon": {"type": "zeroGradient"},
            "nut": {"type": "calculated", "value": "uniform 0"},
        },
        "massFlowInlet": {
            "U": {"type": "fixedValue", "value": "uniform ({velocity_value} 0 0)"},
            "p": {"type": "zeroGradient"},
            "k": {"type": "zeroGradient"},
            "epsilon": {"type": "zeroGradient"},
            "nut": {"type": "calculated", "value": "uniform 0"},
        },
        "wall": {
            "U": {
                "noSlip": {"type": "noSlip"},
                "slip": {"type": "slip"},
                "fixedValue": {"type": "fixedValue", "value": "uniform ({u_ms} {v_ms} {w_ms})"},
            },
            "p": {"type": "zeroGradient"},
            "k": {"type": "wallFunction", "function": "k"},
            "epsilon": {"type": "wallFunction", "function": "epsilon"},
            "nut": {"type": "wallFunction", "function": "nut"},
        },
        "symmetry": {
            "U": {"type": "empty"},
            "p": {"type": "empty"},
            "k": {"type": "empty"},
            "epsilon": {"type": "empty"},
            "nut": {"type": "empty"},
        },
        "noFrictionWall": {
            "U": {"type": "slip"},
            "p": {"type": "zeroGradient"},
            "k": {"type": "zeroGradient"},
            "epsilon": {"type": "zeroGradient"},
            "nut": {"type": "wallFunction", "function": "nut"},
        },
        "uniformNormalFixedValue": {
            "U": {"type": "uniformNormalFixedValue", "value": "uniform {ref_value}"},
            "p": {"type": "zeroGradient"},
            "k": {"type": "zeroGradient"},
            "epsilon": {"type": "zeroGradient"},
            "nut": {"type": "calculated", "value": "uniform 0"},
        },
        "surfaceNormalFixedValue": {
            "U": {"type": "surfaceNormalFixedValue", "refValue": "uniform {ref_value}", "ramp": "table ((0 0) (10 1))"},
            "p": {"type": "zeroGradient"},
            "k": {"type": "zeroGradient"},
            "epsilon": {"type": "zeroGradient"},
            "nut": {"type": "calculated", "value": "uniform 0"},
        },
    },
    "kOmegaSST": {
        "velocityInlet": {
            "U": {"type": "fixedValue", "value": "uniform ({u_ms} {v_ms} {w_ms})"},
            "p": {"type": "zeroGradient"},
            "nut": {"type": "calculated", "value": "uniform 0"},
            "k": {
                "withTurbulence": {"type": "fixedValue", "value": "uniform {k_value}"},
                "default": {"type": "zeroGradient"},
            },
            "omega": {
                "withTurbulence": {"type": "fixedValue", "value": "uniform {omega_value}"},
                "default": {"type": "zeroGradient"},
            },
        },
        "pressureInlet": {
            "U": {"type": "zeroGradient"},
            "p": {"type": "fixedValue", "value": "uniform {p_pa}"},
            "nut": {"type": "calculated", "value": "uniform 0"},
            "k": {
                "withTurbulence": {"type": "fixedValue", "value": "uniform {k_value}"},
                "default": {"type": "zeroGradient"},
            },
            "omega": {
                "withTurbulence": {"type": "fixedValue", "value": "uniform {omega_value}"},
                "default": {"type": "zeroGradient"},
            },
        },
        "pressureOutlet": {
            "U": {"type": "pressureInletOutletVelocity", "value": "uniform ({u_ms} {v_ms} {w_ms})"},
            "p": {"type": "fixedValue", "value": "uniform 0"},
            "k": {"type": "zeroGradient"},
            "omega": {"type": "zeroGradient"},
            "nut": {"type": "calculated", "value": "uniform 0"},
        },
        "massFlowInlet": {
            "U": {"type": "fixedValue", "value": "uniform ({velocity_value} 0 0)"},
            "p": {"type": "zeroGradient"},
            "k": {"type": "zeroGradient"},
            "omega": {"type": "zeroGradient"},
            "nut": {"type": "calculated", "value": "uniform 0"},
        },
        "wall": {
            "U": {
                "noSlip": {"type": "noSlip"},
                "slip": {"type": "slip"},
                "fixedValue": {"type": "fixedValue", "value": "uniform ({u_ms} {v_ms} {w_ms})"},
            },
            "p": {"type": "zeroGradient"},
            "k": {"type": "wallFunction", "function": "k"},
            "omega": {"type": "wallFunction", "function": "omega"},
            "nut": {"type": "wallFunction", "function": "nut"},
        },
        "symmetry": {
            "U": {"type": "empty"},
            "p": {"type": "empty"},
            "k": {"type": "empty"},
            "omega": {"type": "empty"},
            "nut": {"type": "empty"},
        },
        "noFrictionWall": {
            "U": {"type": "slip"},
            "p": {"type": "zeroGradient"},
            "k": {"type": "zeroGradient"},
            "omega": {"type": "zeroGradient"},
            "nut": {"type": "wallFunction", "function": "nut"},
        },
        "uniformNormalFixedValue": {
            "U": {"type": "uniformNormalFixedValue", "value": "uniform {ref_value}"},
            "p": {"type": "zeroGradient"},
            "k": {"type": "zeroGradient"},
            "omega": {"type": "zeroGradient"},
            "nut": {"type": "calculated", "value": "uniform 0"},
        },
        "surfaceNormalFixedValue": {
            "U": {"type": "surfaceNormalFixedValue", "refValue": "uniform {ref_value}", "ramp": "table ((0 0) (10 1))"},
            "p": {"type": "zeroGradient"},
            "k": {"type": "zeroGradient"},
            "omega": {"type": "zeroGradient"},
            "nut": {"type": "calculated", "value": "uniform 0"},
        },
    },
}

# Mapping des types de conditions aux limites aux méthodes de calcul des paramètres
CONDITION_CALCULATORS = {
    "velocityInlet": {
        "calculate": lambda velocity, turbulence_intensity, **kwargs: {
            "u_ms": velocity[0].get_in('m/s'),
            "v_ms": velocity[1].get_in('m/s'),
            "w_ms": velocity[2].get_in('m/s'),
            **({} if not turbulence_intensity else {
                "norm_u": (velocity[0].get_in("m/s")**2 + velocity[1].get_in("m/s")**2 + velocity[2].get_in("m/s")**2) ** 0.5,
                "k_value": 1.5 * ((velocity[0].get_in("m/s")**2 + velocity[1].get_in("m/s")**2 + velocity[2].get_in("m/s")**2) ** 0.5 * turbulence_intensity) ** 2,
                "epsilon_value": (1.5 * ((velocity[0].get_in("m/s")**2 + velocity[1].get_in("m/s")**2 + velocity[2].get_in("m/s")**2) ** 0.5 * turbulence_intensity) ** 2) ** 1.5 / (0.07 * ((velocity[0].get_in("m/s")**2 + velocity[1].get_in("m/s")**2 + velocity[2].get_in("m/s")**2) ** 0.5)),
                "omega_value": (1.5 * ((velocity[0].get_in("m/s")**2 + velocity[1].get_in("m/s")**2 + velocity[2].get_in("m/s")**2) ** 0.5 * turbulence_intensity) ** 2) ** 0.5 / (0.09 * ((velocity[0].get_in("m/s")**2 + velocity[1].get_in("m/s")**2 + velocity[2].get_in("m/s")**2) ** 0.5)), # Simplified for kOmegaSST
            })
        },
        "validate": lambda velocity, **kwargs: all(comp.quantity.check('[length] / [time]') for comp in velocity),
        "error_message": "Each velocity component must have units of length/time."
    },
    "pressureInlet": {
        "calculate": lambda pressure, turbulence_intensity, **kwargs: {
            "p_pa": pressure.get_in('Pa'),
            **({} if not turbulence_intensity else {
                "k_value": 1.5 * (pressure.get_in("Pa") * turbulence_intensity) ** 2,
                "epsilon_value": (1.5 * (pressure.get_in("Pa") * turbulence_intensity) ** 2) ** 1.5 / (0.07 * pressure.get_in("Pa")),
                "omega_value": (1.5 * (pressure.get_in("Pa") * turbulence_intensity) ** 2) ** 0.5 / (0.09 * pressure.get_in("Pa")),
            })
        },
        "validate": lambda pressure, **kwargs: pressure.quantity.check('[mass] / ([length] * [time] ** 2)'),
        "error_message": "Pressure must have units of mass/(length*time²)."
    },
    "pressureOutlet": {
        "calculate": lambda velocity, **kwargs: {
            "u_ms": velocity[0].get_in('m/s'),
            "v_ms": velocity[1].get_in('m/s'),
            "w_ms": velocity[2].get_in('m/s'),
        },
        "validate": lambda velocity, **kwargs: all(comp.quantity.check('[length] / [time]') for comp in velocity),
        "error_message": "Each velocity component must have units of length/time."
    },
    "massFlowInlet": {
        "calculate": lambda mass_flow_rate, density, **kwargs: {
            "velocity_value": (mass_flow_rate.get_in("kg/s") / density.get_in("kg/m^3")),
        },
        "validate": lambda mass_flow_rate, density, **kwargs:
            mass_flow_rate.quantity.check('[mass] / [time]') and density.quantity.check('[mass] / [volume]'),
        "error_message": "Mass flow rate must have units of mass/time and density must have units of mass/volume."
    },
    "wall": {
        "calculate": lambda velocity, **kwargs: {
            **({} if not velocity else {
                "u_ms": velocity[0].get_in('m/s'),
                "v_ms": velocity[1].get_in('m/s'),
                "w_ms": velocity[2].get_in('m/s'),
            })
        },
        "validate": lambda velocity, **kwargs:
            True if not velocity else all(comp.quantity.check('[length] / [time]') for comp in velocity),
        "error_message": "Each velocity component must have units of length/time."
    },
    "uniformNormalFixedValue": {
        "calculate": lambda ref_value, **kwargs: {"ref_value": ref_value},
        "validate": lambda ref_value, **kwargs: isinstance(ref_value, (int, float)),
        "error_message": "ref_value must be a number."
    },
    "surfaceNormalFixedValue": {
        "calculate": lambda ref_value, **kwargs: {"ref_value": ref_value},
        "validate": lambda ref_value, **kwargs: isinstance(ref_value, (int, float)),
        "error_message": "ref_value must be a number."
    },
}
