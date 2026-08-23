from pathlib import Path

CASE = Path(__file__).resolve().parent
ZERO = CASE / '0'
HEADER = '''FoamFile
{{
    format ascii;
    class {cls};
    location "0";
    object {name};
}}

dimensions {dims};

internalField uniform {internal};

boundaryField
{{
{body}
}}
'''

def scalar(name, dims, internal, human='zeroGradient', cls='volScalarField'):
    wall_type = 'fixedFluxPressure' if name == 'p_rgh' else 'zeroGradient'
    human_type = wall_type if name == 'p_rgh' else human
    human_entry = f'''    human {{
        type {human};
        commsDir "${{FOAM_CASE}}/comms";
        file "data";
        initByExternal yes;
        value uniform 307.15;
    }}''' if human == 'externalCoupledTemperature' else f'''    human {{ type {human_type}; value uniform 0; }}'''
    ceiling_type = 'fixedValue' if name == 'p_rgh' else 'zeroGradient'
    body = f'''    inlet {{ type {wall_type}; value uniform 0; }}
    outlet {{ type {wall_type}; value uniform 0; }}
    floor {{ type {wall_type}; value uniform 0; }}
    ceiling {{ type {ceiling_type}; value uniform 0; }}
    sideA {{ type {wall_type}; value uniform 0; }}
    sideB {{ type {wall_type}; value uniform 0; }}
{human_entry}'''
    (ZERO / name).write_text(HEADER.format(cls=cls, name=name, dims=dims, internal=internal, body=body))

scalar('T', '[0 0 0 1 0 0 0]', '293.15', 'externalCoupledTemperature')
scalar('p_rgh', '[1 -1 -2 0 0 0 0]', '0')
scalar('k', '[0 2 -2 0 0 0 0]', '1e-6')
scalar('omega', '[0 0 -1 0 0 0 0]', '1')
scalar('nut', '[0 2 -1 0 0 0 0]', '0')
scalar('alphat', '[1 -1 -1 0 0 0 0]', '0')

u_body = '''    inlet { type noSlip; }
    outlet { type noSlip; }
    floor { type noSlip; }
    ceiling { type pressureInletOutletVelocity; value uniform (0 0 0); }
    sideA { type noSlip; }
    sideB { type noSlip; }
    human { type noSlip; }'''
(ZERO / 'U').write_text(HEADER.format(cls='volVectorField', name='U', dims='[0 1 -1 0 0 0 0]', internal='(0 0 0)', body=u_body))
