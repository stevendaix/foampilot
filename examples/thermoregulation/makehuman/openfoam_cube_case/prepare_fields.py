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
    human_entry = f'''    human {{
        type {human};
        commsDir "${{FOAM_CASE}}/comms";
        file "data";
        initByExternal yes;
        value uniform 307.15;
    }}''' if human == 'externalCoupledTemperature' else f'''    human {{ type {human}; }}'''
    body = f'''    inlet {{ type zeroGradient; }}
    outlet {{ type zeroGradient; }}
    floor {{ type zeroGradient; }}
    ceiling {{ type zeroGradient; }}
    sideA {{ type zeroGradient; }}
    sideB {{ type zeroGradient; }}
{human_entry}'''
    (ZERO / name).write_text(HEADER.format(cls=cls, name=name, dims=dims, internal=internal, body=body))

scalar('T', '[0 0 0 1 0 0 0]', '293.15', 'externalCoupledTemperature')
scalar('p_rgh', '[1 -1 -2 0 0 0 0]', '0')
scalar('k', '[0 2 -2 0 0 0 0]', '1e-6')
scalar('omega', '[0 0 -1 0 0 0 0]', '1')
scalar('nut', '[0 2 -1 0 0 0 0]', '0')
scalar('alphat', '[1 -1 -1 0 0 0 0]', '0')

u_body = '''    inlet { type fixedValue; value uniform (0 0 0); }
    outlet { type zeroGradient; }
    floor { type noSlip; }
    ceiling { type noSlip; }
    sideA { type noSlip; }
    sideB { type noSlip; }
    human { type noSlip; }'''
(ZERO / 'U').write_text(HEADER.format(cls='volVectorField', name='U', dims='[0 1 -1 0 0 0 0]', internal='(0 0 0)', body=u_body))
