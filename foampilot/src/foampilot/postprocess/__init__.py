from foampilot.postprocess.openfoam_pyvista import FoamPostProcessing
from foampilot.postprocess.jos3_openfoam import (
    OpenFOAMJOS3Coupler,
    NodalThermalExchange,
    JOS3_SEGMENT_NAMES,
)
from foampilot.postprocess.openfoam_external_coupled import (
    OpenFOAMExternalCoupledProvider,
    OpenFOAM13TemperatureProvider,
)
from foampilot.postprocess.openfoam_direct import (
    OpenFOAMDirectReader,
    CHTDirectReader,
    read_openfoam,
    read_cht_openfoam,
)
from foampilot.postprocess.boundary_viewer import BoundaryViewer
from foampilot.postprocess.web_presentation import (
    CFDDashboard,
    plotly_contour_from_mesh,
    plotly_velocity_magnitude,
    plotly_temperature_contour,
    plotly_pressure_contour,
)
