# Applied theory: biomedical flow, outdoor wind, and human thermoregulation

This chapter explains **why** a model is selected, **which law** it represents, **what data** it needs, and **when it becomes unreliable**. It is deliberately more detailed than a solver recipe. A CFD case is not defined by the executable name alone; it is defined by geometry, conservation laws, constitutive relations, boundary data, turbulence closure, numerical discretisation, and a validation strategy.

## 1. Model-selection principle

A useful model is not the most complex model available. It is the least complex model that resolves the quantity of interest under the conditions of the experiment or application. The choice should be justified by:

| Question | Consequence |
| --- | --- |
| Is the flow compressible? | Select the density and pressure formulation. |
| Is it steady or transient? | Select a steady RANS, transient RANS, LES, or time-dependent laminar model. |
| Is heat coupled to momentum? | Use passive scalar transport, buoyancy, or a full energy/thermophysical model. |
| Are wall gradients important? | Choose wall resolution, wall functions, and mesh targets consistently. |
| Does viscosity depend on shear rate? | Use Newtonian or non-Newtonian rheology. |
| Are the boundaries measured or idealised? | Use data-driven profiles, tables, or analytical functions and quantify uncertainty. |
| Is the geometry patient-specific or geospatial? | Preserve coordinate systems, units, topology, and provenance. |

A model should always state its **domain of validity**. In particular, a successful tutorial run does not validate the physical assumptions for a biomedical or environmental prediction.

# 2. Biomedical CFD

## 2.1 What is being modelled?

Biomedical CFD can refer to very different problems: blood flow in arteries, respiratory airflow, heat exchange around a body, flow through medical devices, or transport in porous tissue. The laws and boundary data differ between them. This section focuses on vascular flow and the interfaces between patient-specific geometry, hemodynamic models, and FoamPilot utilities.

For a fixed fluid domain, the base equations are mass conservation and momentum conservation:

$$
\frac{\partial\rho}{\partial t}+\nabla\cdot(\rho\mathbf{u})=0,
$$

$$
\rho\left(\frac{\partial\mathbf{u}}{\partial t}+\mathbf{u}\cdot\nabla\mathbf{u}\right)
=-\nabla p+\nabla\cdot\boldsymbol{\tau}+\mathbf{f},
$$

where $\boldsymbol{\tau}$ is the viscous stress tensor and $\mathbf{f}$ includes body forces or modelled sources.

For an incompressible Newtonian fluid:

$$
\boldsymbol{\tau}=2\mu\mathbf{D},
\qquad
\mathbf{D}=\frac12\left(\nabla\mathbf{u}+\nabla\mathbf{u}^{T}\right),
$$

with constant dynamic viscosity $\mu$. For blood, this is an approximation whose suitability depends on vessel size, shear rate, haematocrit, and the output of interest.

## 2.2 Newtonian versus non-Newtonian blood

Blood is often treated as Newtonian in large arteries and high-shear regions because the apparent viscosity approaches a nearly constant value. This simplification reduces cost and makes convergence easier. It can be defensible when the goal is a global pressure drop and the shear rate is high over most of the region.

Blood also has shear-thinning behaviour: apparent viscosity increases at low shear rates and decreases as shear increases. Non-Newtonian models become more important in recirculation zones, aneurysms, near stagnation, distal vessels, and when wall shear stress or residence time is the main quantity of interest. A comparative study of intracranial stenosis models found that Newtonian and non-Newtonian assumptions can have a small effect on pressure ratio while producing more visible differences in low-WSS regions, especially during diastole [1].

### Newtonian law

$$
\mu=\mu_0.
$$

This is the simplest model. The viscosity must be stated with units and temperature assumptions.

### Carreau–Yasuda law

A common shear-thinning form is:

$$
\mu(\dot\gamma)=\mu_\infty+(\mu_0-\mu_\infty)
\left[1+(\lambda\dot\gamma)^a\right]^{(n-1)/a},
$$

where $\dot\gamma$ is the shear-rate magnitude, $\mu_0$ and $\mu_\infty$ are limiting viscosities, $\lambda$ is a time scale, $a$ controls the transition, and $n<1$ produces shear thinning.

### Casson law

The Casson model is another empirical rheology used for blood:

$$
\sqrt{\tau}=\sqrt{\tau_y}+\sqrt{\mu_c\dot\gamma},
$$

where $\tau_y$ is a yield-stress-like parameter and $\mu_c$ controls the high-shear behaviour. The exact regularisation and implementation matter at low shear rate.

### How to choose

| Quantity of interest | First model to test | Additional sensitivity study |
| --- | --- | --- |
| Bulk flow rate or rough pressure ratio | Newtonian | Carreau–Yasuda or Casson if low-shear regions matter. |
| Wall shear stress | Newtonian and non-Newtonian comparison | Report low-WSS and oscillatory-WSS sensitivity. |
| Residence time or thrombosis-related indicator | Non-Newtonian candidate | Check rheological parameters and near-wall resolution. |
| Large artery with high shear | Newtonian may be sufficient | Verify against a non-Newtonian run. |
| Small vessel or strong recirculation | Non-Newtonian is more defensible | Include diameter, haematocrit, temperature, and patient variability. |

FoamPilot’s base `transportProperties` path is naturally suited to constant-property cases. A non-Newtonian law requires a solver and dictionary configuration that actually evaluates viscosity from the local shear rate. Assigning a descriptive Python variable without verifying the generated OpenFOAM dictionary does not activate a rheology model.

## 2.3 Pulsatility and the Womersley number

Blood flow is usually pulsatile. The Womersley number compares unsteady inertia with viscous diffusion:

$$
\alpha=R\sqrt{\frac{\omega\rho}{\mu}},
$$

where $R$ is the vessel radius and $\omega$ is the angular frequency of the cardiac waveform. Low $\alpha$ produces a profile closer to quasi-steady parabolic flow. Higher $\alpha$ produces a flatter core and stronger phase lag between pressure gradient and wall response.

For pulsatile simulations, the inlet should be represented by a measured or synthetic flow-rate waveform. The waveform must be converted into a velocity profile using the actual inlet area and, if possible, a developed or Womersley-consistent profile. A uniform velocity at the entrance of a sharply curved or branching artery can generate artificial entrance effects that contaminate the region of interest.

## 2.4 Boundary conditions in vascular CFD

The most important biomedical uncertainty is often the boundary condition rather than the interior discretisation.

| Boundary | Common data | Physical issue |
| --- | --- | --- |
| Inlet | Flow rate, velocity profile, pressure, or patient waveform | Measured plane may be far from the computational inlet. |
| Outlet | Fixed pressure, traction, resistance, impedance, or Windkessel | Downstream vasculature is truncated. |
| Wall | No-slip rigid wall, moving wall, or fluid-structure coupling | Wall compliance can change pressure and WSS. |
| Branch | Flow split or pressure relation | Patient-specific downstream resistance is uncertain. |

### Windkessel outlet model

A Windkessel model represents the resistance and compliance of the vasculature downstream of a truncated outlet. A common three-element model combines proximal resistance $R_1$, compliance $C$, and distal resistance $R_2$. In a pressure-flow form:

$$
C\frac{dP_c}{dt}=Q-\frac{P_c-P_d}{R_2},
$$

$$
P=P_c+R_1Q,
$$

where $Q$ is outlet flow, $P$ is outlet pressure, $P_c$ is capacitor pressure, and $P_d$ is distal reference pressure. The model is chosen because a fixed pressure outlet cannot reproduce the storage and delayed response of the downstream network.

FoamPilot exposes `WindkesselModel` as a model add-on. Before using it, define the sign convention, units, initial capacitor pressure, pressure reference, and coupling time step. Calibrate $R_1$, $R_2$, and $C$ against measured pressure-flow data or a documented physiological assumption. A Windkessel model is a reduced-order boundary representation; it is not a full cardiovascular circulation model.

## 2.5 Patient-specific geometry and data provenance

A biomedical case commonly begins with CTA, MRI, CT, NIfTI, STL, VTP, or another segmented surface. The pipeline should document:

1. imaging modality, resolution, acquisition date, and orientation;
2. segmentation method and thresholding decisions;
3. smoothing and hole-closing operations;
4. inlet/outlet extension length;
5. surface remeshing tolerance and triangle count;
6. conversion to metric units;
7. branch labels and patch names;
8. mesh quality and volume conservation;
9. boundary-condition source and calibration;
10. anonymisation and data governance.

FoamPilot utilities include NIfTI-to-STL and vascular surface-cleaning helpers. These are geometry-processing tools, not clinical segmentation validators. Inspect the output visually and quantitatively before solving.

## 2.6 Biomedical validation quantities

The following quantities are often reported:

- pressure drop and translesional pressure ratio;
- time-averaged wall shear stress;
- oscillatory shear index;
- relative residence time;
- recirculation volume;
- flow split at branches;
- peak systolic and end-diastolic values;
- conservation of flow through outlets.

Do not interpret one local WSS peak without a mesh-convergence and near-wall study. WSS is a derivative at the wall and is particularly sensitive to surface smoothing, mesh spacing, temporal resolution, and rheology.

# 3. Outdoor wind and atmospheric boundary layers

## 3.1 Why a uniform inlet is often wrong

A building or urban-flow simulation is not simply a car case rotated vertically. Near the ground, the mean wind speed increases with height and turbulence varies with height. Buildings disturb this incoming atmospheric boundary layer, creating acceleration around corners, roof separation, street-canyon recirculation, and wakes.

A uniform inlet can be acceptable for a controlled wind tunnel or for a simplified method study. It is generally not consistent with an atmospheric boundary layer unless the domain and boundary conditions are deliberately constructed so that the profile develops before the region of interest.

## 3.2 Governing equations

For low-speed outdoor wind, air is commonly treated as incompressible:

$$
\nabla\cdot\mathbf{U}=0,
$$

$$
\frac{\partial\mathbf{U}}{\partial t}+\mathbf{U}\cdot\nabla\mathbf{U}
=-\frac{1}{\rho}\nabla p+\nabla\cdot[(\nu+\nu_t)\nabla\mathbf{U}],
$$

with turbulent viscosity $\nu_t$ supplied by a closure such as $k$–$\epsilon$, realizable $k$–$\epsilon$, RNG $k$–$\epsilon$, or $k$–$\omega$ SST.

The correct choice depends on the output. A steady RANS model is efficient for mean wind and mean pressure. URANS is needed when coherent unsteadiness matters. LES or hybrid methods are more appropriate when transient vortices and turbulent fluctuations are primary outputs, but their mesh and timestep costs are much higher.

## 3.3 Logarithmic wind law

The neutral atmospheric boundary layer is often approximated with the logarithmic law:

$$
U(z)=\frac{u_*}{\kappa}\ln\left(\frac{z+z_0}{z_0}\right),
$$

where $u_*$ is friction velocity, $\kappa\approx0.4$ is the von Kármán constant, and $z_0$ is aerodynamic roughness length. If the reference wind is known at height $z_r$:

$$
U(z)=U(z_r)\frac{\ln[(z+z_0)/z_0]}{\ln[(z_r+z_0)/z_0]}.
$$

The logarithmic law is chosen because it represents the mean velocity in the surface layer under neutral, horizontally homogeneous assumptions. It does not automatically represent thermal stratification, forest canopies, complex terrain, or strongly transient weather.

OpenFOAM provides atmospheric boundary-layer conditions based on log-law-type profiles and turbulence quantities. Its documentation describes `atmBoundaryLayerInletVelocity`, `atmBoundaryLayerInletK`, `atmBoundaryLayerInletEpsilon`, and `atmBoundaryLayerInletOmega`, as well as atmospheric wall functions and source terms [2].

## 3.4 Power-law profile

An engineering alternative is:

$$
U(z)=U_r\left(\frac{z}{z_r}\right)^\alpha,
$$

where $\alpha$ is an empirical shear exponent. The power law is convenient when wind data are available at two heights or when a wind-engineering standard provides an exponent. It is less directly connected to surface roughness than the log law and should not be mixed with a roughness length without stating the conversion.

## 3.5 Turbulence inlet data

The inlet must define not only velocity but also turbulence. For a $k$–$\epsilon$ model, a common estimate is:

$$
 k=\frac32(UI)^2,
$$

where $I$ is turbulence intensity. A length scale $L$ can be used to estimate:

$$
\epsilon=C_\mu^{3/4}\frac{k^{3/2}}{L},
$$

and for a $k$–$\omega$ model:

$$
\omega\approx\frac{\sqrt{k}}{C_\mu^{1/4}L}.
$$

These formulas are modelling assumptions, not measurements. The profiles of $U$, $k$, $\epsilon$, or $\omega$ should be mutually compatible; otherwise the atmospheric boundary layer can drift, accelerate, or decay before reaching the buildings.

## 3.6 Stability and buoyancy

Neutral flow neglects thermal stratification. Stable or unstable atmospheric conditions require temperature, buoyancy, and turbulence-production assumptions. The sign and magnitude of the buoyancy term affect vertical mixing, wake recovery, and pedestrian-level wind.

For a simplified thermal urban case, a Boussinesq approximation may be used when temperature differences are small. For larger stratification or density changes, a compressible or variable-density model is more appropriate. The choice must be consistent with the available weather data and the solver’s thermophysical formulation.

## 3.7 Domain and wall modelling

An outdoor domain should provide adequate upstream fetch, downstream wake length, lateral clearance, and top clearance. The ground is not just another wall: its roughness determines the velocity profile and turbulence generation. Wall functions, roughness parameters, first-cell height, and the atmospheric inlet profile must be chosen as one system.

The main reason for using a specific law or boundary condition is **equilibrium consistency**. If the inlet profile implies one roughness and the ground wall implies another, the profile evolves artificially. The first task is therefore to verify a precursor or empty-domain case before adding buildings.

## 3.8 Urban outputs

Relevant outputs include mean speed at pedestrian height, exceedance probability if transient data are available, pressure on façades, wind comfort indicators, roof acceleration, street-canyon circulation, turbulence intensity, and pollutant transport when a scalar equation is coupled.

A wind result should always state the reference height, roughness, wind direction, atmospheric stability, turbulence model, domain dimensions, mesh count, wall treatment, and averaging interval.

# 4. Human thermoregulation

## 4.1 Coupling levels

Thermoregulation can be represented at several levels:

| Level | Description | Suitable use |
| --- | --- | --- |
| Convective boundary condition | Prescribed heat-transfer coefficient or skin temperature. | Simple thermal CFD around a body. |
| Multi-node physiology | Core, blood, muscle, fat, and skin temperatures with regulatory responses. | Coupling CFD environment with human thermal response. |
| Detailed local physiology | Segment-level metabolism, perfusion, sweating, clothing, radiation, and posture. | Research studies requiring local response. |
| Fully coupled human-fluid model | Physiological state changes alter surface fluxes and flow conditions. | Advanced research; requires careful time coupling and validation. |

FoamPilot’s MakeHuman/JOS-3 workflow belongs to the geometry-plus-physiology coupling level. MakeHuman provides a body surface; JOS-3 provides a multi-node thermal response; OpenFOAM resolves the surrounding flow and heat transfer.

## 4.2 JOS-3 model concept

JOS-3 is a numerical human thermoregulation model that predicts quantities such as core temperature, skin temperature, sweating, blood flow, and thermal responses for 17 body segments and the whole body [3] [4]. It derives from earlier multi-node models and uses a physiological network of tissue compartments and regulatory signals.

The model contains heat storage and transfer through body tissue, blood perfusion, metabolic production, respiratory loss, conduction, convection, radiation, and evaporation. Regulatory responses can include vasodilation, vasoconstriction, sweating, shivering, non-shivering thermogenesis, and changes related to activity or posture.

The model should be treated as a lumped or multi-node physiology model, not as a resolved vascular CFD model. A CFD field can provide local air temperature, air speed, humidity-related inputs, and radiative conditions, while JOS-3 returns segment-level skin temperatures and heat-loss signals.

## 4.3 Human heat balance

A simplified human heat balance is:

$$
M-W=Q_{sk}+Q_{res}+S,
$$

where $M$ is metabolic heat production, $W$ is external work, $Q_{sk}$ is total skin heat loss, $Q_{res}$ is respiratory heat loss, and $S$ is body heat storage. Skin heat loss can be decomposed into:

$$
Q_{sk}=Q_{conv}+Q_{rad}+Q_{cond}+Q_{evap}.
$$

The CFD model resolves or approximates convective transfer. Radiation may be modelled with a radiation solver or represented by a mean radiant temperature. Evaporation depends on humidity, clothing, skin wettedness, and vapour-pressure differences; it is not determined by air velocity alone.

## 4.4 Why local 17-zone data matter

A single mean body temperature hides local exposure. A person may have a hot face, cooled hands, an insulated torso, and asymmetric airflow at the same time. JOS-3 accepts local environmental and clothing values for 17 body segments. The FoamPilot geometry workflow creates corresponding surface patches and a `zone_mapping.csv` so that CFD results can be aggregated consistently.

The mapping must document:

- the exact body-part names and order;
- the surface patch name in the STL or OpenFOAM mesh;
- the area represented by each patch;
- whether a patch is exposed, clothed, or occluded;
- how local velocity, temperature, and radiation are averaged;
- the sign convention for heat flux;
- the temporal interpolation between CFD and physiology.

## 4.5 Convection laws around the body

The convective heat flux is often written:

$$
q''_{conv}=h_c(T_{skin}-T_a),
$$

where $h_c$ is a local convective heat-transfer coefficient and $T_a$ is air temperature. In a CFD coupling, $h_c$ may be estimated from the resolved wall heat flux:

$$
 h_c=\frac{q''_{conv}}{T_{skin}-T_a},
$$

or from an empirical correlation based on local velocity, characteristic length, and orientation. The CFD-derived route is more spatially resolved, but it depends on mesh quality, wall treatment, surface temperature boundary conditions, and turbulence modelling.

For a simplified correlation, the Nusselt number may depend on a Reynolds number and Prandtl number:

$$
Nu=\frac{h_cL}{k_a}=f(Re_L,Pr),
$$

with

$$
Re_L=\frac{U L}{\nu_a},
\qquad
Pr=\frac{\nu_a}{\alpha_a}.
$$

The choice between correlation and CFD should be explicit. Correlations are cheap and useful for preliminary design; CFD is useful when flow separation, recirculation, posture, clothing geometry, or spatial asymmetry matter.

## 4.6 Radiation and mean radiant temperature

Radiation is not equivalent to air temperature. A body can be in cool air while receiving strong long-wave or solar radiation from surrounding surfaces. A practical physiology coupling therefore supplies air temperature $T_a$, mean radiant temperature $T_r$, air speed $V_a$, relative humidity, clothing insulation, activity level, and posture.

If the CFD case does not solve radiation, use a documented $T_r$ input rather than silently setting $T_r=T_a$. If solar loading matters, distinguish short-wave solar absorption from long-wave exchange and record surface emissivity and absorptivity.

## 4.7 Data exchange between CFD and JOS-3

A robust coupling loop is:

```text
MakeHuman surface
→ surface cleanup and JOS-3 patch generation
→ CFD mesh and patch mapping
→ OpenFOAM temperature/velocity/radiation solution
→ area-weighted segment averages
→ JOS-3 physiological update
→ updated skin temperature or heat-flux boundary data
→ next CFD interval
```

The coupling can be one-way or two-way:

| Coupling | CFD receives | Physiology receives | Use |
| --- | --- | --- | --- |
| One-way | Fixed skin temperature or heat flux | CFD air conditions | Initial feasibility study. |
| Loose two-way | Updated segment skin temperature or flux | Local CFD temperature, speed, radiation, humidity proxy | Practical transient coupling. |
| Strong two-way | Iterated thermal boundary condition within each timestep | Converged local environmental state | Expensive research coupling. |

The timestep must resolve both CFD transients and physiological response. A physiology update every CFD iteration may be unnecessary; a very large coupling interval can miss rapid exposure changes. Test the coupling interval as a numerical parameter.

## 4.8 Thermoregulation validation

Validation should compare the physiological model and the surrounding CFD model separately before judging the coupled system. For the CFD side, validate velocity, temperature, wall flux, and mesh convergence. For the physiology side, verify baseline skin/core temperatures, metabolic response, sweating, blood flow, and expected responses to controlled thermal exposures.

A thermoregulation output such as mean skin temperature is a model prediction with physiological uncertainty. It should not be presented as a clinical diagnosis or as a validated human response without experimental comparison.

## References

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8450390/ "Liu et al., Comparison of Newtonian and Non-newtonian Fluid Models in Blood Flow Simulation"

[2]: https://www.openfoam.com/news/main-news/openfoam-v20-06/boundary-conditions "OpenFOAM: atmospheric boundary-layer boundary conditions"

[3]: https://github.com/TanabeLab/JOS-3 "TanabeLab/JOS-3: Joint system thermoregulation model"

[4]: https://doi.org/10.1016/j.enbuild.2020.110575 "Takahashi et al., Thermoregulation model JOS-3 with new open source code"
