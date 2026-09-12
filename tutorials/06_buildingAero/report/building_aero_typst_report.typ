
#set document(title: "Building Aerodynamics Analysis", author: "FoamPilot")
#set page(paper: "a4", margin: 2.5cm, numbering: "1 / 1")
#set text(font: "New Computer Modern", size: 11pt, lang: "fr")
#set heading(numbering: "1.1.")
#set par(justify: true)
#show figure.caption: it => [
  #text(weight: "bold", size: 0.9em)[#it.supplement #it.counter.display():] #it.body
]


= Introduction
Urban boundary layer flow simulation around buildings using k-epsilon turbulence model with simpleFoam.

#figure($ u(y) = u_* \frac{\ln(y/y_0)}{\kappa} $, caption: [Logarithmic wind profile]) <eq:log_profile>

#figure(table(columns: 2, stroke: 0.5pt, inset: 7pt, align: center + horizon,
  table.header([* Parameter *], [* Value *]),
  [Parameter],
  [Value],
  [Re],
  [6.7e6],
  [AR],
  [1.0],
  [Cd],
  [1.2],
), caption: [Simulation parameters])

= Mesh Statistics


#figure(table(columns: 2, stroke: 0.5pt, inset: 7pt, align: center + horizon,
  [Statistic],
  [Value],
  [num_points],
  [11459],
  [num_cells],
  [12786],
  [bounds],
  [[0.0, 200.0, 0.0, 100.00399780273438, 0.0, 50.0]],
  [volume],
  [1000000.0684243309],
  [area],
  [None],
), caption: [Mesh quality metrics])

= Velocity Statistics


#figure(table(columns: 2, stroke: 0.5pt, inset: 7pt, align: center + horizon,
  [Statistic],
  [Value],
  [mean],
  [2.4451801776885986],
  [std],
  [4.262539863586426],
  [min],
  [-0.4652418792247772],
  [max],
  [12.615300178527832],
  [volume_weighted_mean],
  [[8.617223661765742, -5.279255594863886e-05, 0.000153282820345665]],
), caption: [Velocity field (U) statistics])