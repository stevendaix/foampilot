# OpenFOAM 13 floating sixDoF plugin

This plugin is a port of the `mooringLine` restraint from the educational
`thesis-FloatingTurbine` repository. It deliberately links against the native
OpenFOAM 13 `libsixDoFRigidBodyMotion` library instead of copying the old
v2012 sixDoF implementation. This avoids the incompatible v2012 dictionary
and `autoPtr` APIs while preserving the catenary restraint physics.

Build after sourcing `/opt/openfoam13/etc/bashrc`:

```bash
wmake
```

The resulting library is `libfloatingSixDoFRigidBodyMotion.so`. Load it in
`dynamicMeshDict` after the native motion-solver library. The original source
license and provenance are retained in the copied source headers.
