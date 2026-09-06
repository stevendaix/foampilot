/* Compatibility adapter for OpenFOAM Foundation 13.
 * The old v12 custom wrapper duplicated distributedTriSurface and depended on
 * the removed triSurfaceMesh.H. Foundation 13 provides the maintained native
 * searchableSurfaces::distributedTriSurface implementation.
 */
#ifndef distributedTriSurfaceMeshBugFix_H
#define distributedTriSurfaceMeshBugFix_H

#include "distributedTriSurface.H"

namespace Foam
{
using distributedTriSurfaceMeshBugFix = searchableSurfaces::distributedTriSurface;
}

#endif
