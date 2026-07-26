from typing import Dict, Optional, Any, List
from foampilot.base.openFOAMFile import OpenFOAMFile


class FvConstraintsFile(OpenFOAMFile):
    """
    OpenFOAM fvConstraints file for embedded boundary and constraint support.

    Supports constraint types such as:
    - pointConstraint: fixes points in space
    - faceConstraint: constrains face motion
    - fvConstraint: generic constraint entry
    """

    def __init__(
        self,
        parent: Optional[Any] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.parent = parent
        self.constraints = constraints or []

        super().__init__(object_name="fvConstraints")

    def add_constraint(
        self,
        name: str,
        constraint_type: str,
        patch: Optional[str] = None,
        **coeffs: Any,
    ) -> None:
        """Add a constraint entry.

        Args:
            name: Name of the constraint entry.
            constraint_type: Type of constraint (pointConstraint, faceConstraint, etc.)
            patch: Optional patch name this constraint applies to.
            **coeffs: Additional coefficients for the constraint.
        """
        entry: Dict[str, Any] = {"type": constraint_type}
        if patch is not None:
            entry["patch"] = patch
        entry.update(coeffs)
        self.constraints.append((name, entry))

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for name, entry in self.constraints:
            result[name] = entry
        return result

    def write(self, filepath: str) -> None:
        self.attributes = self.to_dict()
        self.write_file(filepath)

    @classmethod
    def from_dict(cls, config: Dict[str, Any], parent: Optional[Any] = None) -> "FvConstraintsFile":
        constraints = []
        for name, entry in config.items():
            constraints.append((name, entry))
        return cls(parent=parent, constraints=constraints)