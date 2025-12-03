from foampilot.base.openFOAMFile import OpenFOAMFile

class DecomposeParDictFile(OpenFOAMFile):
    """
    Handler for the decomposeParDict dictionary.
    """

    def __init__(self, parent=None, nb_proc: int = 1):
        super().__init__(
            parent=parent,
            object_name="decomposeParDict",
            location="system",
            class_name="dictionary",
        )

        # Default dictionary for parallel runs
        self.data = {
            "numberOfSubdomains": nb_proc,
            "method": "scotch",
        }

    def set_nb_proc(self, nb_proc: int):
        self.data["numberOfSubdomains"] = nb_proc