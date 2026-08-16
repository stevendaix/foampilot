import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class vmtkBaseScript:
    def __init__(self):
        self._input_members: List[Dict[str, Any]] = []
        self._output_members: List[Dict[str, Any]] = []

    def SetInputMembers(self, members: List[Dict[str, Any]]):
        self._input_members = members

    def SetOutputMembers(self, members: List[Dict[str, Any]]):
        self._output_members = members

    def SetScriptName(self, name: str):
        self._script_name = name

    def SetScriptDoc(self, doc: str):
        self._script_doc = doc

    def Execute(self):
        raise NotImplementedError

    def PrintLog(self, msg: str):
        logger.info(msg)

    def PrintError(self, msg: str):
        logger.error(msg)

    def InputInfo(self, msg: str):
        logger.info(msg)

    def InputText(self, prompt: str) -> str:
        return input(prompt)

    def OutputText(self, msg: str):
        logger.info(msg)

    def OutputProgress(self, progress: float, width: int = 10):
        bar_len = int(width * progress / 100.0)
        bar = "=" * bar_len + "-" * (width - bar_len)
        print(f"[{bar}] {progress:.1f}%")
