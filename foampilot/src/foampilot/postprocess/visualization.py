import os
import logging
import pyvista as pv

logger = logging.getLogger(__name__)


def detect_offscreen_mode() -> bool:
    if os.environ.get("SSH_CONNECTION"):
        return True
    if os.environ.get("DISPLAY") is None:
        return True
    if "WSL" in os.uname().release or "microsoft" in os.uname().release.lower():
        return True
    if os.environ.get("CI"):
        return True
    return False


def get_plotter_kwargs(off_screen: bool = None, **kwargs) -> dict:
    if off_screen is None:
        off_screen = detect_offscreen_mode()
    if off_screen:
        pv.OFF_SCREEN = True
    kwargs["off_screen"] = off_screen
    return kwargs


def create_plotter(off_screen: bool = None, **kwargs):
    kwargs = get_plotter_kwargs(off_screen=off_screen, **kwargs)
    return pv.Plotter(**kwargs)


def check_rendering_health() -> dict:
    result = {
        "off_screen": pv.OFF_SCREEN,
        "display": os.environ.get("DISPLAY", "not set"),
        "ssh": bool(os.environ.get("SSH_CONNECTION")),
        "wsl": "WSL" in os.uname().release or "microsoft" in os.uname().release.lower(),
        "render_window_type": "",
        "gl_capabilities": "",
        "gl_broken": False,
        "render_works": False,
        "recommendation": "",
    }

    try:
        pl = pv.Plotter(off_screen=True)
        result["render_window_type"] = type(pl.ren_win).__name__
        try:
            caps = pl.ren_win.ReportCapabilities()
            result["gl_capabilities"] = caps
            result["gl_broken"] = "display id not set" in caps.lower()
        except Exception:
            result["gl_broken"] = True

        mesh = pv.Sphere()
        pl.add_mesh(mesh)
        pl.ren_win.Render()
        img = pl.screenshot(return_img=True)
        result["render_works"] = img.max() > 0
        pl.close()
    except Exception as e:
        result["gl_capabilities"] = f"Error: {e}"
        result["gl_broken"] = True

    if result["gl_broken"]:
        result["recommendation"] = (
            "GLX context is broken. Use off_screen=True for all Plotter calls. "
            "Set pv.OFF_SCREEN = True globally or use create_plotter()."
        )
    elif result["render_window_type"] == "vtkXOpenGLRenderWindow" and not pv.OFF_SCREEN:
        result["recommendation"] = (
            "Using vtkXOpenGLRenderWindow without off_screen. "
            "In WSL2/SSH this may produce black screens. Use off_screen=True."
        )
    else:
        result["recommendation"] = "Rendering appears functional."

    return result