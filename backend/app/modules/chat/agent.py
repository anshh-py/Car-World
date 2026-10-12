"""Load the existing Gradio agent implementation without duplicating its logic."""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from types import ModuleType

from app.core.config import PROJECT_ROOT, settings

@lru_cache(maxsize=1)
def _load_gradio_agent_module() -> ModuleType:
    """Import app.py as a module; its Gradio UI only launches under __main__."""
    source = PROJECT_ROOT / "app.py"
    if not source.is_file():
        raise RuntimeError("The existing CarWorld agent app.py was not found.")
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    spec = importlib.util.spec_from_file_location("carworld_gradio_agent", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("The existing CarWorld agent could not be loaded.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def get_agent():
    """Return the same agent class and tools used by the Gradio application."""
    if not settings.groq_api_key:
        raise RuntimeError("The backend is missing GROQ_API_KEY configuration.")
    module = _load_gradio_agent_module()
    # app.py reads this process environment when constructing its shared client.
    return module.agent
