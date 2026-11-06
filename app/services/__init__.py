"""Service layer: domain logic between the REST routes and the data layer.

Re-exports are resolved lazily (PEP 562). Eager imports here would make
`app.serializers` -> `app.services.feedback` pull in `evaluation_service` ->
`app.models` at a point where `app.models` is still initialising.
"""

from importlib import import_module
from typing import Any

# NOTE: there is deliberately no `if TYPE_CHECKING:` re-export block here. The
# runtime `__getattr__` below plus `__all__` are the public surface; adding
# eager type-checker imports re-creates the circular-import problem this module
# exists to avoid, and linters reject the redundant-alias workaround outside
# `__init__.py`-style stub files.

_EXPORTS = {
    "resolve_rubric": "app.services.concept_service",
    "EvaluationService": "app.services.evaluation_service",
    "generate_feedback": "app.services.feedback",
    "improvement_tip": "app.services.feedback",
    "get_ocr_service": "app.services.ocr_service",
    "build_recommendations": "app.services.recommendation_service",
    "get_mastery_profile": "app.services.recommendation_service",
    "strength_of": "app.services.recommendation_service",
    "update_mastery": "app.services.recommendation_service",
    "RubricEngine": "app.services.rubric_service",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_path), name)


def __dir__():
    return sorted(__all__)