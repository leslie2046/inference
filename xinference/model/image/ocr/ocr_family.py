# Copyright 2022-2025 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OCR model engine configuration and selection.

This module provides engine configuration for OCR models, following the same
pattern as embedding and rerank models.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

if TYPE_CHECKING:
    from ..core import ImageModelFamilyV2

logger = logging.getLogger(__name__)


# OCR model engine classes
TRANSFORMERS_OCR_CLASSES: List[Type[Any]] = []
VLLM_OCR_CLASSES: List[Type[Any]] = []

# { ocr model name -> { engine name -> engine params } }
OCR_ENGINES: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

# { engine name -> [engine classes] }
OCR_SUPPORTED_ENGINES: Dict[str, List[Type[Any]]] = {}


def check_engine_by_model_name_and_engine(
    model_engine: str,
    model_name: str,
    model_format: Optional[str] = None,
) -> Type[Any]:
    """Find the appropriate OCR model class for the given engine and model."""

    def get_model_engine_from_spell(engine_str: str) -> str:
        """Case-insensitive engine name lookup."""
        if model_name not in OCR_ENGINES:
            return engine_str
        for engine in OCR_ENGINES[model_name].keys():
            if engine.lower() == engine_str.lower():
                return engine
        return engine_str

    if model_name not in OCR_ENGINES:
        raise ValueError(f"OCR model {model_name} not found in engines.")

    model_engine = get_model_engine_from_spell(model_engine)

    if model_engine not in OCR_ENGINES[model_name]:
        raise ValueError(
            f"OCR model {model_name} cannot be run on engine {model_engine}."
        )

    match_params = OCR_ENGINES[model_name][model_engine]
    for param in match_params:
        if model_format and model_format != param.get("model_format"):
            continue
        return param["ocr_class"]

    raise ValueError(f"OCR model {model_name} cannot be run on engine {model_engine}.")


def generate_engine_config_by_model_family(model_family: "ImageModelFamilyV2") -> None:
    """Generate engine configuration for an OCR model family.

    This follows the same pattern as embedding/rerank models.
    """
    model_name = model_family.model_name
    model_ability = getattr(model_family, "model_ability", []) or []

    # Only generate engine config for OCR models
    if "ocr" not in model_ability:
        return

    engines: Dict[str, List[Dict[str, Any]]] = OCR_ENGINES.get(model_name, {})

    for engine_name, engine_classes in OCR_SUPPORTED_ENGINES.items():
        for cls in engine_classes:
            # Check if the class has a match method
            match_func = getattr(cls, "match", None)
            if callable(match_func):
                if match_func(model_family):
                    if engine_name not in engines:
                        engines[engine_name] = []

                    engines[engine_name].append(
                        {
                            "model_name": model_name,
                            "model_format": "pytorch",
                            "ocr_class": cls,
                        }
                    )
                    # Only match the first class for each engine
                    break

    if engines:
        OCR_ENGINES[model_name] = engines


def register_ocr_engines() -> None:
    """Register all OCR engine classes.

    This is called during module initialization.
    """
    from .paddleocr_vl import PaddleOCRVLModel
    from .paddleocr_vl_vllm import PaddleOCRVLVLLMModel

    # Clear and re-register
    TRANSFORMERS_OCR_CLASSES.clear()
    VLLM_OCR_CLASSES.clear()

    TRANSFORMERS_OCR_CLASSES.append(PaddleOCRVLModel)
    VLLM_OCR_CLASSES.append(PaddleOCRVLVLLMModel)

    OCR_SUPPORTED_ENGINES["Transformers"] = TRANSFORMERS_OCR_CLASSES
    OCR_SUPPORTED_ENGINES["vLLM"] = VLLM_OCR_CLASSES


def init_ocr_engines(
    builtin_image_models: Dict[str, List["ImageModelFamilyV2"]]
) -> None:
    """Initialize OCR engines for all built-in image models.

    Args:
        builtin_image_models: Dictionary of image model families
    """
    register_ocr_engines()

    for model_name, model_family_list in builtin_image_models.items():
        for model_family in model_family_list:
            generate_engine_config_by_model_family(model_family)
