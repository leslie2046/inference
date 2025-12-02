# Copyright 2022-2024 XProbe Inc.
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

import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional

from ..._compat import Literal
from ..core import CacheableModelSpec, VirtualEnvSettings
from ..utils import ModelInstanceInfoMixin
from .utils import get_launcher

logger = logging.getLogger(__name__)


class FlexibleModelSpec(CacheableModelSpec, ModelInstanceInfoMixin):
    version: Literal[1, 2] = 2
    model_id: Optional[str]  # type: ignore
    model_description: Optional[str]
    model_uri: Optional[str]
    launcher: str
    launcher_args: Optional[str]
    virtualenv: Optional[VirtualEnvSettings]

    def parser_args(self):
        return json.loads(self.launcher_args)

    class Config:
        extra = "allow"

    def to_description(self):
        return {
            "model_type": "flexible",
            "address": getattr(self, "address", None),
            "accelerators": getattr(self, "accelerators", None),
            "model_name": self.model_name,
            "launcher": self.launcher,
            "launcher_args": self.launcher_args,
        }

    def to_version_info(self):
        return {
            "model_version": self.model_name,
            "cache_status": True,
            "model_file_location": self.model_uri,
            "launcher": self.launcher,
            "launcher_args": self.launcher_args,
        }

    def validate_model(self):
        """Validate the flexible model configuration."""
        # Check that launcher is provided
        if not self.launcher:
            raise ValueError(
                f"You must specify `launcher` when registering flexible models."
            )
        # Check that launcher_args can be parsed if provided
        if self.launcher_args:
            try:
                self.parser_args()
            except Exception as e:
                raise ValueError(
                    f"Invalid launcher args {self.launcher_args}: {str(e)}"
                )


def generate_flexible_model_description(
    model_spec: FlexibleModelSpec,
) -> Dict[str, List[Dict]]:
    res = defaultdict(list)
    res[model_spec.model_name].append(model_spec.to_version_info())
    return res


FLEXIBLE_MODELS: List[FlexibleModelSpec] = []
FLEXIBLE_MODEL_DESCRIPTIONS: Dict[str, List[Dict]] = defaultdict(list)


def get_flexible_model_descriptions():
    import copy

    return copy.deepcopy(FLEXIBLE_MODEL_DESCRIPTIONS)


class FlexibleModel:
    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_family: FlexibleModelSpec,
        device: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        self.model_family = model_family
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        self._config = config

    def load(self):
        """
        Load the model.
        """

    def infer(self, *args, **kwargs):
        """
        Call model to inference.
        """
        raise NotImplementedError("infer method not implemented.")

    @property
    def model_uid(self):
        return self._model_uid

    @property
    def model_path(self):
        return self._model_path

    @property
    def device(self):
        return self._device

    @property
    def config(self):
        return self._config


def match_flexible_model(model_name):
    from .custom import get_flexible_models

    for model_spec in get_flexible_models():
        if model_name == model_spec.model_name:
            return model_spec
    return None


def create_flexible_model_instance(
    model_uid: str,
    model_name: str,
    model_path: Optional[str] = None,
    model_id: Optional[str] = None,
    model_source: Optional[str] = None,
    **kwargs,
) -> FlexibleModel:
    model_spec = match_flexible_model(model_name)
    if model_spec is None:
        raise ValueError(f"Flexible model {model_name} not found")
    
    # Override model_id and model_hub if provided
    if model_id is not None:
        model_spec.model_id = model_id
    if model_source is not None:
        model_spec.model_hub = model_source
    
    if not model_path:
        model_path = model_spec.model_uri
    
    # If model_path is still None, try to download the model using CacheManager
    if not model_path:
        if model_spec.model_id:
            logger.info(f"Model path not found, trying to download model {model_spec.model_name} from {model_spec.model_hub}...")
            from ..cache_manager import CacheManager
            cache_manager = CacheManager(model_spec)
            model_path = cache_manager.cache()
            if model_path:
                logger.info(f"Model downloaded to {model_path}")
                # Update model_uri in model_spec so that launcher can use it
                model_spec.model_uri = model_path
            else:
                logger.error(f"Failed to download model {model_spec.model_name}")
        else:
            logger.warning(f"Model {model_spec.model_name} has no model_uri and no model_id, cannot download.")

    launcher_name = model_spec.launcher
    launcher_args = model_spec.parser_args()
    kwargs.update(launcher_args)

    # Pass model_path explicitly if possible, but launcher usually reads from spec
    # Some launchers might look at kwargs, so we update it just in case, 
    # though modelscope_launcher reads from spec.
    if model_path:
        kwargs['model_path'] = model_path

    model = get_launcher(launcher_name)(
        model_uid=model_uid, model_spec=model_spec, **kwargs
    )

    return model
