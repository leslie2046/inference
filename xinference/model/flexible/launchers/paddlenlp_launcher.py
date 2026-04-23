# Copyright 2022-2026 XProbe Inc.
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

import importlib
import os
from typing import Any, Dict, Optional

from ..core import FlexibleModel, FlexibleModelSpec


def _normalize_device_id(device: Optional[Any]) -> Optional[int]:
    if device is None:
        return None
    if isinstance(device, int):
        return device

    normalized = str(device).strip().lower()
    if normalized == "cpu":
        return -1
    if normalized in {"gpu", "cuda"}:
        return 0
    if ":" in normalized:
        _, _, suffix = normalized.rpartition(":")
        if suffix.isdigit():
            return int(suffix)
    if normalized.isdigit():
        return int(normalized)
    return None


def _normalize_id2label(id2label: Any) -> Any:
    if not isinstance(id2label, dict):
        return id2label

    normalized_id2label = {}
    for key, value in id2label.items():
        if isinstance(key, int):
            normalized_id2label[key] = value
        elif isinstance(key, str) and key.isdigit():
            normalized_id2label[int(key)] = value
        else:
            normalized_id2label[key] = value
    return normalized_id2label


class PaddleNLPTextClassificationModel(FlexibleModel):
    def load(self):
        try:
            Taskflow = getattr(importlib.import_module("paddlenlp"), "Taskflow")
        except ImportError as e:
            error_message = "Failed to import module 'paddlenlp'"
            installation_guide = [
                "Please make sure both 'paddlenlp' and a compatible 'paddlepaddle' package are installed. ",
                "For PaddleNLP text classification flexible models, refer to: ",
                "https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification",
            ]
            raise ImportError(
                f"{error_message}\n\n{''.join(installation_guide)}"
            ) from e
        except AttributeError as e:
            raise ImportError("Failed to import 'Taskflow' from 'paddlenlp'.") from e

        config: Dict[str, Any] = dict(self.config or {})
        config.pop("task", None)
        config.pop("device", None)
        config.pop("task_path", None)

        mode = config.pop("mode", None)
        if "id2label" in config:
            config["id2label"] = _normalize_id2label(config["id2label"])

        device_id = config.get("device_id")
        if device_id is None:
            normalized_device_id = _normalize_device_id(self._device)
            if normalized_device_id is None and self._device is not None:
                raise ValueError(f"Unsupported PaddleNLP device value: {self._device}")
            if normalized_device_id is not None:
                config["device_id"] = normalized_device_id

        self._taskflow = Taskflow(
            "text_classification",
            mode=mode,
            task_path=self._model_path,
            **config,
        )

    def infer(self, *args, **kwargs):
        return self._taskflow(*args, **kwargs)


def launcher(model_uid: str, model_spec: FlexibleModelSpec, **kwargs) -> FlexibleModel:
    task = kwargs.get("task")
    device = kwargs.get("device")

    model_path = model_spec.model_uri
    if model_path is None:
        raise ValueError("model_path required")
    if not os.path.isdir(model_path):
        raise ValueError(
            f"PaddleNLP launcher requires `model_uri` to be an existing local directory, got: {model_path}"
        )

    if task != "text_classification":
        raise ValueError(f"Unknown Task for PaddleNLP launcher: {task}")

    return PaddleNLPTextClassificationModel(
        model_uid=model_uid,
        model_path=model_path,
        model_family=model_spec,
        device=device,
        config=kwargs,
    )
