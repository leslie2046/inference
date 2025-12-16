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

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import PIL.Image

if TYPE_CHECKING:
    from ..core import ImageModelFamilyV2

logger = logging.getLogger(__name__)


class PaddleOCRVLVLLMModel:
    """PaddleOCR-VL model using vLLM backend for OCR, table recognition, formula recognition, and chart recognition."""

    def __init__(
        self,
        model_uid: str,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        model_spec: Optional["ImageModelFamilyV2"] = None,
        **kwargs,
    ):
        self.model_family = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        # model info when loading
        self._model = None
        self._processor = None
        # info
        self._model_spec = model_spec
        self._abilities = model_spec.model_ability or []  # type: ignore
        self._kwargs = kwargs

    @property
    def model_ability(self):
        return self._abilities

    @staticmethod
    def match(model_family: "ImageModelFamilyV2") -> bool:
        """Check if this engine class matches the given model family.

        Args:
            model_family: The image model family to check.

        Returns:
            True if this engine supports the model family.
        """
        model_name = getattr(model_family, "model_name", "")
        model_ability = getattr(model_family, "model_ability", []) or []

        # PaddleOCRVLVLLMModel (vLLM backend) supports PaddleOCR-VL
        return model_name == "PaddleOCR-VL" and "ocr" in model_ability

    def load(self):
        try:
            from vllm import LLM
        except ImportError:
            error_message = "Failed to import module 'vllm'"
            installation_guide = [
                "Please make sure 'vllm' is installed. ",
                "You can install it by `pip install vllm`\n",
            ]
            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        from transformers import AutoProcessor

        logger.info(f"Loading PaddleOCR-VL model with vLLM from {self._model_path}")

        try:
            # Load processor for chat template
            self._processor = AutoProcessor.from_pretrained(
                self._model_path, trust_remote_code=True
            )

            # Prepare vLLM kwargs
            vllm_kwargs = self._kwargs.copy()
            # Remove non-vLLM kwargs
            for key in ["model_engine", "model_format"]:
                vllm_kwargs.pop(key, None)

            # Set default values for vLLM
            if "trust_remote_code" not in vllm_kwargs:
                vllm_kwargs["trust_remote_code"] = True
            if "dtype" not in vllm_kwargs:
                vllm_kwargs["dtype"] = "bfloat16"

            # Load model with vLLM
            self._model = LLM(model=self._model_path, **vllm_kwargs)

            logger.info("PaddleOCR-VL model loaded successfully with vLLM")
        except Exception as e:
            logger.error(f"Failed to load PaddleOCR-VL model with vLLM: {e}")
            raise

    def ocr(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        **kwargs,
    ) -> Union[str, List[str], Dict[str, Any]]:
        """
        Perform OCR, table recognition, formula recognition, or chart recognition.

        Args:
            image: PIL Image or list of PIL Images
            **kwargs: Additional parameters including:
                - task: Task type ('ocr', 'table', 'formula', 'chart'), default: 'ocr'
                - prompt: Custom prompt (optional, overrides task-based prompt)
                - max_new_tokens: Maximum number of tokens to generate (default: 1024)
                - return_dict: Whether to return a dictionary with metadata (default: False)

        Returns:
            OCR results as string, list of strings, or dict
        """
        logger.info("PaddleOCR-VL vLLM kwargs: %s", kwargs)

        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded. Please call load() first.")

        # Extract parameters
        task = kwargs.get("task", "ocr")
        custom_prompt = kwargs.get("prompt", None)
        max_new_tokens = kwargs.get("max_new_tokens", 1024)
        return_dict = kwargs.get("return_dict", False)

        # Define task prompts
        PROMPTS = {
            "ocr": "OCR:",
            "table": "Table Recognition:",
            "formula": "Formula Recognition:",
            "chart": "Chart Recognition:",
        }

        # Use custom prompt if provided, otherwise use task-based prompt
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = PROMPTS.get(task, PROMPTS["ocr"])

        # Handle single image input
        if isinstance(image, PIL.Image.Image):
            result = self._process_single(image, prompt, max_new_tokens)
            if return_dict:
                return {
                    "text": result,
                    "model": "paddleocr-vl-vllm",
                    "task": task,
                    "success": True,
                }
            return result

        # Handle batch image input
        elif isinstance(image, list):
            results = [
                self._process_single(img, prompt, max_new_tokens) for img in image
            ]
            if return_dict:
                return {
                    "text": results,
                    "model": "paddleocr-vl-vllm",
                    "task": task,
                    "success": True,
                    "num_images": len(results),
                }
            return results

        else:
            raise ValueError("Input must be a PIL Image or list of PIL Images")

    def _process_single(
        self, image: PIL.Image.Image, prompt: str, max_new_tokens: int
    ) -> str:
        """Process a single image with the given prompt using vLLM."""
        from vllm import SamplingParams

        # Ensure model and processor are loaded
        assert self._model is not None, "Model not loaded. Call load() first."
        assert self._processor is not None, "Processor not loaded. Call load() first."

        # Convert image to RGB if needed
        if image.mode in ["RGBA", "CMYK"]:
            image = image.convert("RGB")

        # Prepare messages in the format expected by PaddleOCR-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template to get the text prompt
        text_prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Prepare sampling params
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
        )

        # Generate using vLLM
        outputs = self._model.generate(
            {
                "prompt": text_prompt,
                "multi_modal_data": {"image": image},
            },
            sampling_params=sampling_params,
        )

        # Extract generated text
        if outputs and len(outputs) > 0:
            result = outputs[0].outputs[0].text
        else:
            result = ""

        return result
