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

import os
import shutil
import tempfile
import types

import pytest


def test_register_flexible_model():
    from ..core import FlexibleModelSpec
    from ..custom import register_flexible_model, unregister_flexible_model

    tmp_dir = tempfile.mkdtemp()

    model_spec = FlexibleModelSpec(
        model_name="flexible_model",
        model_uri=os.path.abspath(tmp_dir),
        launcher="xinference.model.flexible.launchers.transformers",
    )

    register_flexible_model(model_spec, persist=False)

    unregister_flexible_model("flexible_model")

    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_model():
    from ..core import FlexibleModelSpec
    from ..utils import get_launcher

    launcher = get_launcher("xinference.model.flexible.launchers.transformers")
    model = launcher(
        model_uid="flexible_model",
        model_spec=FlexibleModelSpec(
            model_name="mock",
            model_uri="mock",
            launcher="xinference.model.flexible.launchers.transformers",
        ),
        task="mock",
    )

    model.load()

    result = model.infer(inputs="hello world")
    # assert result == {"inputs": "hello world"}
    assert result is not None
    assert "inputs" in result
    assert result["inputs"] == "hello world"


def test_paddlenlp_text_classification_model(monkeypatch: pytest.MonkeyPatch):
    from ..core import FlexibleModelSpec
    from ..utils import get_launcher

    taskflow_calls = {}

    class FakeTaskflow:
        def __init__(self, task_name, mode=None, **kwargs):
            taskflow_calls["task_name"] = task_name
            taskflow_calls["mode"] = mode
            taskflow_calls["kwargs"] = kwargs

        def __call__(self, *args, **kwargs):
            return {
                "args": list(args),
                "kwargs": kwargs,
                "task_name": taskflow_calls["task_name"],
                "mode": taskflow_calls["mode"],
                "task_path": taskflow_calls["kwargs"]["task_path"],
            }

    monkeypatch.setitem(
        __import__("sys").modules,
        "paddlenlp",
        types.SimpleNamespace(Taskflow=FakeTaskflow),
    )

    tmp_dir = tempfile.mkdtemp()
    try:
        launcher = get_launcher("xinference.model.flexible.launchers.paddlenlp")
        model = launcher(
            model_uid="flexible_model",
            model_spec=FlexibleModelSpec(
                model_name="mock_paddlenlp",
                model_uri=os.path.abspath(tmp_dir),
                launcher="xinference.model.flexible.launchers.paddlenlp",
            ),
            task="text_classification",
            mode="finetune",
            is_static_model=True,
            problem_type="multi_class",
            batch_size=8,
            id2label={"0": "negative", "1": "positive"},
            device="cpu",
        )

        model.load()
        result = model.infer(["hello world"])

        assert taskflow_calls["task_name"] == "text_classification"
        assert taskflow_calls["mode"] == "finetune"
        assert taskflow_calls["kwargs"]["task_path"] == os.path.abspath(tmp_dir)
        assert taskflow_calls["kwargs"]["is_static_model"] is True
        assert taskflow_calls["kwargs"]["problem_type"] == "multi_class"
        assert taskflow_calls["kwargs"]["batch_size"] == 8
        assert taskflow_calls["kwargs"]["id2label"] == {
            0: "negative",
            1: "positive",
        }
        assert taskflow_calls["kwargs"]["device_id"] == -1
        assert result["task_name"] == "text_classification"
        assert result["task_path"] == os.path.abspath(tmp_dir)
        assert result["args"] == [["hello world"]]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_paddlenlp_text_classification_model_invalid_device():
    from ..core import FlexibleModelSpec
    from ..utils import get_launcher

    tmp_dir = tempfile.mkdtemp()
    try:
        launcher = get_launcher("xinference.model.flexible.launchers.paddlenlp")
        model = launcher(
            model_uid="flexible_model",
            model_spec=FlexibleModelSpec(
                model_name="mock_paddlenlp",
                model_uri=os.path.abspath(tmp_dir),
                launcher="xinference.model.flexible.launchers.paddlenlp",
            ),
            task="text_classification",
            device="mps",
        )

        with pytest.raises(ValueError, match="Unsupported PaddleNLP device value"):
            model.load()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_paddlenlp_text_classification_model_without_device(
    monkeypatch: pytest.MonkeyPatch,
):
    from ..core import FlexibleModelSpec
    from ..utils import get_launcher

    taskflow_calls = {}

    class FakeTaskflow:
        def __init__(self, task_name, mode=None, **kwargs):
            taskflow_calls["task_name"] = task_name
            taskflow_calls["mode"] = mode
            taskflow_calls["kwargs"] = kwargs

        def __call__(self, *args, **kwargs):
            return {
                "args": list(args),
                "kwargs": kwargs,
            }

    monkeypatch.setitem(
        __import__("sys").modules,
        "paddlenlp",
        types.SimpleNamespace(Taskflow=FakeTaskflow),
    )

    tmp_dir = tempfile.mkdtemp()
    try:
        launcher = get_launcher("xinference.model.flexible.launchers.paddlenlp")
        model = launcher(
            model_uid="flexible_model",
            model_spec=FlexibleModelSpec(
                model_name="mock_paddlenlp",
                model_uri=os.path.abspath(tmp_dir),
                launcher="xinference.model.flexible.launchers.paddlenlp",
            ),
            task="text_classification",
        )

        model.load()

        assert taskflow_calls["task_name"] == "text_classification"
        assert "device_id" not in taskflow_calls["kwargs"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_paddlenlp_text_classification_model_requires_local_directory():
    from ..core import FlexibleModelSpec
    from ..utils import get_launcher

    launcher = get_launcher("xinference.model.flexible.launchers.paddlenlp")

    with pytest.raises(
        ValueError,
        match="PaddleNLP launcher requires `model_uri` to be an existing local directory",
    ):
        launcher(
            model_uid="flexible_model",
            model_spec=FlexibleModelSpec(
                model_name="mock_paddlenlp",
                model_uri=os.path.abspath("missing_dir_for_paddlenlp_launcher"),
                launcher="xinference.model.flexible.launchers.paddlenlp",
            ),
            task="text_classification",
        )
