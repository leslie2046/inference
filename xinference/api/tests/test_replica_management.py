# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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
# CI: linux

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import APIRouter, FastAPI, HTTPException

from ...core.exceptions import ModelReloadInProgress
from ..restful_api import RESTfulAPI
from ..routers import models


@pytest.fixture
def replica_api():
    api = RESTfulAPI.__new__(RESTfulAPI)
    api._uid_to_model_name = {}
    api._app = SimpleNamespace(router=SimpleNamespace(routes=[]))
    return api


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [None, {"replica": 2}])
async def test_add_model_replica_route_accepts_optional_body(replica_api, payload):
    replica = payload["replica"] if payload else 1
    replicas = [{"replica_id": replica_id} for replica_id in range(replica)]
    supervisor = AsyncMock()
    supervisor.add_model_replicas_with_config.return_value = replicas
    replica_api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    replica_api._app = FastAPI()
    replica_api._router = APIRouter()
    replica_api._auth_service = None
    replica_api._advanced_auth_service = None

    models.register_routes(replica_api)
    replica_api._app.include_router(replica_api._router)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=replica_api._app), base_url="http://test"
    ) as client:
        request_kwargs = {"json": payload} if payload is not None else {}
        response = await client.post("/v1/models/demo/replicas", **request_kwargs)

    assert response.status_code == 200
    assert response.json() == (
        replicas[0] if replica == 1 else {"replica": replica, "replicas": replicas}
    )
    supervisor.add_model_replicas_with_config.assert_awaited_once_with(
        model_uid="demo",
        replica=replica,
        replica_configs=None,
        model_engine=None,
        n_gpu=None,
    )


@pytest.mark.asyncio
async def test_add_model_replica_accepts_empty_body(replica_api):
    supervisor = AsyncMock()
    supervisor.add_model_replicas_with_config.return_value = [
        {
            "replica_id": 1,
            "replica_model_uid": "demo-rep1",
            "worker_address": "127.0.0.1:9978",
        }
    ]
    replica_api._get_supervisor_ref = AsyncMock(return_value=supervisor)

    response = await replica_api.add_model_replica("demo")

    assert response.status_code == 200
    assert (
        json.loads(response.body)
        == supervisor.add_model_replicas_with_config.return_value[0]
    )
    supervisor.add_model_replicas_with_config.assert_awaited_once_with(
        model_uid="demo",
        replica=1,
        replica_configs=None,
        model_engine=None,
        n_gpu=None,
    )


@pytest.mark.asyncio
async def test_add_model_replicas_forwards_count_engine_and_device(replica_api):
    supervisor = AsyncMock()
    supervisor.add_model_replicas_with_config.return_value = [
        {
            "replica_id": replica_id,
            "replica_model_uid": f"demo-rep{replica_id}",
            "worker_address": "127.0.0.1:9978",
        }
        for replica_id in (1, 2)
    ]
    replica_api._get_supervisor_ref = AsyncMock(return_value=supervisor)

    response = await replica_api.add_model_replica(
        "demo",
        {
            "replica": 2,
            "model_engine": "vllm",
            "n_gpu": 1,
        },
    )

    assert json.loads(response.body) == {
        "replica": 2,
        "replicas": supervisor.add_model_replicas_with_config.return_value,
    }
    supervisor.add_model_replicas_with_config.assert_awaited_once_with(
        model_uid="demo",
        replica=2,
        replica_configs=None,
        model_engine="vllm",
        n_gpu=1,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"replica_config": []},
        {"replica_config": {"devices": []}},
        {"replica_config": {"devices": [{"gpu_idx": [0]}]}},
        {"replica": 0},
        {"replica": True},
        {"model_engine": "  "},
        {"n_gpu": -1},
        {"replica": 2, "replica_config": [{"devices": []}]},
    ],
)
async def test_add_model_replica_rejects_invalid_config(replica_api, payload):
    replica_api._get_supervisor_ref = AsyncMock()

    with pytest.raises(HTTPException) as exc_info:
        await replica_api.add_model_replica("demo", payload)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_terminate_last_replica_evicts_api_cache(replica_api, monkeypatch):
    supervisor = AsyncMock()
    supervisor.terminate_model_replica.return_value = 0
    replica_api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    replica_api._uid_to_model_name["demo"] = "demo-model"
    evict_model_cache = Mock()
    monkeypatch.setattr(
        "xinference.api.oauth2.advanced.audit.evict_model_cache", evict_model_cache
    )

    response = await replica_api.terminate_model_replica("demo", 0)

    assert json.loads(response.body) == {"remaining_replicas": 0}
    assert "demo" not in replica_api._uid_to_model_name
    evict_model_cache.assert_called_once_with("demo")


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["add", "remove"])
async def test_resize_during_reload_returns_conflict(replica_api, operation):
    supervisor = AsyncMock()
    supervisor.add_model_replicas.side_effect = ModelReloadInProgress(
        "reload in progress"
    )
    supervisor.terminate_model_replica.side_effect = ModelReloadInProgress(
        "reload in progress"
    )
    replica_api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    with pytest.raises(HTTPException) as exc:
        if operation == "add":
            await replica_api.add_model_replica("demo")
        else:
            await replica_api.terminate_model_replica("demo", 0)
    assert exc.value.status_code == 409
