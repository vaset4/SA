import json

import app.services.lightrag_domain_prompt as domain_prompt
import app.services.lightrag_service as lightrag_module
from app.services.graph.academic_schema import (
    ACADEMIC_ENTITY_EXTRACTION_SYSTEM_PROMPT,
    ACADEMIC_ENTITY_TYPES,
)


def test_install_domain_prompts_replaces_system_prompt(monkeypatch) -> None:
    prompts = {
        "entity_extraction_system_prompt": "original",
        "entity_extraction_examples": ["original example"],
    }
    monkeypatch.setattr(lightrag_module, "LIGHTRAG_PROMPTS", prompts)
    monkeypatch.setattr(domain_prompt, "ORIGINAL_DOMAIN_PROMPTS", {})
    monkeypatch.setattr(domain_prompt, "DOMAIN_PROMPTS_INSTALLED", False)
    monkeypatch.setattr(domain_prompt.settings, "lightrag_use_domain_prompt", True, raising=False)

    assert lightrag_module._install_domain_prompts() is True

    assert prompts["entity_extraction_system_prompt"] == ACADEMIC_ENTITY_EXTRACTION_SYSTEM_PROMPT
    assert prompts["entity_extraction_examples"][0].startswith("<Entity_types>")


def test_install_domain_prompts_disabled_restores_default(monkeypatch) -> None:
    prompts = {
        "entity_extraction_system_prompt": "patched",
        "entity_extraction_examples": ["patched"],
    }
    monkeypatch.setattr(lightrag_module, "LIGHTRAG_PROMPTS", prompts)
    monkeypatch.setattr(
        domain_prompt,
        "ORIGINAL_DOMAIN_PROMPTS",
        {
            "entity_extraction_system_prompt": "original",
            "entity_extraction_examples": ["original example"],
        },
    )
    monkeypatch.setattr(domain_prompt, "DOMAIN_PROMPTS_INSTALLED", True)
    monkeypatch.setattr(domain_prompt.settings, "lightrag_use_domain_prompt", False, raising=False)

    assert lightrag_module._install_domain_prompts() is False

    assert prompts["entity_extraction_system_prompt"] == "original"
    assert prompts["entity_extraction_examples"] == ["original example"]


def test_initialize_passes_academic_addon_params(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class FakeLightRAG:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        async def initialize_storages(self) -> None:
            return None

    async def fake_initialize_pipeline_status() -> None:
        return None

    monkeypatch.setattr(lightrag_module, "LightRAG", FakeLightRAG)
    monkeypatch.setattr(lightrag_module, "initialize_pipeline_status", fake_initialize_pipeline_status)
    monkeypatch.setattr(lightrag_module, "_LIGHTRAG_IMPORT_ERROR", None)
    monkeypatch.setattr(lightrag_module, "_install_domain_prompts", lambda: True)
    monkeypatch.setattr(lightrag_module.settings, "graph_rag_enabled", True, raising=False)
    monkeypatch.setattr(lightrag_module.settings, "data_dir", str(tmp_path), raising=False)

    service = lightrag_module.LightRAGService()

    import asyncio

    asyncio.run(service.initialize("lib_a"))

    assert captured["addon_params"] == {
        "language": "Chinese",
        "entity_types": list(ACADEMIC_ENTITY_TYPES),
    }


def test_reset_runtimes_clears_runtime_cache_and_flag(monkeypatch) -> None:
    service = lightrag_module.LightRAGService()
    service._get_runtime("lib_a")
    monkeypatch.setattr(domain_prompt, "DOMAIN_PROMPTS_INSTALLED", True)

    service.reset_runtimes("toggle domain prompt")

    assert service._runtimes == {}
    assert domain_prompt.DOMAIN_PROMPTS_INSTALLED is False


def test_lint_lightrag_entities_removes_non_academic_types(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(lightrag_module.settings, "data_dir", str(tmp_path), raising=False)
    service = lightrag_module.LightRAGService()
    runtime = service._get_runtime("lib_a")
    runtime.working_dir.mkdir(parents=True, exist_ok=True)
    entities_path = runtime.working_dir / "kv_store_full_entities.json"
    entities_path.write_text(
        json.dumps(
            {
                "method": {"entity_type": "Method", "entity_name": "STGCN"},
                "person": {"entity_type": "Person", "entity_name": "Tang"},
            }
        ),
        encoding="utf-8",
    )

    result = service.lint_lightrag_entities(library_id="lib_a")

    assert result["deleted"] == 1
    assert result["kept"] == 1
    assert result["by_type"] == {"Method": 1}
    assert "person" not in json.loads(entities_path.read_text(encoding="utf-8"))
