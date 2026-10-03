"""Skill de conversación: se carga en el system prompt y los saludos llegan al LLM."""

import uuid

import pytest

from conftest import FakeLLM, make_agent, text_response
from tiendahogar_agent.adapters.outbound.skills.filesystem_skill_repository import FileSystemSkillRepository
from tiendahogar_agent.application.prompts import NO_CONTEXT_MESSAGE
from tiendahogar_agent.config import settings


def test_skill_is_loaded_without_frontmatter():
    skill = FileSystemSkillRepository(settings.skills_dir).get("atencion_al_cliente")
    assert not skill.startswith("---")
    assert "Empatía" in skill


def test_skill_is_injected_after_safety_rules(retriever):
    llm = FakeLLM([text_response("¡Hola! ¿En qué te ayudo?")])
    make_agent(llm, retriever).ask("hola", session_id=str(uuid.uuid4()))
    system = llm.systems[0]
    assert "Escalamiento obligatorio" in system  # reglas de seguridad siguen presentes
    assert "<reglas_de_conversacion>" in system and "Empatía" in system
    assert system.index("Escalamiento obligatorio") < system.index("<reglas_de_conversacion>")


@pytest.mark.parametrize("message", ["hola", "Buenos días", "¡Muchas gracias!", "adiós, hasta luego"])
def test_small_talk_goes_to_llm_instead_of_no_info_template(retriever, message):
    llm = FakeLLM([text_response("¡Hola! Con gusto te ayudo.")])
    result = make_agent(llm, retriever).ask(message, session_id=str(uuid.uuid4()))
    assert len(llm.calls) == 1
    assert result["answer"] != NO_CONTEXT_MESSAGE


def test_long_off_topic_message_starting_with_greeting_is_not_small_talk(retriever):
    llm = FakeLLM([])
    result = make_agent(llm, retriever).ask(
        "hola, recomiéndame una receta de pastel de chocolate con fresas para el domingo",
        session_id=str(uuid.uuid4()),
    )
    assert result["answer"] == NO_CONTEXT_MESSAGE and llm.calls == []
