import os
import httpx
import openai
import asyncio
import requests
from typing import Any
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

import logging

logger = logging.getLogger(__name__)


def get_health_base_url(base_url: str) -> str:
    if base_url.endswith("/v1"):
        return base_url.rstrip("/v1") + "/health"
    return f"{base_url}/health"


def is_model_healthy(
    model_name: str,
    timeout_seconds: float = 60.0,
    base_url: str = "https://llm.graphia-ssh.eu",
    api_key: str = os.getenv("PSNC_SMALL_TOOL_API_KEY"),
    checker_name: str = "generic",
) -> tuple[bool, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    try:
        # Do a minimal chat completion to confirm whether the model is usable.
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Hi!"}],
                "max_tokens": 1
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        if response.status_code == 200:
            logger.info(
                f"[{checker_name}] Health result: healthy (chat completion verification succeeded)"
            )
            return True, f"model {model_name} is healthy"

        logger.info(
            f"[{checker_name}] Health result: unhealthy (chat completion verification failed with status {response.status_code})"
        )
        return False, f"model {model_name} is unhealthy"
    except Exception as exc:
        logger.warning(f"[{checker_name}] Health probe failed with error: {exc}")
        return False, str(exc)


def create_llm(
    base_url: str,
    model: str,
    api_key: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int | None = None,
    extra_body: dict | None = None,
) -> ChatOpenAI:
    http_client = httpx.Client(verify=False)
    return ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body=extra_body,
        http_client=http_client,
    )


def create_openai_client(base_url: str, api_key: str) -> openai.OpenAI:
    return openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def update_runtime_llm(
    runtime_state: Any, llm: ChatOpenAI, llm_tools: list[Any] | None = None
) -> None:
    """Atomically update runtime llm and bound-tools llm."""
    runtime_state.llm = llm
    if llm_tools is not None:
        runtime_state.llm_with_tools = llm.bind_tools(llm_tools)


def update_runtime_tool_client(runtime_state: Any, client: openai.OpenAI, model_name: str) -> None:
    runtime_state.tool_openai_client = client
    runtime_state.tool_model = model_name


async def build_llm_from_health(
    base_url: str,
    api_key: str,
    model_name: str,
    fallback_base_url: str,
    fallback_api_key: str,
    fallback_model_name: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int | None = None,
    extra_body: dict | None = None,
) -> tuple[ChatOpenAI, bool, str]:
    """Build routed LLM (primary+fallback or fallback-only) from health status."""
    is_healthy, reason = await asyncio.to_thread(
        is_model_healthy,
        model_name=model_name,
        timeout_seconds=10.0,
        base_url="https://llm.hpc.psnc.pl/v1",
        api_key=os.getenv("PSNC_HPC_API_KEY"),
        checker_name="startup-main-build",
    )
    if is_healthy:
        llm = create_llm(
            base_url,
            model_name,
            api_key,
            temperature,
            top_p,
            max_tokens,
            extra_body,
        ).with_fallbacks(
            [
                create_llm(
                    fallback_base_url,
                    fallback_model_name,
                    fallback_api_key,
                    temperature,
                    top_p,
                    max_tokens,
                    extra_body,
                )
            ]
        )
    else:
        llm = create_llm(
            fallback_base_url,
            fallback_model_name,
            fallback_api_key,
            temperature,
            top_p,
            max_tokens,
            extra_body,
        )
    return llm, is_healthy, reason


async def build_tool_client_from_health(
    base_url: str,
    api_key: str,
    model_name: str,
    fallback_base_url: str,
    fallback_api_key: str,
    fallback_model_name: str,
) -> tuple[openai.OpenAI, str, bool, str]:
    """Build tool OpenAI client/model from health status."""
    is_healthy, reason = await asyncio.to_thread(
        is_model_healthy,
        model_name=model_name,
        timeout_seconds=10.0,
        checker_name="startup-tools-build",
    )
    if is_healthy:
        return create_openai_client(base_url, api_key), model_name, True, reason
    return (
        create_openai_client(fallback_base_url, fallback_api_key),
        fallback_model_name,
        False,
        reason,
    )


async def periodic_health_check(
    base_url: str,
    api_key: str,
    model_name: str,
    fallback_base_url: str,
    fallback_api_key: str,
    fallback_model_name: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int | None = None,
    extra_body: dict | None = None,
    runtime_state: Any = None,
    llm_tools: list[Any] | None = None,
    interval: int = 120,
):
    """Runs every `interval` seconds and updates runtime_state LLM based on health."""
    last_is_healthy, initial_reason = await asyncio.to_thread(
        is_model_healthy,
        model_name=model_name,
        timeout_seconds=10.0,
        base_url="https://llm.hpc.psnc.pl/v1",
        api_key=os.getenv("PSNC_HPC_API_KEY"),
        checker_name="periodic-main-initial",
    )
    logger.info(
        f"[periodic-main] Started periodic health checker for model='{model_name}' interval={interval}s "
        f"initial_status={last_is_healthy} detail={initial_reason}"
    )
    while True:
        await asyncio.sleep(interval)
        is_healthy, reason = await asyncio.to_thread(
            is_model_healthy,
            model_name=model_name,
            timeout_seconds=10.0,
            checker_name="periodic-main-loop",
        )
        logger.info(
            f"[periodic-main] Health check complete for model='{model_name}': status={is_healthy} detail={reason}",
        )
        if is_healthy == last_is_healthy:
            logger.info(
                f"[periodic-main] No model switch needed for model='{model_name}' "
                f"(status unchanged: {is_healthy})"
            )
            continue

        llm, _, _ = await build_llm_from_health(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            fallback_base_url=fallback_base_url,
            fallback_api_key=fallback_api_key,
            fallback_model_name=fallback_model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        if runtime_state is not None:
            update_runtime_llm(runtime_state, llm, llm_tools)

        last_is_healthy = is_healthy
        if is_healthy:
            logger.info("[periodic-main] Primary LLM recovered, switched back to primary")
        else:
            logger.warning(f"[periodic-main] Primary LLM down ({reason}), switching to fallback")


async def periodic_tool_health_check(
    base_url: str,
    api_key: str,
    model_name: str,
    fallback_base_url: str,
    fallback_api_key: str,
    fallback_model_name: str,
    runtime_state: Any = None,
    interval: int = 120,
):
    """Runs every `interval` seconds and updates runtime_state tool client based on health."""
    last_is_healthy, initial_reason = await asyncio.to_thread(
        is_model_healthy,
        model_name=model_name,
        timeout_seconds=60.0,
        checker_name="periodic-tools-initial",
    )
    logger.info(
        f"[periodic-tools] Started periodic health checker for model='{model_name}' interval={interval}s "
        f"initial_status={last_is_healthy} detail={initial_reason}"
    )
    while True:
        await asyncio.sleep(interval)
        is_healthy, reason = await asyncio.to_thread(
            is_model_healthy,
            model_name=model_name,
            timeout_seconds=60.0,
            checker_name="periodic-tools-loop",
        )
        logger.info(
            f"[periodic-tools] Health check complete for model='{model_name}': status={is_healthy} detail={reason}",
        )
        if is_healthy == last_is_healthy:
            logger.info(
                f"[periodic-tools] No model switch needed for model='{model_name}' "
                f"(status unchanged: {is_healthy})"
            )
            continue

        client, active_model, _, _ = await build_tool_client_from_health(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            fallback_base_url=fallback_base_url,
            fallback_api_key=fallback_api_key,
            fallback_model_name=fallback_model_name,
        )
        if runtime_state is not None:
            update_runtime_tool_client(runtime_state, client, active_model)

        last_is_healthy = is_healthy
        if is_healthy:
            logger.info("[periodic-tools] Primary tool model recovered, switched back to primary")
        else:
            logger.warning(f"[periodic-tools] Primary tool model down ({reason}), switching to fallback")
