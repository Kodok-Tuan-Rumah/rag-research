"""
LLM Fallback Client - Multi-provider sequential fallback

This client provides automatic fallback between multiple LLM providers
when the primary provider fails due to rate limits, timeouts, or errors.

Usage:
    from src.llm.fallback_client import LLMFallbackClient, Provider

    client = LLMFallbackClient()
    response = client.generate("Your prompt here")
"""

import os
import time
from typing import Optional, List, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Provider(Enum):
    """Supported LLM providers"""
    GROQ = "groq"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class LLMProviderError(Exception):
    """Base exception for LLM provider errors"""
    pass


class RateLimitError(LLMProviderError):
    """Provider returned 429 rate limit"""
    pass


class TimeoutError(LLMProviderError):
    """Provider request timed out"""
    pass


class APIError(LLMProviderError):
    """Provider returned API error"""
    pass


class GenerationError(LLMProviderError):
    """All providers failed to generate response"""
    pass


class LLMFallbackClient:
    """
    Multi-provider LLM client with automatic fallback

    Strategy:
    - single: Use only primary provider (fastest, no fallback)
    - fallback: Sequential fallback through providers (recommended)
    """

    def __init__(self):
        self.strategy = os.getenv("LLM_PROVIDER_STRATEGY", "single")
        self.primary = Provider(os.getenv("LLM_PRIMARY_PROVIDER", "groq"))
        self.fallbacks = self._parse_fallbacks()

    def _parse_fallbacks(self) -> List[Provider]:
        """Parse fallback providers from environment variable"""
        fallbacks_str = os.getenv("LLM_FALLBACK_PROVIDERS", "")
        if not fallbacks_str:
            return []

        fallbacks = []
        for p in fallbacks_str.split(","):
            p = p.strip()
            if p:
                try:
                    fallbacks.append(Provider(p))
                except ValueError:
                    logger.warning(f"Unknown provider in fallbacks: {p}")
        return fallbacks

    def generate(
        self,
        prompt: str,
        max_retries: int = 2,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """
        Generate response with automatic fallback

        Args:
            prompt: Input prompt
            max_retries: Retries per provider
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response text

        Raises:
            GenerationError: If all providers fail
        """
        if self.strategy == "single":
            return self._call_provider_with_retry(
                self.primary, prompt, max_retries, temperature, max_tokens
            )

        # Fallback strategy
        providers = [self.primary] + self.fallbacks
        last_error = None

        for i, provider in enumerate(providers):
            try:
                logger.info(f"Attempting provider {provider.name} ({i+1}/{len(providers)})")
                response = self._call_provider_with_retry(
                    provider, prompt, max_retries, temperature, max_tokens
                )
                logger.info(f"Success with provider {provider.name}")
                self._log_provider_used(provider)
                return response
            except (RateLimitError, TimeoutError, APIError) as e:
                last_error = e
                logger.warning(f"Provider {provider.name} failed: {e}")
                self._log_provider_failed(provider, str(e))
                continue

        # All providers failed
        error_msg = f"All LLM providers failed. Last error: {last_error}"
        logger.error(error_msg)
        raise GenerationError(error_msg)

    def _call_provider_with_retry(
        self,
        provider: Provider,
        prompt: str,
        max_retries: int,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call provider with retry logic"""
        for attempt in range(max_retries + 1):
            try:
                return self._call_provider(provider, prompt, temperature, max_tokens)
            except (RateLimitError, TimeoutError) as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retry {attempt + 1}/{max_retries} after {wait_time}s")
                    time.sleep(wait_time)
                else:
                    raise

    def _call_provider(
        self,
        provider: Provider,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call specific provider implementation"""
        if provider == Provider.GROQ:
            return self._call_groq(prompt, temperature, max_tokens)
        elif provider == Provider.ANTHROPIC:
            return self._call_anthropic(prompt, temperature, max_tokens)
        elif provider == Provider.OPENAI:
            return self._call_openai(prompt, temperature, max_tokens)
        elif provider == Provider.OLLAMA:
            return self._call_ollama(prompt, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _call_groq(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> str:
        """Call Groq API"""
        # TODO: Implement Groq API call
        # For now, raise NotImplementedError
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "gsk_your_actual_key_here":
            raise APIError("GROQ_API_KEY not configured")

        # Placeholder implementation
        # In production, use:
        # from groq import Groq
        # client = Groq(api_key=api_key)
        # response = client.chat.completions.create(...)
        raise NotImplementedError("Groq provider not yet implemented")

    def _call_anthropic(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> str:
        """Call Anthropic API"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key == "sk-ant-your_actual_key_here":
            raise APIError("ANTHROPIC_API_KEY not configured")

        # Placeholder implementation
        # In production, use:
        # import anthropic
        # client = anthropic.Anthropic(api_key=api_key)
        # response = client.messages.create(...)
        raise NotImplementedError("Anthropic provider not yet implemented")

    def _call_openai(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> str:
        """Call OpenAI API"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "sk-your_actual_key_here":
            raise APIError("OPENAI_API_KEY not configured")

        # Placeholder implementation
        # In production, use:
        # from openai import OpenAI
        # client = OpenAI(api_key=api_key)
        # response = client.chat.completions.create(...)
        raise NotImplementedError("OpenAI provider not yet implemented")

    def _call_ollama(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> str:
        """Call Ollama local API"""
        # Placeholder implementation
        # In production, use:
        # import requests
        # response = requests.post(...)
        raise NotImplementedError("Ollama provider not yet implemented")

    def _log_provider_used(self, provider: Provider):
        """Log which provider succeeded"""
        # In production, integrate with Prometheus metrics
        logger.info(f"Provider used: {provider.name}")

    def _log_provider_failed(self, provider: Provider, error: str):
        """Log provider failure"""
        # In production, integrate with Prometheus metrics
        logger.warning(f"Provider failed: {provider.name}, error: {error}")
