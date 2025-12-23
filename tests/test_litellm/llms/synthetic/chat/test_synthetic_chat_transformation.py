"""
Unit tests for Synthetic configuration.

These tests validate the SyntheticConfig class which extends OpenAIGPTConfig.
Synthetic is an OpenAI-compatible provider.
"""

import os
import sys
from typing import Dict, List, Optional
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.synthetic.chat.transformation import SyntheticConfig


class TestSyntheticConfig:
    """Test class for SyntheticConfig functionality"""

    def test_validate_environment(self):
        """Test that validate_environment adds correct headers"""
        config = SyntheticConfig()
        headers = {}
        api_key = "fake-synthetic-key"

        result = config.validate_environment(
            headers=headers,
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base="https://api.synthetic.new/v1/",
        )

        # Verify headers
        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"

    def test_missing_api_key(self):
        """Test error handling when API key is missing"""
        config = SyntheticConfig()

        with pytest.raises(ValueError) as excinfo:
            config.validate_environment(
                headers={},
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base="https://api.synthetic.new/v1/",
            )

        assert "Missing Synthetic API Key" in str(excinfo.value)

    def test_inheritance(self):
        """Test proper inheritance from OpenAIGPTConfig"""
        config = SyntheticConfig()

        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

        assert isinstance(config, OpenAIGPTConfig)
        assert hasattr(config, "get_supported_openai_params")

    def test_map_openai_params(self):
        """Test map_openai_params handles parameters correctly"""
        config = SyntheticConfig()

        # Test with supported parameters
        non_default_params = {"temperature": 0.7, "max_tokens": 100}
        optional_params = {}
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gpt-3.5-turbo",
            drop_params=False,
        )
        assert "temperature" in result
        assert result["temperature"] == 0.7
        assert "max_tokens" in result
        assert result["max_tokens"] == 100

    def test_map_openai_params_with_max_completion_tokens(self):
        """Test map_openai_params handles max_completion_tokens correctly"""
        config = SyntheticConfig()

        # Test max_completion_tokens gets mapped to max_tokens
        non_default_params = {"max_completion_tokens": 150}
        optional_params = {}
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gpt-3.5-turbo",
            drop_params=False,
        )
        assert "max_tokens" in result
        assert result["max_tokens"] == 150

    def test_default_api_base(self):
        """Test that default API base is used when none is provided"""
        config = SyntheticConfig()
        headers = {}
        api_key = "fake-synthetic-key"

        # Call validate_environment without specifying api_base
        result = config.validate_environment(
            headers=headers,
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base=None,  # Not providing api_base
        )

        # Verify headers are still set correctly
        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"

        # We can't directly test the api_base value here since validate_environment
        # only returns the headers, but we can verify it doesn't raise an exception
        # which would happen if api_base handling was incorrect

    def test_synthetic_completion_mock(self, respx_mock):
        """
        Mock test for Synthetic completion.
        This test mocks the actual HTTP request to test the integration properly.
        """
        import litellm

        litellm.disable_aiohttp_transport = (
            True  # since this uses respx, we need to set use_aiohttp_transport to False
        )
        from litellm import completion

        # Set up environment variables for the test
        api_key = "fake-synthetic-key"
        api_base = "https://api.synthetic.new/v1"
        model = "synthetic/gpt-3.5-turbo"
        model_name = "gpt-3.5-turbo"  # The actual model name without provider prefix

        # Mock the HTTP request to the Synthetic API
        respx_mock.post(f"{api_base}/chat/completions").respond(
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello! How can I help you today?",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21,
                },
            },
            status_code=200,
        )

        # Make the actual API call through LiteLLM
        response = completion(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            api_key=api_key,
            api_base=api_base,
        )

        # Verify response structure
        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert hasattr(response.choices[0], "message")
        assert hasattr(response.choices[0].message, "content")
        assert response.choices[0].message.content is not None

        # Check for specific content in the response
        assert "Hello" in response.choices[0].message.content
