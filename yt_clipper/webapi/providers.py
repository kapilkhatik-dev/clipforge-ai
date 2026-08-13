"""Provider catalog and process-local configuration profiles."""

from __future__ import annotations

import os
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Literal

from pydantic import SecretStr

from ..config import (
    LLMProvider,
    default_analysis_model,
    resolve_analysis_model,
    resolve_codex_binary,
    resolve_codex_timeout_seconds,
    resolve_llm_api_key,
    resolve_llm_provider,
)
from ..domain.models import PipelineConfig
from ..services.codex_cli import CodexCLIClient
from .models import (
    CredentialState,
    ProviderConfigurationField,
    ProviderConfigurationScalar,
    ProviderConfigurationState,
    ProviderDescriptor,
    ProviderModel,
    ProviderPatch,
    ProviderProfile,
    ProviderRuntimeConfig,
    ProviderTestResult,
)


ProviderTester = Callable[[PipelineConfig], tuple[bool, str]]
ProviderLiveProbe = Callable[[PipelineConfig], tuple[bool, str]]
LOGGER = logging.getLogger(__name__)


FieldSource = Literal["environment", "default"] | None
ResolvedFieldSource = Literal["environment", "runtime", "default"] | None
FieldResolver = Callable[
    [LLMProvider, bool], tuple[ProviderConfigurationScalar | None, FieldSource]
]


@dataclass(frozen=True)
class ProviderFieldDefinition:
    """Public field schema plus its server-side translation contract."""

    descriptor: ProviderConfigurationField
    pipeline_parameter: str
    resolver: FieldResolver
    legacy_value_attribute: str | None = None
    legacy_clear_attribute: str | None = None


@dataclass(frozen=True)
class ProviderDefinition:
    """All provider-specific metadata and configuration live in one registry."""

    provider: LLMProvider
    display_name: str
    description: str
    transport: Literal["local-cli", "hosted-api"]
    model_label: str
    model_prefix: str
    fields: tuple[ProviderFieldDefinition, ...]

    @property
    def credential_field(self) -> ProviderFieldDefinition | None:
        return next(
            (field for field in self.fields if field.pipeline_parameter == "llm_api_key"),
            None,
        )

    def field(self, key: str) -> ProviderFieldDefinition:
        try:
            return next(field for field in self.fields if field.descriptor.key == key)
        except StopIteration as exc:
            raise ValueError(
                f"Configuration field '{key}' is not valid for {self.display_name}"
            ) from exc

    def descriptor(self) -> ProviderDescriptor:
        return ProviderDescriptor(
            id=self.provider,
            display_name=self.display_name,
            description=self.description,
            transport=self.transport,
            requires_credential=self.credential_field is not None,
            default_model=default_analysis_model(self.provider),
            models=[
                ProviderModel(
                    id=default_analysis_model(self.provider),
                    label=self.model_label,
                )
            ],
            allow_custom_model=True,
            capabilities=[
                "clip-analysis",
                "highlight-montage",
                "structured-output",
            ],
            configuration_fields=[field.descriptor for field in self.fields],
        )


def _resolve_api_key(
    provider: LLMProvider,
    include_common: bool,
) -> tuple[str | None, FieldSource]:
    value = resolve_llm_api_key(
        provider,
        include_common_overrides=include_common,
    )
    return value, "environment" if value else None


def _resolve_codex_binary(
    _provider: LLMProvider,
    _include_common: bool,
) -> tuple[str, FieldSource]:
    return (
        resolve_codex_binary(),
        "environment" if os.getenv("CLIPPER_CODEX_BINARY", "").strip() else "default",
    )


def _resolve_codex_timeout(
    _provider: LLMProvider,
    _include_common: bool,
) -> tuple[int, FieldSource]:
    return (
        resolve_codex_timeout_seconds(),
        (
            "environment"
            if os.getenv("CLIPPER_CODEX_TIMEOUT_SECONDS", "").strip()
            else "default"
        ),
    )


def _hosted_api_key_field() -> ProviderFieldDefinition:
    return ProviderFieldDefinition(
        descriptor=ProviderConfigurationField(
            key="apiKey",
            label="API key",
            input_type="secret",
            section="Authentication",
            section_description=(
                "Paste a new key only when you want to replace the existing credential."
            ),
            help_text=(
                "The key is write-only and remains in this local service session. "
                "Use an environment variable for restart-persistent configuration."
            ),
            placeholder="Paste provider API key",
            required=True,
            write_only=True,
            clearable=True,
            min_length=1,
            max_length=8192,
        ),
        pipeline_parameter="llm_api_key",
        resolver=_resolve_api_key,
        legacy_value_attribute="api_key",
        legacy_clear_attribute="clear_api_key",
    )


_CODEX_FIELDS = (
    ProviderFieldDefinition(
        descriptor=ProviderConfigurationField(
            key="codexBinary",
            label="Executable",
            input_type="text",
            section="Local CLI",
            section_description=(
                "ClipForge runs the installed CLI with a restricted analysis session."
            ),
            help_text="Command name or absolute path to the Codex CLI executable.",
            placeholder="codex",
            required=True,
            min_length=1,
            max_length=1024,
        ),
        pipeline_parameter="codex_binary",
        resolver=_resolve_codex_binary,
        legacy_value_attribute="codex_binary",
    ),
    ProviderFieldDefinition(
        descriptor=ProviderConfigurationField(
            key="codexTimeoutSeconds",
            label="Timeout",
            input_type="number",
            section="Local CLI",
            section_description=(
                "ClipForge runs the installed CLI with a restricted analysis session."
            ),
            help_text="Maximum duration allowed for one Codex analysis request.",
            required=True,
            minimum=30,
            maximum=1800,
            step=1,
            suffix="sec",
        ),
        pipeline_parameter="codex_timeout_seconds",
        resolver=_resolve_codex_timeout,
        legacy_value_attribute="codex_timeout_seconds",
    ),
)


def _hosted_definition(
    provider: LLMProvider,
    display_name: str,
    description: str,
    model_label: str,
    model_prefix: str,
) -> ProviderDefinition:
    return ProviderDefinition(
        provider=provider,
        display_name=display_name,
        description=description,
        transport="hosted-api",
        model_label=model_label,
        model_prefix=model_prefix,
        fields=(_hosted_api_key_field(),),
    )


_PROVIDER_DEFINITIONS: dict[LLMProvider, ProviderDefinition] = {
    LLMProvider.CODEX: ProviderDefinition(
        provider=LLMProvider.CODEX,
        display_name="Codex CLI",
        description="Use the locally installed and authenticated Codex CLI.",
        transport="local-cli",
        model_label="Codex default",
        model_prefix="codex/",
        fields=_CODEX_FIELDS,
    ),
    LLMProvider.NVIDIA: _hosted_definition(
        LLMProvider.NVIDIA,
        "NVIDIA NIM",
        "Run analysis with NVIDIA's hosted NIM models.",
        "Nemotron 3 Ultra",
        "nvidia_nim/",
    ),
    LLMProvider.OPENROUTER: _hosted_definition(
        LLMProvider.OPENROUTER,
        "OpenRouter",
        "Access supported third-party models through OpenRouter.",
        "Nemotron 3 Ultra (free)",
        "openrouter/",
    ),
    LLMProvider.OPENAI: _hosted_definition(
        LLMProvider.OPENAI,
        "OpenAI",
        "Use an OpenAI API key with OpenAI models.",
        "GPT-4.1 mini",
        "openai/",
    ),
    LLMProvider.ANTHROPIC: _hosted_definition(
        LLMProvider.ANTHROPIC,
        "Anthropic",
        "Use an Anthropic API key with Claude models.",
        "Claude Sonnet 4.6",
        "anthropic/",
    ),
}


def provider_descriptors() -> list[ProviderDescriptor]:
    return [_PROVIDER_DEFINITIONS[provider].descriptor() for provider in LLMProvider]


def _default_tester(config: PipelineConfig) -> tuple[bool, str]:
    if config.llm_provider == LLMProvider.CODEX:
        identity = CodexCLIClient(
            config.codex_binary,
            timeout_seconds=config.codex_timeout_seconds,
        ).cache_identity(config.model)
        return True, f"Codex CLI is available ({identity})."
    if config.get_llm_api_key():
        return True, "Provider credentials and model configuration are ready."
    return False, "No API key is configured for this provider."


def _default_live_probe(config: PipelineConfig) -> tuple[bool, str]:
    """Make one explicit, timeout-bounded model request when the user opts in."""
    if config.llm_provider == LLMProvider.CODEX:
        result = CodexCLIClient(
            config.codex_binary,
            timeout_seconds=min(config.codex_timeout_seconds, 30),
        ).request(
            messages=[{"role": "user", "content": "Return {\"ok\": true}."}],
            model=config.model,
            output_schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
            description="Verify authenticated access to the selected Codex model.",
            max_attempts=1,
        )
        return (
            isinstance(result.payload, dict) and result.payload.get("ok") is True,
            "Codex model access verified.",
        )

    api_key = config.get_llm_api_key()
    if not api_key:
        return False, "No API key is configured for this provider."
    # Import lazily so local Codex-only startup does not initialize a hosted SDK.
    from litellm import completion

    completion(
        model=config.model,
        messages=[{"role": "user", "content": "Reply with OK."}],
        api_key=api_key,
        max_tokens=16,
        timeout=20.0,
        num_retries=0,
    )
    return True, "Live model access verified."


class ProviderRegistry:
    """Thread-safe runtime overrides layered over existing environment settings."""

    def __init__(
        self,
        tester: ProviderTester | None = None,
        live_probe: ProviderLiveProbe | None = None,
    ) -> None:
        self._lock = RLock()
        self._tester = tester or _default_tester
        self._live_probe = live_probe or _default_live_probe
        self._startup_provider = resolve_llm_provider()
        self._active = self._startup_provider
        self._model_overrides: dict[LLMProvider, str] = {}
        self._field_overrides: dict[
            LLMProvider,
            dict[str, ProviderConfigurationScalar],
        ] = {provider: {} for provider in LLMProvider}
        self._tests: dict[LLMProvider, ProviderTestResult] = {}
        self._revisions: dict[LLMProvider, int] = {
            provider: 0 for provider in LLMProvider
        }
        # The environment-selected startup profile remains usable without forcing
        # an interactive setup check. Any later edit creates a new revision that
        # must be checked and explicitly activated before it can generate.
        self._active_revision = self._revisions[self._active]

    @staticmethod
    def profile_id(provider: LLMProvider) -> str:
        return f"{provider.value}-default"

    @classmethod
    def provider_from_profile(cls, profile_id: str) -> LLMProvider:
        for provider in LLMProvider:
            if profile_id == cls.profile_id(provider):
                return provider
        raise KeyError(profile_id)

    @property
    def active_profile_id(self) -> str:
        with self._lock:
            return self.profile_id(self._active)

    def descriptors(self) -> list[ProviderDescriptor]:
        return provider_descriptors()

    @staticmethod
    def _definition(provider: LLMProvider) -> ProviderDefinition:
        return _PROVIDER_DEFINITIONS[provider]

    def _field_value(
        self,
        provider: LLMProvider,
        field: ProviderFieldDefinition,
    ) -> tuple[ProviderConfigurationScalar | None, ResolvedFieldSource]:
        overrides = self._field_overrides[provider]
        if field.descriptor.key in overrides:
            return overrides[field.descriptor.key], "runtime"
        include_common = provider == self._startup_provider
        return field.resolver(provider, include_common)

    def _configuration_state(
        self,
        provider: LLMProvider,
    ) -> dict[str, ProviderConfigurationState]:
        states: dict[str, ProviderConfigurationState] = {}
        for field in self._definition(provider).fields:
            value, source = self._field_value(provider, field)
            configured = value is not None and (not isinstance(value, str) or bool(value))
            states[field.descriptor.key] = ProviderConfigurationState(
                value=None if field.descriptor.write_only else value,
                configured=configured,
                source=source,
            )
        return states

    def _credential_state(self, provider: LLMProvider) -> CredentialState:
        credential_field = self._definition(provider).credential_field
        if credential_field is None:
            return CredentialState(configured=True)
        value, source = self._field_value(provider, credential_field)
        configured = bool(value)
        credential_source: Literal["environment", "runtime"] | None = (
            source if source in ("environment", "runtime") else None
        )
        return CredentialState(
            configured=configured,
            source=credential_source if configured else None,
        )

    def _profile(self, provider: LLMProvider) -> ProviderProfile:
        definition = self._definition(provider)
        configuration = self._configuration_state(provider)
        codex_config = None
        if provider == LLMProvider.CODEX:
            codex_config = ProviderRuntimeConfig(
                codex_binary=str(configuration["codexBinary"].value),
                codex_timeout_seconds=int(
                    configuration["codexTimeoutSeconds"].value or 300
                ),
            )
        credential = self._credential_state(provider)
        model = self._model_overrides.get(
            provider,
            resolve_analysis_model(
                provider,
                include_common_overrides=provider == self._startup_provider,
            ),
        )
        try:
            self._validate_model(provider, model)
            model_is_valid = True
        except ValueError:
            model_is_valid = False
        last_test = self._tests.get(provider)
        generation_ready = (
            provider == self._active
            and self._active_revision == self._revisions[provider]
            and credential.configured
            and model_is_valid
            and (last_test is None or last_test.status == "healthy")
        )
        return ProviderProfile(
            id=self.profile_id(provider),
            provider_id=provider,
            name=definition.display_name,
            model=model,
            active=provider == self._active,
            generation_ready=generation_ready,
            credential=credential,
            config=codex_config,
            configuration=configuration,
            last_test=last_test,
        )

    def profiles(self) -> list[ProviderProfile]:
        with self._lock:
            return [self._profile(provider) for provider in LLMProvider]

    def profile(self, profile_id: str) -> ProviderProfile:
        with self._lock:
            return self._profile(self.provider_from_profile(profile_id))

    def update(self, profile_id: str, patch: ProviderPatch) -> ProviderProfile:
        provider = self.provider_from_profile(profile_id)
        with self._lock:
            configuration = self._validate_patch(provider, patch)
            if patch.model is not None:
                self._model_overrides[provider] = patch.model
            overrides = self._field_overrides[provider]
            for key, value in configuration.items():
                if value is None:
                    overrides.pop(key, None)
                else:
                    overrides[key] = value
            self._tests.pop(provider, None)
            self._revisions[provider] += 1
            return self._profile(provider)

    @classmethod
    def _validate_patch(
        cls,
        provider: LLMProvider,
        patch: ProviderPatch,
    ) -> dict[str, ProviderConfigurationScalar | None]:
        """Validate a whole patch before mutating any process-local settings."""
        if patch.model is not None:
            cls._validate_model(provider, patch.model)
        if patch.api_key is not None and patch.clear_api_key:
            raise ValueError("apiKey and clearApiKey cannot be submitted together")

        definition = cls._definition(provider)
        changes = dict(patch.configuration)
        accepted_legacy_attributes: set[str] = set()
        for field in definition.fields:
            value_attribute = field.legacy_value_attribute
            clear_attribute = field.legacy_clear_attribute
            if value_attribute is not None:
                accepted_legacy_attributes.add(value_attribute)
                legacy_value = getattr(patch, value_attribute)
                if legacy_value is not None:
                    if field.descriptor.key in changes:
                        raise ValueError(
                            f"Configuration field '{field.descriptor.key}' was submitted twice"
                        )
                    changes[field.descriptor.key] = (
                        legacy_value.get_secret_value()
                        if isinstance(legacy_value, SecretStr)
                        else legacy_value
                    )
            if clear_attribute is not None:
                accepted_legacy_attributes.add(clear_attribute)
                if getattr(patch, clear_attribute):
                    if field.descriptor.key in changes:
                        raise ValueError(
                            f"Configuration field '{field.descriptor.key}' was submitted twice"
                        )
                    changes[field.descriptor.key] = None

        legacy_submissions = {
            "api_key": patch.api_key is not None,
            "clear_api_key": patch.clear_api_key,
            "codex_binary": patch.codex_binary is not None,
            "codex_timeout_seconds": patch.codex_timeout_seconds is not None,
        }
        unsupported = next(
            (
                name
                for name, submitted in legacy_submissions.items()
                if submitted and name not in accepted_legacy_attributes
            ),
            None,
        )
        if unsupported is not None:
            public_name = {
                "api_key": "apiKey",
                "clear_api_key": "clearApiKey",
                "codex_binary": "codexBinary",
                "codex_timeout_seconds": "codexTimeoutSeconds",
            }[unsupported]
            raise ValueError(
                f"{public_name} is not valid for the {definition.display_name} provider"
            )

        validated: dict[str, ProviderConfigurationScalar | None] = {}
        for key, value in changes.items():
            field = definition.field(key)
            descriptor = field.descriptor
            if value is None:
                if not descriptor.clearable:
                    raise ValueError(f"Configuration field '{key}' cannot be cleared")
                validated[key] = None
                continue
            if descriptor.input_type in {"text", "secret"}:
                if not isinstance(value, str):
                    raise ValueError(f"Configuration field '{key}' must be text")
                normalized = value.strip()
                if descriptor.required and not normalized:
                    raise ValueError(f"Configuration field '{key}' is required")
                if (
                    descriptor.min_length is not None
                    and len(normalized) < descriptor.min_length
                ):
                    raise ValueError(f"Configuration field '{key}' is too short")
                if (
                    descriptor.max_length is not None
                    and len(normalized) > descriptor.max_length
                ):
                    raise ValueError(f"Configuration field '{key}' is too long")
                validated[key] = normalized
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Configuration field '{key}' must be a number")
            if descriptor.step == 1 and not float(value).is_integer():
                raise ValueError(f"Configuration field '{key}' must be a whole number")
            if descriptor.minimum is not None and value < descriptor.minimum:
                raise ValueError(
                    f"Configuration field '{key}' must be at least {descriptor.minimum:g}"
                )
            if descriptor.maximum is not None and value > descriptor.maximum:
                raise ValueError(
                    f"Configuration field '{key}' must be at most {descriptor.maximum:g}"
                )
            validated[key] = int(value) if descriptor.step == 1 else value
        return validated

    @staticmethod
    def _validate_model(provider: LLMProvider, model: str) -> None:
        expected_prefix = _PROVIDER_DEFINITIONS[provider].model_prefix
        if not model.startswith(expected_prefix):
            raise ValueError(
                f"The selected model must begin with {expected_prefix} for this provider"
            )

    def activate(self, profile_id: str) -> str:
        provider = self.provider_from_profile(profile_id)
        with self._lock:
            profile = self._profile(provider)
            if profile.credential.configured is False:
                raise ValueError("Configure this provider before making it active")
            if profile.last_test is None or profile.last_test.status != "healthy":
                raise ValueError("Run a successful setup check before making this provider active")
            self._active = provider
            self._active_revision = self._revisions[provider]
            return self.profile_id(provider)

    def pipeline_config(self, profile_id: str, **options: Any) -> PipelineConfig:
        provider = self.provider_from_profile(profile_id)
        with self._lock:
            include_common = provider == self._startup_provider
            model = self._model_overrides.get(
                provider,
                resolve_analysis_model(
                    provider,
                    include_common_overrides=include_common,
                ),
            )
            self._validate_model(provider, model)
            provider_parameters: dict[str, ProviderConfigurationScalar | None] = {}
            for field in self._definition(provider).fields:
                value, _source = self._field_value(provider, field)
                provider_parameters[field.pipeline_parameter] = value
            api_key = provider_parameters.pop("llm_api_key", None)
            return PipelineConfig(
                llm_provider=provider,
                llm_api_key=SecretStr(str(api_key)) if api_key is not None else None,
                model=model,
                **provider_parameters,
                **options,
            )

    def generation_config(self, profile_id: str, **options: Any) -> PipelineConfig:
        """Build a provider-isolated config that is ready to start paid work."""
        provider = self.provider_from_profile(profile_id)
        with self._lock:
            profile = self._profile(provider)
            if not profile.credential.configured:
                raise ValueError("Configure this provider before generating clips")
            if provider != self._active:
                raise ValueError("Make this provider active before generating clips")
            # Validate the provider/model boundary before reporting readiness so a
            # bad environment override remains directly actionable.
            config = self.pipeline_config(profile_id, **options)
            if not profile.generation_ready:
                raise ValueError(
                    "Run a successful setup check and re-apply this active provider "
                    "before generating clips"
                )
            if (
                config.llm_provider != LLMProvider.CODEX
                and config.get_llm_api_key() is None
            ):
                raise ValueError("Configure this provider before generating clips")
            return config

    def test(self, profile_id: str) -> ProviderTestResult:
        provider = self.provider_from_profile(profile_id)
        with self._lock:
            revision = self._revisions[provider]
            try:
                config = self.pipeline_config(profile_id, analyze_only=True)
            except ValueError:
                result = ProviderTestResult(
                    status="unhealthy",
                    tested_at=datetime.now(timezone.utc),
                    message="The model identifier does not match this provider.",
                )
                self._tests[provider] = result
                return result
        started = time.perf_counter()
        try:
            healthy, _ = self._tester(config)
        except Exception as exc:
            # Provider/CLI exceptions can include upstream response bodies, local
            # paths, or credentials. Keep those details out of persistent logs.
            LOGGER.warning(
                "Provider setup check failed for %s (%s)",
                provider.value,
                type(exc).__name__,
            )
            healthy = False
        message = (
            "The Codex CLI executable is available. Authentication and model "
            "access will be verified when analysis starts."
            if healthy and provider == LLMProvider.CODEX
            else (
                "Provider credentials and model configuration are ready."
                if healthy
                else (
                    "The Codex CLI executable could not be verified. Check its installation."
                    if provider == LLMProvider.CODEX
                    else "This provider configuration could not be verified."
                )
            )
        )
        result = ProviderTestResult(
            status="healthy" if healthy else "unhealthy",
            tested_at=datetime.now(timezone.utc),
            latency_ms=max(0, round((time.perf_counter() - started) * 1000)),
            message=message,
        )
        with self._lock:
            if self._revisions[provider] != revision:
                return ProviderTestResult(
                    status="unhealthy",
                    tested_at=datetime.now(timezone.utc),
                    message="Settings changed during the check. Save and check setup again.",
                )
            self._tests[provider] = result
        return result

    def probe(self, profile_id: str) -> ProviderTestResult:
        """Run an opt-in live request without changing setup/activation state."""
        provider = self.provider_from_profile(profile_id)
        with self._lock:
            revision = self._revisions[provider]
            profile = self._profile(provider)
            if not profile.credential.configured:
                return ProviderTestResult(
                    status="unhealthy",
                    tested_at=datetime.now(timezone.utc),
                    message="Configure this provider before running a live API test.",
                )
            try:
                config = self.pipeline_config(profile_id, analyze_only=True)
            except ValueError:
                return ProviderTestResult(
                    status="unhealthy",
                    tested_at=datetime.now(timezone.utc),
                    message="The model identifier does not match this provider.",
                )

        started = time.perf_counter()
        try:
            healthy, _ = self._live_probe(config)
        except Exception as exc:
            LOGGER.warning(
                "Live provider probe failed for %s (%s)",
                provider.value,
                type(exc).__name__,
            )
            healthy = False
        result = ProviderTestResult(
            status="healthy" if healthy else "unhealthy",
            tested_at=datetime.now(timezone.utc),
            latency_ms=max(0, round((time.perf_counter() - started) * 1000)),
            message=(
                "A live request to the selected model completed successfully."
                if healthy
                else "The live model request failed. Check credentials, model access, and provider status."
            ),
        )
        with self._lock:
            if self._revisions[provider] != revision:
                return ProviderTestResult(
                    status="unhealthy",
                    tested_at=datetime.now(timezone.utc),
                    message="Settings changed during the live test. Save and try again.",
                )
        return result

    def diagnostics(self) -> list[tuple[str, str, str]]:
        """Return display-safe readiness without mutating saved setup-check state."""
        results: list[tuple[str, str, str]] = []
        for profile in self.profiles():
            provider = profile.provider_id
            if provider == LLMProvider.CODEX:
                try:
                    config = self.pipeline_config(profile.id, analyze_only=True)
                    healthy, _ = self._tester(config)
                except Exception:
                    healthy = False
                results.append(
                    (
                        profile.id,
                        "healthy" if healthy else "unhealthy",
                        (
                            "The Codex CLI is available."
                            if healthy
                            else "The Codex CLI is not ready."
                        ),
                    )
                )
            elif profile.credential.configured:
                results.append(
                    (profile.id, "healthy", "Provider configuration is present.")
                )
            else:
                results.append(
                    (
                        profile.id,
                        "unconfigured",
                        "No provider credential is configured.",
                    )
                )
        return results
