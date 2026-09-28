"""Typed request and response classes, generated from docs/openapi.json by scripts/generate_types.py; do not edit."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Union

@dataclass(kw_only=True)
class AdapterGenerationSelection:
    """An exact immutable LoRA adapter generation."""

    generation: str


@dataclass(kw_only=True)
class AdapterSpec:
    """Runtime LoRA adapter management; listing adapters is always allowed."""

    root: str | None = None
    runtime_updates: bool | None = None


class AgentPermission(str, Enum):
    AUTO = "auto"
    ASK = "ask"
    DENY = "deny"


@dataclass(kw_only=True)
class AgenticSpec:
    agent_permission: AgentPermission | None = None
    max_tool_rounds: int | None = None
    tool_dispatch_url: str | None = None


@dataclass(kw_only=True)
class AllowedToolChoiceFunction:
    name: str
    type: Literal["function"] = "function"


@dataclass(kw_only=True)
class AllowedToolChoiceWebSearchPreview:
    type: Literal["web_search_preview"] = "web_search_preview"


@dataclass(kw_only=True)
class AllowedToolChoiceCodeInterpreter:
    type: Literal["code_interpreter"] = "code_interpreter"


@dataclass(kw_only=True)
class AllowedToolChoiceShell:
    type: Literal["shell"] = "shell"


class AllowedToolsMode(str, Enum):
    AUTO = "auto"
    REQUIRED = "required"


@dataclass(kw_only=True)
class AllowedToolsToolChoice:
    mode: AllowedToolsMode
    tools: list[AllowedToolChoice]
    type: AllowedToolsToolChoiceType = "allowed_tools"


class AllowedToolsToolChoiceType(str, Enum):
    ALLOWED_TOOLS = "allowed_tools"


@dataclass(kw_only=True)
class AnthropicContainer:
    skills: list[AnthropicSkillReference] | None = None


@dataclass(kw_only=True)
class AnthropicContentBlock:
    cache_control: dict[str, Any] | None = None
    citations: dict[str, Any] | None = None
    content: dict[str, Any] | None = None
    id: str | None = None
    input: dict[str, Any] | None = None
    is_error: bool | None = None
    name: str | None = None
    signature: str | None = None
    source: AnthropicImageSource | None = None
    text: str | None = None
    thinking: str | None = None
    tool_use_id: str | None = None
    type: str


@dataclass(kw_only=True)
class AnthropicCountTokensResponse:
    input_tokens: int


@dataclass(kw_only=True)
class AnthropicError:
    error: AnthropicErrorBody
    type: str | None = None


@dataclass(kw_only=True)
class AnthropicErrorBody:
    message: str
    type: str


@dataclass(kw_only=True)
class AnthropicImageSource:
    data: str | None = None
    media_type: str | None = None
    type: str
    url: str | None = None


@dataclass(kw_only=True)
class AnthropicJsonOutputFormat:
    schema: dict[str, Any]
    type: str


@dataclass(kw_only=True)
class AnthropicMessage:
    content: AnthropicMessageContent
    role: str


@dataclass(kw_only=True)
class AnthropicMessageResponse:
    content: list[AnthropicResponseContentBlock]
    files: list[Any] | None = None
    id: str
    model: str
    role: str | None = None
    session_id: str | None = None
    stop_reason: str
    stop_sequence: str | None = None
    type: str | None = None
    usage: AnthropicUsage


@dataclass(kw_only=True)
class AnthropicMessagesRequest:
    agent_permission: str | None = None
    code_execution_permission: str | None = None
    container: AnthropicContainer | None = None
    dry_allowed_length: int | None = None
    dry_base: float | None = None
    dry_multiplier: float | None = None
    dry_sequence_breakers: list[str] | None = None
    enable_code_execution: bool | None = None
    enable_thinking: bool | None = None
    files: list[Any] | None = None
    frequency_penalty: float | None = None
    grammar: Grammar | None = None
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
    max_tokens: int | None = None
    max_tool_rounds: int | None = None
    messages: list[AnthropicMessage]
    metadata: dict[str, Any] | None = None
    min_p: float | None = None
    model: str | None = None
    output_config: AnthropicOutputConfig | None = None
    presence_penalty: float | None = None
    reasoning_effort: ReasoningEffort | None = None
    repetition_penalty: float | None = None
    response_format: ResponseFormat | None = None
    session_id: str | None = None
    stop_sequences: list[str] | None = None
    stream: bool | None = None
    system: AnthropicSystem | None = None
    temperature: float | None = None
    thinking: AnthropicThinking | None = None
    tool_choice: AnthropicToolChoice | None = None
    tools: list[AnthropicTool] | None = None
    top_k: int | None = None
    top_logprobs: int | None = None
    top_p: float | None = None
    truncate_sequence: bool | None = None
    web_search_options: WebSearchOptions | None = None


@dataclass(kw_only=True)
class AnthropicOutputConfig:
    effort: str | None = None
    format: AnthropicJsonOutputFormat | None = None


@dataclass(kw_only=True)
class AnthropicResponseContentBlock:
    id: str | None = None
    input: dict[str, Any] | None = None
    name: str | None = None
    signature: str | None = None
    text: str | None = None
    thinking: str | None = None
    type: str


@dataclass(kw_only=True)
class AnthropicSkillListObject:
    data: list[AnthropicSkillObject]
    has_more: bool
    next_page: str | None = None


@dataclass(kw_only=True)
class AnthropicSkillObject:
    created_at: str
    display_title: str
    id: str
    latest_version: str
    source: str
    type: str
    updated_at: str


@dataclass(kw_only=True)
class AnthropicSkillReference:
    skill_id: str
    type: str
    version: Any = None


@dataclass(kw_only=True)
class AnthropicSkillVersionListObject:
    data: list[AnthropicSkillVersionObject]
    has_more: bool
    next_page: str | None = None


@dataclass(kw_only=True)
class AnthropicSkillVersionObject:
    created_at: str
    description: str
    directory: str
    id: str
    name: str
    skill_id: str
    type: str
    version: str


@dataclass(kw_only=True)
class AnthropicThinking:
    budget_tokens: int | None = None
    display: str | None = None
    type: str


@dataclass(kw_only=True)
class AnthropicTool:
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    max_uses: int | None = None
    name: str | None = None
    type: str | None = None
    user_location: AnthropicWebSearchUserLocation | None = None


@dataclass(kw_only=True)
class AnthropicToolChoice:
    name: str | None = None
    type: str


@dataclass(kw_only=True)
class AnthropicUsage:
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    input_tokens: int
    output_tokens: int


@dataclass(kw_only=True)
class AnthropicWebSearchUserLocation:
    city: str | None = None
    country: str | None = None
    region: str | None = None
    timezone: str | None = None
    type: str


class ApprovalDecision(str, Enum):
    APPROVE = "approve"
    DENY = "deny"


@dataclass(kw_only=True)
class ApprovalDecisionRequest:
    """Decision payload for a pending agentic tool approval."""

    decision: ApprovalDecision
    message: str | None = None
    remember_for_session: bool | None = None


@dataclass(kw_only=True)
class ApprovalDecisionResponse:
    status: str


@dataclass(kw_only=True)
class ApproximateUserLocation:
    city: str | None = None
    country: str | None = None
    region: str | None = None
    timezone: str | None = None


class AudioResponseFormat(str, Enum):
    """Audio format options for speech generation responses."""

    MP3 = "mp3"
    OPUS = "opus"
    AAC = "aac"
    FLAC = "flac"
    WAV = "wav"
    PCM = "pcm"


@dataclass(kw_only=True)
class BuiltinToolChoice:
    type: BuiltinToolChoiceType


class BuiltinToolChoiceType(str, Enum):
    WEB_SEARCH_PREVIEW = "web_search_preview"
    CODE_INTERPRETER = "code_interpreter"
    SHELL = "shell"


@dataclass(kw_only=True)
class CalibrationApplyRequest:
    """Request body for applying online calibration."""

    save_cimatrix: str | None = None


@dataclass(kw_only=True)
class CalibrationStatus:
    collecting: bool
    layers: int
    layers_tracking: int
    max_rows: int
    min_rows: int
    total_rows: int


@dataclass(kw_only=True)
class ChatCompletionChunkChoice:
    delta: ChatCompletionChunkDelta
    finish_reason: str | None = None
    index: int
    logprobs: Any = None


@dataclass(kw_only=True)
class ChatCompletionChunkDelta:
    content: str | None = None
    reasoning_content: str | None = None
    role: str
    tool_calls: list[Any] | None = None


@dataclass(kw_only=True)
class ChatCompletionChunkResponse:
    adapter_generation: str | None = None
    choices: list[ChatCompletionChunkChoice]
    created: int
    id: str
    model: str
    object: str
    session_id: str | None = None
    system_fingerprint: str
    usage: CompletionUsageResponse | None = None


@dataclass(kw_only=True)
class ChatCompletionRequest:
    """Chat completion request following OpenAI's specification"""

    adapter: AdapterSelection | None = None
    agent_permission: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    code_execution_permission: str | None = None
    dry_allowed_length: int | None = None
    dry_base: float | None = None
    dry_multiplier: float | None = None
    dry_sequence_breakers: list[str] | None = None
    enable_shell: bool | None = None
    enable_thinking: bool | None = None
    files: list[Any] | None = None
    frequency_penalty: float | None = None
    grammar: Grammar | None = None
    ignore_eos: bool | None = None
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
    max_tokens: int | None = None
    max_tool_rounds: int | None = None
    messages: Union[list[Message], str]
    min_p: float | None = None
    model: str | None = None
    n: int | None = None
    presence_penalty: float | None = None
    reasoning_effort: ReasoningEffort | None = None
    repetition_penalty: float | None = None
    response_format: ResponseFormat | None = None
    seed: int | None = None
    session_id: str | None = None
    stop: StopTokens | None = None
    stream: bool | None = None
    temperature: float | None = None
    tool_choice: ToolChoice | None = None
    tools: list[OpenAiTool] | None = None
    top_k: int | None = None
    top_logprobs: int | None = None
    top_p: float | None = None
    truncate_sequence: bool | None = None
    web_search_options: WebSearchOptions | None = None


@dataclass(kw_only=True)
class ChatCompletionResponse:
    adapter_generation: str | None = None
    agentic_tool_calls: list[Any] | None = None
    choices: list[ChatCompletionResponseChoice]
    created: int
    files: list[Any] | None = None
    id: str
    model: str
    object: str
    session_id: str | None = None
    system_fingerprint: str
    usage: CompletionUsageResponse


@dataclass(kw_only=True)
class ChatCompletionResponseChoice:
    finish_reason: str
    index: int
    logprobs: Any = None
    message: ChatCompletionResponseMessage


@dataclass(kw_only=True)
class ChatCompletionResponseMessage:
    content: str | None = None
    reasoning_content: str | None = None
    role: str
    tool_calls: list[Any] | None = None


@dataclass(kw_only=True)
class CompletionChunkChoice:
    finish_reason: str | None = None
    index: int
    logprobs: Any = None
    text: str


@dataclass(kw_only=True)
class CompletionChunkResponse:
    adapter_generation: str | None = None
    choices: list[CompletionChunkChoice]
    created: int
    id: str
    model: str
    object: str
    system_fingerprint: str


@dataclass(kw_only=True)
class CompletionRequest:
    """Legacy OpenAI compatible text completion request"""

    adapter: AdapterSelection | None = None
    best_of: int | None = None
    dry_allowed_length: int | None = None
    dry_base: float | None = None
    dry_multiplier: float | None = None
    dry_sequence_breakers: list[str] | None = None
    echo: bool | None = None
    frequency_penalty: float | None = None
    grammar: Grammar | None = None
    ignore_eos: bool | None = None
    logit_bias: dict[str, float] | None = None
    logprobs: int | None = None
    max_tokens: int | None = None
    min_p: float | None = None
    model: str | None = None
    n: int | None = None
    presence_penalty: float | None = None
    prompt: str
    repetition_penalty: float | None = None
    seed: int | None = None
    stop: StopTokens | None = None
    stream: bool | None = None
    suffix: str | None = None
    temperature: float | None = None
    tool_choice: ToolChoice | None = None
    tools: list[Tool] | None = None
    top_k: int | None = None
    top_p: float | None = None
    truncate_sequence: bool | None = None
    user: str | None = None


@dataclass(kw_only=True)
class CompletionResponse:
    adapter_generation: str | None = None
    choices: list[CompletionResponseChoice]
    created: int
    id: str
    model: str
    object: str
    system_fingerprint: str
    usage: CompletionUsageResponse


@dataclass(kw_only=True)
class CompletionResponseChoice:
    finish_reason: str
    index: int
    logprobs: Any = None
    text: str


@dataclass(kw_only=True)
class CompletionUsageResponse:
    avg_compl_tok_per_sec: float
    avg_prompt_tok_per_sec: float
    avg_tok_per_sec: float
    completion_tokens: int
    prompt_tokens: int
    prompt_tokens_details: PromptTokensDetailsResponse | None = None
    total_completion_time_sec: float
    total_prompt_time_sec: float
    total_time_sec: float
    total_tokens: int


@dataclass(kw_only=True)
class ContainerFileListObject:
    data: list[ContainerFileMetadata]
    object: str


@dataclass(kw_only=True)
class ContainerFileMetadata:
    """OpenAI-compatible container file metadata backed by the same in-process file store."""

    bytes: int
    container_id: str
    created_at: int
    filename: str
    format: str | None = None
    id: str
    mime_type: str
    object: str
    source: SourceMeta


class DiffusionLoaderType(str, Enum):
    """The architecture to load the diffusion model as."""

    FLUX = "flux"
    FLUX_OFFLOADED = "flux-offloaded"


@dataclass(kw_only=True)
class EmbeddingData:
    embedding: EmbeddingVector
    index: int
    object: str


class EmbeddingEncodingFormat(str, Enum):
    FLOAT = "float"
    BASE64 = "base64"


class EmbeddingLoaderType(str, Enum):
    """The architecture to load the embedding model as."""

    EMBEDDINGGEMMA = "embeddinggemma"
    QWEN3EMBEDDING = "qwen3embedding"


@dataclass(kw_only=True)
class EmbeddingRequest:
    dimensions: int | None = None
    encoding_format: EmbeddingEncodingFormat | None = None
    input: EmbeddingInput
    model: str | None = None
    truncate_sequence: bool | None = None
    user: str | None = None


@dataclass(kw_only=True)
class EmbeddingResponse:
    data: list[EmbeddingData]
    model: str
    object: str
    usage: EmbeddingUsage


@dataclass(kw_only=True)
class EmbeddingUsage:
    prompt_tokens: int
    total_tokens: int


@dataclass(kw_only=True)
class EngineSpec:
    """What to load and how to run it: the JSON form of the options `inference serve` takes."""

    adapters: AdapterSpec | None = None
    agentic: AgenticSpec | None = None
    model: ModelSelected
    model_id: str | None = None
    runtime: RuntimeSpec | None = None
    skills: SkillsSpec | None = None


@dataclass(kw_only=True)
class FileCitation:
    """File citation details"""

    file_id: str
    quote: str | None = None


@dataclass(kw_only=True)
class FileDeleted:
    deleted: bool
    id: str
    object: str


@dataclass(kw_only=True)
class FileListObject:
    data: list[FileMetadata]
    object: str


@dataclass(kw_only=True)
class FileMetadata:
    """OpenAI file metadata + inference.rs extensions (`format`, `mime_type`, `source`, `truncated`)."""

    bytes: int
    created_at: int
    filename: str
    format: str | None = None
    id: str
    mime_type: str
    object: str
    purpose: str
    source: SourceMeta
    truncated: bool | None = None


@dataclass(kw_only=True)
class FilePathInfo:
    """File path information"""

    file_id: str


@dataclass(kw_only=True)
class Function:
    """Function definition for a tool"""

    description: str | None = None
    name: str
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


@dataclass(kw_only=True)
class FunctionCalled:
    """Represents a function call made by the assistant"""

    arguments: str
    name: str


@dataclass(kw_only=True)
class GrammarRegex:
    type: Literal["regex"] = "regex"
    value: str


@dataclass(kw_only=True)
class GrammarJsonSchema:
    type: Literal["json_schema"] = "json_schema"
    value: dict[str, Any]


@dataclass(kw_only=True)
class GrammarLlguidanceValueGrammars:
    """Grammar configuration with lexer settings"""

    json_schema: dict[str, Any] | None = None
    lark_grammar: str | None = None
    name: str | None = None


@dataclass(kw_only=True)
class GrammarLlguidanceValue:
    """Top-level grammar configuration for LLGuidance"""

    grammars: list[GrammarLlguidanceValueGrammars]
    max_tokens: int | None = None


@dataclass(kw_only=True)
class GrammarLlguidance:
    type: Literal["llguidance"] = "llguidance"
    value: GrammarLlguidanceValue


@dataclass(kw_only=True)
class GrammarLark:
    type: Literal["lark"] = "lark"
    value: str


@dataclass(kw_only=True)
class ImageChoice:
    b64_json: str | None = None
    url: str | None = None


@dataclass(kw_only=True)
class ImageGenerationRequest:
    """Image generation request"""

    height: int | None = None
    model: str | None = None
    n: int | None = None
    prompt: str
    response_format: ImageGenerationResponseFormat | None = None
    width: int | None = None


@dataclass(kw_only=True)
class ImageGenerationResponse:
    created: int
    data: list[ImageChoice]


class ImageGenerationResponseFormat(str, Enum):
    """Image generation response format"""

    URL = "Url"
    B64JSON = "B64Json"


class IncludeOption(str, Enum):
    """Include options for response content."""

    FILE_SEARCH_CALL_RESULTS = "file_search_call.results"
    MESSAGE_INPUT_IMAGE_IMAGE_URL = "message.input_image.image_url"
    COMPUTER_CALL_OUTPUT_OUTPUT_IMAGE_URL = "computer_call_output.output.image_url"
    REASONING_ENCRYPTED_CONTENT = "reasoning.encrypted_content"


@dataclass(kw_only=True)
class IncompleteDetails:
    """Details about incomplete responses"""

    reason: IncompleteReason


class IncompleteReason(str, Enum):
    """Reason for incomplete response"""

    MAX_OUTPUT_TOKENS = "max_output_tokens"
    CONTENT_FILTER = "content_filter"
    INTERRUPTED = "interrupted"


@dataclass(kw_only=True)
class InputTokensDetails:
    """Detailed input token breakdown"""

    audio_tokens: int | None = None
    cached_tokens: int | None = None
    image_tokens: int | None = None
    text_tokens: int | None = None


class IsqOrganization(str, Enum):
    DEFAULT = "default"
    MOQE = "moqe"


@dataclass(kw_only=True)
class JsonSchemaResponseFormat:
    """JSON Schema for structured responses"""

    name: str
    schema: Any


@dataclass(kw_only=True)
class LoadLoraAdapterRequest:
    expected_generation: str | None = None
    load_inplace: bool | None = None
    lora_name: str
    lora_path: str
    model: str | None = None


@dataclass(kw_only=True)
class LoraAdapterListResponse:
    data: list[LoraAdapterObject]
    generations: list[LoraResidentGenerationObject]
    max_adapters: int
    max_bytes: int
    max_rank: int
    object: str
    resident_bytes: int
    resident_generations: int
    retired_generations: int


@dataclass(kw_only=True)
class LoraAdapterObject:
    bytes: int
    generation: str
    id: str
    object: str
    rank: int
    revision: str | None = None
    source: str | None = None


@dataclass(kw_only=True)
class LoraAdapterSpec:
    """Alias and source used to preload a LoRA adapter."""

    alias: str
    base_model_name: str | None = None
    revision: str | None = None
    source: str


@dataclass(kw_only=True)
class LoraResidentGenerationObject:
    active_leases: int
    aliases: list[str]
    bytes: int
    generation: str
    rank: int
    retired: bool


@dataclass(kw_only=True)
class LoraRuntimeConfig:
    """Admission limits for a dynamic LoRA runtime."""

    max_adapters: int
    max_bytes: int
    max_rank: int


@dataclass(kw_only=True)
class Message:
    """Represents a single message in a conversation"""

    content: MessageContent | None = None
    name: str | None = None
    reasoning_content: str | None = None
    role: str
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


class ModelDType(str, Enum):
    """DType for the model."""

    AUTO = "auto"
    BF16 = "bf16"
    F16 = "f16"
    F32 = "f32"


@dataclass(kw_only=True)
class ModelObject:
    """Model information metadata about an available mode"""

    adapter_generation: str | None = None
    created: int
    id: str
    mcp_servers_connected: int | None = None
    mcp_tools_count: int | None = None
    object: str
    owned_by: str
    parent: str | None = None
    root: str | None = None
    status: str | None = None
    tools_available: bool | None = None


@dataclass(kw_only=True)
class ModelObjects:
    """Collection of available models"""

    data: list[ModelObject]
    object: str


@dataclass(kw_only=True)
class ModelOperationRequest:
    """The body of an unload, reload or status request."""

    model_id: str


@dataclass(kw_only=True)
class ModelSelectedRun:
    """Select a model for running via auto loader"""

    calibration_file: str | None = None
    dtype: ModelDType | None = None
    from_uqff: str | None = None
    hf_cache_path: str | None = None
    imatrix: str | None = None
    matformer_config_path: str | None = None
    matformer_slice_name: str | None = None
    max_batch_size: int | None = None
    max_edge: int | None = None
    max_image_length: int | None = None
    max_num_images: int | None = None
    max_seq_len: int | None = None
    model_id: str
    organization: IsqOrganization | None = None
    tokenizer_json: str | None = None
    topology: str | None = None
    write_uqff: UqffWriteSpec | None = None
    _external = 'Run'


@dataclass(kw_only=True)
class ModelSelectedPlain:
    """Select a plain model, without quantization or adapters"""

    arch: NormalLoaderType | None = None
    calibration_file: str | None = None
    dtype: ModelDType | None = None
    from_uqff: str | None = None
    hf_cache_path: str | None = None
    imatrix: str | None = None
    matformer_config_path: str | None = None
    matformer_slice_name: str | None = None
    max_batch_size: int | None = None
    max_seq_len: int | None = None
    model_id: str
    organization: IsqOrganization | None = None
    tokenizer_json: str | None = None
    topology: str | None = None
    write_uqff: UqffWriteSpec | None = None
    _external = 'Plain'


@dataclass(kw_only=True)
class ModelSelectedXLora:
    """Select an X-LoRA architecture"""

    arch: NormalLoaderType | None = None
    dtype: ModelDType
    from_uqff: str | None = None
    hf_cache_path: str | None = None
    max_batch_size: int
    max_seq_len: int
    model_id: str | None = None
    order: str
    organization: IsqOrganization | None = None
    tgt_non_granular_index: int | None = None
    tokenizer_json: str | None = None
    topology: str | None = None
    write_uqff: UqffWriteSpec | None = None
    xlora_model_id: str
    _external = 'XLora'


@dataclass(kw_only=True)
class ModelSelectedLora:
    """Select a LoRA architecture"""

    adapters: list[LoraAdapterSpec]
    arch: NormalLoaderType | None = None
    calibration_file: str | None = None
    dtype: ModelDType
    from_uqff: str | None = None
    hf_cache_path: str | None = None
    imatrix: str | None = None
    matformer_config_path: str | None = None
    matformer_slice_name: str | None = None
    max_batch_size: int
    max_edge: int | None = None
    max_image_length: int | None = None
    max_num_images: int | None = None
    max_seq_len: int
    model_id: str
    organization: IsqOrganization | None = None
    runtime_config: LoraRuntimeConfig
    tokenizer_json: str | None = None
    topology: str | None = None
    write_uqff: UqffWriteSpec | None = None
    _external = 'Lora'


@dataclass(kw_only=True)
class ModelSelectedGGUF:
    """Select a GGUF model."""

    calibration_file: str | None = None
    dtype: ModelDType
    hf_cache_path: str | None = None
    imatrix: str | None = None
    lora_adapters: list[LoraAdapterSpec] | None = None
    lora_runtime_config: LoraRuntimeConfig | None = None
    matformer_config_path: str | None = None
    matformer_slice_name: str | None = None
    max_batch_size: int
    max_edge: int | None = None
    max_image_length: int | None = None
    max_num_images: int | None = None
    max_seq_len: int
    mmproj_filename: str | None = None
    organization: IsqOrganization | None = None
    quantized_filename: str
    quantized_model_id: str
    tok_model_id: str | None = None
    tokenizer_json: str | None = None
    topology: str | None = None
    write_uqff: UqffWriteSpec | None = None
    _external = 'GGUF'


@dataclass(kw_only=True)
class ModelSelectedXLoraGGUF:
    """Select a GGUF model with X-LoRA."""

    calibration_file: str | None = None
    dtype: ModelDType
    hf_cache_path: str | None = None
    imatrix: str | None = None
    matformer_config_path: str | None = None
    matformer_slice_name: str | None = None
    max_batch_size: int
    max_seq_len: int
    order: str
    organization: IsqOrganization | None = None
    quantized_filename: str
    quantized_model_id: str
    tgt_non_granular_index: int | None = None
    tok_model_id: str | None = None
    tokenizer_json: str | None = None
    topology: str | None = None
    write_uqff: UqffWriteSpec | None = None
    xlora_model_id: str
    _external = 'XLoraGGUF'


@dataclass(kw_only=True)
class ModelSelectedLoraGGUF:
    """Select a GGUF model with LoRA."""

    adapters_model_id: str
    calibration_file: str | None = None
    dtype: ModelDType
    hf_cache_path: str | None = None
    imatrix: str | None = None
    matformer_config_path: str | None = None
    matformer_slice_name: str | None = None
    max_batch_size: int
    max_seq_len: int
    order: str
    organization: IsqOrganization | None = None
    quantized_filename: str
    quantized_model_id: str
    tok_model_id: str | None = None
    tokenizer_json: str | None = None
    topology: str | None = None
    write_uqff: UqffWriteSpec | None = None
    _external = 'LoraGGUF'


@dataclass(kw_only=True)
class ModelSelectedGGML:
    """Select a GGML model."""

    dtype: ModelDType
    gqa: int
    max_batch_size: int
    max_seq_len: int
    quantized_filename: str
    quantized_model_id: str
    tok_model_id: str
    tokenizer_json: str | None = None
    topology: str | None = None
    _external = 'GGML'


@dataclass(kw_only=True)
class ModelSelectedXLoraGGML:
    """Select a GGML model with X-LoRA."""

    dtype: ModelDType
    gqa: int
    max_batch_size: int
    max_seq_len: int
    order: str
    quantized_filename: str
    quantized_model_id: str
    tgt_non_granular_index: int | None = None
    tok_model_id: str | None = None
    tokenizer_json: str | None = None
    topology: str | None = None
    xlora_model_id: str
    _external = 'XLoraGGML'


@dataclass(kw_only=True)
class ModelSelectedLoraGGML:
    """Select a GGML model with LoRA."""

    adapters_model_id: str
    dtype: ModelDType
    gqa: int
    max_batch_size: int
    max_seq_len: int
    order: str
    quantized_filename: str
    quantized_model_id: str
    tok_model_id: str | None = None
    tokenizer_json: str | None = None
    topology: str | None = None
    _external = 'LoraGGML'


@dataclass(kw_only=True)
class ModelSelectedMultimodalPlain:
    """Select a multimodal plain model, without quantization or adapters"""

    arch: MultimodalLoaderType | None = None
    calibration_file: str | None = None
    dtype: ModelDType | None = None
    from_uqff: str | None = None
    hf_cache_path: str | None = None
    imatrix: str | None = None
    matformer_config_path: str | None = None
    matformer_slice_name: str | None = None
    max_batch_size: int | None = None
    max_edge: int | None = None
    max_image_length: int | None = None
    max_num_images: int | None = None
    max_seq_len: int | None = None
    model_id: str
    organization: IsqOrganization | None = None
    tokenizer_json: str | None = None
    topology: str | None = None
    write_uqff: UqffWriteSpec | None = None
    _external = 'MultimodalPlain'


@dataclass(kw_only=True)
class ModelSelectedDiffusionPlain:
    """Select a diffusion model, without quantization or adapters"""

    arch: DiffusionLoaderType
    dtype: ModelDType
    model_id: str
    _external = 'DiffusionPlain'


@dataclass(kw_only=True)
class ModelSelectedSpeech:
    arch: SpeechLoaderType
    dac_model_id: str | None = None
    dtype: ModelDType
    model_id: str
    _external = 'Speech'


@dataclass(kw_only=True)
class ModelSelectedMultiModel:
    """Select multi-model mode with configuration file"""

    config: str
    default_model_id: str | None = None
    _external = 'MultiModel'


@dataclass(kw_only=True)
class ModelSelectedEmbedding:
    """Select an embedding model, without quantization or adapters"""

    arch: EmbeddingLoaderType | None = None
    calibration_file: str | None = None
    dtype: ModelDType | None = None
    from_uqff: str | None = None
    hf_cache_path: str | None = None
    imatrix: str | None = None
    model_id: str
    tokenizer_json: str | None = None
    topology: str | None = None
    write_uqff: UqffWriteSpec | None = None
    _external = 'Embedding'


class ModelStatus(str, Enum):
    LOADED = "loaded"
    UNLOADED = "unloaded"
    RELOADING = "reloading"


@dataclass(kw_only=True)
class ModelStatusResponse:
    model_id: str
    status: ModelStatus


class MultimodalLoaderType(str, Enum):
    """The architecture to load the multimodal model as."""

    PHI3V = "phi3v"
    IDEFICS2 = "idefics2"
    LLAVA_NEXT = "llava_next"
    LLAVA = "llava"
    LFM2VL = "lfm2vl"
    VLLAMA = "vllama"
    QWEN2VL = "qwen2vl"
    IDEFICS3 = "idefics3"
    MINICPMO = "minicpmo"
    PHI4MM = "phi4mm"
    QWEN2_5VL = "qwen2_5vl"
    GEMMA3 = "gemma3"
    MISTRAL3 = "mistral3"
    LLAMA4 = "llama4"
    GEMMA3N = "gemma3n"
    QWEN3VL = "qwen3vl"
    QWEN3VLMOE = "qwen3vlmoe"
    QWEN3_5 = "qwen3_5"
    QWEN3_5MOE = "qwen3_5moe"
    VOXTRAL = "voxtral"
    GEMMA4 = "gemma4"
    MUSE_GLIMMER = "muse_glimmer"
    DIFFUSIONGEMMA = "diffusiongemma"
    PADDLEOCR_VL = "paddleocr_vl"


@dataclass(kw_only=True)
class NamedFunctionToolChoice:
    name: str
    type: ToolType = "function"


class NormalLoaderType(str, Enum):
    """The architecture to load the normal model as."""

    MISTRAL = "mistral"
    GEMMA = "gemma"
    MIXTRAL = "mixtral"
    LLAMA = "llama"
    PHI2 = "phi2"
    PHI3 = "phi3"
    QWEN2 = "qwen2"
    GEMMA2 = "gemma2"
    STARCODER2 = "starcoder2"
    PHI3_5MOE = "phi3.5moe"
    DEEPSEEKV2 = "deepseekv2"
    DEEPSEEKV3 = "deepseekv3"
    QWEN3 = "qwen3"
    GLM4 = "glm4"
    GLM4MOELITE = "glm4moelite"
    GLM4MOE = "glm4moe"
    QWEN3MOE = "qwen3moe"
    SMOLLM3 = "smollm3"
    GRANITEMOEHYBRID = "granitemoehybrid"
    GPT_OSS = "gpt_oss"
    HUNYUANV1DENSE = "hunyuanv1dense"
    HUNYUANV1MOE = "hunyuanv1moe"
    QWEN3NEXT = "qwen3next"
    QWEN3_5 = "qwen3_5"
    LFM2 = "lfm2"
    LFM2_MOE = "lfm2_moe"


@dataclass(kw_only=True)
class OpenAiCodeInterpreterAutoContainer:
    file_ids: list[str] | None = None
    memory_limit: str | None = None
    type: OpenAiCodeInterpreterContainerType = "auto"


class OpenAiCodeInterpreterContainerType(str, Enum):
    AUTO = "auto"


@dataclass(kw_only=True)
class OpenAiCodeInterpreterTool:
    container: OpenAiCodeInterpreterContainer
    type: OpenAiCodeInterpreterToolType = "code_interpreter"


class OpenAiCodeInterpreterToolType(str, Enum):
    CODE_INTERPRETER = "code_interpreter"


class OpenAiFunctionToolType(str, Enum):
    FUNCTION = "function"


@dataclass(kw_only=True)
class OpenAiNamespaceTool:
    """A grouping of client-executed tools; the model sees each inner tool as `<namespace>.<name>`."""

    description: str | None = None
    name: str
    tools: list[OpenAiNamespaceEntry] | None = None
    type: OpenAiNamespaceToolType = "namespace"


class OpenAiNamespaceToolType(str, Enum):
    NAMESPACE = "namespace"


@dataclass(kw_only=True)
class OpenAiResponsesFunctionTool:
    description: str | None = None
    name: str
    parameters: dict[str, Any] | None = None
    strict: bool | None = None
    type: OpenAiFunctionToolType = "function"


@dataclass(kw_only=True)
class OpenAiShellEnvironmentContainerAuto:
    skills: list[OpenAiShellSkill] | None = None
    type: Literal["container_auto"] = "container_auto"


@dataclass(kw_only=True)
class OpenAiShellEnvironmentLocal:
    path: str
    type: Literal["local"] = "local"


@dataclass(kw_only=True)
class OpenAiShellEnvironmentContainerReference:
    container_id: str
    type: Literal["container_reference"] = "container_reference"


@dataclass(kw_only=True)
class OpenAiShellSkillSkillReference:
    skill_id: str
    type: Literal["skill_reference"] = "skill_reference"
    version: Any = None


@dataclass(kw_only=True)
class OpenAiShellSkillLocal:
    path: str
    type: Literal["local"] = "local"


@dataclass(kw_only=True)
class OpenAiShellTool:
    environment: OpenAiShellEnvironment
    type: OpenAiShellToolType = "shell"


class OpenAiShellToolType(str, Enum):
    SHELL = "shell"


@dataclass(kw_only=True)
class OpenAiWebSearchTool:
    external_web_access: bool | None = None
    filters: WebSearchFilters | None = None
    image_settings: WebSearchImageSettings | None = None
    return_token_budget: WebSearchReturnTokenBudget | None = None
    search_content_types: list[WebSearchContentType] | None = None
    search_context_size: SearchContextSize | None = None
    type: OpenAiWebSearchToolType
    user_location: OpenAiWebSearchUserLocation | None = None


class OpenAiWebSearchToolType(str, Enum):
    WEB_SEARCH = "web_search"
    WEB_SEARCH_PREVIEW = "web_search_preview"


@dataclass(kw_only=True)
class OpenAiWebSearchUserLocationApproximate:
    city: str | None = None
    country: str | None = None
    region: str | None = None
    timezone: str | None = None
    type: Literal["approximate"] = "approximate"


@dataclass(kw_only=True)
class OpenResponsesCreateRequest:
    """OpenResponses API create request"""

    adapter: AdapterSelection | None = None
    background: bool | None = None
    dry_allowed_length: int | None = None
    dry_base: float | None = None
    dry_multiplier: float | None = None
    dry_sequence_breakers: list[str] | None = None
    files: list[Any] | None = None
    frequency_penalty: float | None = None
    grammar: Grammar | None = None
    ignore_eos: bool | None = None
    include: list[IncludeOption] | None = None
    input: OpenResponsesInput
    instructions: str | None = None
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    max_tool_rounds: int | None = None
    metadata: Any = None
    min_p: float | None = None
    model: str | None = None
    n: int | None = None
    parallel_tool_calls: bool | None = None
    presence_penalty: float | None = None
    previous_response_id: str | None = None
    reasoning: ReasoningConfig | None = None
    repetition_penalty: float | None = None
    response_format: ResponseFormat | None = None
    seed: int | None = None
    stop: StopTokens | None = None
    store: bool | None = None
    stream: bool | None = None
    stream_options: StreamOptions | None = None
    temperature: float | None = None
    text: TextConfig | None = None
    tool_choice: ToolChoice | None = None
    tools: list[OpenAiTool] | None = None
    top_k: int | None = None
    top_logprobs: int | None = None
    top_p: float | None = None
    truncation: TruncationStrategy | None = None


@dataclass(kw_only=True)
class OpenResponsesInputMessageInputTextFileCitation:
    """File citation annotation"""

    end_index: int
    file_citation: FileCitation
    start_index: int
    text: str
    type: Literal["file_citation"] = "file_citation"


@dataclass(kw_only=True)
class OpenResponsesInputMessageInputTextUrlCitation:
    """URL citation annotation"""

    end_index: int
    start_index: int
    text: str
    type: Literal["url_citation"] = "url_citation"
    url_citation: UrlCitation


@dataclass(kw_only=True)
class OpenResponsesInputMessageInputTextFilePath:
    """File path annotation"""

    end_index: int
    file_path: FilePathInfo
    start_index: int
    text: str
    type: Literal["file_path"] = "file_path"


@dataclass(kw_only=True)
class OpenResponsesInputMessageInputTextContainerFileCitation:
    """Container file citation annotation"""

    container_id: str
    end_index: int
    file_id: str
    filename: str
    index: int | None = None
    start_index: int
    type: Literal["container_file_citation"] = "container_file_citation"


@dataclass(kw_only=True)
class OpenResponsesInputMessageInputText:
    annotations: list[Union[OpenResponsesInputMessageInputTextFileCitation, OpenResponsesInputMessageInputTextUrlCitation, OpenResponsesInputMessageInputTextFilePath, OpenResponsesInputMessageInputTextContainerFileCitation]] | None = None
    text: str
    type: Literal["input_text"] = "input_text"


@dataclass(kw_only=True)
class OpenResponsesInputMessageInputImage:
    image_url: str | None = None
    type: Literal["input_image"] = "input_image"


@dataclass(kw_only=True)
class OpenResponsesInputMessageInputAudio:
    data: str
    format: str
    type: Literal["input_audio"] = "input_audio"


@dataclass(kw_only=True)
class OpenResponsesInputMessageInputFile:
    file_data: str | None = None
    file_id: str | None = None
    file_url: str | None = None
    filename: str | None = None
    type: Literal["input_file"] = "input_file"


@dataclass(kw_only=True)
class OpenResponsesInputMessage:
    content: Union[str, list[Union[OpenResponsesInputMessageInputText, OpenResponsesInputMessageInputImage, OpenResponsesInputMessageInputAudio, OpenResponsesInputMessageInputFile]]]
    role: str
    type: Literal["message"] = "message"


@dataclass(kw_only=True)
class OpenResponsesInputItemReference:
    id: str
    type: Literal["item_reference"] = "item_reference"


@dataclass(kw_only=True)
class OpenResponsesInputFunctionCall:
    arguments: str
    call_id: str
    name: str
    namespace: str | None = None
    type: Literal["function_call"] = "function_call"


@dataclass(kw_only=True)
class OpenResponsesInputFunctionCallOutput:
    call_id: str
    output: str
    type: Literal["function_call_output"] = "function_call_output"


@dataclass(kw_only=True)
class OpenResponsesInputReasoningReasoningText:
    text: str
    type: Literal["reasoning_text"] = "reasoning_text"


@dataclass(kw_only=True)
class OpenResponsesInputReasoningSummaryText:
    text: str
    type: Literal["summary_text"] = "summary_text"


@dataclass(kw_only=True)
class OpenResponsesInputReasoning:
    content: list[OpenResponsesInputReasoningReasoningText] | None = None
    id: str | None = None
    summary: list[OpenResponsesInputReasoningSummaryText] | None = None
    type: Literal["reasoning"] = "reasoning"


@dataclass(kw_only=True)
class OpenResponsesStreamEventResponseCreated:
    """Response created event"""

    response: ResponseResource
    sequence_number: int
    type: Literal["response.created"] = "response.created"


@dataclass(kw_only=True)
class OpenResponsesStreamEventResponseInProgress:
    """Response in progress event"""

    response: ResponseResource
    sequence_number: int
    type: Literal["response.in_progress"] = "response.in_progress"


@dataclass(kw_only=True)
class OpenResponsesStreamEventResponseOutputItemAdded:
    """Output item added event"""

    item: OutputItem
    output_index: int
    sequence_number: int
    type: Literal["response.output_item.added"] = "response.output_item.added"


@dataclass(kw_only=True)
class OpenResponsesStreamEventResponseContentPartAdded:
    """Content part added event"""

    content_index: int
    output_index: int
    part: OutputContent
    sequence_number: int
    type: Literal["response.content_part.added"] = "response.content_part.added"


@dataclass(kw_only=True)
class OpenResponsesStreamEventResponseOutputTextDelta:
    """Text delta event"""

    content_index: int
    delta: str
    output_index: int
    sequence_number: int
    type: Literal["response.output_text.delta"] = "response.output_text.delta"


@dataclass(kw_only=True)
class OpenResponsesStreamEventResponseContentPartDone:
    """Content part done event"""

    content_index: int
    output_index: int
    part: OutputContent
    sequence_number: int
    type: Literal["response.content_part.done"] = "response.content_part.done"


@dataclass(kw_only=True)
class OpenResponsesStreamEventResponseOutputItemDone:
    """Output item done event"""

    item: OutputItem
    output_index: int
    sequence_number: int
    type: Literal["response.output_item.done"] = "response.output_item.done"


@dataclass(kw_only=True)
class OpenResponsesStreamEventResponseFunctionCallArgumentsDelta:
    """Function call arguments delta"""

    call_id: str
    delta: str
    output_index: int
    sequence_number: int
    type: Literal["response.function_call_arguments.delta"] = "response.function_call_arguments.delta"


@dataclass(kw_only=True)
class OpenResponsesStreamEventResponseFunctionCallArgumentsDone:
    """Function call arguments done"""

    arguments: str
    call_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.function_call_arguments.done"] = "response.function_call_arguments.done"


@dataclass(kw_only=True)
class OpenResponsesStreamEventResponseReasoningTextDelta:
    """Reasoning text delta"""

    content_index: int
    delta: str
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.reasoning_text.delta"] = "response.reasoning_text.delta"


@dataclass(kw_only=True)
class OpenResponsesStreamEventResponseReasoningTextDone:
    """Reasoning text done"""

    content_index: int
    item_id: str
    output_index: int
    sequence_number: int
    text: str
    type: Literal["response.reasoning_text.done"] = "response.reasoning_text.done"


@dataclass(kw_only=True)
class OpenResponsesStreamEventResponseCompleted:
    """Response completed event"""

    response: ResponseResource
    sequence_number: int
    type: Literal["response.completed"] = "response.completed"


@dataclass(kw_only=True)
class OpenResponsesStreamEventResponseFailed:
    """Response failed event"""

    response: ResponseResource
    sequence_number: int
    type: Literal["response.failed"] = "response.failed"


@dataclass(kw_only=True)
class OpenResponsesStreamEventResponseIncomplete:
    """Response incomplete event"""

    response: ResponseResource
    sequence_number: int
    type: Literal["response.incomplete"] = "response.incomplete"


@dataclass(kw_only=True)
class OpenResponsesStreamEventError:
    """Error event"""

    code: str
    message: str
    param: str | None = None
    sequence_number: int
    type: Literal["error"] = "error"


@dataclass(kw_only=True)
class OutputContentOutputText:
    text: str
    type: Literal["output_text"] = "output_text"


@dataclass(kw_only=True)
class OutputContentRefusal:
    refusal: str
    type: Literal["refusal"] = "refusal"


@dataclass(kw_only=True)
class OutputItemMessageOutputText:
    text: str
    type: Literal["output_text"] = "output_text"


@dataclass(kw_only=True)
class OutputItemMessageRefusal:
    refusal: str
    type: Literal["refusal"] = "refusal"


@dataclass(kw_only=True)
class OutputItemMessage:
    content: list[Union[OutputItemMessageOutputText, OutputItemMessageRefusal]]
    id: str
    role: str
    status: str
    type: Literal["message"] = "message"


@dataclass(kw_only=True)
class OutputItemFunctionCall:
    arguments: str
    call_id: str
    id: str
    name: str
    namespace: str | None = None
    status: str
    type: Literal["function_call"] = "function_call"


@dataclass(kw_only=True)
class OutputItemShellCall:
    action: dict[str, Any]
    call_id: str
    id: str
    status: str
    type: Literal["shell_call"] = "shell_call"


@dataclass(kw_only=True)
class OutputItemShellCallOutput:
    call_id: str
    id: str
    output: list[dict[str, Any]]
    status: str
    type: Literal["shell_call_output"] = "shell_call_output"


@dataclass(kw_only=True)
class OutputItemReasoningReasoningText:
    text: str
    type: Literal["reasoning_text"] = "reasoning_text"


@dataclass(kw_only=True)
class OutputItemReasoningSummaryText:
    text: str
    type: Literal["summary_text"] = "summary_text"


@dataclass(kw_only=True)
class OutputItemReasoning:
    content: list[OutputItemReasoningReasoningText]
    id: str
    status: str
    summary: list[OutputItemReasoningSummaryText]
    type: Literal["reasoning"] = "reasoning"


@dataclass(kw_only=True)
class OutputTokensDetails:
    """Detailed output token breakdown"""

    audio_tokens: int | None = None
    reasoning_tokens: int | None = None
    text_tokens: int | None = None


@dataclass(kw_only=True)
class PromptTokensDetailsResponse:
    cached_tokens: int


@dataclass(kw_only=True)
class ReIsqRequest:
    ggml_type: str


@dataclass(kw_only=True)
class ReasoningConfig:
    """Reasoning configuration for models that support extended thinking"""

    effort: ReasoningEffort | None = None
    summary: ReasoningSummary | None = None


class ReasoningEffort(str, Enum):
    """Reasoning effort. `none` aliases `off`; `max` aliases `xhigh`."""

    OFF = "off"
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class ReasoningSummary(str, Enum):
    """Reasoning summary configuration"""

    CONCISE = "concise"
    DETAILED = "detailed"
    AUTO = "auto"


@dataclass(kw_only=True)
class ResponseDeleted:
    """What deleting a response returns."""

    deleted: bool
    id: str
    object: str


@dataclass(kw_only=True)
class ResponseError:
    """Error information for a response"""

    code: str
    message: str


@dataclass(kw_only=True)
class ResponseFormatText:
    """Free-form text response"""

    type: Literal["text"] = "text"


@dataclass(kw_only=True)
class ResponseFormatJsonObject:
    """Structured response as any JSON object"""

    type: Literal["json_object"] = "json_object"


@dataclass(kw_only=True)
class ResponseFormatJsonSchema:
    """Structured response following a JSON schema"""

    json_schema: JsonSchemaResponseFormat
    type: Literal["json_schema"] = "json_schema"


@dataclass(kw_only=True)
class ResponseResource:
    """The main response resource returned by the OpenResponses API"""

    adapter_generation: str | None = None
    background: bool | None = None
    completed_at: int | None = None
    created_at: int
    error: ResponseError | None = None
    frequency_penalty: float | None = None
    id: str
    incomplete_details: IncompleteDetails | None = None
    instructions: str | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    metadata: Any = None
    model: str
    object: str
    output: list[OutputItem]
    output_text: str | None = None
    parallel_tool_calls: bool | None = None
    presence_penalty: float | None = None
    previous_response_id: str | None = None
    reasoning: str | None = None
    status: ResponseStatus
    store: bool | None = None
    temperature: float | None = None
    text: TextConfig | None = None
    tool_choice: ToolChoice | None = None
    tools: list[OpenAiTool] | None = None
    top_logprobs: int | None = None
    top_p: float | None = None
    truncation: TruncationStrategy | None = None
    usage: ResponseUsage | None = None


class ResponseStatus(str, Enum):
    """Status of a response in the OpenResponses API"""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"
    CANCELLED = "cancelled"


@dataclass(kw_only=True)
class ResponseUsage:
    """Usage information for a response"""

    input_tokens: int
    input_tokens_details: InputTokensDetails | None = None
    output_tokens: int
    output_tokens_details: OutputTokensDetails | None = None
    total_tokens: int


@dataclass(kw_only=True)
class ResponsesAnnotation:
    """Response annotation"""

    end_index: int
    start_index: int
    text: str
    type: str


@dataclass(kw_only=True)
class ResponsesChunk:
    """Response streaming chunk"""

    chunk_type: str
    created_at: float
    delta: ResponsesDelta | None = None
    id: str
    metadata: Any = None
    model: str
    object: str
    usage: ResponsesUsage | None = None


@dataclass(kw_only=True)
class ResponsesContent:
    """Response content item"""

    annotations: list[ResponsesAnnotation] | None = None
    text: str | None = None
    type: str


@dataclass(kw_only=True)
class ResponsesCreateRequest:
    """Response creation request"""

    adapter: AdapterSelection | None = None
    dry_allowed_length: int | None = None
    dry_base: float | None = None
    dry_multiplier: float | None = None
    dry_sequence_breakers: list[str] | None = None
    enable_thinking: bool | None = None
    frequency_penalty: float | None = None
    grammar: Grammar | None = None
    ignore_eos: bool | None = None
    input: ResponsesMessages
    instructions: str | None = None
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
    max_tokens: int | None = None
    max_tool_calls: int | None = None
    metadata: Any = None
    min_p: float | None = None
    modalities: list[str] | None = None
    model: str | None = None
    n: int | None = None
    output_token_details: bool | None = None
    parallel_tool_calls: bool | None = None
    presence_penalty: float | None = None
    previous_response_id: str | None = None
    reasoning_effort: str | None = None
    reasoning_enabled: bool | None = None
    reasoning_max_tokens: int | None = None
    reasoning_top_logprobs: int | None = None
    repetition_penalty: float | None = None
    response_format: ResponseFormat | None = None
    seed: int | None = None
    stop: StopTokens | None = None
    store: bool | None = None
    stream: bool | None = None
    temperature: float | None = None
    tool_choice: ToolChoice | None = None
    tools: list[OpenAiTool] | None = None
    top_k: int | None = None
    top_logprobs: int | None = None
    top_p: float | None = None
    truncate_sequence: bool | None = None
    truncation: dict[str, Any] | None = None


@dataclass(kw_only=True)
class ResponsesDelta:
    """Response delta for streaming"""

    output: list[ResponsesDeltaOutput] | None = None
    status: str | None = None


@dataclass(kw_only=True)
class ResponsesDeltaContent:
    """Response delta content item"""

    text: str | None = None
    type: str


@dataclass(kw_only=True)
class ResponsesDeltaOutput:
    """Response delta output item"""

    content: list[ResponsesDeltaContent] | None = None
    id: str
    type: str


@dataclass(kw_only=True)
class ResponsesError:
    """Response error"""

    message: str
    type: str


@dataclass(kw_only=True)
class ResponsesIncompleteDetails:
    """Incomplete details for incomplete responses"""

    reason: str


@dataclass(kw_only=True)
class ResponsesInputTokensDetails:
    """Input tokens details"""

    audio_tokens: int | None = None
    cached_tokens: int | None = None
    image_tokens: int | None = None
    text_tokens: int | None = None


@dataclass(kw_only=True)
class ResponsesObject:
    """Response object"""

    created_at: float
    error: ResponsesError | None = None
    id: str
    incomplete_details: ResponsesIncompleteDetails | None = None
    instructions: str | None = None
    metadata: Any = None
    model: str
    object: str
    output: list[ResponsesOutput]
    output_text: str | None = None
    status: str
    usage: ResponsesUsage | None = None


@dataclass(kw_only=True)
class ResponsesOutput:
    """Response output item"""

    content: list[ResponsesContent]
    id: str
    role: str
    status: str | None = None
    type: str


@dataclass(kw_only=True)
class ResponsesOutputTokensDetails:
    """Output tokens details"""

    audio_tokens: int | None = None
    reasoning_tokens: int | None = None
    text_tokens: int | None = None


@dataclass(kw_only=True)
class ResponsesUsage:
    """Response usage information"""

    input_tokens: int
    input_tokens_details: ResponsesInputTokensDetails | None = None
    output_tokens: int
    output_tokens_details: ResponsesOutputTokensDetails | None = None
    total_tokens: int


@dataclass(kw_only=True)
class RuntimeSpec:
    chat_template: str | None = None
    device: str | None = None
    isq: str | None = None
    jinja_explicit: str | None = None
    max_model_len: int | None = None
    max_seqs: int | None = None
    no_kv_cache: bool | None = None
    paged_attn: bool | None = None
    prefix_cache_n: int | None = None
    seed: int | None = None
    token_source: str | None = None


class SearchContextSize(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(kw_only=True)
class SerializedSession:
    """Wire format. Images and video frames are base64 PNGs."""

    files: list[Any] | None = None
    images: list[str] | None = None
    messages: list[Any]
    videos: list[SerializedVideo] | None = None


@dataclass(kw_only=True)
class SerializedVideo:
    fps: float
    frames: list[str]
    sampled_indices: list[int]
    total_num_frames: int


@dataclass(kw_only=True)
class SkillListObject:
    data: list[SkillObject]
    object: str


@dataclass(kw_only=True)
class SkillListQuery:
    limit: int | None = None
    page: str | None = None
    source: str | None = None


@dataclass(kw_only=True)
class SkillObject:
    created_at: int
    description: str
    id: str
    latest_version: int
    name: str
    object: str


@dataclass(kw_only=True)
class SkillVersionObject:
    created_at: int
    description: str
    id: str
    name: str
    object: str
    skill_id: str
    version: int


@dataclass(kw_only=True)
class SkillsSpec:
    """Where uploaded skills are kept; requests reference them from the shell tool."""

    root: str | None = None


@dataclass(kw_only=True)
class SourceMeta:
    """Which agentic tool produced the file, and when in the session."""

    round: int
    tool: str
    turn: int


@dataclass(kw_only=True)
class SpeechGenerationRequest:
    """Speech generation request"""

    input: str
    model: str | None = None
    response_format: AudioResponseFormat


class SpeechLoaderType(str, Enum):
    DIA = "dia"


@dataclass(kw_only=True)
class StreamOptions:
    """Stream options configuration"""

    include_usage: bool | None = None


@dataclass(kw_only=True)
class TextConfig:
    """Text output configuration"""

    format: TextFormat | None = None


@dataclass(kw_only=True)
class TextFormatText:
    """Plain text output"""

    type: Literal["text"] = "text"


@dataclass(kw_only=True)
class TextFormatJsonSchema:
    """JSON output with optional schema"""

    name: str
    schema: Any = None
    strict: bool | None = None
    type: Literal["json_schema"] = "json_schema"


@dataclass(kw_only=True)
class TextFormatJsonObject:
    """JSON object output"""

    type: Literal["json_object"] = "json_object"


@dataclass(kw_only=True)
class Tool:
    """Tool definition"""

    function: Function
    type: ToolType = "function"


@dataclass(kw_only=True)
class ToolCall:
    """Represents a tool call made by the assistant"""

    function: FunctionCalled
    id: str | None = None
    type: ToolType = "function"


class ToolType(str, Enum):
    """Type of tool"""

    FUNCTION = "function"


class TruncationStrategy(str, Enum):
    """Truncation strategy for input"""

    AUTO = "auto"
    DISABLED = "disabled"


@dataclass(kw_only=True)
class TuneModelRequest:
    cpu: bool | None = None
    dtype: str | None = None
    hf_revision: str | None = None
    max_batch_size: int | None = None
    max_image_length: int | None = None
    max_num_images: int | None = None
    max_seq_len: int | None = None
    model_id: str
    profile: TuneProfileRequest | None = None
    requested_isq: str | None = None
    token_source: str | None = None


class TuneProfileRequest(str, Enum):
    QUALITY = "quality"
    BALANCED = "balanced"
    FAST = "fast"


@dataclass(kw_only=True)
class UnloadLoraAdapterRequest:
    expected_generation: str | None = None
    lora_int_id: int | None = None
    lora_name: str
    model: str | None = None


@dataclass(kw_only=True)
class UqffWriteSpecConfig:
    base_model: str | None = None
    output: str
    repo_id: str | None = None
    types: list[str] | None = None


@dataclass(kw_only=True)
class UrlCitation:
    """URL citation details"""

    title: str | None = None
    url: str


class WebSearchContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"


@dataclass(kw_only=True)
class WebSearchFilters:
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None


@dataclass(kw_only=True)
class WebSearchImageSettings:
    caption: bool | None = None
    max_results: int | None = None


@dataclass(kw_only=True)
class WebSearchOptions:
    external_web_access: bool | None = None
    extract_description: str | None = None
    filters: WebSearchFilters | None = None
    image_settings: WebSearchImageSettings | None = None
    return_token_budget: WebSearchReturnTokenBudget | None = None
    search_content_types: list[WebSearchContentType] | None = None
    search_context_size: SearchContextSize | None = None
    search_description: str | None = None
    user_location: WebSearchUserLocation | None = None


class WebSearchReturnTokenBudget(str, Enum):
    DEFAULT = "default"
    UNLIMITED = "unlimited"


@dataclass(kw_only=True)
class WebSearchUserLocationApproximate:
    approximate: ApproximateUserLocation
    type: Literal["approximate"] = "approximate"


AdapterSelection = Union[str, AdapterGenerationSelection]
AllowedToolChoice = Union[AllowedToolChoiceFunction, AllowedToolChoiceWebSearchPreview, AllowedToolChoiceCodeInterpreter, AllowedToolChoiceShell]
AnthropicMessageContent = Union[str, list[AnthropicContentBlock]]
AnthropicSystem = Union[str, list[AnthropicContentBlock]]
EmbeddingInput = Union[str, list[str], list[int], list[list[int]]]
EmbeddingVector = Union[list[float], str]
Grammar = Union[GrammarRegex, GrammarJsonSchema, GrammarLlguidance, GrammarLark]
MessageInnerContent = Union[str, dict[str, str]]
MessageContent = Union[str, list[dict[str, MessageInnerContent]]]
ModelSelected = Union[ModelSelectedRun, ModelSelectedPlain, ModelSelectedXLora, ModelSelectedLora, ModelSelectedGGUF, ModelSelectedXLoraGGUF, ModelSelectedLoraGGUF, ModelSelectedGGML, ModelSelectedXLoraGGML, ModelSelectedLoraGGML, ModelSelectedMultimodalPlain, ModelSelectedDiffusionPlain, ModelSelectedSpeech, ModelSelectedMultiModel, ModelSelectedEmbedding]
OpenAiCodeInterpreterContainer = Union[str, OpenAiCodeInterpreterAutoContainer]
OpenAiNamespaceEntry = Union[OpenAiResponsesFunctionTool, Any]
OpenAiShellEnvironment = Union[OpenAiShellEnvironmentContainerAuto, OpenAiShellEnvironmentLocal, OpenAiShellEnvironmentContainerReference]
OpenAiShellSkill = Union[OpenAiShellSkillSkillReference, OpenAiShellSkillLocal]
OpenAiTool = Union[Tool, OpenAiResponsesFunctionTool, OpenAiWebSearchTool, OpenAiCodeInterpreterTool, OpenAiShellTool, OpenAiNamespaceTool]
OpenAiWebSearchUserLocation = OpenAiWebSearchUserLocationApproximate
OpenResponsesInput = Union[str, list[Union[OpenResponsesInputMessage, OpenResponsesInputItemReference, OpenResponsesInputFunctionCall, OpenResponsesInputFunctionCallOutput, OpenResponsesInputReasoning]]]
OpenResponsesStreamEvent = Union[OpenResponsesStreamEventResponseCreated, OpenResponsesStreamEventResponseInProgress, OpenResponsesStreamEventResponseOutputItemAdded, OpenResponsesStreamEventResponseContentPartAdded, OpenResponsesStreamEventResponseOutputTextDelta, OpenResponsesStreamEventResponseContentPartDone, OpenResponsesStreamEventResponseOutputItemDone, OpenResponsesStreamEventResponseFunctionCallArgumentsDelta, OpenResponsesStreamEventResponseFunctionCallArgumentsDone, OpenResponsesStreamEventResponseReasoningTextDelta, OpenResponsesStreamEventResponseReasoningTextDone, OpenResponsesStreamEventResponseCompleted, OpenResponsesStreamEventResponseFailed, OpenResponsesStreamEventResponseIncomplete, OpenResponsesStreamEventError]
OutputContent = Union[OutputContentOutputText, OutputContentRefusal]
OutputItem = Union[OutputItemMessage, OutputItemFunctionCall, OutputItemShellCall, OutputItemShellCallOutput, OutputItemReasoning]
ResponseFormat = Union[ResponseFormatText, ResponseFormatJsonObject, ResponseFormatJsonSchema]
ResponsesMessages = Union[list[Message], str]
StopTokens = Union[list[str], str]
TextFormat = Union[TextFormatText, TextFormatJsonSchema, TextFormatJsonObject]
ToolChoice = Union[Literal["none"], Literal["auto"], Literal["required"], AllowedToolsToolChoice, BuiltinToolChoice, Tool, NamedFunctionToolChoice]
UqffWriteSpec = Union[str, UqffWriteSpecConfig]
WebSearchUserLocation = WebSearchUserLocationApproximate
