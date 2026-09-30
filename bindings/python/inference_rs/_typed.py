"""The engine with typed requests and responses, over the JSON engine (`Engine.json`)."""

import json
from collections.abc import Sequence

from . import types
from ._callbacks import HostCallbacks
from ._codec import from_data, from_json, to_json
from ._engine import JsonEngine, MediaAttachment, SkillFile, Stream
from ._owned import Blob


def _named(schema: dict):
    """Reads a stream event's data by its name; events the schema does not describe stay as parsed JSON."""
    return lambda name, data: from_data(schema[name], data) if name in schema else data


CHAT_EVENTS = _named({"chunk": types.ChatCompletionChunkResponse})
COMPLETION_EVENTS = _named({"chunk": types.CompletionChunkResponse})


def RESPONSE_EVENTS(name, data):
    return from_data(types.OpenResponsesStreamEvent, data)


def _parsed(stream: Stream, parse) -> Stream:
    stream.parse = parse
    return stream


class Engine:
    """A loaded model serving typed requests; each takes a class from `inference_rs.types` or its JSON string.

    Responses come back as those classes, with anything the schema leaves open as parsed JSON. Failures raise
    InferenceError with the protocol's error JSON. Calls block and release the GIL, so several threads may share an
    engine. Close it, or use `with`. `json` serves the same operations as JSON strings.
    """

    def __init__(self, spec: types.EngineSpec | dict | str, callbacks: HostCallbacks | None = None):
        """`spec` is what to load and how to run it: an EngineSpec, or its JSON as a dict or string."""
        self.json = JsonEngine(to_json(spec), callbacks)

    abi_version = staticmethod(JsonEngine.abi_version)
    build_version = staticmethod(JsonEngine.build_version)

    def close(self):
        self.json.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def chat(
        self,
        request: types.ChatCompletionRequest | str,
        media: Sequence[MediaAttachment] = (),
    ) -> types.ChatCompletionResponse:
        return from_json(types.ChatCompletionResponse, self.json.chat(to_json(request), media))

    def chat_stream(
        self,
        request: types.ChatCompletionRequest | str,
        media: Sequence[MediaAttachment] = (),
    ) -> Stream:
        """Events named `chunk` carry a ChatCompletionChunkResponse."""
        return _parsed(self.json.chat_stream(to_json(request), media), CHAT_EVENTS)

    def completion(self, request: types.CompletionRequest | str) -> types.CompletionResponse:
        return from_json(types.CompletionResponse, self.json.completion(to_json(request)))

    def completion_stream(self, request: types.CompletionRequest | str) -> Stream:
        """Events named `chunk` carry a CompletionChunkResponse."""
        return _parsed(self.json.completion_stream(to_json(request)), COMPLETION_EVENTS)

    def embeddings(self, request: types.EmbeddingRequest | str) -> types.EmbeddingResponse:
        return from_json(types.EmbeddingResponse, self.json.embeddings(to_json(request)))

    def anthropic_messages(self, request: types.AnthropicMessagesRequest | str) -> types.AnthropicMessageResponse:
        return from_json(
            types.AnthropicMessageResponse,
            self.json.anthropic_messages(to_json(request)),
        )

    def anthropic_messages_stream(self, request: types.AnthropicMessagesRequest | str) -> Stream:
        """Anthropic stream events, as parsed JSON."""
        return self.json.anthropic_messages_stream(to_json(request))

    def create_response(self, request: types.OpenResponsesCreateRequest | str) -> types.ResponseResource:
        return from_json(types.ResponseResource, self.json.create_response(to_json(request)))

    def response_stream(self, request: types.OpenResponsesCreateRequest | str) -> Stream:
        """OpenResponses events, each read as its variant of OpenResponsesStreamEvent."""
        return _parsed(self.json.response_stream(to_json(request)), RESPONSE_EVENTS)

    def get_response(self, response_id: str) -> types.ResponseResource:
        return from_json(types.ResponseResource, self.json.get_response(response_id))

    def delete_response(self, response_id: str) -> types.ResponseDeleted:
        return from_json(types.ResponseDeleted, self.json.delete_response(response_id))

    def cancel_response(self, response_id: str) -> types.ResponseResource:
        return from_json(types.ResponseResource, self.json.cancel_response(response_id))

    def re_isq(self, ggml_type: str) -> types.ReIsqResponse:
        """Requantizes a model that loaded with ISQ; answers once the engine has queued it."""
        return from_json(types.ReIsqResponse, self.json.re_isq(json.dumps({"ggml_type": ggml_type})))

    def calibration_start(self) -> types.CalibrationStatus:
        """Starts collecting activation statistics from the requests the engine serves."""
        return from_json(types.CalibrationStatus, self.json.calibration_start())

    def calibration_status(self) -> types.CalibrationStatus:
        return from_json(types.CalibrationStatus, self.json.calibration_status())

    def cache_stats(self) -> types.CacheStats:
        return from_json(types.CacheStats, self.json.cache_stats())

    def calibration_apply(self, save_cimatrix: str | None = None) -> types.CalibrationStatus:
        """Requantizes from the collected statistics; returns the status as it stood before."""
        request = {} if save_cimatrix is None else {"save_cimatrix": str(save_cimatrix)}
        return from_json(types.CalibrationStatus, self.json.calibration_apply(json.dumps(request)))

    def list_sessions(self) -> types.SessionList:
        return from_json(types.SessionList, self.json.list_sessions())

    def get_session(self, session_id: str) -> types.SerializedSession:
        return from_json(types.SerializedSession, self.json.get_session(session_id))

    def put_session(self, session_id: str, session: types.SerializedSession | str) -> types.SessionStored:
        """Imports a session under `session_id`, replacing any session there."""
        return from_json(types.SessionStored, self.json.put_session(session_id, to_json(session)))

    def delete_session(self, session_id: str) -> types.SessionDeleted:
        return from_json(types.SessionDeleted, self.json.delete_session(session_id))

    def tokenize(self, text: str, add_special_tokens: bool = True, model: str | None = None) -> list[int]:
        request = {"text": text, "add_special_tokens": add_special_tokens, "model": model}
        return from_json(types.TokenizeResponse, self.json.tokenize(json.dumps(request))).tokens

    def detokenize(self, tokens: Sequence[int], skip_special_tokens: bool = True, model: str | None = None) -> str:
        request = {"tokens": list(tokens), "skip_special_tokens": skip_special_tokens, "model": model}
        return from_json(types.DetokenizeResponse, self.json.detokenize(json.dumps(request))).text

    def list_models(self) -> types.ModelObjects:
        return from_json(types.ModelObjects, self.json.list_models())

    def unload_model(self, model_id: str) -> types.ModelStatusResponse:
        return from_json(types.ModelStatusResponse, self.json.unload_model(_model(model_id)))

    def reload_model(self, model_id: str) -> types.ModelStatusResponse:
        return from_json(types.ModelStatusResponse, self.json.reload_model(_model(model_id)))

    def model_status(self, model_id: str) -> types.ModelStatusResponse:
        return from_json(types.ModelStatusResponse, self.json.model_status(_model(model_id)))

    def list_lora_adapters(self, model: str | None = None) -> types.LoraAdapterListResponse:
        request = json.dumps({} if model is None else {"model": model})
        return from_json(types.LoraAdapterListResponse, self.json.list_lora_adapters(request))

    def load_lora_adapter(self, request: types.LoadLoraAdapterRequest | str) -> types.LoraAdapterObject:
        return from_json(types.LoraAdapterObject, self.json.load_lora_adapter(to_json(request)))

    def unload_lora_adapter(self, request: types.UnloadLoraAdapterRequest | str) -> types.LoraAdapterObject:
        return from_json(types.LoraAdapterObject, self.json.unload_lora_adapter(to_json(request)))

    def image_generation(self, request: types.ImageGenerationRequest | str) -> types.ImageGenerationResponse:
        return from_json(types.ImageGenerationResponse, self.json.image_generation(to_json(request)))

    def speech_generation(self, request: types.SpeechGenerationRequest | str) -> Blob:
        """The audio; its MIME type carries the sample rate and channel count."""
        return self.json.speech_generation(to_json(request))

    def resolve_approval(
        self, approval_id: str, decision: types.ApprovalDecisionRequest | str
    ) -> types.ApprovalDecisionResponse:
        return from_json(
            types.ApprovalDecisionResponse,
            self.json.resolve_approval(approval_id, to_json(decision)),
        )

    def upload_file(self, data: bytes, filename: str, purpose: str, mime_type: str | None = None) -> types.FileMetadata:
        return from_json(
            types.FileMetadata,
            self.json.upload_file(data, filename, purpose, mime_type),
        )

    def list_files(self) -> types.FileListObject:
        return from_json(types.FileListObject, self.json.list_files())

    def get_file(self, file_id: str) -> types.FileMetadata:
        return from_json(types.FileMetadata, self.json.get_file(file_id))

    def delete_file(self, file_id: str) -> types.FileDeleted:
        return from_json(types.FileDeleted, self.json.delete_file(file_id))

    def file_content(self, file_id: str) -> Blob:
        return self.json.file_content(file_id)

    def upload_skill(self, files: Sequence[SkillFile]) -> types.SkillObject:
        return from_json(types.SkillObject, self.json.upload_skill(files))

    def upload_skill_version(self, skill_id: str, files: Sequence[SkillFile]) -> types.SkillVersionObject:
        return from_json(types.SkillVersionObject, self.json.upload_skill_version(skill_id, files))

    def list_skills(self) -> types.SkillListObject:
        return from_json(types.SkillListObject, self.json.list_skills())

    def list_skill_versions(self, skill_id: str) -> types.AnthropicSkillVersionListObject:
        return from_json(
            types.AnthropicSkillVersionListObject,
            self.json.list_skill_versions(skill_id),
        )


def _model(model_id: str) -> str:
    return to_json(types.ModelOperationRequest(model_id=model_id))
