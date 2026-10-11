//! /v1/realtime transcription sessions end to end on a tiny random-weight streaming Parakeet, over a real socket.

use base64::Engine as _;
use futures::{SinkExt, StreamExt};
use inference_server_core::inference_server_router_builder::InferenceRsServerRouterBuilder;
use serde_json::{Value, json};
use tokio_tungstenite::tungstenite::Message;

use crate::parakeet_support;

const RATE: u32 = 16_000;
const CHIRP_START_HZ: f64 = 200.0;
const CHIRP_RISE_HZ: f64 = 600.0;
const AMPLITUDE: f64 = 0.4;
// 100 ms appends
const APPEND_SAMPLES: usize = 1600;
// speech, then enough silence for the default 500 ms of server VAD to end the turn
const SPEECH_SECONDS: f64 = 2.0;
const SILENCE_SECONDS: f64 = 1.0;
// the beta's pcm16 rate
const BETA_RATE: u32 = 24_000;
const KEYS: &str = "team-a = key-alpha\n";
const KEY: &str = "key-alpha";

type Socket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

struct Server {
    address: std::net::SocketAddr,
    engine: inference_api::Engine,
    _checkpoint: tempfile::TempDir,
}

async fn server() -> anyhow::Result<Server> {
    serve(false).await.map(|(server, _)| server)
}

// the server, and with `keyed` its router, to sign a browser in through
async fn serve(keyed: bool) -> anyhow::Result<(Server, axum::Router)> {
    let checkpoint = parakeet_support::tiny_parakeet_checkpoint("nemotron_asr_streaming")?;
    let spec = serde_json::from_value(json!({
        "model": {"Transcription": {"model_id": checkpoint.path().to_string_lossy(), "dtype": "f32"}},
        "runtime": {"device": "cpu"},
    }))?;
    let engine = inference_api::Engine::load(spec).await?;
    let mut builder = InferenceRsServerRouterBuilder::new().with_engine(&engine);
    if keyed {
        builder = builder.with_api_keys(inference_server_core::auth::ApiKeys::parse(KEYS)?);
    }
    let router = builder.build().await?;
    let served = router.clone();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    tokio::spawn(async move { axum::serve(listener, served).await });
    Ok((
        Server {
            address,
            engine,
            _checkpoint: checkpoint,
        },
        router,
    ))
}

async fn connect(server: &Server, query: &str) -> anyhow::Result<Socket> {
    let url = format!("ws://{}/v1/realtime?{query}", server.address);
    Ok(tokio_tungstenite::connect_async(url).await?.0)
}

async fn next(socket: &mut Socket) -> anyhow::Result<Value> {
    loop {
        match socket.next().await {
            Some(Ok(Message::Text(text))) => return Ok(serde_json::from_str(text.as_str())?),
            Some(Ok(_)) => continue,
            other => anyhow::bail!("the socket ended: {other:?}"),
        }
    }
}

async fn send(socket: &mut Socket, event: Value) -> anyhow::Result<()> {
    Ok(socket.send(Message::Text(event.to_string().into())).await?)
}

// events until one of `kind`, which is returned with all those before it
async fn until(socket: &mut Socket, kind: &str) -> anyhow::Result<Vec<Value>> {
    let mut events = Vec::new();
    loop {
        let event = next(socket).await?;
        let done = event["type"] == kind;
        events.push(event);
        if done {
            return Ok(events);
        }
    }
}

fn chirp(seconds: f64) -> Vec<f32> {
    chirp_at(seconds, RATE)
}

fn chirp_at(seconds: f64, rate: u32) -> Vec<f32> {
    (0..(seconds * f64::from(rate)) as usize)
        .map(|i| {
            let t = i as f64 / f64::from(rate);
            let phase =
                2.0 * std::f64::consts::PI * (CHIRP_START_HZ * t + CHIRP_RISE_HZ * t * t / 2.0);
            (AMPLITUDE * phase.sin()) as f32
        })
        .collect()
}

fn appends(pcm: &[f32]) -> Vec<Value> {
    pcm.chunks(APPEND_SAMPLES)
        .map(|chunk| json!({"type": "input_audio_buffer.append", "audio": pcm16(chunk)}))
        .collect()
}

fn deltas(events: &[Value]) -> String {
    events
        .iter()
        .filter(|e| e["type"] == "conversation.item.input_audio_transcription.delta")
        .filter_map(|e| e["delta"].as_str())
        .collect()
}

// a manually committed item transcribes as an upload of its audio does
#[tokio::test]
async fn a_committed_item_transcribes_as_its_upload() -> anyhow::Result<()> {
    let server = server().await?;
    let pcm = chirp(SPEECH_SECONDS);
    let mut wav = Vec::new();
    inference_models_speech::utils::write_pcm_as_wav(&mut wav, &pcm, RATE, 1)?;
    let mut request = inference_api::openai::TranscriptionRequest::new(
        inference_api::openai::TranscriptionResponseFormat::Json,
    );
    request.model = "default".to_string();
    let whole: Value =
        serde_json::from_str(&server.engine.transcription(request, &wav).await?.body)?;

    let mut socket = connect(&server, "intent=transcription").await?;
    let created = next(&mut socket).await?;
    assert_eq!(created["type"], "session.created");
    assert_eq!(created["session"]["type"], "transcription");
    send(
        &mut socket,
        json!({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {
            "format": {"type": "audio/pcm", "rate": RATE},
            "transcription": {"model": "gpt-4o-transcribe"},
            "turn_detection": null,
        }}}}),
    )
    .await?;
    let updated = next(&mut socket).await?;
    assert_eq!(updated["type"], "session.updated");
    assert_eq!(updated["session"]["audio"]["input"]["format"]["rate"], RATE);
    assert_eq!(
        updated["session"]["audio"]["input"]["turn_detection"],
        Value::Null
    );
    for append in appends(&pcm) {
        send(&mut socket, append).await?;
    }
    send(&mut socket, json!({"type": "input_audio_buffer.commit"})).await?;
    let events = until(
        &mut socket,
        "conversation.item.input_audio_transcription.completed",
    )
    .await?;
    let committed = events
        .iter()
        .find(|e| e["type"] == "input_audio_buffer.committed")
        .expect("a committed event");
    let completed = events.last().expect("a completed event");
    assert_eq!(completed["item_id"], committed["item_id"]);
    assert_eq!(completed["transcript"], whole["text"]);
    assert_eq!(
        deltas(&events).trim_end(),
        whole["text"].as_str().unwrap_or_default()
    );
    assert_eq!(completed["usage"]["type"], "duration");
    assert!(
        (completed["usage"]["seconds"].as_f64().unwrap_or_default() - SPEECH_SECONDS).abs() < 1e-6
    );

    // an empty commit and an unknown event are errors that leave the session open
    send(
        &mut socket,
        json!({"type": "input_audio_buffer.commit", "event_id": "mine"}),
    )
    .await?;
    let error = next(&mut socket).await?;
    assert_eq!(error["type"], "error");
    assert_eq!(error["error"]["code"], "input_audio_buffer_commit_empty");
    assert_eq!(error["error"]["event_id"], "mine");
    send(&mut socket, json!({"type": "response.create"})).await?;
    assert_eq!(next(&mut socket).await?["error"]["code"], "unknown_event");
    send(&mut socket, json!({"type": "input_audio_buffer.clear"})).await?;
    assert_eq!(
        next(&mut socket).await?["type"],
        "input_audio_buffer.cleared"
    );
    Ok(())
}

// with server VAD, a stretch of speech and the silence after it make a turn the server commits itself; the beta's
// events are answered in kind
#[tokio::test]
async fn server_vad_commits_a_turn_after_its_silence() -> anyhow::Result<()> {
    let server = server().await?;
    let mut socket = connect(&server, "intent=transcription").await?;
    next(&mut socket).await?;
    send(
        &mut socket,
        json!({"type": "transcription_session.update", "session": {
            "input_audio_format": "pcm16",
            "turn_detection": {"type": "server_vad", "silence_duration_ms": 500},
        }}),
    )
    .await?;
    let updated = next(&mut socket).await?;
    assert_eq!(updated["type"], "transcription_session.updated");
    assert_eq!(updated["session"]["input_audio_format"], "pcm16");
    let mut audio = chirp_at(SPEECH_SECONDS, BETA_RATE);
    audio.extend(std::iter::repeat_n(
        0.0,
        (SILENCE_SECONDS * f64::from(BETA_RATE)) as usize,
    ));
    for append in appends(&audio) {
        send(&mut socket, append).await?;
    }
    let events = until(
        &mut socket,
        "conversation.item.input_audio_transcription.completed",
    )
    .await?;
    let kinds: Vec<&str> = events.iter().filter_map(|e| e["type"].as_str()).collect();
    let position = |kind: &str| kinds.iter().position(|k| *k == kind).expect(kind);
    assert!(
        position("input_audio_buffer.speech_started")
            < position("input_audio_buffer.speech_stopped")
    );
    assert!(
        position("input_audio_buffer.speech_stopped") < position("input_audio_buffer.committed")
    );
    let stopped = &events[position("input_audio_buffer.speech_stopped")];
    let end_ms = stopped["audio_end_ms"].as_u64().unwrap_or_default();
    // speech ends at 2 s of 24 kHz audio; the turn ends once the 500 ms of silence after it are in
    assert!((2400..=2700).contains(&end_ms), "{end_ms}");
    Ok(())
}

// a session for anything but transcription is refused before the upgrade
#[tokio::test]
async fn only_transcription_sessions_are_served() -> anyhow::Result<()> {
    let server = server().await?;
    let refused = connect(&server, "model=gpt-realtime&intent=conversation").await;
    let Err(error) = refused else {
        anyhow::bail!("a conversation session was upgraded");
    };
    assert!(error.to_string().contains("400"), "{error}");
    Ok(())
}

fn pcm16(audio: &[f32]) -> String {
    let bytes: Vec<u8> = audio
        .iter()
        .flat_map(|s| ((s * f32::from(i16::MAX)) as i16).to_le_bytes())
        .collect();
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

// two turns inside one append are two items, each with its own audio, and nothing opens an item before speech
#[tokio::test]
async fn turns_inside_one_append_are_items_of_their_own() -> anyhow::Result<()> {
    let server = server().await?;
    let mut socket = connect(&server, "intent=transcription").await?;
    next(&mut socket).await?;
    send(
        &mut socket,
        json!({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {
            "format": {"type": "audio/pcm", "rate": RATE},
            "turn_detection": {"type": "server_vad", "silence_duration_ms": 400, "prefix_padding_ms": 0},
        }}}}),
    )
    .await?;
    next(&mut socket).await?;
    let silence = |seconds: f64| vec![0.0f32; (seconds * f64::from(RATE)) as usize];
    let mut audio = silence(0.5);
    audio.extend(chirp(1.0));
    audio.extend(silence(0.6));
    audio.extend(chirp(1.0));
    audio.extend(silence(0.6));
    send(
        &mut socket,
        json!({"type": "input_audio_buffer.append", "audio": pcm16(&audio)}),
    )
    .await?;
    let completed = "conversation.item.input_audio_transcription.completed";
    let first = until(&mut socket, completed).await?;
    let second = until(&mut socket, completed).await?;
    let started = |events: &[Value]| {
        events
            .iter()
            .find(|e| e["type"] == "input_audio_buffer.speech_started")
            .and_then(|e| e["audio_start_ms"].as_u64())
    };
    // speech opens at 0.5 s and 2.1 s, to the 20 ms frame
    assert_eq!(started(&first), Some(500));
    assert_eq!(started(&second), Some(2100));
    let (a, b) = (first.last().unwrap(), second.last().unwrap());
    assert_ne!(a["item_id"], b["item_id"]);
    for done in [a, b] {
        // each item holds its turn and the 400 ms of silence that ended it, not the half second before the first
        let seconds = done["usage"]["seconds"].as_f64().unwrap_or_default();
        assert!((1.38..=1.42).contains(&seconds), "{seconds}");
    }
    let added = second
        .iter()
        .find(|e| e["type"] == "conversation.item.added")
        .expect("an added item");
    assert_eq!(added["previous_item_id"], a["item_id"]);
    Ok(())
}

// malformed events and audio are errors the session survives, and a refused update changes nothing
#[tokio::test]
async fn bad_events_are_errors_the_session_survives() -> anyhow::Result<()> {
    let server = server().await?;
    let mut socket = connect(&server, "intent=transcription").await?;
    next(&mut socket).await?;
    socket.send(Message::Text("not json".into())).await?;
    assert_eq!(next(&mut socket).await?["error"]["code"], "invalid_json");
    socket.send(Message::Binary(vec![0u8; 4].into())).await?;
    assert_eq!(next(&mut socket).await?["error"]["code"], "invalid_message");
    send(
        &mut socket,
        json!({"type": "input_audio_buffer.append", "audio": ""}),
    )
    .await?;
    assert_eq!(next(&mut socket).await?["error"]["code"], "invalid_value");
    send(
        &mut socket,
        json!({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {
            "format": {"type": "audio/pcm", "rate": 4000}, "turn_detection": null,
        }}}}),
    )
    .await?;
    assert_eq!(next(&mut socket).await?["error"]["code"], "invalid_value");
    send(
        &mut socket,
        json!({"type": "session.update", "session": {"type": "transcription"}}),
    )
    .await?;
    let input = next(&mut socket).await?["session"]["audio"]["input"].clone();
    assert_eq!(input["format"]["rate"], BETA_RATE);
    assert_eq!(input["turn_detection"]["type"], "server_vad");
    Ok(())
}

// on a keyed server a socket needs a key, which a browser offers as a subprotocol; a signed-in browser's cookie
// opens one only from the server's own origin
#[tokio::test(flavor = "multi_thread")]
async fn a_keyed_server_takes_a_key_or_a_same_origin_cookie() -> anyhow::Result<()> {
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    let (server, router) = serve(true).await?;
    assert!(connect(&server, "intent=transcription").await.is_err());
    let url = format!("ws://{}/v1/realtime?intent=transcription", server.address);
    let mut request = url.as_str().into_client_request()?;
    request.headers_mut().insert(
        "sec-websocket-protocol",
        format!("realtime, openai-insecure-api-key.{KEY}").parse()?,
    );
    let (mut socket, response) = tokio_tungstenite::connect_async(request).await?;
    assert_eq!(response.headers()["sec-websocket-protocol"], "realtime");
    assert_eq!(next(&mut socket).await?["type"], "session.created");

    let cookie = crate::keyed::sign_in(&router, KEY).await?;
    let with_origin = |origin: String| -> anyhow::Result<_> {
        let mut request = url.as_str().into_client_request()?;
        request.headers_mut().insert("cookie", cookie.parse()?);
        request.headers_mut().insert("origin", origin.parse()?);
        Ok(request)
    };
    let foreign = with_origin("http://127.0.0.1:1".to_string())?;
    assert!(tokio_tungstenite::connect_async(foreign).await.is_err());
    let own = with_origin(format!("http://{}", server.address))?;
    let (mut socket, _) = tokio_tungstenite::connect_async(own).await?;
    assert_eq!(next(&mut socket).await?["type"], "session.created");
    Ok(())
}
