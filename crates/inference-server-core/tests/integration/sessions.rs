//! Session import on a server shared by clients that only know their own ids.

use inference_api::files::FileUpload;
use inference_core::{SerializedSession, files::FileContent};

use crate::cancel::tiny_engine;

fn session_with(file: inference_core::files::File) -> SerializedSession {
    SerializedSession {
        messages: Vec::new(),
        images: Vec::new(),
        videos: Vec::new(),
        files: vec![file],
    }
}

fn text_of(file: &inference_core::files::File) -> Option<&str> {
    match &file.content {
        FileContent::Text { text, .. } => text.as_deref(),
        _ => None,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_session_import_keeps_the_body_of_a_file_already_stored() -> anyhow::Result<()> {
    let (_dir, engine) = tiny_engine().await?;
    let uploaded = engine
        .upload_file(FileUpload {
            filename: "notes.txt".to_string(),
            mime_type: Some("text/plain".to_string()),
            purpose: "user_data".to_string(),
            bytes: b"original".to_vec(),
        })
        .map_err(anyhow::Error::msg)?;
    let mut forged = (*engine
        .state()
        .find_file(&uploaded.id, None)
        .expect("uploaded"))
    .clone();
    forged.content = FileContent::Text {
        text: Some("replaced".to_string()),
        preview: None,
    };

    // An agentic request citing the upload as an input file tags it with that request's session.
    for session in ["session_other", "session_citing"] {
        if session == "session_citing" {
            assert!(engine.state().try_tag_file(&uploaded.id, session, None)?);
        }
        engine
            .put_session(session, session_with(forged.clone()))
            .map_err(anyhow::Error::msg)?;
        let kept = engine
            .state()
            .find_file(&uploaded.id, None)
            .expect("still stored");
        assert_eq!(text_of(&kept), Some("original"), "{session}");
    }

    forged.id = "file_restored".to_string();
    engine
        .put_session("session_mine", session_with(forged.clone()))
        .map_err(anyhow::Error::msg)?;
    let restored = engine
        .state()
        .find_file("file_restored", None)
        .expect("restored");
    assert_eq!(text_of(&restored), Some("replaced"));
    let listed = engine.session("session_mine").map_err(anyhow::Error::msg)?;
    assert_eq!(listed.files.len(), 1);
    Ok(())
}
