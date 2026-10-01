//! What the file store and the session store keep, seeded the way agent runs and image generation fill them.

use inference_core::{SerializedSession, files::FileContent};

use super::tiny_engine;
use crate::files::FileUpload;

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

fn upload(name: &str, mime_type: &str, bytes: &[u8]) -> FileUpload {
    FileUpload {
        filename: name.to_string(),
        mime_type: Some(mime_type.to_string()),
        purpose: "user_data".to_string(),
        bytes: bytes.to_vec(),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_session_import_keeps_the_body_of_a_file_already_stored() -> anyhow::Result<()> {
    let (_dir, engine) = tiny_engine().await?;
    let uploaded = engine
        .upload_file(upload("notes.txt", "text/plain", b"original"))
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

#[tokio::test(flavor = "multi_thread")]
async fn a_container_lists_and_serves_only_the_files_its_run_cited() -> anyhow::Result<()> {
    let (_dir, engine) = tiny_engine().await?;
    let uploaded = engine
        .upload_file(upload("table.csv", "text/csv", b"a,b\n1,2\n"))
        .map_err(anyhow::Error::msg)?;
    let listed = |container: &str| -> anyhow::Result<Vec<String>> {
        let files = engine
            .container_files(container)
            .map_err(anyhow::Error::msg)?;
        Ok(files.data.into_iter().map(|file| file.id).collect())
    };
    assert!(listed("cntr_mine")?.is_empty());
    // A Responses run tags the files it cites with its container id.
    assert!(
        engine
            .state()
            .try_tag_file(&uploaded.id, "cntr_mine", None)?
    );
    assert_eq!(listed("cntr_mine")?, vec![uploaded.id.clone()]);
    assert!(listed("cntr_other")?.is_empty());
    let cited = engine
        .container_file("cntr_mine", &uploaded.id)
        .map_err(anyhow::Error::msg)?;
    assert_eq!(cited.id, uploaded.id);
    let body = engine
        .container_file_content("cntr_mine", &uploaded.id)
        .map_err(anyhow::Error::msg)?;
    assert_eq!(body.bytes, b"a,b\n1,2\n");
    for container in ["cntr_other", "cntr_unused"] {
        assert!(engine.container_file(container, &uploaded.id).is_err());
        assert!(
            engine
                .container_file_content(container, &uploaded.id)
                .is_err()
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_generated_image_url_is_its_file_content() -> anyhow::Result<()> {
    let (_dir, engine) = tiny_engine().await?;
    let png = inference_core::images::encode_png(&image::DynamicImage::new_rgb8(5, 3))?;
    let url = crate::files::store_generated_image(engine.state(), None, png.clone(), None)
        .map_err(anyhow::Error::msg)?;
    let id = url
        .strip_prefix("/v1/files/")
        .and_then(|rest| rest.strip_suffix("/content"))
        .unwrap_or_else(|| panic!("{url} is a file content url"));
    let body = engine.file_content(id).map_err(anyhow::Error::msg)?;
    assert_eq!((body.mime_type.as_str(), body.bytes), ("image/png", png));
    Ok(())
}
