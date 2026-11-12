use std::path::PathBuf;

use futures::future::BoxFuture;
use serde_json::{Value, json};

use super::{Tool, ToolContext, ToolKind, ToolOutcome, str_arg, usize_arg};
use crate::session::image_type;

const DEFAULT_LINES: usize = 2000;
const MAX_LINE_CHARS: usize = 2000;
// past this an image is refused rather than sent with every later request
const MAX_IMAGE_BYTES: u64 = 20 << 20;

pub struct ReadTool;

impl Tool for ReadTool {
    fn name(&self) -> &'static str {
        "read"
    }

    fn description(&self) -> &'static str {
        "Read a text file, with line numbers. `offset` (1-based) and `limit` select a range of lines; long files are \
         cut at 2000 lines unless a range is given. When you can see images, reading an image file (png, jpg, gif, \
         webp, bmp) shows you the image, for instance a screenshot you took to check your work."
    }

    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {
            "path": {"type": "string", "description": "File path, relative to the project or absolute"},
            "offset": {"type": "integer", "description": "First line to read, 1-based"},
            "limit": {"type": "integer", "description": "Number of lines to read"}},
            "required": ["path"]})
    }

    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }

    fn paths(&self, args: &Value, ctx: &ToolContext) -> Vec<PathBuf> {
        str_arg(args, "path")
            .map(|p| vec![ctx.resolve(p)])
            .unwrap_or_default()
    }

    fn run<'a>(&'a self, args: Value, ctx: &'a ToolContext) -> BoxFuture<'a, ToolOutcome> {
        Box::pin(async move {
            let path = match str_arg(&args, "path") {
                Ok(path) => ctx.resolve(path),
                Err(error) => return error,
            };
            if image_type(&path).is_some() {
                return read_image(path, ctx).await;
            }
            let text = match tokio::fs::read(&path).await {
                Ok(bytes) => match String::from_utf8(bytes) {
                    Ok(text) => text,
                    Err(_) => {
                        return ToolOutcome::error(format!(
                            "{} is not UTF-8 text",
                            ctx.display(&path)
                        ));
                    }
                },
                Err(error) => {
                    return ToolOutcome::error(format!(
                        "cannot read {}: {error}",
                        ctx.display(&path)
                    ));
                }
            };
            let offset = usize_arg(&args, "offset").unwrap_or(1).max(1);
            let limit = usize_arg(&args, "limit").unwrap_or(DEFAULT_LINES);
            let total = text.lines().count();
            let mut out = String::new();
            for (n, line) in text.lines().enumerate().skip(offset - 1).take(limit) {
                let line = match line.char_indices().nth(MAX_LINE_CHARS) {
                    Some((cut, _)) => format!("{}...", &line[..cut]),
                    None => line.to_string(),
                };
                out.push_str(&format!("{:>6}\t{line}\n", n + 1));
            }
            let last = (offset - 1 + limit).min(total);
            if total == 0 {
                out.push_str("(empty file)\n");
            } else if last < total {
                out.push_str(&format!(
                    "... {} more lines (read with offset {})\n",
                    total - last,
                    last + 1
                ));
            }
            ToolOutcome::ok(out)
        })
    }
}

// an image goes to the model after the round's tool results; the result itself only says what it is
async fn read_image(path: PathBuf, ctx: &ToolContext) -> ToolOutcome {
    let shown = ctx.display(&path);
    if !ctx.images {
        return ToolOutcome::error(format!(
            "{shown} is an image, and this model can't see images"
        ));
    }
    let size = match tokio::fs::metadata(&path).await {
        Ok(meta) if meta.is_file() => meta.len(),
        Ok(_) => return ToolOutcome::error(format!("{shown} is not a file")),
        Err(error) => return ToolOutcome::error(format!("cannot read {shown}: {error}")),
    };
    if size > MAX_IMAGE_BYTES {
        return ToolOutcome::error(format!(
            "{shown} is {size} bytes, over the {MAX_IMAGE_BYTES}-byte limit for an image"
        ));
    }
    ToolOutcome::with_images(
        format!("{shown}: an image of {size} bytes, shown to you after the tool results"),
        vec![path],
    )
}

#[cfg(test)]
mod tests {
    use super::ReadTool;
    use crate::tools::{Tool, ToolContext};
    use serde_json::json;

    #[tokio::test]
    async fn an_image_is_shown_to_a_model_that_sees_images_and_refused_otherwise() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("shot.PNG"), b"\x89PNG\r\n\x1a\n").unwrap();
        let mut ctx = ToolContext {
            root: dir.path().to_path_buf(),
            images: true,
        };
        let out = ReadTool.run(json!({"path": "shot.PNG"}), &ctx).await;
        assert!(!out.is_error, "{}", out.text);
        assert_eq!(out.images, [dir.path().join("shot.PNG")]);
        ctx.images = false;
        let out = ReadTool.run(json!({"path": "shot.PNG"}), &ctx).await;
        assert!(out.is_error && out.images.is_empty());
    }

    #[tokio::test]
    async fn reads_a_numbered_range() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "one\ntwo\nthree\nfour\n").unwrap();
        let ctx = ToolContext {
            root: dir.path().to_path_buf(),
            images: false,
        };
        let out = ReadTool
            .run(json!({"path": "f.txt", "offset": 2, "limit": 2}), &ctx)
            .await;
        assert!(!out.is_error);
        assert_eq!(
            out.text,
            "     2\ttwo\n     3\tthree\n... 1 more lines (read with offset 4)\n"
        );
        assert!(
            ReadTool
                .run(json!({"path": "missing"}), &ctx)
                .await
                .is_error
        );
    }
}
