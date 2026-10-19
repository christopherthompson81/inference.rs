//! Which tool calls run on their own and which ask the user first.

use std::collections::HashSet;
use std::str::FromStr;

use serde_json::Value;

use crate::tools::{Tool, ToolContext, ToolKind};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PermissionMode {
    /// Reads inside the project run; edits, commands and anything outside the project ask.
    #[default]
    Ask,
    /// Everything runs.
    Auto,
    /// Reads inside the project run, reads outside ask, edits and commands are refused.
    ReadOnly,
}

impl FromStr for PermissionMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ask" => Ok(Self::Ask),
            "auto" => Ok(Self::Auto),
            "read-only" | "readonly" => Ok(Self::ReadOnly),
            other => Err(format!(
                "unknown permission mode `{other}` (ask, auto, read-only)"
            )),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Ask,
    Deny(String),
}

/// The user's answer to an approval prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Approval {
    Once,
    /// Allow this tool inside the project for the rest of the session.
    Always,
    Deny,
}

#[derive(Debug, Default)]
pub struct Permissions {
    pub mode: PermissionMode,
    always: HashSet<String>,
}

impl Permissions {
    pub fn new(mode: PermissionMode) -> Self {
        Self {
            mode,
            always: HashSet::new(),
        }
    }

    pub fn check(&self, tool: &dyn Tool, args: &Value, ctx: &ToolContext) -> Decision {
        let inside = tool.paths(args, ctx).iter().all(|path| ctx.is_inside(path));
        match (self.mode, tool.kind()) {
            (PermissionMode::Auto, _) => Decision::Allow,
            (_, ToolKind::Read) if inside => Decision::Allow,
            (_, ToolKind::Read) => Decision::Ask,
            (PermissionMode::ReadOnly, _) => Decision::Deny("the session is read-only".to_string()),
            // a blanket yes covers the project; a path outside it asks every time
            (PermissionMode::Ask, _) if inside && self.always.contains(tool.name()) => {
                Decision::Allow
            }
            (PermissionMode::Ask, _) => Decision::Ask,
        }
    }

    pub fn allow_always(&mut self, tool: &str) {
        self.always.insert(tool.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::{Decision, PermissionMode, Permissions};
    use crate::tools::{BashTool, EditTool, ReadTool, ToolContext};
    use serde_json::json;
    use std::path::PathBuf;

    #[test]
    fn reads_inside_run_and_the_rest_asks_until_allowed() {
        let ctx = ToolContext {
            root: PathBuf::from("/work/repo"),
        };
        let mut permissions = Permissions::new(PermissionMode::Ask);
        let inside = json!({"path": "src/lib.rs"});
        let outside = json!({"path": "../secrets"});
        assert_eq!(permissions.check(&ReadTool, &inside, &ctx), Decision::Allow);
        assert_eq!(permissions.check(&ReadTool, &outside, &ctx), Decision::Ask);
        assert_eq!(permissions.check(&EditTool, &inside, &ctx), Decision::Ask);
        permissions.allow_always("edit");
        assert_eq!(permissions.check(&EditTool, &inside, &ctx), Decision::Allow);
        assert_eq!(permissions.check(&EditTool, &outside, &ctx), Decision::Ask);
        assert_eq!(
            permissions.check(&BashTool, &json!({"command": "ls"}), &ctx),
            Decision::Ask
        );
        let read_only = Permissions::new(PermissionMode::ReadOnly);
        assert!(matches!(
            read_only.check(&EditTool, &inside, &ctx),
            Decision::Deny(_)
        ));
        assert_eq!(
            Permissions::new(PermissionMode::Auto).check(&EditTool, &outside, &ctx),
            Decision::Allow
        );
    }
}
