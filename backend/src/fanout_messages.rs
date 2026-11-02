//! Inbox, toast and push copy as localizable messages.
//!
//! Core renders a notify hint's `i18n` text per recipient (the inbox row and
//! toast in the node's `ui-language`, each push in the locale its device
//! registered with); the English rendering still goes out as the plain
//! `title`/`body`, which an older Core delivers as is.
//!
//! Each `*_MESSAGE` is `format!` with one named argument per placeholder
//! (`name = arg("name")`) so the i18n extractor catalogs the ICU source;
//! `tools/generate-core-i18n.mjs` copies the translations into Core's catalog.
//! Every message is a whole sentence. Core's renderer has no plural selector,
//! so counts are written as `label: {count}`.
#![allow(clippy::useless_format)]

use ryu_app_events::{NotifyHint, NotifyMessage, NotifyTemplate as Template};

pub(crate) const AUTO_FIX_GAVE_UP_MESSAGE: Template = |_| format!("Auto-fix gave up");
pub(crate) const RUN_DIAGNOSED_MESSAGE: Template = |_| format!("A failed run was diagnosed");
pub(crate) const FIX_WAITING_MESSAGE: Template = |arg| {
    format!(
        "Fix waiting in your inbox: {diagnosis}",
        diagnosis = arg("diagnosis")
    )
};

pub(crate) const HEAL_GAVE_UP_NOTE_MESSAGE: Template = |arg| {
    format!(
        "Auto-fix attempts used: {count}. Review it manually.",
        count = arg("count")
    )
};

/// The note for a run auto-fix gave up on after `attempts` attempts.
pub(crate) fn gave_up_note(attempts: u32) -> NotifyMessage {
    NotifyMessage::new(HEAL_GAVE_UP_NOTE_MESSAGE).count("count", i64::from(attempts))
}

/// Auto-fix ran out of attempts after `attempts` tries.
pub(crate) fn attempts_exhausted(attempts: u32) -> NotifyHint {
    NotifyHint::localized(
        NotifyMessage::new(AUTO_FIX_GAVE_UP_MESSAGE),
        vec![gave_up_note(attempts)],
    )
    .with_level("error")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gave_up_note_is_a_localizable_message_with_a_count() {
        let hint = attempts_exhausted(3);
        assert_eq!(hint.title, "Auto-fix gave up");
        assert_eq!(
            hint.body.as_deref(),
            Some("Auto-fix attempts used: 3. Review it manually.")
        );
        let wire = serde_json::to_value(&hint).unwrap();
        let body = &wire["i18n"]["body"][0];
        assert_eq!(
            body["message"],
            "Auto-fix attempts used: {count}. Review it manually."
        );
        assert_eq!(body["args"]["count"]["number"], 3);
        assert!(body.get("relay").is_none());
        assert_eq!(wire["level"], "error");
    }
}

/// A failed run was diagnosed and its fix waits for the user's approval.
pub(crate) fn run_diagnosed(diagnosis: &str) -> NotifyHint {
    NotifyHint::localized(
        NotifyMessage::new(RUN_DIAGNOSED_MESSAGE),
        vec![NotifyMessage::new(FIX_WAITING_MESSAGE).text("diagnosis", diagnosis)],
    )
}
