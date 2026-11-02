//! `approvals.request` bodies the sidecar sends to Core.

use serde_json::{json, Value};

/// The body for a heal that gave up. Core words the summary from `attempts`
/// in the reader's language; an older Core, which ignores the field, shows
/// the English `note`.
pub(crate) fn heal_exhausted(source_id: &str, note: &str, attempts: Option<u32>) -> Value {
    let mut body = json!({ "kind": "heal_exhausted", "source_id": source_id, "note": note });
    if let Some(attempts) = attempts {
        body["attempts"] = json!(attempts);
    }
    body
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gave_up_body_carries_the_attempt_count_beside_the_english_note() {
        assert_eq!(
            heal_exhausted(
                "conv-1",
                "Auto-fix attempts used: 2. Review it manually.",
                Some(2)
            ),
            json!({
                "kind": "heal_exhausted",
                "source_id": "conv-1",
                "note": "Auto-fix attempts used: 2. Review it manually.",
                "attempts": 2,
            })
        );
        assert!(heal_exhausted("conv-1", "n", None)
            .get("attempts")
            .is_none());
    }
}
