//! Which language a diagnosis is written in.
//!
//! The diagnosis is model output the user reads verbatim: in the Inbox row, the
//! approval body and the push. No catalog can translate it, so the system prompt
//! has to name the language. Core puts the node's `ui-language` on the
//! `core#run.failed` payload as `locale` (already resolved to a catalog locale);
//! the delivery route scopes the heal with [`with_locale`], and
//! [`system_prompt`] appends the language line to the diagnosis prompt.
//!
//! The corrected instruction is not translated: it is re-run as the user's own
//! instruction, so it stays in the language that instruction was written in. JSON
//! keys stay English because the engine parses them.
//!
//! This mirrors `english_language_name` / `reply_language_instruction` in Core's
//! `apps/core/src/i18n/reply_language.rs`; the satellite cannot link Core. A test
//! pins the table to the catalog's locale list so the two stay in sync. The line is
//! English prompt scaffolding, not user-facing copy, so it is not a catalog message.
//!
//! An older Core sends no `locale`; the prompt then stays exactly as it was (English).

use std::borrow::Cow;
use std::future::Future;

/// English names of the catalog locales, as a model is most reliably told them.
const LANGUAGE_NAMES: &[(&str, &str)] = &[
    ("ar", "Arabic"),
    ("de", "German"),
    ("en", "English"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("hi", "Hindi"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("pt-BR", "Brazilian Portuguese"),
    ("ru", "Russian"),
    ("zh-CN", "Simplified Chinese"),
];

tokio::task_local! {
    /// The UI locale of the failed run being healed, when Core sent one.
    static REPLY_LOCALE: Option<String>;
}

/// Run `heal` with `locale` as the language its diagnosis is written in.
pub async fn with_locale<F: Future>(locale: Option<String>, heal: F) -> F::Output {
    REPLY_LOCALE.scope(locale, heal).await
}

/// Whether a tag asks for Traditional Chinese, which the catalog does not ship.
fn is_traditional_chinese(tag: &str) -> bool {
    tag.split(['-', '_'])
        .skip(1)
        .any(|subtag| ["hant", "tw", "hk", "mo"].contains(&subtag.to_ascii_lowercase().as_str()))
}

/// The English name of the catalog language `tag` names (`ja` and `ja-JP` are
/// `Japanese`; an unsupported or blank tag is `English`).
pub fn english_language_name(tag: &str) -> &'static str {
    let tag = tag.trim().replace('_', "-");
    if let Some((_, name)) = LANGUAGE_NAMES
        .iter()
        .find(|(code, _)| code.eq_ignore_ascii_case(&tag))
    {
        return name;
    }
    let language = tag.split('-').next().unwrap_or_default();
    if language.eq_ignore_ascii_case("zh") && is_traditional_chinese(&tag) {
        return "English";
    }
    LANGUAGE_NAMES
        .iter()
        .find(|(code, _)| {
            code.split('-')
                .next()
                .is_some_and(|base| base.eq_ignore_ascii_case(language))
        })
        .map_or("English", |(_, name)| name)
}

/// The prompt line asking for the diagnosis in `tag`'s language, or `None` for
/// English (the prompt is already English, so it stays exactly as it was).
pub fn reply_language_instruction(tag: &str) -> Option<String> {
    let name = english_language_name(tag);
    (name != "English").then(|| {
        format!(
            "Write the diagnosis in {name}. Keep corrected_prompt in the language of the \
             original instruction, and keep the JSON keys exactly as given, in English."
        )
    })
}

/// `base` for a heal with no locale or an English one; otherwise `base` plus the
/// language line for `locale`.
pub fn system_prompt_for<'a>(base: &'a str, locale: Option<&str>) -> Cow<'a, str> {
    match locale.and_then(reply_language_instruction) {
        Some(line) => Cow::Owned(format!("{base}\n\n{line}")),
        None => Cow::Borrowed(base),
    }
}

/// [`system_prompt_for`] with the locale of the heal in scope (see [`with_locale`]).
pub fn system_prompt(base: &str) -> Cow<'_, str> {
    let locale = REPLY_LOCALE.try_with(Clone::clone).ok().flatten();
    system_prompt_for(base, locale.as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = "Respond with JSON.";

    #[test]
    fn japanese_asks_for_a_japanese_diagnosis_and_keeps_the_instruction_language() {
        let prompt = system_prompt_for(BASE, Some("ja"));
        assert!(prompt.starts_with(BASE));
        assert!(prompt.contains("Write the diagnosis in Japanese."));
        assert!(
            prompt.contains("Keep corrected_prompt in the language of the original instruction")
        );
    }

    #[test]
    fn english_and_a_missing_locale_leave_the_prompt_unchanged() {
        assert_eq!(system_prompt_for(BASE, Some("en")), BASE);
        assert_eq!(system_prompt_for(BASE, Some("en-GB")), BASE);
        assert_eq!(system_prompt_for(BASE, None), BASE);
        assert_eq!(system_prompt_for(BASE, Some("  ")), BASE);
        assert_eq!(system_prompt_for(BASE, Some("tlh")), BASE);
    }

    #[test]
    fn tags_resolve_to_catalog_languages() {
        assert_eq!(english_language_name("pt-BR"), "Brazilian Portuguese");
        assert_eq!(english_language_name("pt_br"), "Brazilian Portuguese");
        assert_eq!(english_language_name("ja-JP"), "Japanese");
        assert_eq!(english_language_name("zh-CN"), "Simplified Chinese");
        assert_eq!(english_language_name("zh-TW"), "English");
        assert_eq!(english_language_name("zh-Hant"), "English");
    }

    /// The table covers exactly English plus the catalog's locales (Core's
    /// `catalog::LOCALES`), so it stays in sync with Core's own table.
    #[test]
    fn table_matches_the_catalog_locales() {
        const CATALOG: &[&str] = &[
            "ar", "de", "es", "fr", "hi", "it", "ja", "ko", "pt-BR", "ru", "zh-CN",
        ];
        let mut expected: Vec<&str> = CATALOG.iter().copied().chain(["en"]).collect();
        expected.sort_unstable();
        let mut codes: Vec<&str> = LANGUAGE_NAMES.iter().map(|(code, _)| *code).collect();
        codes.sort_unstable();
        assert_eq!(codes, expected);
        for code in CATALOG {
            assert_ne!(english_language_name(code), "English", "{code}");
        }
    }

    #[tokio::test]
    async fn the_scoped_locale_reaches_the_system_prompt() {
        assert_eq!(system_prompt(BASE), BASE);
        let scoped = with_locale(Some("de".to_owned()), async {
            system_prompt(BASE).into_owned()
        })
        .await;
        assert!(scoped.contains("Write the diagnosis in German."));
        let unscoped = with_locale(None, async { system_prompt(BASE).into_owned() }).await;
        assert_eq!(unscoped, BASE);
    }
}
