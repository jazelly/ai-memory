//! Keep work state out of the memory that is injected back into sessions.
//!
//! Work state is what is true about one piece of in-flight work: a PR number,
//! a commit SHA, a branch ref, a Noop dispatch/thread/message id. It goes
//! stale the moment the work lands, and a future session that reads it as
//! standing context acts on a fact that is no longer true.
//!
//! [`strip_work_state`] drops every line that carries such a token from pages
//! under the injected prefixes. It never errors and never rewrites a line;
//! a line is kept whole or removed whole.
//!
//! Token rules and the false positives they avoid:
//! - Commit SHA: a word of 7 to 40 lowercase hex characters holding at least
//!   one digit and one `a`-`f` letter. All-digit numbers (`1234567`) and
//!   all-letter words (`deadbeef`, `facade`) are kept. A run preceded by `#`
//!   is a CSS color (`#ff0000ff`) and is kept; 3/6-digit colors are too short
//!   to match anyway. A UUID's first segment satisfies the rule, so a line
//!   holding a UUID is dropped: in these pages a UUID is an instance id.
//!   A 64-character sha256 checksum is one word longer than 40, so it is kept.
//! - GitHub URLs to a pull request, issue, or commit.
//! - `PR #123`, `PR 123`, `pull/123`.
//! - `#123` only when it is clearly an issue ref: after `issue`, `fixes`,
//!   `closes`, or `resolves`, or qualified as `owner/repo#123`. A bare `#123`
//!   is kept, because `#333` is a color and `#1` an ordinal.
//! - `refs/heads/...` and `refs/pull/...`.
//! - Noop ids: `dispatch:<id>`, `thread:<id>` (no space after the colon, so
//!   prose like `thread: the worker` and `std::thread::spawn` are kept), and
//!   `msg-<hex>` with at least 8 hex characters (Noop's message id shape, so
//!   `msg-box` is kept).

use std::sync::LazyLock;

use regex::Regex;

/// Page prefixes whose bodies are injected as standing context and so must
/// never hold work state. `sessions/` and everything else is left as written.
const FILTERED_PREFIXES: &[&str] = &["_rules/", "procedures/", "profile/", "_slots/"];

static WORK_STATE_TOKEN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"https?://github\.com/[^/\s]+/[^/\s]+/(?:pull|issues|commit)/",
        r"|(?i:\bPR\s*#?\d+\b)",
        r"|\bpull/\d+\b",
        r"|(?i:\b(?:issues?|fix(?:es|ed)?|close[sd]?|resolve[sd]?)\s+#\d+\b)",
        r"|\b[\w.-]+/[\w.-]+#\d+\b",
        r"|\brefs/(?:heads|pull)/",
        r"|\b(?:dispatch|thread):[A-Za-z0-9]",
        r"|\bmsg-[0-9a-f]{8,}\b",
    ))
    .unwrap()
});

static HEX_WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b[0-9a-f]{7,40}\b").unwrap());

/// Remove every line of `body` that carries a work-state token, when `path`
/// is under a filtered prefix. Any other path returns `body` unchanged.
pub fn strip_work_state(path: &str, body: &str) -> String {
    if !FILTERED_PREFIXES.iter().any(|p| path.starts_with(p)) {
        return body.to_string();
    }
    body.split_inclusive('\n')
        .filter(|line| !has_work_state(line))
        .collect()
}

fn has_work_state(line: &str) -> bool {
    WORK_STATE_TOKEN.is_match(line) || HEX_WORD.find_iter(line).any(|m| is_sha(line, m))
}

fn is_sha(line: &str, m: regex::Match<'_>) -> bool {
    let word = m.as_str();
    !line[..m.start()].ends_with('#')
        && word.bytes().any(|b| b.is_ascii_digit())
        && word.bytes().any(|b| b.is_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strip_rule(body: &str) -> String {
        strip_work_state("_rules/git.md", body)
    }

    #[test]
    fn drops_lines_with_work_state_tokens() {
        let body = "# Git\n\
            - Squash before merge.\n\
            - Fixed in 3f9c2ab.\n\
            - See https://github.com/acme/app/pull/42 for context.\n\
            - Waiting on dispatch:01HXYZ to finish.\n\
            - PR #17 is in review.\n\
            - closes #88\n\
            - Pushed refs/heads/feature-x.\n\
            - Reply went to msg-1a2b3c4d.\n\
            - Owned by thread:abc123.\n\
            - Tracked as acme/app#12.\n";
        assert_eq!(strip_rule(body), "# Git\n- Squash before merge.\n");
    }

    #[test]
    fn keeps_lines_that_only_look_like_ids() {
        let body = "- Use #ff0000ff for errors and #333 for text.\n\
            - Retry up to 1234567 times, never deadbeef.\n\
            - The #1 cause is a stale lockfile.\n\
            - Spawn with std::thread::spawn; thread: one per core.\n\
            - Close the msg-box on submit.\n\
            - Verify the 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08 checksum.\n";
        assert_eq!(strip_rule(body), body);
    }

    #[test]
    fn drops_a_uuid_line() {
        assert_eq!(
            strip_rule("- Run 550e8400-e29b-41d4-a716-446655440000 failed.\n- keep\n"),
            "- keep\n"
        );
    }

    #[test]
    fn filters_only_injected_prefixes() {
        let body = "- Fixed in 3f9c2ab.\n";
        for path in [
            "procedures/a.md",
            "profile/me.md",
            "_slots/current-focus.md",
        ] {
            assert_eq!(strip_work_state(path, body), "", "{path}");
        }
        for path in ["sessions/s.md", "gotchas/a.md", "notes/a.md"] {
            assert_eq!(strip_work_state(path, body), body, "{path}");
        }
    }
}
