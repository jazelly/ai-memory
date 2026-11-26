//! Integration tests for `/session/bind`: precedence over markers, startup
//! injection, active-project pointer, `SessionEnd` cleanup, per-operator
//! isolation, `AuthLevel::User` grants, and 409 on already-captured sessions.

use ai_memory_core::{
    ActorKey, AuthLevel, AuthorizedViewer, IdentityKey, NewUser, ObservationKind, ProjectId,
    SessionId, UserRole, WorkspaceId,
};
use ai_memory_store::{AccessMode, GrantLevel};
use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use tempfile::TempDir;
use tower::ServiceExt;

use crate::router::tests::{make_state, wait_for_kinds};
use crate::router::{HandoffQuery, HookState, hook_router, resolve_authorized_startup_scope};

fn app(
    state: &HookState,
    level: Option<AuthLevel>,
    actor: Option<&IdentityKey>,
    viewer: Option<AuthorizedViewer>,
) -> Router {
    let mut router = hook_router(state.clone());
    if let Some(level) = level {
        router = router.layer(axum::Extension(level));
    }
    if let Some(actor) = actor {
        router = router.layer(axum::Extension(actor.to_actor_context()));
    }
    if let Some(viewer) = viewer {
        router = router.layer(axum::Extension(viewer));
    }
    router
}

async fn send(
    app: Router,
    method: &str,
    uri: &str,
    body: Option<serde_json::Value>,
) -> (StatusCode, serde_json::Value) {
    let mut req = Request::builder().method(method).uri(uri);
    let body = match body {
        Some(json) => {
            req = req.header("content-type", "application/json");
            Body::from(serde_json::to_vec(&json).unwrap())
        }
        None => Body::empty(),
    };
    let response = app.oneshot(req.body(body).unwrap()).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .unwrap();
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json)
}

async fn post_hook(
    state: &HookState,
    auth: (
        Option<AuthLevel>,
        Option<&IdentityKey>,
        Option<AuthorizedViewer>,
    ),
    event: &str,
    agent: &str,
    sid: &str,
    expected: &[ObservationKind],
) {
    let (level, actor, viewer) = auth;
    let hook_event_name = match event {
        "session-end" => "SessionEnd",
        _ => "UserPromptSubmit",
    };
    let body = serde_json::json!({
        "session_id": sid,
        "cwd": "/repo/app",
        "hook_event_name": hook_event_name,
        "prompt": "hello from the bot",
    });
    let (status, _) = send(
        app(state, level, actor, viewer),
        "POST",
        &format!(
            "/hook?event={event}&agent={agent}&cwd=/repo/app&workspace=default&project=scratch&project_src=marker"
        ),
        Some(body),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    wait_for_kinds(state, sid, expected).await;
}

async fn scope_ids(state: &HookState, workspace: &str, project: &str) -> (WorkspaceId, ProjectId) {
    let ws = state
        .writer
        .get_or_create_workspace(workspace.to_owned())
        .await
        .unwrap();
    let proj = state
        .writer
        .get_or_create_project(ws, project, None)
        .await
        .unwrap();
    (ws, proj)
}

async fn session_scope(state: &HookState, sid: &str) -> (WorkspaceId, ProjectId) {
    let (ws, proj, _) = state
        .reader
        .find_session_scope(SessionId::from_native(sid))
        .await
        .unwrap()
        .expect("session captured");
    (ws, proj)
}

#[tokio::test]
async fn bound_session_captures_starts_updates_pointer_and_clears_on_session_end() {
    let tmp = TempDir::new().unwrap();
    let state = make_state(&tmp).await;
    let (status, body) = send(
        app(&state, None, None, None),
        "POST",
        "/session/bind",
        Some(serde_json::json!({
            "session_id": "bot-session-1",
            "agent": "claude-code",
            "workspace": "bot-42",
            "project": "research-bot",
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["binding"]["workspace"], "bot-42");
    assert_eq!(body["binding"]["project"], "research-bot");

    let bound = scope_ids(&state, "bot-42", "research-bot").await;
    let marked = scope_ids(&state, "default", "scratch").await;

    // SessionStart resolves to the bound scope; unbound session falls back to marker.
    let base_query = HandoffQuery {
        session_id: Some("bot-session-1".into()),
        agent: Some("claude-code".into()),
        cwd: Some("/repo/app".into()),
        workspace: Some("default".into()),
        project: Some("scratch".into()),
        ..Default::default()
    };
    assert_eq!(
        resolve_authorized_startup_scope(&state, &base_query, None, None)
            .await
            .unwrap(),
        bound
    );
    assert_eq!(
        resolve_authorized_startup_scope(
            &state,
            &HandoffQuery {
                session_id: Some("unbound-session".into()),
                ..base_query
            },
            None,
            None,
        )
        .await
        .unwrap(),
        marked
    );

    // Capture overrides the marker and updates the MCP active-project pointer.
    post_hook(
        &state,
        (None, None, None),
        "user-prompt-submit",
        "claude-code",
        "bot-session-1",
        &[ObservationKind::UserPrompt],
    )
    .await;
    assert_eq!(session_scope(&state, "bot-session-1").await, bound);
    assert_eq!(
        state.active_project.get_for(&ActorKey {
            user: None,
            session_id: Some("bot-session-1".into()),
        }),
        Some(bound)
    );

    // Unbound session from the same directory still uses the marker.
    post_hook(
        &state,
        (None, None, None),
        "user-prompt-submit",
        "claude-code",
        "plain-session",
        &[ObservationKind::UserPrompt],
    )
    .await;
    assert_eq!(session_scope(&state, "plain-session").await, marked);

    // SessionEnd lands in the bound scope and deletes the binding.
    post_hook(
        &state,
        (None, None, None),
        "session-end",
        "claude-code",
        "bot-session-1",
        &[ObservationKind::UserPrompt, ObservationKind::SessionEnd],
    )
    .await;
    assert_eq!(session_scope(&state, "bot-session-1").await, bound);
    let (status, _) = send(
        app(&state, None, None, None),
        "GET",
        "/session/bind?session_id=bot-session-1",
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    // Resuming the same session after SessionEnd requires re-binding first so
    // the resumed prompt stays in `bound` instead of reverting to the marker.
    let (status, _) = send(
        app(&state, None, None, None),
        "POST",
        "/session/bind",
        Some(serde_json::json!({
            "session_id": "bot-session-1",
            "workspace": "bot-42",
            "project": "research-bot",
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    post_hook(
        &state,
        (None, None, None),
        "user-prompt-submit",
        "claude-code",
        "bot-session-1",
        &[
            ObservationKind::UserPrompt,
            ObservationKind::SessionEnd,
            ObservationKind::UserPrompt,
        ],
    )
    .await;
    assert_eq!(session_scope(&state, "bot-session-1").await, bound);
}

#[tokio::test]
async fn harness_restricted_binding_ignores_other_harnesses() {
    let tmp = TempDir::new().unwrap();
    let state = make_state(&tmp).await;
    let (status, _) = send(
        app(&state, None, None, None),
        "POST",
        "/session/bind",
        Some(serde_json::json!({
            "session_id": "codex-only",
            "agent": "codex",
            "workspace": "bot-1",
            "project": "notes",
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    post_hook(
        &state,
        (None, None, None),
        "user-prompt-submit",
        "claude-code",
        "codex-only",
        &[ObservationKind::UserPrompt],
    )
    .await;
    assert_eq!(
        session_scope(&state, "codex-only").await,
        scope_ids(&state, "default", "scratch").await
    );
}

#[tokio::test]
async fn user_auth_enforces_grants_isolation_and_rejects_already_captured_sessions() {
    let tmp = TempDir::new().unwrap();
    let state = make_state(&tmp).await;

    let alice_id = state
        .writer
        .create_human_user(
            NewUser {
                username: "alice".into(),
                name: None,
                email: None,
            },
            UserRole::User,
            None,
            false,
        )
        .await
        .unwrap();
    let bob_id = state
        .writer
        .create_human_user(
            NewUser {
                username: "bob".into(),
                name: None,
                email: None,
            },
            UserRole::User,
            None,
            false,
        )
        .await
        .unwrap();
    let alice = IdentityKey::User("alice".into());
    let bob = IdentityKey::User("bob".into());
    let alice_viewer = AuthorizedViewer(alice_id);
    let bob_viewer = AuthorizedViewer(bob_id);

    // Restricted target project where only Alice holds Write.
    let (_bot_ws, secret_proj) = scope_ids(&state, "bot-1", "secret").await;
    state
        .writer
        .set_access_mode(secret_proj, AccessMode::Restricted)
        .await
        .unwrap();
    state
        .writer
        .grant_memory(alice_id, secret_proj, GrantLevel::Write, None)
        .await
        .unwrap();

    // Anonymous caller is refused with 401 when operators are distinguished.
    let (status, _) = send(
        app(&state, Some(AuthLevel::Anonymous), None, None),
        "POST",
        "/session/bind",
        Some(serde_json::json!({"session_id": "s1", "workspace": "bot-1", "project": "secret"})),
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    // Bob (AuthLevel::User without grant) gets 403 on restricted project.
    let (status, _) = send(
        app(&state, Some(AuthLevel::User), Some(&bob), Some(bob_viewer)),
        "POST",
        "/session/bind",
        Some(serde_json::json!({"session_id": "s1", "workspace": "bot-1", "project": "secret"})),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);

    // Alice (AuthLevel::User with Write grant) succeeds; Bob gets 409 on GET/POST/DELETE,
    // while an unbound id returns 404 on GET (consistent 404 vs 409).
    let (status, _) = send(
        app(
            &state,
            Some(AuthLevel::User),
            Some(&alice),
            Some(alice_viewer),
        ),
        "POST",
        "/session/bind",
        Some(serde_json::json!({"session_id": "s1", "workspace": "bot-1", "project": "secret"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = send(
        app(
            &state,
            Some(AuthLevel::User),
            Some(&alice),
            Some(alice_viewer),
        ),
        "GET",
        "/session/bind?session_id=s1",
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // Alice's hook capture follows her binding; Bob's hook capture with a
    // distinct session_id cannot collide or use Alice's binding, and even with
    // the same native session_id on a fresh capture Bob ignores Alice's bind.
    post_hook(
        &state,
        (Some(AuthLevel::User), Some(&alice), Some(alice_viewer)),
        "user-prompt-submit",
        "claude-code",
        "s1",
        &[ObservationKind::UserPrompt],
    )
    .await;
    assert_eq!(
        session_scope(&state, "s1").await,
        scope_ids(&state, "bot-1", "secret").await
    );
    let (status, _) = send(
        app(&state, Some(AuthLevel::User), Some(&alice), Some(alice_viewer)),
        "POST",
        "/session/bind",
        Some(serde_json::json!({"session_id": "s-alice-only", "workspace": "bot-1", "project": "secret"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    post_hook(
        &state,
        (Some(AuthLevel::User), Some(&bob), Some(bob_viewer)),
        "user-prompt-submit",
        "claude-code",
        "s-alice-only",
        &[ObservationKind::UserPrompt],
    )
    .await;
    assert_eq!(
        session_scope(&state, "s-alice-only").await,
        scope_ids(&state, "default", "scratch").await
    );
    for (method, uri, body) in [
        ("GET", "/session/bind?session_id=s1", None),
        (
            "POST",
            "/session/bind",
            Some(serde_json::json!({"session_id": "s1", "workspace": "bot-2", "project": "notes"})),
        ),
        ("DELETE", "/session/bind?session_id=s1", None),
    ] {
        let (status, _) = send(
            app(&state, Some(AuthLevel::User), Some(&bob), Some(bob_viewer)),
            method,
            uri,
            body,
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT, "{method} {uri}");
    }
    let (status, _) = send(
        app(&state, Some(AuthLevel::User), Some(&bob), Some(bob_viewer)),
        "GET",
        "/session/bind?session_id=never-bound",
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    // Alice captures an unbound session under default/scratch; binding it late
    // to another scope returns 409, and Bob trying to bind it returns 409 too.
    post_hook(
        &state,
        (Some(AuthLevel::User), Some(&alice), Some(alice_viewer)),
        "user-prompt-submit",
        "claude-code",
        "early-captured",
        &[ObservationKind::UserPrompt],
    )
    .await;
    let (status, _) = send(
        app(&state, Some(AuthLevel::User), Some(&alice), Some(alice_viewer)),
        "POST",
        "/session/bind",
        Some(serde_json::json!({"session_id": "early-captured", "workspace": "bot-1", "project": "secret"})),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT);
    let (status, _) = send(
        app(&state, Some(AuthLevel::User), Some(&bob), Some(bob_viewer)),
        "POST",
        "/session/bind",
        Some(serde_json::json!({"session_id": "early-captured", "workspace": "default", "project": "scratch"})),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT);

    // Legacy unattributed session (actor_user = None) cannot be bound by a DB user.
    let legacy_sid = SessionId::from_native("legacy-unattributed");
    let (def_ws, def_proj) = scope_ids(&state, "default", "scratch").await;
    state
        .writer
        .begin_session(ai_memory_core::NewSession {
            id: legacy_sid,
            workspace_id: def_ws,
            project_id: def_proj,
            agent_kind: ai_memory_core::AgentKind::ClaudeCode,
            cwd: None,
            actor_user: None,
            occurred_at: Some(1_000),
            origin: None,
        })
        .await
        .unwrap();
    let (status, _) = send(
        app(
            &state,
            Some(AuthLevel::User),
            Some(&alice),
            Some(alice_viewer),
        ),
        "POST",
        "/session/bind",
        Some(serde_json::json!({
            "session_id": "legacy-unattributed",
            "workspace": "default",
            "project": "scratch",
        })),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT);
}

#[tokio::test]
async fn invalid_bind_requests_are_refused() {
    let tmp = TempDir::new().unwrap();
    let state = make_state(&tmp).await;
    for (label, body) in [
        (
            "reserved prefix",
            serde_json::json!({"session_id": "s", "workspace": "bot-1", "project": "_global"}),
        ),
        (
            "profile prefix",
            serde_json::json!({"session_id": "s", "workspace": "default", "project": "_profile"}),
        ),
        (
            "uppercase",
            serde_json::json!({"session_id": "s", "workspace": "Bot", "project": "notes"}),
        ),
        (
            "slash",
            serde_json::json!({"session_id": "s", "workspace": "bot-1", "project": "a/b"}),
        ),
        (
            "empty session",
            serde_json::json!({"session_id": " ", "workspace": "bot-1", "project": "notes"}),
        ),
        (
            "unknown agent",
            serde_json::json!({"session_id": "s", "agent": "nope", "workspace": "bot-1", "project": "notes"}),
        ),
    ] {
        let (status, _) = send(
            app(&state, None, None, None),
            "POST",
            "/session/bind",
            Some(body),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{label}");
    }
}
