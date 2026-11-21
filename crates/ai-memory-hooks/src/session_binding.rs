//! `POST|GET|DELETE /session/bind`: per-session scope binding.
//!
//! Binds one native harness session id to a workspace/project before its first
//! hook fires. Hook capture and session-start injection consult the binding
//! ahead of stickiness and cwd / marker / git-remote derivation (only a managed
//! run's pinned scope outranks it). The binding is removed when the session
//! ends or after [`DEFAULT_BINDING_TTL_SECS`].

use std::sync::Arc;

use ai_memory_core::{
    AgentKind, AuthLevel, IdentityKey, NativeSessionIdentity, ProjectId, SessionId, WorkspaceId,
};
use ai_memory_store::{
    NewSessionScopeBinding, ProjectAccess, ResolvedScope, SessionScopeBinding, StoreError,
    authorize_scope_for, create_explicit_scope_guarded,
};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::router::{HookState, actor_identity, owner_stamp_for_event};

/// Longest workspace or project name accepted, matching `.ai-memory.toml` routes.
pub const MAX_BOUND_NAME_BYTES: usize = 512;

/// Bounded lifetime (7 days) so crashed sessions that never fire `SessionEnd`
/// do not accumulate rows indefinitely.
pub const DEFAULT_BINDING_TTL_SECS: i64 = 7 * 24 * 60 * 60;

/// Marker-file route grammar: `^[a-z0-9][a-z0-9._-]*$`, at most 512 bytes.
/// Leading `_` is forbidden, so reserved names (`_global`, `_profile*`) are
/// rejected here without a separate check.
#[must_use]
pub fn is_valid_bound_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_BOUND_NAME_BYTES
        && name.bytes().enumerate().all(|(index, byte)| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || (index > 0 && matches!(byte, b'.' | b'_' | b'-'))
        })
}

fn now_micros() -> i64 {
    jiff::Timestamp::now().as_microsecond()
}

pub(crate) async fn bound_scope_for(
    state: &HookState,
    native_session: Option<&str>,
    agent: AgentKind,
    actor: Option<&IdentityKey>,
) -> Result<Option<(WorkspaceId, ProjectId)>, StoreError> {
    let Some(native) = native_session.filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    let binding = state
        .reader
        .session_scope_binding(SessionId::from_native(native), now_micros())
        .await?;
    let owner = actor.map(IdentityKey::storage_key);
    Ok(binding
        .filter(|binding| binding.applies_to(owner.as_deref(), agent))
        .map(|binding| (binding.workspace_id, binding.project_id)))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BindSessionRequest {
    session_id: String,
    #[serde(default)]
    agent: Option<String>,
    workspace: String,
    project: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SessionBindingQuery {
    session_id: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct SessionBindingView {
    session_id: String,
    session_uuid: String,
    agent: Option<String>,
    workspace: String,
    project: String,
    workspace_id: String,
    project_id: String,
    created_at: String,
    updated_at: String,
    expires_at: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct BindSessionResponse {
    binding: SessionBindingView,
}

type Refusal = (StatusCode, Json<serde_json::Value>);

fn refusal(status: StatusCode, message: impl Into<String>) -> Refusal {
    (status, Json(serde_json::json!({ "error": message.into() })))
}

fn error(status: StatusCode, message: impl Into<String>) -> Response {
    refusal(status, message).into_response()
}

/// Fail closed when operator distinction is active and the caller has no
/// authenticated identity (or is `AuthLevel::Anonymous`). On single-operator
/// unauthenticated installs (`require_bearer` disabled), `Anonymous` is allowed.
async fn authorize_caller(
    state: &HookState,
    level: Option<Extension<AuthLevel>>,
    actor: Option<&IdentityKey>,
) -> Result<(), Refusal> {
    let level = level.map_or(AuthLevel::Anonymous, |Extension(l)| l);
    if level == AuthLevel::User && actor.is_none() {
        return Err(refusal(
            StatusCode::UNAUTHORIZED,
            "user authentication requires an actor identity",
        ));
    }
    if level == AuthLevel::Anonymous
        && state
            .reader
            .distinguishes_operators(state.trusted_proxy_identity)
            .await
            .map_err(|_| refusal(StatusCode::INTERNAL_SERVER_ERROR, "auth check failed"))?
    {
        return Err(refusal(
            StatusCode::UNAUTHORIZED,
            "authentication required for session binding",
        ));
    }
    Ok(())
}

fn micros_to_rfc3339(micros: i64) -> String {
    jiff::Timestamp::from_microsecond(micros)
        .map_or_else(|_| micros.to_string(), |ts| ts.to_string())
}

async fn view(state: &HookState, binding: &SessionScopeBinding) -> SessionBindingView {
    let workspace = state
        .reader
        .workspace_name_by_id(binding.workspace_id)
        .await
        .ok()
        .flatten()
        .unwrap_or_else(|| binding.workspace_id.to_string());
    let project = state
        .reader
        .project_name_by_id(binding.workspace_id, binding.project_id)
        .await
        .ok()
        .flatten()
        .unwrap_or_else(|| binding.project_id.to_string());
    SessionBindingView {
        session_id: binding.native_session_id.clone(),
        session_uuid: binding.session_id.to_string(),
        agent: binding.agent.map(|agent| agent.as_str().to_owned()),
        workspace,
        project,
        workspace_id: binding.workspace_id.to_string(),
        project_id: binding.project_id.to_string(),
        created_at: micros_to_rfc3339(binding.created_at),
        updated_at: micros_to_rfc3339(binding.updated_at),
        expires_at: micros_to_rfc3339(binding.expires_at),
    }
}

fn parse_session_id(state: &HookState, raw: &str) -> Result<NativeSessionIdentity, Refusal> {
    NativeSessionIdentity::parse(raw, &state.sanitizer).map_err(|_| {
        refusal(
            StatusCode::BAD_REQUEST,
            "session_id must be a non-empty native session id of at most 512 bytes with no \
             control characters or secret-looking content",
        )
    })
}

fn parse_agent(raw: Option<&str>) -> Result<Option<AgentKind>, Refusal> {
    let Some(raw) = raw.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    let agent = AgentKind::from_wire(raw);
    if agent == AgentKind::Other && raw != "other" {
        return Err(refusal(
            StatusCode::BAD_REQUEST,
            format!("unknown agent {raw:?}"),
        ));
    }
    Ok(Some(agent))
}

fn store_error(err: &StoreError) -> Response {
    match err {
        StoreError::SessionCollision => error(
            StatusCode::CONFLICT,
            "this session is bound or owned by another operator",
        ),
        StoreError::InvalidState(reason) => error(StatusCode::CONFLICT, reason.clone()),
        other => {
            warn!(error = %other, "session binding failed");
            error(StatusCode::INTERNAL_SERVER_ERROR, "session binding failed")
        }
    }
}

/// `POST /session/bind`: bind one harness session to a workspace/project
/// before its first hook fires.
pub(crate) async fn handle_bind(
    State(state): State<Arc<HookState>>,
    level: Option<Extension<AuthLevel>>,
    actor_ext: Option<Extension<ai_memory_core::ActorContext>>,
    viewer_ext: Option<Extension<ai_memory_core::AuthorizedViewer>>,
    Json(request): Json<BindSessionRequest>,
) -> Response {
    let actor = actor_identity(actor_ext);
    if let Err(response) = authorize_caller(&state, level, actor.as_ref()).await {
        return response.into_response();
    }
    let native = match parse_session_id(&state, &request.session_id) {
        Ok(native) => native,
        Err(response) => return response.into_response(),
    };
    let agent = match parse_agent(request.agent.as_deref()) {
        Ok(agent) => agent,
        Err(response) => return response.into_response(),
    };
    let workspace = request.workspace.trim();
    let project = request.project.trim();
    for (label, value) in [("workspace", workspace), ("project", project)] {
        if !is_valid_bound_name(value) {
            return error(
                StatusCode::BAD_REQUEST,
                format!(
                    "{label} {value:?} is not valid: use lowercase letters, digits, '.', '_' and \
                     '-', starting with a letter or digit, at most {MAX_BOUND_NAME_BYTES} bytes"
                ),
            );
        }
    }

    let viewer = viewer_ext.map(|Extension(viewer)| viewer.user());
    let session_id = SessionId::from_native(native.as_str());

    let resolved = match create_explicit_scope_guarded(
        &state.reader,
        &state.writer,
        workspace,
        project,
        viewer,
    )
    .await
    {
        Ok(resolved) => resolved,
        Err(err) if err.is_forbidden() => return error(StatusCode::FORBIDDEN, err.to_string()),
        Err(err) => return error(StatusCode::BAD_REQUEST, err.to_string()),
    };
    if resolved.promoted_from.is_some()
        && let Err(failure) = state
            .wiki
            .refresh_renamed_scope(resolved.scope.workspace_id, resolved.scope.project_id)
            .await
    {
        warn!(
            error = %failure,
            workspace_id = %resolved.scope.workspace_id,
            project_id = %resolved.scope.project_id,
            "project name promotion committed; manifest refresh failed; startup backfill can repair"
        );
    }

    let owner_stamp = owner_stamp_for_event(&state, actor.as_ref()).await;
    let now = now_micros();
    let expires_at = now.saturating_add(DEFAULT_BINDING_TTL_SECS * 1_000_000);
    let stored = match state
        .writer
        .upsert_session_scope_binding(NewSessionScopeBinding {
            session_id,
            native_session_id: native.into_string(),
            owner_user: actor.as_ref().map(IdentityKey::storage_key),
            owner_stamp,
            agent,
            project_id: resolved.scope.project_id,
            now,
            expires_at,
        })
        .await
    {
        Ok(stored) => stored,
        Err(err) => return store_error(&err),
    };

    info!(
        session = %session_id,
        workspace,
        project,
        agent = agent.map(AgentKind::as_str),
        "session scope bound"
    );
    (
        StatusCode::OK,
        Json(BindSessionResponse {
            binding: view(&state, &stored).await,
        }),
    )
        .into_response()
}

/// `GET /session/bind?session_id=...`: caller's live binding, 409 if held by
/// another operator, or 404 when absent/expired.
pub(crate) async fn handle_inspect(
    State(state): State<Arc<HookState>>,
    level: Option<Extension<AuthLevel>>,
    actor_ext: Option<Extension<ai_memory_core::ActorContext>>,
    viewer_ext: Option<Extension<ai_memory_core::AuthorizedViewer>>,
    Query(query): Query<SessionBindingQuery>,
) -> Response {
    let actor = actor_identity(actor_ext);
    if let Err(response) = authorize_caller(&state, level, actor.as_ref()).await {
        return response.into_response();
    }
    let native = match parse_session_id(&state, &query.session_id) {
        Ok(native) => native,
        Err(response) => return response.into_response(),
    };
    let owner = actor.as_ref().map(IdentityKey::storage_key);
    let viewer = viewer_ext.map(|Extension(v)| v.user());
    match state
        .reader
        .session_scope_binding(SessionId::from_native(native.as_str()), now_micros())
        .await
    {
        Ok(Some(binding)) if binding.owner_user.as_deref() == owner.as_deref() => {
            if viewer.is_some() {
                match authorize_scope_for(
                    &state.reader,
                    None,
                    ResolvedScope {
                        workspace_id: binding.workspace_id,
                        project_id: binding.project_id,
                    },
                    viewer,
                    ProjectAccess::Read,
                )
                .await
                {
                    Ok(_) => {}
                    Err(err) if err.is_forbidden() => {
                        return error(StatusCode::FORBIDDEN, err.to_string());
                    }
                    Err(err) => return error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
                }
            }
            (StatusCode::OK, Json(view(&state, &binding).await)).into_response()
        }
        Ok(Some(_)) => store_error(&StoreError::SessionCollision),
        Ok(None) => error(StatusCode::NOT_FOUND, "session is not bound"),
        Err(err) => store_error(&err),
    }
}

/// `DELETE /session/bind?session_id=...`: remove the caller's binding; 409 if
/// held by another operator.
pub(crate) async fn handle_unbind(
    State(state): State<Arc<HookState>>,
    level: Option<Extension<AuthLevel>>,
    actor_ext: Option<Extension<ai_memory_core::ActorContext>>,
    Query(query): Query<SessionBindingQuery>,
) -> Response {
    let actor = actor_identity(actor_ext);
    if let Err(response) = authorize_caller(&state, level, actor.as_ref()).await {
        return response.into_response();
    }
    let native = match parse_session_id(&state, &query.session_id) {
        Ok(native) => native,
        Err(response) => return response.into_response(),
    };
    let owner = actor.as_ref().map(IdentityKey::storage_key);
    match state
        .writer
        .delete_session_scope_binding(SessionId::from_native(native.as_str()), owner, now_micros())
        .await
    {
        Ok(removed) => (
            StatusCode::OK,
            Json(serde_json::json!({ "removed": removed })),
        )
            .into_response(),
        Err(err) => store_error(&err),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bound_names_follow_marker_grammar_and_reject_reserved_prefixes() {
        for ok in ["bot-1", "a", "0", "my.bot_name-2", "x".repeat(512).as_str()] {
            assert!(is_valid_bound_name(ok), "{ok}");
        }
        for bad in [
            "",
            "Bot",
            "-bot",
            ".bot",
            "_global",
            "_profile",
            "_profile.abc",
            "bot name",
            "bot/one",
            "bøt",
            "x".repeat(513).as_str(),
        ] {
            assert!(!is_valid_bound_name(bad), "{bad}");
        }
    }

    #[test]
    fn agent_names_are_parsed_strictly() {
        assert_eq!(parse_agent(None).unwrap(), None);
        assert_eq!(parse_agent(Some(" ")).unwrap(), None);
        assert_eq!(
            parse_agent(Some("claude-code")).unwrap(),
            Some(AgentKind::ClaudeCode)
        );
        assert_eq!(
            parse_agent(Some("cursor")).unwrap(),
            Some(AgentKind::Cursor)
        );
        assert!(parse_agent(Some("not-a-harness")).is_err());
    }
}
