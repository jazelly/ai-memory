//! Per-session scope bindings (V114).
//!
//! Stores only `project_id`; the workspace is read through `projects`, so
//! `move-project` keeps the binding attached and `merge-project` updates it.

use ai_memory_core::{AgentKind, ProjectId, SessionId, WorkspaceId};
use rusqlite::{Connection, OptionalExtension, params};

use crate::error::{StoreError, StoreResult};

/// A binding to persist. `owner_user` is the binder's qualified storage key,
/// or `None` on an unauthenticated server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewSessionScopeBinding {
    /// Deterministic internal session UUID.
    pub session_id: SessionId,
    /// Raw external session identifier supplied by the caller.
    pub native_session_id: String,
    /// Qualified storage key of the binding owner, or `None` when unauthenticated.
    pub owner_user: Option<String>,
    /// Effective `sessions.actor_user` stamp for the caller (for legacy/owner checks).
    pub owner_stamp: Option<String>,
    /// Optional agent restriction; `None` matches any harness.
    pub agent: Option<AgentKind>,
    /// Target project ID (workspace is joined from `projects`).
    pub project_id: ProjectId,
    /// Current timestamp in epoch microseconds.
    pub now: i64,
    /// Expiry timestamp in epoch microseconds.
    pub expires_at: i64,
}

/// A stored, unexpired binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionScopeBinding {
    /// Deterministic internal session UUID.
    pub session_id: SessionId,
    /// Raw external session identifier supplied by the caller.
    pub native_session_id: String,
    /// Qualified storage key of the binding owner, or `None` when unauthenticated.
    pub owner_user: Option<String>,
    /// Optional agent restriction; `None` matches any harness.
    pub agent: Option<AgentKind>,
    /// Target workspace ID resolved via `projects`.
    pub workspace_id: WorkspaceId,
    /// Target project ID.
    pub project_id: ProjectId,
    /// Creation timestamp in epoch microseconds.
    pub created_at: i64,
    /// Last-update timestamp in epoch microseconds.
    pub updated_at: i64,
    /// Expiry timestamp in epoch microseconds.
    pub expires_at: i64,
}

/// Result of a combined capture-path scope lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureScopeLookup {
    /// Live per-session scope binding applies to this event.
    Bound(WorkspaceId, ProjectId),
    /// No applicable binding; existing session row provides sticky scope.
    Session(WorkspaceId, ProjectId, Option<String>),
    /// Neither a binding nor an existing session row exists.
    None,
}

impl SessionScopeBinding {
    /// Returns whether this binding matches the caller's operator key and agent harness.
    #[must_use]
    pub fn applies_to(&self, owner: Option<&str>, agent: AgentKind) -> bool {
        self.owner_user.as_deref().unwrap_or("") == owner.unwrap_or("")
            && self.agent.is_none_or(|bound| bound == agent)
    }
}

fn owner_column(owner: Option<&str>) -> &str {
    owner.unwrap_or("")
}

pub(crate) fn upsert_session_scope_binding(
    conn: &mut Connection,
    binding: &NewSessionScopeBinding,
) -> StoreResult<SessionScopeBinding> {
    let tx = conn.transaction()?;
    tx.execute(
        "DELETE FROM session_scope_bindings WHERE expires_at <= ?1",
        params![binding.now],
    )?;
    let existing_owner: Option<String> = tx
        .query_row(
            "SELECT owner_user FROM session_scope_bindings WHERE session_id = ?1",
            params![binding.session_id.as_bytes()],
            |row| row.get(0),
        )
        .optional()?;
    let owner = owner_column(binding.owner_user.as_deref());
    if existing_owner.is_some_and(|existing| existing != owner) {
        return Err(StoreError::SessionCollision);
    }
    let session_row: Option<(Vec<u8>, Option<String>)> = tx
        .query_row(
            "SELECT project_id, actor_user FROM sessions WHERE id = ?1",
            params![binding.session_id.as_bytes()],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    if let Some((proj_bytes, actor_user)) = session_row {
        if actor_user.as_deref() != binding.owner_stamp.as_deref() {
            return Err(StoreError::SessionCollision);
        }
        if ProjectId::from_slice(&proj_bytes)? != binding.project_id {
            return Err(StoreError::InvalidState(
                "session already has captured events in another scope; bind before the session's first hook"
                    .into(),
            ));
        }
    }
    let outside_obs: bool = tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM observations WHERE session_id = ?1 AND project_id != ?2)",
        params![binding.session_id.as_bytes(), binding.project_id.as_bytes()],
        |row| row.get(0),
    )?;
    let claimed_handoff: bool = tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM handoffs WHERE state = 'accepted' AND accepted_by_session = ?1 AND project_id != ?2)",
        params![binding.session_id.as_bytes(), binding.project_id.as_bytes()],
        |row| row.get(0),
    )?;
    if outside_obs || claimed_handoff {
        return Err(StoreError::InvalidState(
            "session already has captured events or a claimed handoff in another scope; bind before the session's first hook"
                .into(),
        ));
    }
    tx.execute(
        "INSERT INTO session_scope_bindings \
             (session_id, native_session_id, owner_user, agent_kind, project_id, \
              created_at, updated_at, expires_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6, ?7) \
         ON CONFLICT(session_id) DO UPDATE SET \
             native_session_id = excluded.native_session_id, \
             owner_user = excluded.owner_user, \
             agent_kind = excluded.agent_kind, \
             project_id = excluded.project_id, \
             updated_at = excluded.updated_at, \
             expires_at = excluded.expires_at",
        params![
            binding.session_id.as_bytes(),
            binding.native_session_id,
            owner,
            binding.agent.map(AgentKind::as_str),
            binding.project_id.as_bytes(),
            binding.now,
            binding.expires_at,
        ],
    )?;
    let stored = read_binding(&tx, binding.session_id, binding.now)?
        .ok_or_else(|| StoreError::NotFound("session scope binding project".into()))?;
    tx.commit()?;
    Ok(stored)
}

pub(crate) fn delete_session_scope_binding(
    conn: &mut Connection,
    session_id: SessionId,
    owner: Option<&str>,
    now: i64,
) -> StoreResult<bool> {
    let tx = conn.transaction()?;
    let row: Option<(String, i64)> = tx
        .query_row(
            "SELECT owner_user, expires_at FROM session_scope_bindings WHERE session_id = ?1",
            params![session_id.as_bytes()],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    let Some((stored_owner, expires_at)) = row else {
        return Ok(false);
    };
    let live = expires_at > now;
    if live && stored_owner != owner_column(owner) {
        return Err(StoreError::SessionCollision);
    }
    let removed = tx.execute(
        "DELETE FROM session_scope_bindings WHERE session_id = ?1",
        params![session_id.as_bytes()],
    )?;
    tx.commit()?;
    Ok(removed > 0 && live)
}

pub(crate) fn session_scope_binding(
    conn: &Connection,
    session_id: SessionId,
    now: i64,
) -> StoreResult<Option<SessionScopeBinding>> {
    read_binding(conn, session_id, now)
}

/// Combined capture-path lookup: returns an applicable binding if one is live,
/// otherwise falls back to reading the existing session row for stickiness —
/// using a single pool checkout.
pub(crate) fn resolve_capture_binding_or_session_scope(
    conn: &Connection,
    session_id: SessionId,
    has_native_session: bool,
    owner: Option<&str>,
    agent: AgentKind,
    now: i64,
) -> StoreResult<CaptureScopeLookup> {
    if has_native_session
        && let Some(binding) = read_binding(conn, session_id, now)?
        && binding.applies_to(owner, agent)
    {
        return Ok(CaptureScopeLookup::Bound(
            binding.workspace_id,
            binding.project_id,
        ));
    }
    let row = conn
        .query_row(
            "SELECT workspace_id, project_id, cwd FROM sessions WHERE id = ?1",
            params![session_id.as_bytes()],
            |row| {
                Ok((
                    row.get::<_, Vec<u8>>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            },
        )
        .optional()?;
    Ok(match row {
        Some((ws, proj, cwd)) => CaptureScopeLookup::Session(
            WorkspaceId::from_slice(&ws)?,
            ProjectId::from_slice(&proj)?,
            cwd,
        ),
        None => CaptureScopeLookup::None,
    })
}

fn read_binding(
    conn: &Connection,
    session_id: SessionId,
    now: i64,
) -> StoreResult<Option<SessionScopeBinding>> {
    let row = conn
        .query_row(
            "SELECT b.native_session_id, b.owner_user, b.agent_kind, p.workspace_id, \
                    b.project_id, b.created_at, b.updated_at, b.expires_at \
             FROM session_scope_bindings b \
             JOIN projects p ON p.id = b.project_id \
             WHERE b.session_id = ?1 AND b.expires_at > ?2",
            params![session_id.as_bytes(), now],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, Vec<u8>>(3)?,
                    row.get::<_, Vec<u8>>(4)?,
                    row.get::<_, i64>(5)?,
                    row.get::<_, i64>(6)?,
                    row.get::<_, i64>(7)?,
                ))
            },
        )
        .optional()?;
    let Some((native, owner, agent, ws, proj, created_at, updated_at, expires_at)) = row else {
        return Ok(None);
    };
    Ok(Some(SessionScopeBinding {
        session_id,
        native_session_id: native,
        owner_user: (!owner.is_empty()).then_some(owner),
        agent: agent.as_deref().map(AgentKind::from_wire),
        workspace_id: WorkspaceId::from_slice(&ws)?,
        project_id: ProjectId::from_slice(&proj)?,
        created_at,
        updated_at,
        expires_at,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{
        Compaction, PurgeMode, get_or_create_project, get_or_create_workspace, merge_project,
        purge_session,
    };

    fn db() -> (Connection, WorkspaceId, ProjectId, ProjectId) {
        let mut conn = Connection::open_in_memory().unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        crate::migrations::run(&mut conn).unwrap();
        let ws = get_or_create_workspace(&mut conn, "bot-1").unwrap();
        let a = get_or_create_project(&mut conn, &ws, "alpha", None).unwrap();
        let b = get_or_create_project(&mut conn, &ws, "beta", None).unwrap();
        (conn, ws, a, b)
    }

    fn binding(project_id: ProjectId, owner: Option<&str>, now: i64) -> NewSessionScopeBinding {
        NewSessionScopeBinding {
            session_id: SessionId::from_native("native-1"),
            native_session_id: "native-1".into(),
            owner_user: owner.map(str::to_owned),
            owner_stamp: owner.map(str::to_owned),
            agent: Some(AgentKind::ClaudeCode),
            project_id,
            now,
            expires_at: now + 1000,
        }
    }

    #[test]
    fn bind_expiry_owner_harness_and_project_follow() {
        let (mut conn, ws, a, b) = db();
        let sid = SessionId::from_native("native-1");

        let stored = upsert_session_scope_binding(&mut conn, &binding(a, Some("u:1"), 10)).unwrap();
        assert_eq!((stored.workspace_id, stored.project_id), (ws, a));
        assert!(stored.applies_to(Some("u:1"), AgentKind::ClaudeCode));
        assert!(!stored.applies_to(Some("u:2"), AgentKind::ClaudeCode));
        assert!(!stored.applies_to(Some("u:1"), AgentKind::Codex));

        // Another operator cannot replace or delete a live binding.
        assert!(matches!(
            upsert_session_scope_binding(&mut conn, &binding(b, Some("u:2"), 20)),
            Err(StoreError::SessionCollision)
        ));
        assert!(matches!(
            delete_session_scope_binding(&mut conn, sid, Some("u:2"), 20),
            Err(StoreError::SessionCollision)
        ));

        // Expiry makes it invisible and lets upsert prune and replace it.
        assert!(session_scope_binding(&conn, sid, 1010).unwrap().is_none());
        let taken =
            upsert_session_scope_binding(&mut conn, &binding(b, Some("u:2"), 1010)).unwrap();
        assert_eq!(taken.project_id, b);

        // merge-project moves the binding to the target project instead of cascading it.
        merge_project(&mut conn, ws, b, a, &[], true).unwrap();
        assert_eq!(
            session_scope_binding(&conn, sid, 1020)
                .unwrap()
                .unwrap()
                .project_id,
            a
        );

        // purge_session removes the binding too.
        conn.execute(
            "INSERT INTO sessions (id, workspace_id, project_id, agent_kind, started_at) \
             VALUES (?1, ?2, ?3, 'claude-code', 100)",
            params![sid.as_bytes(), ws.as_bytes(), a.as_bytes()],
        )
        .unwrap();
        purge_session(
            &mut conn,
            ws,
            a,
            sid,
            None,
            Compaction::Skip,
            PurgeMode::Commit,
        )
        .unwrap();
        assert!(session_scope_binding(&conn, sid, 1030).unwrap().is_none());

        // Observations or accepted handoffs outside the target project reject bind even when
        // the sessions row itself matches the target project, while same-project ones pass.
        let c = get_or_create_project(&mut conn, &ws, "gamma", None).unwrap();
        let sid_obs = SessionId::from_native("native-obs");
        let sid_ho = SessionId::from_native("native-ho");
        for s in [sid_obs, sid_ho] {
            conn.execute(
                "INSERT INTO sessions (id, workspace_id, project_id, agent_kind, started_at)                  VALUES (?1, ?2, ?3, 'claude-code', 100)",
                params![s.as_bytes(), ws.as_bytes(), c.as_bytes()],
            )
            .unwrap();
        }
        conn.execute(
            "INSERT INTO observations              (id, session_id, workspace_id, project_id, kind, title, body, created_at)              VALUES (?1, ?2, ?3, ?4, 'user_prompt', 't', 'b', 100)",
            params![b"obs-1".as_slice(), sid_obs.as_bytes(), ws.as_bytes(), a.as_bytes()],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO handoffs              (id, workspace_id, project_id, from_agent, summary, state, created_at, accepted_by_session)              VALUES (?1, ?2, ?3, 'claude-code', 's', 'accepted', 100, ?4)",
            params![b"ho-1".as_slice(), ws.as_bytes(), a.as_bytes(), sid_ho.as_bytes()],
        )
        .unwrap();
        let mut b_obs = binding(c, None, 1040);
        b_obs.session_id = sid_obs;
        b_obs.native_session_id = "native-obs".into();
        assert!(matches!(
            upsert_session_scope_binding(&mut conn, &b_obs),
            Err(StoreError::InvalidState(_))
        ));
        let mut b_ho = binding(c, None, 1040);
        b_ho.session_id = sid_ho;
        b_ho.native_session_id = "native-ho".into();
        assert!(matches!(
            upsert_session_scope_binding(&mut conn, &b_ho),
            Err(StoreError::InvalidState(_))
        ));
        // Positive control: same session with handoff/session in project `a` binds to `a`.
        let sid_ok = SessionId::from_native("native-ok");
        conn.execute(
            "INSERT INTO sessions (id, workspace_id, project_id, agent_kind, started_at)              VALUES (?1, ?2, ?3, 'claude-code', 100)",
            params![sid_ok.as_bytes(), ws.as_bytes(), a.as_bytes()],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO handoffs              (id, workspace_id, project_id, from_agent, summary, state, created_at, accepted_by_session)              VALUES (?1, ?2, ?3, 'claude-code', 's', 'accepted', 100, ?4)",
            params![b"ho-2".as_slice(), ws.as_bytes(), a.as_bytes(), sid_ok.as_bytes()],
        )
        .unwrap();
        let mut b_ok = binding(a, None, 1040);
        b_ok.session_id = sid_ok;
        b_ok.native_session_id = "native-ok".into();
        assert_eq!(
            upsert_session_scope_binding(&mut conn, &b_ok)
                .unwrap()
                .project_id,
            a
        );
    }
}
