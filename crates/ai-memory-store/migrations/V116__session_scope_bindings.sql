-- Per-session scope bindings: pin one native harness session to a project
-- without a marker file or process-wide AI_MEMORY_HOME. Deleted on SessionEnd
-- or when expires_at passes; project deletion cascades it out.

CREATE TABLE session_scope_bindings (
    session_id        BLOB PRIMARY KEY NOT NULL,
    native_session_id TEXT NOT NULL,
    owner_user        TEXT NOT NULL DEFAULT '',
    agent_kind        TEXT,
    project_id        BLOB NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    created_at        INTEGER NOT NULL,
    updated_at        INTEGER NOT NULL,
    expires_at        INTEGER NOT NULL
) WITHOUT ROWID;

CREATE INDEX idx_session_scope_bindings_project
    ON session_scope_bindings(project_id);
