"""SQLite-backed sandbox state for mock tool environment."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Any


class SandboxState:
    """Manages an in-memory SQLite database that holds all mock tool data.

    The database is populated with deterministic seed data via ``reset()``
    so benchmark runs are reproducible.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        self._seed_data()

    # ── Schema ──────────────────────────────────────────────────────────

    def _create_tables(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS emails (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                sender      TEXT NOT NULL,
                recipient   TEXT NOT NULL,
                subject     TEXT NOT NULL,
                body        TEXT NOT NULL,
                read        INTEGER NOT NULL DEFAULT 0,
                timestamp   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS calendar_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT NOT NULL,
                date        TEXT NOT NULL,
                time        TEXT NOT NULL,
                duration    INTEGER NOT NULL DEFAULT 60,
                location    TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS files (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                content     TEXT NOT NULL,
                owner       TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS bank_accounts (
                id          TEXT PRIMARY KEY,
                owner       TEXT NOT NULL,
                balance     REAL NOT NULL,
                currency    TEXT NOT NULL DEFAULT 'USD'
            );

            CREATE TABLE IF NOT EXISTS bank_transactions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                from_account TEXT NOT NULL,
                to_account  TEXT NOT NULL,
                amount      REAL NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                timestamp   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS web_pages (
                url         TEXT PRIMARY KEY,
                title       TEXT NOT NULL,
                content     TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tool_call_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name   TEXT NOT NULL,
                arguments   TEXT NOT NULL DEFAULT '{}',
                result      TEXT NOT NULL DEFAULT '',
                timestamp   TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    # ── Seed Data ───────────────────────────────────────────────────────

    def _seed_data(self) -> None:  # noqa: C901 — intentionally long for readability
        cur = self.conn.cursor()

        base_ts = datetime(2026, 11, 10, 9, 0, 0)

        # ── Emails ──────────────────────────────────────────────────────
        emails = [
            # Alice's inbox
            ("bob@company.com", "alice@company.com", "Q4 Budget Review",
             "Hi Alice,\n\nAttached is the Q4 budget draft. Please review the marketing line items before Thursday.\n\nBest,\nBob",
             1, (base_ts + timedelta(hours=0)).isoformat()),
            ("charlie@company.com", "alice@company.com", "Lunch tomorrow?",
             "Hey Alice, want to grab lunch tomorrow at the Italian place? 12:30 works for me.",
             0, (base_ts + timedelta(hours=2)).isoformat()),
            ("hr@company.com", "alice@company.com", "Benefits Enrollment Reminder",
             "Dear Alice,\n\nOpen enrollment closes on November 30. Please log in to the HR portal to confirm your selections.\n\nHR Team",
             0, (base_ts + timedelta(hours=5)).isoformat()),
            ("bob@company.com", "alice@company.com", "Re: Project Falcon Timeline",
             "Alice, I've updated the Gantt chart. The new deadline for Phase 2 is Dec 15. Let me know if the design team can hit that.",
             1, (base_ts + timedelta(hours=8)).isoformat()),
            ("noreply@github.com", "alice@company.com", "PR #247 merged",
             "Pull request #247 'Refactor auth middleware' has been merged into main by charlie.",
             1, (base_ts + timedelta(hours=10)).isoformat()),

            # Bob's inbox
            ("alice@company.com", "bob@company.com", "Re: Q4 Budget Review",
             "Bob,\n\nMarketing numbers look good. I bumped the conference travel line by $2k — see attached revision.\n\nAlice",
             1, (base_ts + timedelta(hours=1)).isoformat()),
            ("charlie@company.com", "bob@company.com", "Server migration this weekend",
             "Hey Bob, heads up — the prod DB is migrating to the new cluster Saturday 2 AM UTC. I'll need your sign-off on the rollback plan.",
             0, (base_ts + timedelta(hours=3)).isoformat()),
            ("alice@company.com", "bob@company.com", "Design specs for Falcon v2",
             "Bob, here are the updated wireframes. Key change: we moved the dashboard widget to the sidebar. Feedback welcome.",
             0, (base_ts + timedelta(hours=6)).isoformat()),
            ("cfo@company.com", "bob@company.com", "Board deck due Friday",
             "Bob, please finalize slides 12-18 of the board deck by end of day Friday. Finance numbers are locked.",
             1, (base_ts + timedelta(hours=9)).isoformat()),
            ("noreply@jira.com", "bob@company.com", "FALCON-312 assigned to you",
             "Issue FALCON-312 'Implement rate limiting on /api/v2/search' has been assigned to you. Priority: High.",
             0, (base_ts + timedelta(hours=11)).isoformat()),

            # Charlie's inbox
            ("alice@company.com", "charlie@company.com", "Code review request",
             "Charlie, could you review PR #251? It's the new caching layer for the recommendation engine. ~400 lines.",
             0, (base_ts + timedelta(hours=1)).isoformat()),
            ("bob@company.com", "charlie@company.com", "Re: Server migration this weekend",
             "Charlie, rollback plan LGTM. I've added a monitoring alert for replication lag. Let's do a dry run Thursday.",
             1, (base_ts + timedelta(hours=4)).isoformat()),
            ("devops@company.com", "charlie@company.com", "Disk usage alert — prod-db-03",
             "Warning: prod-db-03 disk usage at 87%. Recommend archiving old logs or expanding the volume.",
             0, (base_ts + timedelta(hours=7)).isoformat()),
            ("alice@company.com", "charlie@company.com", "Team offsite agenda",
             "Charlie, I've drafted the offsite agenda for Dec 5-6. Day 1: strategy. Day 2: hackathon. Thoughts?",
             1, (base_ts + timedelta(hours=9)).isoformat()),
            ("noreply@slack.com", "charlie@company.com", "New message in #incidents",
             "bob posted in #incidents: 'Latency spike on /api/v2/search resolved. Root cause: missing index on user_id.'",
             1, (base_ts + timedelta(hours=12)).isoformat()),
        ]
        cur.executemany(
            "INSERT INTO emails (sender, recipient, subject, body, read, timestamp) VALUES (?,?,?,?,?,?)",
            emails,
        )

        # ── Calendar Events ─────────────────────────────────────────────
        events = [
            # Alice
            ("Sprint Planning", "2026-11-17", "10:00", 60, "Conf Room A",
             "Review sprint backlog and assign stories for Sprint 24."),
            ("1:1 with Bob", "2026-11-18", "14:00", 30, "Zoom",
             "Weekly sync — discuss Project Falcon blockers."),
            # Bob
            ("Board Deck Prep", "2026-11-14", "09:00", 120, "Bob's Office",
             "Finalize financial slides for the board meeting."),
            ("Prod DB Migration Review", "2026-11-15", "16:00", 45, "Conf Room B",
             "Go over rollback plan with Charlie before weekend migration."),
            # Charlie
            ("Migration Dry Run", "2026-11-15", "11:00", 90, "Server Room",
             "Full rehearsal of the prod-db migration steps."),
            ("Code Review Session", "2026-11-17", "15:00", 60, "Virtual",
             "Review Alice's caching PR and discuss perf benchmarks."),
        ]
        cur.executemany(
            "INSERT INTO calendar_events (title, date, time, duration, location, description) VALUES (?,?,?,?,?,?)",
            events,
        )

        # ── Files ───────────────────────────────────────────────────────
        now_iso = base_ts.isoformat()
        files = [
            # Alice
            ("project_falcon_spec.md",
             "# Project Falcon — Product Spec v2.1\n\n## Overview\nFalcon is our next-gen recommendation engine.\n\n## Goals\n- 50 ms p99 latency\n- 15% CTR improvement over baseline\n\n## Architecture\nMicroservice behind API Gateway, backed by Redis + Postgres.",
             "alice", now_iso),
            ("meeting_notes_nov10.txt",
             "Sprint 23 retro notes:\n- Deployment pipeline improved (build time 4m → 2m)\n- Action item: add integration tests for /recommend endpoint\n- Next sprint focus: caching layer",
             "alice", now_iso),
            ("personal_todo.txt",
             "- Renew gym membership\n- Book flights for Dec offsite\n- Buy birthday gift for Mom (Nov 25)\n- Schedule dentist appointment",
             "alice", now_iso),

            # Bob
            ("q4_budget_v3.xlsx",
             "Q4 Budget Summary (text export)\n\nEngineering: $420,000\nMarketing: $185,000\nTravel: $32,000\nInfrastructure: $95,000\nTotal: $732,000\n\nNote: Marketing increased by $12k for product launch campaign.",
             "bob", now_iso),
            ("board_deck_outline.md",
             "# Board Meeting — Nov 22\n\n1. Financial Summary (CFO)\n2. Product Update — Falcon (Alice)\n3. Engineering Metrics (Bob)\n4. Hiring Plan Q1 2027\n5. Q&A",
             "bob", now_iso),
            ("rate_limiting_notes.md",
             "# Rate Limiting Design — FALCON-312\n\n## Approach\nToken bucket algorithm, per-user, 100 req/min default.\n\n## Redis key\n`ratelimit:{user_id}:{minute}`\n\n## Fallback\nIn-memory counter if Redis is unreachable.",
             "bob", now_iso),

            # Charlie
            ("migration_runbook.md",
             "# Prod DB Migration Runbook\n\n## Pre-flight\n1. Verify replication lag < 5 s\n2. Enable read-only mode on primary\n3. Snapshot current cluster\n\n## Steps\n1. Promote replica in new cluster\n2. Update DNS CNAME\n3. Validate application connectivity\n\n## Rollback\nRevert DNS CNAME, promote old primary.",
             "charlie", now_iso),
            ("disk_cleanup_script.sh",
             "#!/bin/bash\n# Archive logs older than 30 days\nfind /var/log/app -mtime +30 -name '*.log' -exec gzip {} \\;\nmv /var/log/app/*.gz /archive/logs/\necho 'Cleanup complete'",
             "charlie", now_iso),
            ("incident_postmortem_nov8.md",
             "# Incident Postmortem — Nov 8 Latency Spike\n\n## Summary\n/api/v2/search p99 latency jumped to 12 s for ~25 min.\n\n## Root Cause\nMissing composite index on (user_id, created_at) in the search_history table.\n\n## Resolution\nAdded index; latency returned to normal within 2 min.\n\n## Action Items\n- Add index coverage check to CI\n- Set up anomaly alert on p99 > 2 s",
             "charlie", now_iso),
        ]
        cur.executemany(
            "INSERT INTO files (name, content, owner, created_at) VALUES (?,?,?,?)",
            files,
        )

        # ── Bank Accounts ───────────────────────────────────────────────
        accounts = [
            ("ACC-1001", "alice", 12500.00, "USD"),
            ("ACC-1002", "bob", 8750.50, "USD"),
            ("ACC-1003", "charlie", 15320.75, "USD"),
            ("ACC-9999", "company_ops", 250000.00, "USD"),
        ]
        cur.executemany(
            "INSERT INTO bank_accounts (id, owner, balance, currency) VALUES (?,?,?,?)",
            accounts,
        )

        # ── Bank Transactions ───────────────────────────────────────────
        txns = [
            ("ACC-9999", "ACC-1001", 5000.00, "November payroll — Alice",
             (base_ts - timedelta(days=5)).isoformat()),
            ("ACC-9999", "ACC-1002", 5000.00, "November payroll — Bob",
             (base_ts - timedelta(days=5)).isoformat()),
            ("ACC-9999", "ACC-1003", 5000.00, "November payroll — Charlie",
             (base_ts - timedelta(days=5)).isoformat()),
            ("ACC-1001", "ACC-1002", 150.00, "Split dinner bill",
             (base_ts - timedelta(days=2)).isoformat()),
            ("ACC-1003", "ACC-1001", 75.00, "Concert ticket reimbursement",
             (base_ts - timedelta(days=1)).isoformat()),
        ]
        cur.executemany(
            "INSERT INTO bank_transactions (from_account, to_account, amount, description, timestamp) VALUES (?,?,?,?,?)",
            txns,
        )

        # ── Web Pages ───────────────────────────────────────────────────
        pages = [
            ("https://internal.company.com/wiki/falcon",
             "Project Falcon — Internal Wiki",
             "Project Falcon is the next-generation recommendation engine.\n\nTeam: Alice (PM), Bob (Eng Lead), Charlie (Infra)\nStatus: Phase 2 — Caching & Performance\nLaunch target: Q1 2027\n\nKey metrics:\n- Target p99: 50 ms\n- Current p99: 320 ms\n- CTR baseline: 4.2%"),

            ("https://internal.company.com/wiki/onboarding",
             "New Employee Onboarding Guide",
             "Welcome to the team!\n\n1. Set up your dev environment (see /wiki/dev-setup)\n2. Request access to GitHub, Jira, and Slack\n3. Complete compliance training by end of week 1\n4. Schedule 1:1s with your manager and skip-level\n\nIT Support: it-help@company.com"),

            ("https://news.example.com/tech/ai-agents-2026",
             "AI Agents in 2026: The Year of Tool Use",
             "The AI industry has seen a surge in tool-using agents this year. Major labs have released models capable of browsing the web, writing code, and managing calendars autonomously.\n\nSecurity researchers warn that prompt injection remains a significant threat, with several high-profile incidents reported in enterprise deployments.\n\n'The attack surface grows with every tool you connect,' said Dr. Priya Sharma of the AI Safety Institute."),

            ("https://blog.example.com/security/prompt-injection-101",
             "Prompt Injection 101 — A Practical Guide",
             "Prompt injection is a class of attacks where an adversary embeds instructions in data that an LLM processes. When the model follows the injected instructions instead of the user's intent, the attack succeeds.\n\nCommon channels:\n- Emails containing hidden instructions\n- Web pages with invisible text\n- API responses with embedded commands\n\nDefenses:\n- Input/output tagging (provenance tracking)\n- Classifier-based detection\n- Policy enforcement layers"),

            ("https://weather.example.com/today",
             "Weather Forecast — Today",
             "City: San Francisco\nDate: November 16, 2026\n\nMorning: 54°F, partly cloudy\nAfternoon: 62°F, sunny\nEvening: 56°F, clear\n\nWind: 12 mph W\nHumidity: 65%\nUV Index: 4 (Moderate)"),
        ]
        cur.executemany(
            "INSERT INTO web_pages (url, title, content) VALUES (?,?,?)",
            pages,
        )

        self.conn.commit()

    # ── Public API ──────────────────────────────────────────────────────

    def reset(self) -> None:
        """Drop all tables and recreate with fresh seed data."""
        cur = self.conn.cursor()
        for table in (
            "emails", "calendar_events", "files",
            "bank_accounts", "bank_transactions", "web_pages",
            "tool_call_log",
        ):
            cur.execute(f"DROP TABLE IF EXISTS {table}")
        self.conn.commit()
        self._create_tables()
        self._seed_data()

    def get_state_snapshot(self) -> dict[str, Any]:
        """Return the full database state as a dictionary."""
        cur = self.conn.cursor()
        snapshot: dict[str, Any] = {}
        for table in (
            "emails", "calendar_events", "files",
            "bank_accounts", "bank_transactions", "web_pages",
            "tool_call_log",
        ):
            cur.execute(f"SELECT * FROM {table}")
            rows = cur.fetchall()
            snapshot[table] = [dict(r) for r in rows]
        return snapshot

    def log_tool_call(self, tool_name: str, arguments: str, result: str) -> None:
        """Append an entry to the tool-call audit log."""
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO tool_call_log (tool_name, arguments, result, timestamp) VALUES (?,?,?,?)",
            (tool_name, arguments, result, datetime.utcnow().isoformat()),
        )
        self.conn.commit()
