"""Initial schema: create all platform tables.

Revision ID: 0001_initial
Revises:
Create Date: 2024-01-01 00:00:00

This baseline migration creates all tables from the application's ORM
metadata. Using the metadata directly guarantees the schema matches the models
exactly across SQLite (dev) and PostgreSQL (prod). Run
`alembic upgrade head` to create the schema in a fresh database.
"""
from __future__ import annotations

import sys
import os

from alembic import op
import sqlalchemy as sa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    from app.database import Base
    import app.models  # noqa: F401

    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    from app.database import Base
    import app.models  # noqa: F401

    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
