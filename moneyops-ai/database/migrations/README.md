# Migrations

Schema version 1 is created by `Base.metadata.create_all` (`scripts/initialize_database.py`)
and recorded in the `schema_version` table. When the schema changes, add a numbered
script here (e.g. `0002_add_column.py`) that alters the tables and inserts the new version,
or adopt Alembic (`alembic init`) pointing at `database.models.Base.metadata`.
