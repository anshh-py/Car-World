"""Create manufacturer, model, and variant catalog tables.

Revision ID: 20261001_car_catalog
Revises:
Create Date: 2026-10-01
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20261001_car_catalog"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "manufacturers",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.UniqueConstraint("name", name="uq_manufacturers_name"),
    )
    op.create_table(
        "models",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("manufacturer_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.ForeignKeyConstraint(["manufacturer_id"], ["manufacturers.id"]),
        sa.UniqueConstraint(
            "manufacturer_id", "name", name="uq_models_manufacturer_name"
        ),
    )
    op.create_index("ix_models_manufacturer_id", "models", ["manufacturer_id"])
    op.create_table(
        "variants",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("ex_showroom_price", sa.Numeric(precision=12, scale=0), nullable=False),
        sa.Column("displacement", sa.Integer(), nullable=False),
        sa.Column("cylinders", sa.SmallInteger(), nullable=False),
        sa.Column("fuel_tank_capacity", sa.Numeric(precision=8, scale=2), nullable=False),
        sa.Column("fuel_type", sa.String(length=100), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("length", sa.Numeric(precision=8, scale=2), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("body_type", sa.String(length=100), nullable=False),
        sa.Column("seating_capacity", sa.SmallInteger(), nullable=False),
        sa.Column("transmission_type", sa.String(length=100), nullable=False),
        sa.Column("power", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("torque", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"]),
        sa.UniqueConstraint("model_id", "name", name="uq_variants_model_name"),
    )
    op.create_index("ix_variants_model_id", "variants", ["model_id"])


def downgrade() -> None:
    op.drop_index("ix_variants_model_id", table_name="variants")
    op.drop_table("variants")
    op.drop_index("ix_models_manufacturer_id", table_name="models")
    op.drop_table("models")
    op.drop_table("manufacturers")