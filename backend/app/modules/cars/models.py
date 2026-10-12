from decimal import Decimal

from sqlalchemy import ForeignKey, Index, Integer, Numeric, SmallInteger, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Manufacturer(Base):
	__tablename__ = "manufacturers"
	__table_args__ = (UniqueConstraint("name", name="uq_manufacturers_name"),)

	id: Mapped[int] = mapped_column(Integer, primary_key=True)
	name: Mapped[str] = mapped_column(String(255), nullable=False)

	models: Mapped[list["CarModel"]] = relationship(back_populates="manufacturer")


class CarModel(Base):
	__tablename__ = "models"
	__table_args__ = (
		UniqueConstraint("manufacturer_id", "name", name="uq_models_manufacturer_name"),
		Index("ix_models_manufacturer_id", "manufacturer_id"),
	)

	id: Mapped[int] = mapped_column(Integer, primary_key=True)
	manufacturer_id: Mapped[int] = mapped_column(
		ForeignKey("manufacturers.id"), nullable=False
	)
	name: Mapped[str] = mapped_column(String(255), nullable=False)

	manufacturer: Mapped[Manufacturer] = relationship(back_populates="models")
	variants: Mapped[list["Variant"]] = relationship(back_populates="model")


class Variant(Base):
	__tablename__ = "variants"
	__table_args__ = (
		UniqueConstraint("model_id", "name", name="uq_variants_model_name"),
		Index("ix_variants_model_id", "model_id"),
	)

	id: Mapped[int] = mapped_column(Integer, primary_key=True)
	model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), nullable=False)
	name: Mapped[str] = mapped_column(String(255), nullable=False)
	ex_showroom_price: Mapped[Decimal] = mapped_column(Numeric(12, 0), nullable=False)
	displacement: Mapped[int] = mapped_column(Integer, nullable=False)
	cylinders: Mapped[int] = mapped_column(SmallInteger, nullable=False)
	fuel_tank_capacity: Mapped[Decimal] = mapped_column(Numeric(8, 2), nullable=False)
	fuel_type: Mapped[str] = mapped_column(String(100), nullable=False)
	height: Mapped[int] = mapped_column(Integer, nullable=False)
	length: Mapped[Decimal] = mapped_column(Numeric(8, 2), nullable=False)
	width: Mapped[int] = mapped_column(Integer, nullable=False)
	body_type: Mapped[str] = mapped_column(String(100), nullable=False)
	seating_capacity: Mapped[int] = mapped_column(SmallInteger, nullable=False)
	transmission_type: Mapped[str] = mapped_column(String(100), nullable=False)
	power: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
	torque: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)

	model: Mapped[CarModel] = relationship(back_populates="variants")