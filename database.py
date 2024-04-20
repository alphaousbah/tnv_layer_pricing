from typing import Final, List

from sqlalchemy import Column, ForeignKey, String, Table, create_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    declared_attr,
    mapped_column,
    relationship,
)

# --------------------------------------
# Create the SQLite database
# --------------------------------------


# Declare the models
class Base(DeclarativeBase):
    pass


class CommonMixin:
    @declared_attr.directive
    def __tablename__(cls) -> str:
        return cls.__name__.lower()

    # Define a standard representation of an instance
    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(id={self.id!r})"


class Layer(CommonMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    occ_limit: Mapped[int]
    occ_deduct: Mapped[int]
    agg_limit: Mapped[int]
    agg_deduct: Mapped[int]

    # Define the 1-to-many relationship between Layer and LayerReinstatement
    reinstatements: Mapped[List["LayerReinstatement"]] = relationship(
        back_populates="layer", cascade="all, delete-orphan"
    )

    # Get the modelfiles associated to the layer through the association layer_modelfile_table
    modelfiles: Mapped[List["ModelFile"]] = relationship(
        secondary=lambda: layer_modelfile_table
    )

    # Define the 1-to-many relationship between Layer and LayerYearLoss
    yearlosses: Mapped[List["LayerYearLoss"]] = relationship(
        back_populates="layer", cascade="all, delete-orphan"
    )


class LayerReinstatement(CommonMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    order: Mapped[int]
    number: Mapped[int]  # -1 for unlimited
    rate: Mapped[int]

    # Define the 1-to-many relationship between Layer and LayerReinstatement
    layer_id: Mapped[int] = mapped_column(ForeignKey("layer.id"))
    layer: Mapped["Layer"] = relationship(back_populates="reinstatements")


class ModelFile(CommonMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    years_simulated: Mapped[int]

    # Define the 1-to-many relationship between ModelFile and ModelYearLoss
    yearlosses: Mapped[List["ModelYearLoss"]] = relationship(
        back_populates="modelfile", cascade="all, delete-orphan"
    )


class ModelYearLoss(CommonMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    year: Mapped[int]
    day: Mapped[int]
    loss_type: Mapped[str] = mapped_column(String(50))  # type = Cat/Non cat
    loss: Mapped[float]

    # Define the 1-to-many relationship between ModelFile and ModelYearLoss
    modelfile_id: Mapped[int] = mapped_column(ForeignKey("modelfile.id"))
    modelfile: Mapped["ModelFile"] = relationship(back_populates="yearlosses")


layer_modelfile_table: Final[Table] = Table(
    "layer_modelfile",
    Base.metadata,
    Column("layer_id", ForeignKey("layer.id"), primary_key=True),
    Column("modelfile_id", ForeignKey("modelfile.id"), primary_key=True),
)


class LayerYearLoss(CommonMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int]
    year: Mapped[int]
    day: Mapped[int]
    loss_type: Mapped[str] = mapped_column(String(50))  # Cat/Non cat
    gross: Mapped[int]
    ceded: Mapped[int]
    net: Mapped[int]

    # Define the 1-to-many relationship between Layer and ResultYearLoss
    layer_id: Mapped[int] = mapped_column(ForeignKey("layer.id"))
    layer: Mapped["Layer"] = relationship(back_populates="yearlosses")


# Create an engine connected to a SQLite database
# engine = create_engine("sqlite://", echo=True)
engine = create_engine("sqlite://")

# Create the database tables
Base.metadata.create_all(engine)
