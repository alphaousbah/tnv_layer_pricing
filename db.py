from typing import Final, List, Optional

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


class LayerMixin:
    # # pvse: Mapped[str] = mapped_column(String(21))  # T1E1234-2024-01-01-00
    # # option: Mapped[str] = mapped_column(String(2))  # 01
    # # name: Mapped[str] = mapped_column(String(50))
    # # coverage: Mapped[str] = mapped_column(String(50))  # coverage = Branche fine AGIR
    # # cat_cover: Mapped[bool]
    # premium: Mapped[int]
    occ_limit: Mapped[int]
    occ_deduct: Mapped[int]
    agg_limit: Mapped[int]
    agg_deduct: Mapped[int]
    # brokerage: Mapped[float]
    # taxes: Mapped[float]
    # overriding_commission: Mapped[float]
    # other_loadings: Mapped[float]
    # treaty_rate: Mapped[float]
    # participation: Mapped[float]
    # inception: Mapped[datetime]
    # expiry: Mapped[datetime]
    # currency: Mapped[str] = mapped_column(String(3))
    # category: Mapped[int]
    # # category gets its values in RefCategory.id
    # layer_class: Mapped[int]
    # # layer_class gets its values in RefClass.id
    # exposure_region: Mapped[int]
    # # exposure_region gets its values in RefExposureRegion.id
    # status: Mapped[int]
    # # status gets its values in RefStatus.id
    # display_order: Mapped[int]


# Approx. 10k records per year
class Layer(CommonMixin, LayerMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    # created_on: Mapped[datetime]
    # created_by: Mapped[int]  # User ID
    # modified_on: Mapped[datetime]
    # modified_by: Mapped[int]  # User ID

    # Define the 1-to-many relationship between Layer and LayerReinstatement
    reinstatements: Mapped[List["LayerReinstatement"]] = relationship(
        back_populates="layer", cascade="all, delete-orphan"
    )

    # Get the modelfiles associated to the layer through the association layer_modelfile_table
    modelfiles: Mapped[List["ModelFile"]] = relationship(
        secondary=lambda: layer_modelfile
    )

    # Define the 1-to-many relationship between Layer and LayerYearLoss
    yearlosses: Mapped[List["LayerYearLoss"]] = relationship(
        back_populates="layer", cascade="all, delete-orphan"
    )


class LayerReinstatementMixin:
    order: Mapped[int]
    number: Mapped[int]  # -1 for unlimited
    rate: Mapped[int]


# Approx. 30k records per year
class LayerReinstatement(CommonMixin, LayerReinstatementMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    # created_on: Mapped[datetime]
    # created_by: Mapped[int]  # User ID
    # modified_on: Mapped[datetime]
    # modified_by: Mapped[int]  # User ID

    # Define the 1-to-many relationship between Layer and LayerReinstatement
    layer_id: Mapped[int] = mapped_column(ForeignKey("layer.id"))
    layer: Mapped["Layer"] = relationship(back_populates="reinstatements")


class LayerYearLoss(CommonMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    # created_on: Mapped[datetime]
    # created_by: Mapped[int]  # User ID
    year: Mapped[int]
    day: Mapped[int]
    gross: Mapped[int]
    ceded: Mapped[int]
    net: Mapped[int]
    reinstated: Mapped[int]
    reinst_premium: Mapped[int]
    loss_type: Mapped[str] = mapped_column(String(50))  # Cat/Non cat
    # peril_id: Mapped[int]
    # peril: Mapped[str] = mapped_column(String(50))
    # model_id: Mapped[int]
    # model: Mapped[str] = mapped_column(String(50))
    # region: Mapped[str] = mapped_column(String(50))
    # line_of_business: Mapped[str] = mapped_column(String(50))

    # Define the 1-to-many relationship between Layer and ResultYearLoss
    layer_id: Mapped[int] = mapped_column(ForeignKey("layer.id"))
    layer: Mapped["Layer"] = relationship(back_populates="yearlosses")


class ModelFile(CommonMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    # created_on: Mapped[datetime]
    # created_by: Mapped[int]  # User ID
    # modified_on: Mapped[datetime]
    # modified_by: Mapped[int]  # User ID
    # currency: Mapped[str] = mapped_column(String(3))
    # model_class: Mapped[str] = mapped_column(String(50))
    # # model_class (code/digit) = class of the underlying model = Empiricial/Frequency-Severity/Exposure-based/blending/Target premium
    # name: Mapped[str] = mapped_column(String(50))
    # description: Mapped[str] = mapped_column(String(1000))
    # loss_type: Mapped[str] = mapped_column(String(50))  # loss_type = Cat/Non cat
    years_simulated: Mapped[int]

    # Define the 1-to-many relationship between ModelFile and ModelYearLoss
    yearlosses: Mapped[List["ModelYearLoss"]] = relationship(
        back_populates="modelfile", cascade="all, delete-orphan"
    )


class ModelYearLoss(CommonMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    # created_on: Mapped[datetime]
    # created_by: Mapped[int]  # User ID
    year: Mapped[int]
    day: Mapped[int]
    loss: Mapped[float]
    loss_type: Mapped[str] = mapped_column(String(50))  # Cat/Non cat
    # peril_id: Mapped[int]
    # peril: Mapped[str] = mapped_column(String(50))
    # model_id: Mapped[int]
    # model: Mapped[str] = mapped_column(String(50))
    # region: Mapped[str] = mapped_column(String(50))
    # line_of_business: Mapped[str] = mapped_column(String(50))

    # Define the 1-to-many relationship between ModelFile and ModelYearLoss
    modelfile_id: Mapped[int] = mapped_column(ForeignKey("modelfile.id"))
    modelfile: Mapped["ModelFile"] = relationship(back_populates="yearlosses")


layer_modelfile: Final[Table] = Table(
    "layer_modelfile",
    Base.metadata,
    Column("layer_id", ForeignKey("layer.id"), primary_key=True),
    Column("modelfile_id", ForeignKey("modelfile.id"), primary_key=True),
)


class ResultInstance(CommonMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    # created_on: Mapped[datetime]
    # created_by: Mapped[int]  # User ID
    # modified_on: Mapped[datetime]
    # modified_by: Mapped[int]  # User ID
    name: Mapped[str] = mapped_column(String(50))

    # Define the 1-to-many relationship between ResultInstance and ResultLayer
    layers: Mapped[List["ResultLayer"]] = relationship(back_populates="instance")


class ResultLayer(CommonMixin, LayerMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int]
    # source_id = ID of the layer from which resultlayer was copied

    # Define the 1-to-many relationship between ResultInstance and ResultLayer
    instance_id: Mapped[int] = mapped_column(ForeignKey("resultinstance.id"))
    instance: Mapped["ResultInstance"] = relationship(back_populates="layers")

    # Define the 1-to-many relationship between ResultLayer and ResultLayerReinstatement
    reinstatements: Mapped[List["ResultLayerReinstatement"]] = relationship(
        back_populates="layer", cascade="all, delete-orphan"
    )

    # Get the modelfiles associated to the resultlayer through the association resultlayer_modelfile
    modelfiles: Mapped[List["ModelFile"]] = relationship(
        secondary=lambda: resultlayer_modelfile
    )

    # Define the 1-to-many relationship between ResultLayer and ResultLayerStatisticLoss
    percentilelosses: Mapped[List["ResultLayerStatisticLoss"]] = relationship(
        back_populates="layer", cascade="all, delete-orphan"
    )


class ResultLayerReinstatement(CommonMixin, LayerReinstatementMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)

    # Define the 1-to-many relationship between ResultLayer and ResultLayerReinstatement
    layer_id: Mapped[int] = mapped_column(ForeignKey("resultlayer.id"))
    layer: Mapped["ResultLayer"] = relationship(back_populates="reinstatements")


resultlayer_modelfile: Final[Table] = Table(
    "resultlayer_modelfile",
    Base.metadata,
    Column("resultlayer_id", ForeignKey("resultlayer.id"), primary_key=True),
    Column("modelfile_id", ForeignKey("modelfile.id"), primary_key=True),
)


class ResultLayerStatisticLoss(CommonMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    statistic: Mapped[str] = mapped_column(String(50))
    # statistic = AEP/OEP/Expected Loss Agg/Expected Loss Occ
    percentile: Mapped[Optional[float]]
    loss: Mapped[int]

    # Define the 1-to-many relationship between ResultLayer and ResultLayerStatisticLoss
    layer_id: Mapped[int] = mapped_column(ForeignKey("resultlayer.id"))
    layer: Mapped["ResultLayer"] = relationship(back_populates="percentilelosses")


# Create an engine connected to a SQLite database
engine = create_engine("sqlite://")

# Create the database tables
Base.metadata.create_all(engine)
