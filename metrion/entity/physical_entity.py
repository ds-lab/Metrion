from metrion.entity.entity import Entity


class PhysicalEntity(Entity):
    """
    Represents a physical entity that can generate measurements for logical entites or can have own measurements.
    """

    def __init__(
        self,
        id: str | None = None,
        parent_id: str | None = None,
    ):
        """
        Initializes a PhysicalEntity instance.
        """
        super().__init__(id=id, parent_id=parent_id)
