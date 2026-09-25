"""Base model for everything the API sends or receives."""

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class ApiModel(BaseModel):
    """JSON field names are camelCase on the wire and snake_case in Python.

    The contract describes one schema per model for both directions
    (`separate_input_output_schemas=False`), so a field with a default is
    optional there. Response models therefore set every field explicitly.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, frozen=True)
