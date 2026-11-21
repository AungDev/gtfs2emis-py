"""gtfs2emis-py: estimate public transport emissions from GTFS data.

Python port of the R `gtfs2emis` package (https://github.com/ipeaGIT/gtfs2emis).

Copyright (c) 2026 gtfs2emis-py contributors.

SPDX-License-Identifier: MIT

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__version__ = "0.1.0"

from gtfs2emis_py.analysis import emis_grid, emis_summary
from gtfs2emis_py.emission_model import (  # noqa: E402
    EF_MODELS,
    EmissionModelError,
    EmissionResult,
    emission_model,
)
from gtfs2emis_py.emission_writer import (
    EmissionManifest,
    EmissionWriteError,
    read_emission_manifest,
    write_emission_output,
)
from gtfs2emis_py.fleet import FleetError, FleetTable, load_fleet  # noqa: E402
from gtfs2emis_py.gtfs.writer import ManifestError, TransportWriteError  # noqa: E402
from gtfs2emis_py.multiply import MultiplyError, multiply_ef  # noqa: E402
from gtfs2emis_py.transport_model import (  # noqa: E402
    BUS_ROUTE_TYPES,
    ShapeResult,
    TransportManifest,
    TransportModel,
    TransportModelError,
    transport_model,
)

__all__ = [
    "BUS_ROUTE_TYPES",
    "EF_MODELS",
    "EmissionManifest",
    "EmissionModelError",
    "EmissionResult",
    "EmissionWriteError",
    "FleetError",
    "FleetTable",
    "ManifestError",
    "MultiplyError",
    "ShapeResult",
    "TransportManifest",
    "TransportModel",
    "TransportModelError",
    "TransportWriteError",
    "__version__",
    "emis_grid",
    "emis_summary",
    "emission_model",
    "load_fleet",
    "multiply_ef",
    "read_emission_manifest",
    "write_emission_output",
    "transport_model",
]
