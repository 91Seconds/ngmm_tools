from dataclasses import dataclass
from typing import ClassVar, Iterable, List, Tuple

import numpy as np
import pandas as pd
import pyproj


def get_utmProj(utm_zone: str):
    if utm_zone.upper().endswith("S"):
        is_south = " +south"
    else:
        is_south = ""
    if pyproj.__version__ == "2.6.1post1":
        utmProj = pyproj.Proj(
            "+proj=utm +zone="
            + utm_zone
            + is_south
            + ", +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        )
    else:
        utmProj = pyproj.Proj(
            "proj=utm zone=" + utm_zone[0:-1] + is_south,
            " ellps=WGS84 datum=WGS84 units=m no_defs",
        )
    return utmProj


@dataclass
class Cells:
    """
    cell schematic
       (7)-----(8)  (surface, top face)
      / |     / |
    (5)-----(6) |
     |  |    |  |
     | (3)---|-(4)  (bottom face)
     |/      |/
    (1)-----(2)
    """

    utm_zone: str
    cells: np.array
    cell_ids: List[int]
    cell_names: List[str]
    grid_lims_x: Tuple[float, float]
    grid_lims_y: Tuple[float, float]
    grid_lims_z: Tuple[float, float]
    cell_size: Tuple[float, float, float]
    cell_q_names: ClassVar[List[str]] = [
        "q1X",
        "q1Y",
        "q1Z",
        "q2X",
        "q2Y",
        "q2Z",
        "q3X",
        "q3Y",
        "q3Z",
        "q4X",
        "q4Y",
        "q4Z",
        "q5X",
        "q5Y",
        "q5Z",
        "q6X",
        "q6Y",
        "q6Z",
        "q7X",
        "q7Y",
        "q7Z",
        "q8X",
        "q8Y",
        "q8Z",
        "mptX",
        "mptY",
        "mptZ",
    ]

    def to_dataframe(self):
        utmProj = get_utmProj(utm_zone=self.utm_zone)
        # cell names
        df_data1 = pd.DataFrame({"cellid": self.cell_ids, "cellname": self.cell_names})
        # cell coordinates
        df_data2 = pd.DataFrame(self.cells, columns=self.cell_q_names)
        df_cellinfo = pd.merge(df_data1, df_data2, left_index=True, right_index=True)
        del df_data1, df_data2
        # add cell utm zone
        df_cellinfo.loc[:, "UTMzone"] = self.utm_zone

        # cell lat\lon verticies
        for q in range(1, 9):
            c_X = ["q%iX" % q, "q%iY" % q]
            c_latlon = ["q%iLat" % q, "q%iLon" % q]

            df_cellinfo.loc[:, c_latlon] = np.flip(
                np.array(
                    [
                        utmProj(pt_xy[0] * 1e3, pt_xy[1] * 1e3, inverse=True)
                        for _, pt_xy in df_cellinfo[c_X].iterrows()
                    ]
                ),
                axis=1,
            )
        # cell midpoints
        c_X = ["mptX", "mptY"]
        c_latlon = ["mptLat", "mptLon"]
        df_cellinfo.loc[:, c_latlon] = np.flip(
            np.array(
                [
                    utmProj(pt_xy[0] * 1e3, pt_xy[1] * 1e3, inverse=True)
                    for _, pt_xy in df_cellinfo[c_X].iterrows()
                ]
            ),
            axis=1,
        )
        return df_cellinfo

    @classmethod
    def from_bounds(
        cls,
        utm_zone: str,
        grid_lims_x: Iterable[float],
        grid_lims_y: Iterable[float],
        grid_lims_z: Iterable[float],
        cell_size: Iterable[float],
    ):
        # grid spacing
        grid_x = np.arange(grid_lims_x[0], grid_lims_x[1] + 0.1, cell_size[0])
        grid_y = np.arange(grid_lims_y[0], grid_lims_y[1] + 0.1, cell_size[1])
        grid_z = np.arange(grid_lims_z[0], grid_lims_z[1] + 0.1, cell_size[2])

        # create cells
        cells = []
        for j1 in range(len(grid_x) - 1):
            for j2 in range(len(grid_y) - 1):
                for j3 in range(len(grid_z) - 1):
                    # cell corners (bottom-face)
                    cell_c1 = [grid_x[j1], grid_y[j2], grid_z[j3]]
                    cell_c2 = [grid_x[j1 + 1], grid_y[j2], grid_z[j3]]
                    cell_c3 = [grid_x[j1], grid_y[j2 + 1], grid_z[j3]]
                    cell_c4 = [grid_x[j1 + 1], grid_y[j2 + 1], grid_z[j3]]
                    # cell corners (top-face)
                    cell_c5 = [grid_x[j1], grid_y[j2], grid_z[j3 + 1]]
                    cell_c6 = [grid_x[j1 + 1], grid_y[j2], grid_z[j3 + 1]]
                    cell_c7 = [grid_x[j1], grid_y[j2 + 1], grid_z[j3 + 1]]
                    cell_c8 = [grid_x[j1 + 1], grid_y[j2 + 1], grid_z[j3 + 1]]
                    # cell center
                    cell_cent = np.mean(
                        np.stack(
                            [
                                cell_c1,
                                cell_c2,
                                cell_c3,
                                cell_c4,
                                cell_c5,
                                cell_c6,
                                cell_c7,
                                cell_c8,
                            ]
                        ),
                        axis=0,
                    ).tolist()
                    # summarize all cell coordinates in a list
                    cell_info = (
                        cell_c1
                        + cell_c2
                        + cell_c3
                        + cell_c4
                        + cell_c5
                        + cell_c6
                        + cell_c7
                        + cell_c8
                        + cell_cent
                    )
                    # add cell info
                    cells.append(cell_info)
        del j1, j2, j3, cell_info
        del cell_c1, cell_c2, cell_c3, cell_c4, cell_c5, cell_c6, cell_c7, cell_c8
        cells = np.array(cells)
        n_cells = len(cells)

        # cell info
        cell_ids = np.arange(n_cells)
        cell_names = ["c.%i" % (i) for i in cell_ids]
        return cls(
            utm_zone=utm_zone,
            grid_lims_x=grid_lims_x,
            grid_lims_y=grid_lims_y,
            grid_lims_z=grid_lims_z,
            cell_size=cell_size,
            cells=cells,
            cell_ids=cell_ids,
            cell_names=cell_names,
        )


def get_source_station_matrix(cells: Cells, ground_motions: pd.DataFrame):
    # create matrix with source and station locations
    data4celldist = ground_motions.loc[:, ["eqX", "eqY", "eqZ", "staX", "staY"]].values
    # add column for elevation for stations, assume 0
    data4celldist = np.hstack([data4celldist, np.zeros([len(data4celldist), 1])])

    # check that all coordinates are inside the grid
    assert np.logical_and(
        data4celldist[:, [0, 3]].min() >= cells.grid_lims_x[0],
        data4celldist[:, [0, 3]].max() <= cells.grid_lims_x[1],
    ), "Error. Events or Sations outside grid cell in x direction."
    assert np.logical_and(
        data4celldist[:, [1, 4]].min() >= cells.grid_lims_y[0],
        data4celldist[:, [1, 4]].max() <= cells.grid_lims_y[1],
    ), "Error. Events or Sations outside grid cell in y direction."
    assert np.logical_and(
        data4celldist[:, [2, 5]].min() >= cells.grid_lims_z[0],
        data4celldist[:, [2, 5]].max() <= cells.grid_lims_z[1],
    ), "Error. Events or Sations outside grid cell in z direction."
    return data4celldist
