import numpy as np
from itertools import compress
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

class Dist_matrix:
    def __init__(self, variant):
        self.variant = variant
        if self.variant == 0 or self.variant == 1:
            # cell indices:
            # |  0 |  1 |  2 |  3 |  4 |
            # |  5 |  6 |  7 |  8 |  9 |
            # | 10 | 11 | 12 | 13 | 14 |
            # | 15 | 16 | 17 | 18 | 19 |
            # | 20 | 21 | 22 | 23 | 24 |

            neighbor_matrix = np.zeros((25, 25), int)
            for i in range(25):
                for j in range(i+1, 25):
                    i_vert = int(i / 5)
                    i_hori = i % 5
                    j_vert = int(j / 5)
                    j_hori = j % 5
                    dist_vert = j_vert - i_vert
                    dist_hori = j_hori - i_hori
                    if (dist_vert == 0 and dist_hori == 1) or (dist_vert == 1 and dist_hori == 0):
                        neighbor_matrix[i, j] = 1

        else:
            # cell indices:
            # |  0 |  X |  7 | 12 | 13 |
            # |  1 |  X |  8 |  X | 14 |
            # |  2 |  X |  9 |  X | 15 |
            # |  3 |  5 | 10 |  X | 16 |
            # |  4 |  6 | 11 |  X | 17 |

            self.mapping = [(0,0), (1,0), (2,0), (3,0), (4,0),
                    (3,1), (4,1),
                    (0,2), (1,2), (2,2), (3,2), (4,2),
                    (0,3),
                    (0,4), (1,4), (2,4), (3,4), (4,4)]

            neighbor_matrix = np.zeros((18, 18), int)
            neighbor_matrix[0, 1] = 1
            neighbor_matrix[1, 2] = 1
            neighbor_matrix[2, 3] = 1
            neighbor_matrix[3, 4] = 1
            neighbor_matrix[3, 5] = 1
            neighbor_matrix[4, 6] = 1
            neighbor_matrix[5, 6] = 1
            neighbor_matrix[5, 10] = 1
            neighbor_matrix[6, 11] = 1
            neighbor_matrix[7, 8] = 1
            neighbor_matrix[7, 12] = 1
            neighbor_matrix[8, 9] = 1
            neighbor_matrix[9, 10] = 1
            neighbor_matrix[10, 11] = 1
            neighbor_matrix[12, 13] = 1
            neighbor_matrix[13, 14] = 1
            neighbor_matrix[14, 15] = 1
            neighbor_matrix[15, 16] = 1
            neighbor_matrix[16, 17] = 1

        # run Dijkstra's algorithm
        graph = csr_matrix(neighbor_matrix)
        self.dist_matrix, self.predecessors = dijkstra(csgraph=graph, directed=False, return_predecessors=True, unweighted=True)

    def convert_coord_to_idx(self, coord):
        if self.variant == 0 or self.variant == 1:
            idx = coord[0] * 5 + coord[1]
        else:
            idx = self.mapping.index(coord)
        return idx

    # convert cell index z to the corresponding coordinates (x, y)
    def convert_idx_to_coord(self, idx):
        if self.variant == 0 or self.variant == 1:
            coord_vert = int(idx / 5)
            coord_hori = idx % 5
            coord = (coord_vert, coord_hori)
        else:
            coord = self.mapping[idx]
        return coord

    # get the distance between two cells given their coordinates based on the previously computed distance matrix
    def get_dist_from_coord(self, coord1, coord2):
        idx1 = self.convert_coord_to_idx(coord1)
        idx2 = self.convert_coord_to_idx(coord2)
        return self.dist_matrix[idx1, idx2]